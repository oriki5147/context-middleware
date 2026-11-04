"""Raw-material cache: everything model-derived, nothing selection-specific.

Unlike ``candidates`` (which stores final scores for fixed chunk configs), this
stores per question the texts, document positions, token counts, bi-encoder
embeddings and cross-encoder scores of ALL chunks and sentences of its
passages, under the production chunking (64 tokens, no overlap). Experiments
that change retrieval, scoring or output assembly (BM25 fusion, neighbor
sentences, query rewrites, ...) then run without re-encoding the corpus.

``derive`` reproduces the production pipeline from the raw material and
returns ``Candidates``, so the existing sweep/report code applies unchanged.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from context_middleware import (
    BGE_QUERY_PREFIX,
    MarkItDownPreprocessor,
    ONNXBiEncoderRetriever,
    ONNXCrossEncoderReranker,
    SelectiveSentenceCompressor,
    TextSegment,
)

from context_middleware.compressor import Reranker

from .answers import contains_answer
from .candidates import RETRIEVAL_TOKENS, Candidates
from .nq_data import Example


@dataclass(frozen=True, slots=True)
class RawQuestion:
    qid: int
    kind: str
    question: str
    answers: tuple[str, ...]
    gold_doc: int  # index of the gold passage, -1 for unanswerable kinds
    query_embedding: NDArray[np.float32]
    # Chunks, in document order (doc 0's chunks first).
    chunk_doc: NDArray[np.int32]
    chunk_text: list[str]
    chunk_title: list[str]
    chunk_tokens: NDArray[np.int32]
    chunk_embedding: NDArray[np.float32]
    # Sentences as the production compressor splits each chunk, in document order.
    sent_chunk: NDArray[np.int32]
    sent_doc: NDArray[np.int32]
    sent_pos: NDArray[np.int32]  # 0-based order within its document
    sent_text: list[str]
    sent_tokens: NDArray[np.int32]
    sent_has_answer: NDArray[np.bool_]
    sent_embedding: NDArray[np.float32]
    sent_rerank: NDArray[np.float32]  # cross-encoder score of every sentence


def build(
    examples: Iterable[Example],
    model_dir: Path,
    reranker_dir: Path,
    chunk_size: int = 64,
    overlap: int = 0,
    threads: int = 8,
    log: Callable[[str], None] = print,
) -> list[RawQuestion]:
    encoder = ONNXBiEncoderRetriever.from_pretrained(
        model_dir, query_prefix=BGE_QUERY_PREFIX, intra_op_num_threads=threads
    )
    reranker = ONNXCrossEncoderReranker.from_pretrained(reranker_dir, intra_op_num_threads=threads)
    preprocessor = MarkItDownPreprocessor.from_pretrained(model_dir, chunk_size, overlap)
    splitter = SelectiveSentenceCompressor(encoder)  # only its split() is used

    out = []
    examples = list(examples)
    start = time.perf_counter()
    for n, ex in enumerate(examples, 1):
        chunks = [
            chunk
            for j, doc in enumerate(ex.docs)
            for chunk in preprocessor.chunk(doc.text, f"d{j}", title=doc.title)
        ]
        sentences, sent_chunk, sent_pos = [], [], []
        position: dict[str, int] = {}
        for c, chunk in enumerate(chunks):
            for sentence in splitter.split([chunk]):
                sentences.append(sentence)
                sent_chunk.append(c)
                sent_pos.append(position.get(chunk.source_doc, 0))
                position[chunk.source_doc] = sent_pos[-1] + 1

        def doc_index(source_doc: str) -> int:
            return int(source_doc.removeprefix("d"))

        out.append(
            RawQuestion(
                qid=ex.qid,
                kind=ex.kind,
                question=ex.question,
                answers=ex.answers,
                gold_doc=next((j for j, d in enumerate(ex.docs) if d.is_gold), -1),
                query_embedding=encoder.encode_query(ex.question),
                chunk_doc=np.array([doc_index(c.source_doc) for c in chunks], dtype=np.int32),
                chunk_text=[c.content for c in chunks],
                chunk_title=[c.title for c in chunks],
                chunk_tokens=np.array([preprocessor.count_tokens(c.content) for c in chunks], dtype=np.int32),
                chunk_embedding=encoder.embed_segments(chunks),
                sent_chunk=np.array(sent_chunk, dtype=np.int32),
                sent_doc=np.array([doc_index(s.source_doc) for s in sentences], dtype=np.int32),
                sent_pos=np.array(sent_pos, dtype=np.int32),
                sent_text=[s.content for s in sentences],
                sent_tokens=np.array([preprocessor.count_tokens(s.content) for s in sentences], dtype=np.int32),
                sent_has_answer=np.array(
                    [ex.answerable and contains_answer(s.content, ex.answers) for s in sentences],
                    dtype=np.bool_,
                ),
                sent_embedding=encoder.embed_segments(sentences),
                sent_rerank=reranker.score(ex.question, sentences),
            )
        )
        if n % 50 == 0 or n == len(examples):
            rate = (time.perf_counter() - start) / n
            log(f"  {n}/{len(examples)}  {rate:.2f}s/question  ~{rate * (len(examples) - n) / 60:.0f} min left")
    return out


def derive(
    raw: RawQuestion,
    chunk_scores: NDArray[np.float32] | None = None,
    sentence_scores: NDArray[np.float32] | None = None,
    rerank_scores: NDArray[np.float32] | None = None,
    retrieval_tokens: int = RETRIEVAL_TOKENS,
) -> Candidates:
    """Production pipeline from raw material: retrieve chunks up to a token
    budget, then the retrieved chunks' sentences become the candidates.

    ``chunk_scores`` / ``sentence_scores`` / ``rerank_scores`` (indexed like
    the raw chunks / sentences) override the model scores, e.g. for hybrid
    BM25 fusion; defaults reproduce production exactly.
    """
    if sentence_scores is None:
        sentence_scores = raw.sent_embedding @ raw.query_embedding
    if rerank_scores is None:
        rerank_scores = raw.sent_rerank
    idx, retrieved, used = candidate_indices(raw, chunk_scores, retrieval_tokens)
    text = "\n\n".join(raw.chunk_text[c] for c in retrieved)
    answerable = raw.kind == "answerable"
    return Candidates(
        original_tokens=used,
        gold_retrieved=raw.gold_doc >= 0 and any(int(raw.chunk_doc[c]) == raw.gold_doc for c in retrieved),
        answer_in_retrieved=answerable and contains_answer(text, raw.answers),
        scores=sentence_scores[idx].astype(np.float32),
        rerank_scores=rerank_scores[idx].astype(np.float32),
        reader_scores=None,
        tokens=raw.sent_tokens[idx],
        has_answer=raw.sent_has_answer[idx],
    )


def candidate_indices(
    raw: RawQuestion,
    chunk_scores: NDArray[np.float32] | None = None,
    retrieval_tokens: int = RETRIEVAL_TOKENS,
) -> tuple[NDArray[np.int64], list[int], int]:
    """Raw sentence indices of the candidates (compressor.split order), the
    retrieved chunk indices (best first), and the retrieved token count."""
    if chunk_scores is None:
        chunk_scores = raw.chunk_embedding @ raw.query_embedding
    retrieved: list[int] = []
    used = 0
    for c in np.argsort(-chunk_scores, kind="stable"):
        if retrieved and used + int(raw.chunk_tokens[c]) > retrieval_tokens:
            break
        retrieved.append(int(c))
        used += int(raw.chunk_tokens[c])

    # Candidates in compressor.split order: by retrieved chunk, then position,
    # dropping repeats of the same text within a document.
    rank = {c: r for r, c in enumerate(retrieved)}
    seen: set[tuple[int, str]] = set()
    candidates = []
    for i in sorted(
        (i for i in range(len(raw.sent_text)) if int(raw.sent_chunk[i]) in rank),
        key=lambda i: (rank[int(raw.sent_chunk[i])], i),
    ):
        key = (int(raw.sent_doc[i]), raw.sent_text[i])
        if key not in seen:
            seen.add(key)
            candidates.append(i)
    return np.array(candidates, dtype=np.int64), retrieved, used


def rescore(
    raws: Iterable[RawQuestion], reranker: Reranker, log: Callable[[str], None] = print
) -> list[RawQuestion]:
    """Copies of ``raws`` whose ``sent_rerank`` comes from another cross-encoder
    (e.g. the trained answerability gate), so every report runs on it unchanged."""
    raws = list(raws)
    out = []
    for n, r in enumerate(raws, 1):
        sentences = [
            TextSegment(f"s{i}", text, f"d{d}", title=r.chunk_title[c])
            for i, (text, d, c) in enumerate(zip(r.sent_text, r.sent_doc.tolist(), r.sent_chunk.tolist()))
        ]
        out.append(replace(r, sent_rerank=reranker.score(r.question, sentences)))
        if n % 200 == 0 or n == len(raws):
            log(f"  {n}/{len(raws)}")
    return out


def save(raws: list[RawQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(raws, f)


def load(path: Path) -> list[RawQuestion]:
    with path.open("rb") as f:
        raws: list[RawQuestion] = pickle.load(f)
    return raws
