"""Expensive phase: run the real models once per question, cache what the sweep needs.

For every chunking config, each question's 20 passages are chunked with the
production preprocessor, the best chunks are retrieved by inner product, and
the production compressor splits them into candidate sentences. We store each
candidate's bi-encoder score, cross-encoder and reader scores (bi-encoder
shortlist only), token count and whether it contains an answer, so selection rules (scorer,
thresholds, caps, budgets) can later be swept without any model.

Model outputs are computed once per unique (title, text) per question, so
chunk configs share the cost of their (identical) sentences.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
from .reader import ONNXExtractiveReader
from .nq_data import Example


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    name: str
    size: int
    overlap: int


# "doc" (size 512 > any passage) keeps each ~130-token passage whole.
CHUNK_CONFIGS = [
    ChunkConfig("doc", 512, 0),
    ChunkConfig("120", 120, 0),
    ChunkConfig("120+ov", 120, 30),
    ChunkConfig("96", 96, 0),
    ChunkConfig("96+ov", 96, 24),
    ChunkConfig("64", 64, 0),
    ChunkConfig("64+ov", 64, 16),
]

# Every config retrieves its best chunks up to this many tokens (instead of a
# fixed K), so all configs hand the compressor equal amounts of text and
# differ only in how it was chunked.
RETRIEVAL_TOKENS = 1200
# The cross-encoder and reader score this many bi-encoder-best sentences; the
# sweep can then emulate any production ``rerank_top_n`` up to this value.
RERANK_TOP_N = 50

_Key = tuple[str, str]  # (title, content)


@dataclass(frozen=True, slots=True)
class Candidates:
    """One question under one chunk config; sentence arrays share an index."""

    original_tokens: int  # tokens in the retrieved chunks
    gold_retrieved: bool  # a chunk from the gold passage was retrieved
    answer_in_retrieved: bool  # an answer string occurs in the retrieved chunks
    scores: NDArray[np.float32]  # bi-encoder sentence-query inner products
    rerank_scores: NDArray[np.float32]  # cross-encoder scores; NaN if not shortlisted
    reader_scores: NDArray[np.float32] | None  # reader margins; NaN if not shortlisted; None if not computed
    tokens: NDArray[np.int32]
    has_answer: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    qid: int
    kind: str  # see nq_data.Example.kind
    by_config: dict[str, Candidates]

    @property
    def answerable(self) -> bool:
        return self.kind == "answerable"


def build(
    examples: Iterable[Example],
    model_dir: Path,
    reranker_dir: Path,
    reader_dir: Path,
    configs: list[ChunkConfig] = CHUNK_CONFIGS,
    threads: int = 8,
    log: Callable[[str], None] = print,
) -> list[QuestionRecord]:
    encoder = ONNXBiEncoderRetriever.from_pretrained(
        model_dir, query_prefix=BGE_QUERY_PREFIX, intra_op_num_threads=threads
    )
    second_stage: dict[str, Reranker] = {
        "rerank": ONNXCrossEncoderReranker.from_pretrained(reranker_dir, intra_op_num_threads=threads),
        "reader": ONNXExtractiveReader.from_pretrained(reader_dir, intra_op_num_threads=threads),
    }
    preprocessors = {
        c.name: MarkItDownPreprocessor.from_pretrained(model_dir, c.size, c.overlap) for c in configs
    }
    count_tokens = next(iter(preprocessors.values())).count_tokens
    splitter = SelectiveSentenceCompressor(encoder)  # only its split() is used

    records = []
    seconds = {name: 0.0 for name in second_stage}
    calls = 0
    start = time.perf_counter()
    examples = list(examples)
    for n, ex in enumerate(examples, 1):
        bi = _BiScores(encoder, encoder.encode_query(ex.question))
        retrieved, sentences = {}, {}
        for c in configs:
            chunks = [
                chunk
                for j, doc in enumerate(ex.docs)
                for chunk in preprocessors[c.name].chunk(doc.text, f"d{j}", title=doc.title)
            ]
            retrieved[c.name] = _retrieve(chunks, bi.scores(chunks), count_tokens)
            sentences[c.name] = splitter.split(retrieved[c.name])
        bi.scores([s for c in configs for s in sentences[c.name]])  # one batched pass

        gold = {f"d{j}" for j, d in enumerate(ex.docs) if d.is_gold}
        stage_scores: dict[str, dict[_Key, float]] = {name: {} for name in second_stage}
        by_config = {}
        for c in configs:
            top_chunks, sents = retrieved[c.name], sentences[c.name]
            text = "\n\n".join(ch.content for ch in top_chunks)
            bi_scores = bi.scores(sents)
            shortlist = [sents[i] for i in np.argsort(-bi_scores, kind="stable")[:RERANK_TOP_N]]
            todo = [s for s in shortlist if _key(s) not in stage_scores["rerank"]]
            if todo:
                calls += 1
                for name, model in second_stage.items():
                    t = time.perf_counter()
                    values = model.score(ex.question, todo)
                    seconds[name] += time.perf_counter() - t
                    stage_scores[name].update(zip(map(_key, todo), map(float, values)))
            listed = {_key(s) for s in shortlist}
            by_config[c.name] = Candidates(
                original_tokens=count_tokens(text),
                gold_retrieved=any(ch.source_doc in gold for ch in top_chunks),
                answer_in_retrieved=ex.answerable and contains_answer(text, ex.answers),
                scores=bi_scores,
                rerank_scores=_shortlisted(stage_scores["rerank"], sents, listed),
                reader_scores=_shortlisted(stage_scores["reader"], sents, listed),
                tokens=np.array([count_tokens(s.content) for s in sents], dtype=np.int32),
                has_answer=np.array(
                    [ex.answerable and contains_answer(s.content, ex.answers) for s in sents],
                    dtype=np.bool_,
                ),
            )
        records.append(QuestionRecord(ex.qid, ex.kind, by_config))
        if n % 50 == 0 or n == len(examples):
            rate = (time.perf_counter() - start) / n
            log(f"  {n}/{len(examples)}  {rate:.2f}s/question  ~{rate * (len(examples) - n) / 60:.0f} min left")
    for name, total in seconds.items():
        if calls:
            log(f"{name}: {1000 * total / calls:.0f} ms per call (<= {RERANK_TOP_N} sentences)")
    return records


def _shortlisted(
    scores: dict[_Key, float], sentences: list[TextSegment], listed: set[_Key]
) -> NDArray[np.float32]:
    return np.array(
        [scores[_key(s)] if _key(s) in listed else np.nan for s in sentences], dtype=np.float32
    )


def _key(segment: TextSegment) -> _Key:
    return (segment.title, segment.content)


def _retrieve(
    chunks: list[TextSegment], scores: NDArray[np.float32], count_tokens: Callable[[str], int]
) -> list[TextSegment]:
    """Best chunks until RETRIEVAL_TOKENS would be exceeded (at least one)."""
    out: list[TextSegment] = []
    used = 0
    for i in np.argsort(-scores, kind="stable"):
        cost = count_tokens(chunks[i].content)
        if out and used + cost > RETRIEVAL_TOKENS:
            break
        out.append(chunks[i])
        used += cost
    return out


class _BiScores:
    """Inner products with one query, embedding each unique (title, text) once."""

    def __init__(self, encoder: ONNXBiEncoderRetriever, query: NDArray[np.float32]) -> None:
        self._encoder = encoder
        self._query = query
        self._scores: dict[_Key, float] = {}

    def scores(self, segments: list[TextSegment]) -> NDArray[np.float32]:
        missing = list({_key(s): s for s in segments if _key(s) not in self._scores}.values())
        if missing:
            values = self._encoder.embed_segments(missing) @ self._query
            self._scores.update(zip(map(_key, missing), map(float, values)))
        return np.array([self._scores[_key(s)] for s in segments], dtype=np.float32)


def save(records: list[QuestionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f)


def load(path: Path) -> list[QuestionRecord]:
    with path.open("rb") as f:
        records: list[QuestionRecord] = pickle.load(f)
    return records
