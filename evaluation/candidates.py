"""Expensive phase: run the real encoder once per question, cache what the sweep needs.

For every chunking config, each question's 20 passages are chunked with the
production preprocessor, the top-K chunks are retrieved by inner product, and
the production compressor splits them into candidate sentences. We store each
candidate's score, token count and whether it contains an answer, so selection
rules (thresholds, caps, budgets) can later be swept without the encoder.

Embeddings are computed once per unique (title, text) per question, so chunk
configs share the cost of their (identical) sentences.
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
    SelectiveSentenceCompressor,
    TextSegment,
)

from .answers import contains_answer
from .nq_data import Example


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    name: str
    size: int
    overlap: int
    top_k: int


# K scales inversely with chunk size so every config retrieves ~1,200 tokens.
# "doc" (size 512 > any passage) keeps each ~130-token passage whole.
CHUNK_CONFIGS = [
    ChunkConfig("doc", 512, 0, 10),
    ChunkConfig("120", 120, 0, 10),
    ChunkConfig("120+ov", 120, 30, 10),
    ChunkConfig("96", 96, 0, 13),
    ChunkConfig("96+ov", 96, 24, 13),
    ChunkConfig("64", 64, 0, 19),
    ChunkConfig("64+ov", 64, 16, 19),
]


@dataclass(frozen=True, slots=True)
class Candidates:
    """One question under one chunk config; sentence arrays share an index."""

    original_tokens: int  # tokens in the retrieved top-K chunks
    gold_retrieved: bool  # a chunk from the gold passage is in the top-K
    answer_in_retrieved: bool  # an answer string occurs in the top-K chunks
    scores: NDArray[np.float32]  # sentence-query inner products
    tokens: NDArray[np.int32]
    has_answer: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    qid: int
    answerable: bool
    by_config: dict[str, Candidates]


def build(
    examples: Iterable[Example],
    model_dir: Path,
    configs: list[ChunkConfig] = CHUNK_CONFIGS,
    threads: int = 8,
    log: Callable[[str], None] = print,
) -> list[QuestionRecord]:
    encoder = ONNXBiEncoderRetriever.from_pretrained(
        model_dir, query_prefix=BGE_QUERY_PREFIX, intra_op_num_threads=threads
    )
    preprocessors = {
        c.name: MarkItDownPreprocessor.from_pretrained(model_dir, c.size, c.overlap) for c in configs
    }
    count_tokens = next(iter(preprocessors.values())).count_tokens
    splitter = SelectiveSentenceCompressor(encoder)  # only its split() is used

    records = []
    start = time.perf_counter()
    examples = list(examples)
    for n, ex in enumerate(examples, 1):
        query = encoder.encode_query(ex.question)
        chunked = {
            c.name: [
                chunk
                for j, doc in enumerate(ex.docs)
                for chunk in preprocessors[c.name].chunk(doc.text, f"d{j}", title=doc.title)
            ]
            for c in configs
        }
        retrieved = {}
        sentences = {}
        cache = _ScoreCache(encoder, query)
        for c in configs:
            chunks = chunked[c.name]
            scores = cache.scores(chunks)
            top = np.argsort(-scores, kind="stable")[: c.top_k]
            retrieved[c.name] = [chunks[i] for i in top]
            sentences[c.name] = splitter.split(retrieved[c.name])
        cache.scores([s for c in configs for s in sentences[c.name]])  # one batched pass

        gold = {f"d{j}" for j, d in enumerate(ex.docs) if d.is_gold}
        by_config = {}
        for c in configs:
            top_chunks, sents = retrieved[c.name], sentences[c.name]
            text = "\n\n".join(ch.content for ch in top_chunks)
            by_config[c.name] = Candidates(
                original_tokens=count_tokens(text),
                gold_retrieved=any(ch.source_doc in gold for ch in top_chunks),
                answer_in_retrieved=ex.answerable and contains_answer(text, ex.answers),
                scores=cache.scores(sents),
                tokens=np.array([count_tokens(s.content) for s in sents], dtype=np.int32),
                has_answer=np.array(
                    [ex.answerable and contains_answer(s.content, ex.answers) for s in sents],
                    dtype=np.bool_,
                ),
            )
        records.append(QuestionRecord(ex.qid, ex.answerable, by_config))
        if n % 50 == 0 or n == len(examples):
            rate = (time.perf_counter() - start) / n
            log(f"  {n}/{len(examples)}  {rate:.2f}s/question  ~{rate * (len(examples) - n) / 60:.0f} min left")
    return records


class _ScoreCache:
    """Inner products with one query, embedding each unique (title, text) once."""

    def __init__(self, encoder: ONNXBiEncoderRetriever, query: NDArray[np.float32]) -> None:
        self._encoder = encoder
        self._query = query
        self._scores: dict[tuple[str, str], float] = {}

    def scores(self, segments: list[TextSegment]) -> NDArray[np.float32]:
        missing = list({(s.title, s.content): s for s in segments if (s.title, s.content) not in self._scores}.values())
        if missing:
            values = self._encoder.embed_segments(missing) @ self._query
            self._scores.update(zip(((s.title, s.content) for s in missing), map(float, values)))
        return np.array([self._scores[(s.title, s.content)] for s in segments], dtype=np.float32)


def save(records: list[QuestionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f)


def load(path: Path) -> list[QuestionRecord]:
    with path.open("rb") as f:
        records: list[QuestionRecord] = pickle.load(f)
    return records
