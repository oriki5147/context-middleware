"""Phase 3: extractive, selective sentence-level compression (RECOMP).

Purely extractive: sentences are kept verbatim or dropped, never rewritten.
Each sentence is scored by inner product with the query using the retriever's
own encoder (title-prefixed, as RECOMP does for decontextualization).

With an optional cross-encoder ``reranker``, the bi-encoder only shortlists
the top ``rerank_top_n`` sentences, and selection runs on reranker scores.

Thresholds are on the active scorer's scale (bi-encoder inner product, or
reranker probability), so they are model-specific and must be calibrated on
data; the defaults are placeholders.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .models import Embedding, TextSegment
from .sentences import SentenceSplitter, pysbd_splitter, sentence_spans


def select_sentences(
    scores: NDArray[np.float32],
    sentence_threshold: float,
    empty_threshold: float,
    min_sentences: int = 1,
    max_sentences: int | None = None,
) -> list[int]:
    """Indices of kept sentences, best first (ties keep input order).

    Pure function so the evaluation harness sweeps exactly the production rule.
    """
    if len(scores) == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    if scores[order[0]] <= empty_threshold:
        return []
    keep = max(int((scores >= sentence_threshold).sum()), min_sentences)
    if max_sentences is not None:
        keep = min(keep, max_sentences)
    return [int(i) for i in order[:keep]]


class SegmentEncoder(Protocol):
    """What the compressor needs from ``ONNXBiEncoderRetriever``."""

    def embed_segments(self, segments: Sequence[TextSegment]) -> NDArray[np.float32]: ...


class Reranker(Protocol):
    """What the compressor needs from ``ONNXCrossEncoderReranker``."""

    def score(self, query: str, segments: Sequence[TextSegment]) -> NDArray[np.float32]: ...


class SelectiveSentenceCompressor:
    def __init__(
        self,
        encoder: SegmentEncoder,
        sentence_threshold: float = 0.65,
        empty_threshold: float = 0.45,
        min_sentences: int = 1,
        max_sentences: int | None = None,
        sentence_splitter: SentenceSplitter | None = None,
        reranker: Reranker | None = None,
        rerank_top_n: int = 20,
    ) -> None:
        if empty_threshold > sentence_threshold:
            raise ValueError("empty_threshold must not exceed sentence_threshold")
        if min_sentences < 0:
            raise ValueError("min_sentences must be non-negative")
        if max_sentences is not None and max_sentences < max(min_sentences, 1):
            raise ValueError("max_sentences must be >= max(min_sentences, 1)")
        if rerank_top_n <= 0:
            raise ValueError("rerank_top_n must be positive")
        self.encoder = encoder
        self.sentence_threshold = sentence_threshold
        self.empty_threshold = empty_threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self._split = sentence_splitter or pysbd_splitter()

    def compress(
        self, query_embedding: Embedding, chunks: Sequence[TextSegment], query: str | None = None
    ) -> list[TextSegment]:
        """Return kept sentences, best first, each carrying its ``score``.

        ``query_embedding`` comes from the same encoder's ``encode_query``, so
        the engine embeds the query once for both retrieval and compression.
        ``query`` (the text) is required only when a reranker is configured.

        - If even the best sentence scores ``<= empty_threshold``, return ``[]``
          (selective augmentation: let the LLM answer from parametric knowledge).
        - Otherwise keep every sentence scoring ``>= sentence_threshold``, but at
          least the top ``min_sentences`` and at most ``max_sentences``.
        """
        sentences = self.split(chunks)
        if not sentences:
            return []
        scores = self.encoder.embed_segments(sentences) @ query_embedding
        if self.reranker is not None:
            if query is None:
                raise ValueError("compress() needs the query text when a reranker is set")
            shortlist = np.argsort(-scores, kind="stable")[: self.rerank_top_n]
            sentences = [sentences[i] for i in shortlist]
            scores = self.reranker.score(query, sentences)
        kept = select_sentences(
            scores,
            self.sentence_threshold,
            self.empty_threshold,
            self.min_sentences,
            self.max_sentences,
        )
        return [replace(sentences[i], score=float(scores[i])) for i in kept]

    def split(self, chunks: Sequence[TextSegment]) -> list[TextSegment]:
        """Split chunks into sentences, dropping repeats from chunk overlap.

        The first occurrence wins, so a sentence keeps the id of the
        highest-ranked chunk it appeared in.
        """
        seen: set[tuple[str, str]] = set()
        sentences: list[TextSegment] = []
        for chunk in chunks:
            for j, (start, end) in enumerate(sentence_spans(chunk.content, self._split)):
                text = chunk.content[start:end]
                if (chunk.source_doc, text) in seen:
                    continue
                seen.add((chunk.source_doc, text))
                sentences.append(
                    TextSegment(
                        segment_id=f"{chunk.segment_id}:{j}",
                        content=text,
                        source_doc=chunk.source_doc,
                        title=chunk.title,
                    )
                )
        return sentences
