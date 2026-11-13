"""Phase 3: extractive, selective sentence-level compression (RECOMP).

Purely extractive: sentences are kept verbatim or dropped, never rewritten.
Each sentence is scored by inner product with the query using the retriever's
own encoder (title-prefixed, as RECOMP does for decontextualization).

With an optional cross-encoder ``reranker``, the bi-encoder only shortlists
the top ``rerank_top_n`` sentences, and selection runs on reranker scores.
With ``rerank_context`` the reranker reads each shortlisted sentence together
with its previous and next sentence in the document; the output is still the
sentence alone.

An optional ``second_reranker`` (a larger cross-encoder) then rescores only
the first reranker's best ``second_top_n`` sentences, and selection runs on
its scores: a cascade spends the expensive model on the few sentences that
can still make the cut.

Thresholds are on their scorer's scale (bi-encoder inner product, or
reranker probability), so they are model-specific and must be calibrated on
data; defaults are calibrated for bge-small-en-v1.5 and the distilled context
reranker (INT8).
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
    sentence_threshold: float | None,
    empty_threshold: float,
    min_sentences: int = 1,
    max_sentences: int | None = None,
) -> list[int]:
    """Indices of kept sentences, best first (ties keep input order).

    ``sentence_threshold=None`` keeps every sentence up to ``max_sentences``.
    Pure function so the evaluation harness sweeps exactly the production rule.
    """
    if len(scores) == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    if scores[order[0]] <= empty_threshold:
        return []
    passing = len(scores) if sentence_threshold is None else int((scores >= sentence_threshold).sum())
    keep = max(passing, min_sentences)
    if max_sentences is not None:
        keep = min(keep, max_sentences)
    return [int(i) for i in order[:keep]]


def with_neighbors(sentence: str, previous: str, following: str) -> str:
    """Reranker input under ``rerank_context``: the sentence, then its document
    neighbors ("" for a missing one). The context reranker is trained on
    exactly this format (evaluation/train_distill.py)."""
    around = " ... ".join(s for s in (previous, following) if s)
    return f"{sentence}\nContext: {around}" if around else sentence


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
        sentence_threshold: float | None = None,
        empty_threshold: float = 0.64,
        min_sentences: int = 1,
        max_sentences: int | None = 3,
        sentence_splitter: SentenceSplitter | None = None,
        reranker: Reranker | None = None,
        rerank_top_n: int = 20,
        rerank_empty_threshold: float = 0.01,
        rerank_context: bool = True,
        second_reranker: Reranker | None = None,
        second_top_n: int = 8,
        second_context: bool = False,
    ) -> None:
        if min_sentences < 0:
            raise ValueError("min_sentences must be non-negative")
        if max_sentences is not None and max_sentences < max(min_sentences, 1):
            raise ValueError("max_sentences must be >= max(min_sentences, 1)")
        if rerank_top_n <= 0:
            raise ValueError("rerank_top_n must be positive")
        if second_top_n <= 0:
            raise ValueError("second_top_n must be positive")
        if second_reranker is not None and reranker is None:
            raise ValueError("a second_reranker needs a first reranker")
        self.encoder = encoder
        self.sentence_threshold = sentence_threshold
        self.empty_threshold = empty_threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.rerank_empty_threshold = rerank_empty_threshold
        self.rerank_context = rerank_context
        self.second_reranker = second_reranker
        self.second_top_n = second_top_n
        self.second_context = second_context
        self._split = sentence_splitter or pysbd_splitter()
        # Ingest-time caches (see precompute): sentence texts per chunk, and
        # bi-encoder embeddings per (title, sentence). Pure optimizations:
        # anything missing is computed on demand with identical results.
        self._chunk_sentences: dict[tuple[str, str], list[str]] = {}
        self._embeddings: dict[tuple[str, str], NDArray[np.float32]] = {}
        # Ingest-time index for rerank_context: (previous, next) sentence per
        # (document, sentence). Not recomputable at query time, when only the
        # retrieved chunks are seen: a sentence missing here is reranked alone.
        self._around: dict[tuple[str, str], tuple[str, str]] = {}
        self._position: dict[tuple[str, str], int] = {}  # ingest order, for the output layout

    def precompute(self, chunks: Sequence[TextSegment]) -> None:
        """Split and embed ``chunks``' sentences ahead of queries (call at ingest).

        Moves sentence segmentation and embedding, the bulk of compress()'s
        cost, out of the query path; only the optional reranker then runs per
        query. Also records each sentence's document neighbors for
        ``rerank_context``, so pass a document's chunks together, in order.
        """
        sentences = self.split(chunks)
        self._embed(sentences)
        for i, s in enumerate(sentences):
            self._position.setdefault((s.source_doc, s.content), len(self._position))
            before = sentences[i - 1] if i > 0 else None
            after = sentences[i + 1] if i + 1 < len(sentences) else None
            self._around.setdefault(
                (s.source_doc, s.content),
                (
                    before.content if before is not None and before.source_doc == s.source_doc else "",
                    after.content if after is not None and after.source_doc == s.source_doc else "",
                ),
            )

    def clear_cache(self) -> None:
        self._chunk_sentences.clear()
        self._embeddings.clear()
        self._around.clear()
        self._position.clear()

    def compress(
        self, query_embedding: Embedding, chunks: Sequence[TextSegment], query: str | None = None
    ) -> list[TextSegment]:
        """Return kept sentences, best first, each carrying its ``score`` (and
        its ``position`` in the document, when the document was precomputed).

        ``query_embedding`` comes from the same encoder's ``encode_query``, so
        the engine embeds the query once for both retrieval and compression.
        ``query`` (the text) is required only when a reranker is configured.

        - If even the best sentence's bi-encoder score is ``<= empty_threshold``,
          return ``[]`` (selective augmentation: let the LLM answer from
          parametric knowledge). This is the off-topic rule; with a reranker it
          runs first, so such queries never pay for reranking.
        - With a reranker, also return ``[]`` when its best score is
          ``<= rerank_empty_threshold``.
        - With a second reranker, only the first one's best ``second_top_n``
          sentences remain candidates, scored by the second.
        - Otherwise keep the best ``max_sentences``; with a ``sentence_threshold``,
          drop those below it, keeping at least the top ``min_sentences``.
        """
        sentences = self.split(chunks)
        if not sentences:
            return []
        scores = self._embed(sentences) @ query_embedding
        empty_threshold = self.empty_threshold
        if self.reranker is not None:
            if query is None:
                raise ValueError("compress() needs the query text when a reranker is set")
            if scores.max() <= self.empty_threshold:
                return []
            shortlist = np.argsort(-scores, kind="stable")[: self.rerank_top_n]
            sentences = [sentences[i] for i in shortlist]
            scores = self.reranker.score(query, self._in_context(sentences) if self.rerank_context else sentences)
            empty_threshold = self.rerank_empty_threshold
            if self.second_reranker is not None and scores.max() > empty_threshold:
                head = np.argsort(-scores, kind="stable")[: self.second_top_n]
                sentences = [sentences[i] for i in head]
                scores = self.second_reranker.score(
                    query, self._in_context(sentences) if self.second_context else sentences
                )
                empty_threshold = -np.inf  # emptiness was the first reranker's call
        kept = select_sentences(
            scores,
            self.sentence_threshold,
            empty_threshold,
            self.min_sentences,
            self.max_sentences,
        )
        return [
            replace(
                sentences[i],
                score=float(scores[i]),
                position=self._position.get((sentences[i].source_doc, sentences[i].content)),
            )
            for i in kept
        ]

    def split(self, chunks: Sequence[TextSegment]) -> list[TextSegment]:
        """Split chunks into sentences, dropping repeats from chunk overlap.

        The first occurrence wins, so a sentence keeps the id of the
        highest-ranked chunk it appeared in.
        """
        seen: set[tuple[str, str]] = set()
        sentences: list[TextSegment] = []
        for chunk in chunks:
            for j, text in enumerate(self._sentences_of(chunk)):
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

    def _in_context(self, sentences: Sequence[TextSegment]) -> list[TextSegment]:
        """Copies for the reranker to read: each sentence with its document neighbors."""
        return [
            replace(s, content=with_neighbors(s.content, *self._around.get((s.source_doc, s.content), ("", ""))))
            for s in sentences
        ]

    def _sentences_of(self, chunk: TextSegment) -> list[str]:
        key = (chunk.segment_id, chunk.content)
        if key not in self._chunk_sentences:
            self._chunk_sentences[key] = [
                chunk.content[start:end] for start, end in sentence_spans(chunk.content, self._split)
            ]
        return self._chunk_sentences[key]

    def _embed(self, sentences: Sequence[TextSegment]) -> NDArray[np.float32]:
        """Embeddings for ``sentences`` from the cache, encoding (and caching) misses."""
        missing = list({_key(s): s for s in sentences if _key(s) not in self._embeddings}.values())
        if missing:
            self._embeddings.update(zip(map(_key, missing), self.encoder.embed_segments(missing)))
        if not sentences:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self._embeddings[_key(s)] for s in sentences])


def _key(segment: TextSegment) -> tuple[str, str]:
    """What a sentence's embedding depends on: its title and its text."""
    return (segment.title, segment.content)
