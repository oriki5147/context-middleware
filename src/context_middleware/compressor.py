"""Phase 3: extractive, selective sentence-level compression (RECOMP).

Purely extractive: sentences are kept verbatim or dropped, never rewritten.
Each sentence is scored by inner product with the query using the retriever's
own encoder (title-prefixed, as RECOMP does for decontextualization).

Thresholds are on the encoder's raw score scale, so they are encoder-specific
and must be calibrated on data; the defaults are placeholders.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .models import Embedding, TextSegment
from .sentences import SentenceSplitter, pysbd_splitter, sentence_spans


class SegmentEncoder(Protocol):
    """What the compressor needs from ``ONNXBiEncoderRetriever``."""

    def embed_segments(self, segments: Sequence[TextSegment]) -> NDArray[np.float32]: ...


class SelectiveSentenceCompressor:
    def __init__(
        self,
        encoder: SegmentEncoder,
        sentence_threshold: float = 0.65,
        empty_threshold: float = 0.45,
        min_sentences: int = 1,
        max_sentences: int | None = None,
        sentence_splitter: SentenceSplitter | None = None,
    ) -> None:
        if empty_threshold > sentence_threshold:
            raise ValueError("empty_threshold must not exceed sentence_threshold")
        if min_sentences < 0:
            raise ValueError("min_sentences must be non-negative")
        if max_sentences is not None and max_sentences < max(min_sentences, 1):
            raise ValueError("max_sentences must be >= max(min_sentences, 1)")
        self.encoder = encoder
        self.sentence_threshold = sentence_threshold
        self.empty_threshold = empty_threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self._split = sentence_splitter or pysbd_splitter()

    def compress(
        self, query_embedding: Embedding, chunks: Sequence[TextSegment]
    ) -> list[TextSegment]:
        """Return kept sentences, best first, each carrying its ``score``.

        ``query_embedding`` comes from the same encoder's ``encode_query``, so
        the engine embeds the query once for both retrieval and compression.

        - If even the best sentence scores ``<= empty_threshold``, return ``[]``
          (selective augmentation: let the LLM answer from parametric knowledge).
        - Otherwise keep every sentence scoring ``>= sentence_threshold``, but at
          least the top ``min_sentences`` and at most ``max_sentences``.
        """
        sentences = self.split(chunks)
        if not sentences:
            return []

        scores = self.encoder.embed_segments(sentences) @ query_embedding
        order = np.argsort(-scores, kind="stable")
        if scores[order[0]] <= self.empty_threshold:
            return []

        keep = max(int((scores >= self.sentence_threshold).sum()), self.min_sentences)
        if self.max_sentences is not None:
            keep = min(keep, self.max_sentences)
        return [replace(sentences[i], score=float(scores[i])) for i in order[:keep]]

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
