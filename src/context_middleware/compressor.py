"""Phase 3: extractive, selective sentence-level compression (RECOMP).

Purely extractive: sentences are kept verbatim or dropped, never rewritten.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import TextSegment
from .retriever import ONNXBiEncoderRetriever


class SelectiveSentenceCompressor:
    def __init__(
        self,
        encoder: ONNXBiEncoderRetriever,
        sentence_threshold: float = 0.65,
        empty_threshold: float = 0.45,
    ) -> None:
        self.encoder = encoder
        self.sentence_threshold = sentence_threshold
        self.empty_threshold = empty_threshold

    def compress(self, query: str, chunks: Sequence[TextSegment]) -> list[TextSegment]:
        """Return scored sentences passing the threshold.

        An empty list signals selective augmentation (all scores <= empty_threshold).
        """
        raise NotImplementedError
