"""Core data structures shared across the middleware pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Embeddings stay as contiguous float32 arrays end-to-end so similarity scoring
# is a single BLAS matmul on CPU, never a Python-level loop over lists.
Embedding = NDArray[np.float32]


@dataclass(slots=True)
class TextSegment:
    """A unit of text flowing through the pipeline: a chunk or a sentence."""

    segment_id: str
    content: str
    source_doc: str
    # Excluded from __eq__/repr: comparing ndarrays yields an array, not a bool.
    embedding: Embedding | None = field(default=None, compare=False, repr=False)
    score: float = 0.0
    # Document title, kept out of ``content`` so content stays a verbatim slice.
    # Prepended when embedding to decontextualize the passage (DPR, RECOMP).
    title: str = ""
    # Sentences only: order within the ingested documents (consecutive numbers
    # are adjacent sentences), None when the document was not ingested.
    position: int | None = None


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """Final output handed to the target LLM.

    ``compression_ratio`` is the fraction of tokens kept
    (``compressed / original``), so 0.05 means a 95% reduction.
    """

    compressed_prompt: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    is_empty: bool

    def __post_init__(self) -> None:
        if self.original_token_count < 0 or self.compressed_token_count < 0:
            raise ValueError("token counts must be non-negative")
        if self.is_empty and (self.compressed_prompt or self.compressed_token_count):
            raise ValueError("an empty result must have no prompt and zero tokens")

    @classmethod
    def from_counts(
        cls, compressed_prompt: str, original_token_count: int, compressed_token_count: int
    ) -> CompressionResult:
        ratio = compressed_token_count / original_token_count if original_token_count else 0.0
        return cls(
            compressed_prompt=compressed_prompt,
            original_token_count=original_token_count,
            compressed_token_count=compressed_token_count,
            compression_ratio=ratio,
            is_empty=False,
        )

    @classmethod
    def empty(cls, original_token_count: int) -> CompressionResult:
        """Selective-augmentation result: no context, let the LLM use parametric knowledge."""
        return cls(
            compressed_prompt="",
            original_token_count=original_token_count,
            compressed_token_count=0,
            compression_ratio=0.0,
            is_empty=True,
        )
