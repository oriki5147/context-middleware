"""Pipeline tunables, defaulted to the values in Architecture.md."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MiddlewareConfig:
    # Phase 1: fixed-length chunking with a ~25% sliding-window overlap.
    chunk_size_tokens: int = 120
    chunk_overlap_tokens: int = 30
    # Phase 2: number of candidate chunks from dense retrieval.
    top_k: int = 10
    # Phase 3: keep sentences scoring >= sentence_threshold; if every sentence
    # scores <= empty_threshold, return an empty context instead.
    sentence_threshold: float = 0.65
    empty_threshold: float = 0.45

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be in [0, chunk_size_tokens)")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= self.empty_threshold <= self.sentence_threshold <= 1.0:
            raise ValueError("require -1 <= empty_threshold <= sentence_threshold <= 1")
