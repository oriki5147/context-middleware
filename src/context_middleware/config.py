"""Pipeline tunables, defaulted to the values in Architecture.md."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .reorderer import ReorderStrategy


class BudgetStrategy(str, Enum):
    """What to do, best sentence first, when a sentence exceeds the remaining budget."""

    FILL = "fill"  # skip it; shorter lower-ranked sentences may fill the rest
    STOP = "stop"  # stop there; the context is always the top-N, with no gaps


@dataclass(frozen=True, slots=True)
class MiddlewareConfig:
    # Phase 1: sentence-aligned chunks of at most chunk_size_tokens, with at
    # least chunk_overlap_tokens of whole sentences carried over.
    chunk_size_tokens: int = 120
    chunk_overlap_tokens: int = 30
    # Phase 2: number of candidate chunks from dense retrieval.
    top_k: int = 10
    # Phase 3: keep sentences scoring >= sentence_threshold (at least
    # min_sentences, at most max_sentences); if every sentence scores
    # <= empty_threshold, return an empty context instead. Scores are on the
    # encoder's own scale, so these are placeholders to calibrate per encoder.
    sentence_threshold: float = 0.65
    empty_threshold: float = 0.45
    min_sentences: int = 1
    max_sentences: int | None = None
    # Token budget for the final context, filled best sentence first.
    max_context_tokens: int | None = None
    budget_strategy: BudgetStrategy = BudgetStrategy.FILL
    # Phase 4.
    reorder_strategy: ReorderStrategy = ReorderStrategy.BEST_FIRST

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be in [0, chunk_size_tokens)")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.empty_threshold > self.sentence_threshold:
            raise ValueError("empty_threshold must not exceed sentence_threshold")
        if self.min_sentences < 0:
            raise ValueError("min_sentences must be non-negative")
        if self.max_sentences is not None and self.max_sentences < max(self.min_sentences, 1):
            raise ValueError("max_sentences must be >= max(min_sentences, 1)")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
