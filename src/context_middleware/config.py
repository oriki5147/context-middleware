"""Pipeline tunables.

Defaults follow the NQ multi-document dev evaluation (see evaluation/), not the
original illustrative values in Architecture.md.
"""

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
    # least chunk_overlap_tokens of whole sentences carried over. Small chunks
    # without overlap retrieved the answer most often at equal token budgets.
    chunk_size_tokens: int = 64
    chunk_overlap_tokens: int = 0
    # Phase 2: number of candidate chunks; 25 x ~47-token chunks ~= the
    # 1,200 tokens retrieved in the evaluation.
    top_k: int = 25
    # Phase 3: selection is driven by the cap: keep the best max_sentences.
    # sentence_threshold (optional) additionally drops sentences below it,
    # down to min_sentences. Evaluation showed caps, not thresholds, set the
    # compression/answer trade-off.
    max_sentences: int | None = 3
    sentence_threshold: float | None = None
    min_sentences: int = 1
    # Empty context when even the best sentence scores <= the threshold of
    # the active scorer. This reliably catches off-topic queries, NOT on-topic
    # queries whose answer is absent. Calibrated for bge-small-en-v1.5 INT8
    # and ms-marco-MiniLM-L6-v2 INT8; recalibrate when changing models.
    empty_threshold: float = 0.64
    rerank_empty_threshold: float = 0.01
    # Optional cross-encoder stage: rerank the bi-encoder's top-N sentences.
    rerank_top_n: int = 20
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
        if self.min_sentences < 0:
            raise ValueError("min_sentences must be non-negative")
        if self.max_sentences is not None and self.max_sentences < max(self.min_sentences, 1):
            raise ValueError("max_sentences must be >= max(min_sentences, 1)")
        if self.rerank_top_n <= 0:
            raise ValueError("rerank_top_n must be positive")
        if self.max_context_tokens is not None and self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
