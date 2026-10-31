"""Cheap phase: sweep selection rules over cached candidates and report metrics.

Selection uses the production functions (``select_sentences``, ``fit_budget``),
so a row here is exactly what the middleware would output with that config.
"""

from __future__ import annotations

import csv
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from context_middleware import BudgetStrategy
from context_middleware.compressor import select_sentences
from context_middleware.engine import fit_budget

from .candidates import Candidates, QuestionRecord

NO_EMPTY = float("-inf")  # disables the empty-context rule


# ----------------------------------------------------------------- retrieval


@dataclass(frozen=True, slots=True)
class RetrievalRow:
    chunk_config: str
    questions: int
    gold_retrieved: float  # gold passage among the top-K chunks
    answer_in_retrieved: float  # answer string reaches the compressor
    answer_rank_1: float  # best-scored candidate sentence contains the answer
    answer_rank_le_3: float
    median_answer_rank: float  # among questions whose answer was retrieved
    median_candidates: float
    mean_original_tokens: float


def retrieval_report(records: Sequence[QuestionRecord]) -> list[RetrievalRow]:
    rows = []
    answerable = [r for r in records if r.answerable]
    for name in answerable[0].by_config:
        cands = [r.by_config[name] for r in answerable]
        ranks = [_answer_rank(c) for c in cands]
        found = [r for r in ranks if r is not None]
        rows.append(
            RetrievalRow(
                chunk_config=name,
                questions=len(cands),
                gold_retrieved=_mean(c.gold_retrieved for c in cands),
                answer_in_retrieved=_mean(c.answer_in_retrieved for c in cands),
                answer_rank_1=_mean(r == 1 for r in ranks),
                answer_rank_le_3=_mean(r is not None and r <= 3 for r in ranks),
                median_answer_rank=float(np.median(found)) if found else float("nan"),
                median_candidates=float(np.median([len(c.scores) for c in cands])),
                mean_original_tokens=_mean(c.original_tokens for c in cands),
            )
        )
    return rows


def _answer_rank(c: Candidates) -> int | None:
    """1-based rank of the best-ranked answer-bearing sentence, or None."""
    order = np.argsort(-c.scores, kind="stable")
    hits = np.flatnonzero(c.has_answer[order])
    return int(hits[0]) + 1 if len(hits) else None


# ------------------------------------------------------------- empty context


@dataclass(frozen=True, slots=True)
class EmptyRow:
    chunk_config: str
    empty_threshold: float
    false_empty: float  # answerable questions wrongly given no context
    true_empty: float  # unanswerable questions correctly given no context


def empty_report(
    records: Sequence[QuestionRecord], thresholds: Iterable[float]
) -> list[EmptyRow]:
    rows = []
    thresholds = list(thresholds)
    for name in records[0].by_config:
        best = {
            flag: np.array([_best(r.by_config[name]) for r in records if r.answerable is flag])
            for flag in (True, False)
        }
        for t in thresholds:
            rows.append(
                EmptyRow(
                    chunk_config=name,
                    empty_threshold=t,
                    false_empty=float((best[True] <= t).mean()) if len(best[True]) else float("nan"),
                    true_empty=float((best[False] <= t).mean()) if len(best[False]) else float("nan"),
                )
            )
    return rows


def _best(c: Candidates) -> float:
    return float(c.scores.max()) if len(c.scores) else NO_EMPTY


# --------------------------------------------------------------- compression


@dataclass(frozen=True, slots=True)
class CompressionRow:
    chunk_config: str
    sentence_threshold: float
    max_sentences: int | None
    max_context_tokens: int | None
    budget_strategy: str
    answer_kept: float  # answer string survives compression (upper bound on LLM accuracy)
    answer_kept_given_retrieved: float  # same, only where retrieval delivered it
    kept_fraction: float  # mean compressed / original tokens
    mean_compressed_tokens: float


def compression_sweep(
    records: Sequence[QuestionRecord],
    sentence_thresholds: Iterable[float],
    max_sentences: Iterable[int | None],
    budgets: Iterable[tuple[int | None, BudgetStrategy]],
) -> list[CompressionRow]:
    """Answerable questions only; the empty rule is disabled (see empty_report)."""
    answerable = [r for r in records if r.answerable]
    grid = list(itertools.product(sentence_thresholds, max_sentences, budgets))
    rows = []
    for name in answerable[0].by_config:
        cands = [r.by_config[name] for r in answerable]
        for threshold, cap, (budget, strategy) in grid:
            kept_answer, kept_tokens, fractions = [], [], []
            for c in cands:
                chosen = select_sentences(c.scores, threshold, NO_EMPTY, 1, cap)
                chosen = [chosen[i] for i in fit_budget(c.tokens[chosen].tolist(), budget, strategy)]
                tokens = int(c.tokens[chosen].sum())
                kept_answer.append(bool(c.has_answer[chosen].any()))
                kept_tokens.append(tokens)
                fractions.append(tokens / c.original_tokens if c.original_tokens else 0.0)
            retrieved = [k for k, c in zip(kept_answer, cands) if c.answer_in_retrieved]
            rows.append(
                CompressionRow(
                    chunk_config=name,
                    sentence_threshold=round(threshold, 4),
                    max_sentences=cap,
                    max_context_tokens=budget,
                    budget_strategy=strategy.value if budget is not None else "-",
                    answer_kept=_mean(kept_answer),
                    answer_kept_given_retrieved=_mean(retrieved),
                    kept_fraction=_mean(fractions),
                    mean_compressed_tokens=_mean(kept_tokens),
                )
            )
    return rows


def pareto(rows: Sequence[CompressionRow]) -> list[CompressionRow]:
    """Rows no other row beats on both answer_kept (higher) and kept_fraction (lower)."""
    front: list[CompressionRow] = []
    for row in sorted(rows, key=lambda r: (r.kept_fraction, -r.answer_kept)):
        if not front or row.answer_kept > front[-1].answer_kept:
            front.append(row)
    return front


# ------------------------------------------------------------------ helpers


def _mean(values: Iterable[bool | int | float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def write_csv(rows: Sequence[object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [asdict(r) for r in rows]  # type: ignore[call-overload]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dicts[0]))
        writer.writeheader()
        writer.writerows(dicts)
