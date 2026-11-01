"""Cheap phase: sweep selection rules over cached candidates and report metrics.

Selection uses the production functions (``select_sentences``, ``fit_budget``),
so a row here is exactly what the middleware would output with that config.
Scorers are compared on identical candidates:
  - "bi":        bi-encoder scores over all candidate sentences;
  - "rerank<N>": cross-encoder scores over the bi-encoder's top-N shortlist
                 (production ``rerank_top_n = N``);
  - "reader<N>": extractive-reader answer margins over the same shortlist.
Only scorers whose scores are in the cache are reported.
"""

from __future__ import annotations

import csv
import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from context_middleware import BudgetStrategy
from context_middleware.compressor import select_sentences
from context_middleware.engine import fit_budget

from .candidates import Candidates, QuestionRecord

NO_EMPTY = float("-inf")  # disables the empty-context rule
SHORTLISTS = (10, 20, 30, 50)
SCORERS = ("bi", *(f"{stage}{n}" for stage in ("rerank", "reader") for n in SHORTLISTS))
NEGATIVE_KINDS = ("unrelated", "distractors")


@dataclass(frozen=True, slots=True)
class View:
    """The candidate set one scorer selects from, with aligned arrays."""

    scores: NDArray[np.float32]
    tokens: NDArray[np.int32]
    has_answer: NDArray[np.bool_]


def available_scorers(records: Sequence[QuestionRecord]) -> list[str]:
    """Scorers with cached scores (caches built before the reader lack it)."""
    sample = next(iter(records[0].by_config.values()))
    return [
        s for s in SCORERS
        if s == "bi" or getattr(sample, f"{s.rstrip('0123456789')}_scores", None) is not None
    ]


def _scorers(records: Sequence[QuestionRecord], grids: Mapping[str, object]) -> list[str]:
    """Available scorers that the caller supplied thresholds for."""
    return [s for s in available_scorers(records) if s in grids]


def view(c: Candidates, scorer: str) -> View:
    if scorer == "bi":
        return View(c.scores, c.tokens, c.has_answer)
    stage = scorer.rstrip("0123456789")
    top_n = int(scorer.removeprefix(stage))
    stage_scores: NDArray[np.float32] | None = getattr(c, f"{stage}_scores")
    if stage_scores is None:
        raise ValueError(f"{scorer} scores were not computed for this cache")
    # Same shortlist as production, kept in bi-encoder order so that ties in
    # second-stage scores break the same way select_sentences breaks them there.
    shortlist = np.argsort(-c.scores, kind="stable")[:top_n]
    if np.isnan(stage_scores[shortlist]).any():
        raise ValueError(f"{scorer} exceeds the cached shortlist")
    return View(stage_scores[shortlist], c.tokens[shortlist], c.has_answer[shortlist])


# ----------------------------------------------------------------- retrieval


@dataclass(frozen=True, slots=True)
class RetrievalRow:
    chunk_config: str
    scorer: str
    questions: int
    gold_retrieved: float  # gold passage among the retrieved chunks
    answer_in_retrieved: float  # answer string reaches the compressor
    answer_in_candidates: float  # ... and is among this scorer's candidates
    answer_rank_1: float  # best-scored candidate sentence contains the answer
    answer_rank_le_3: float
    median_answer_rank: float  # among questions whose answer is a candidate
    median_candidates: float
    mean_original_tokens: float


def retrieval_report(records: Sequence[QuestionRecord]) -> list[RetrievalRow]:
    rows = []
    answerable = [r for r in records if r.answerable]
    for name, scorer in itertools.product(answerable[0].by_config, available_scorers(records)):
        cands = [r.by_config[name] for r in answerable]
        views = [view(c, scorer) for c in cands]
        ranks = [_answer_rank(v) for v in views]
        found = [r for r in ranks if r is not None]
        rows.append(
            RetrievalRow(
                chunk_config=name,
                scorer=scorer,
                questions=len(cands),
                gold_retrieved=_mean(c.gold_retrieved for c in cands),
                answer_in_retrieved=_mean(c.answer_in_retrieved for c in cands),
                answer_in_candidates=_mean(r is not None for r in ranks),
                answer_rank_1=_mean(r == 1 for r in ranks),
                answer_rank_le_3=_mean(r is not None and r <= 3 for r in ranks),
                median_answer_rank=float(np.median(found)) if found else float("nan"),
                median_candidates=float(np.median([len(v.scores) for v in views])),
                mean_original_tokens=_mean(c.original_tokens for c in cands),
            )
        )
    return rows


def _answer_rank(v: View) -> int | None:
    """1-based rank of the best-ranked answer-bearing sentence, or None."""
    order = np.argsort(-v.scores, kind="stable")
    hits = np.flatnonzero(v.has_answer[order])
    return int(hits[0]) + 1 if len(hits) else None


# ------------------------------------------------------------- empty context


@dataclass(frozen=True, slots=True)
class EmptyRow:
    chunk_config: str
    scorer: str
    empty_threshold: float
    false_empty: float  # answerable questions wrongly given no context
    true_empty_unrelated: float  # off-topic negatives correctly given no context
    true_empty_distractors: float  # on-topic negatives correctly given no context


def empty_report(
    records: Sequence[QuestionRecord], thresholds: Mapping[str, Iterable[float]]
) -> list[EmptyRow]:
    rows = []
    for name, scorer in itertools.product(records[0].by_config, _scorers(records, thresholds)):
        best = {
            kind: np.array([_best(view(r.by_config[name], scorer)) for r in records if r.kind == kind])
            for kind in ("answerable", *NEGATIVE_KINDS)
        }
        for t in thresholds[scorer]:
            rate = {k: float((b <= t).mean()) if len(b) else float("nan") for k, b in best.items()}
            rows.append(
                EmptyRow(
                    chunk_config=name,
                    scorer=scorer,
                    empty_threshold=t,
                    false_empty=rate["answerable"],
                    true_empty_unrelated=rate["unrelated"],
                    true_empty_distractors=rate["distractors"],
                )
            )
    return rows


def _best(v: View) -> float:
    return float(v.scores.max()) if len(v.scores) else NO_EMPTY


# --------------------------------------------------------------- compression


@dataclass(frozen=True, slots=True)
class CompressionRow:
    chunk_config: str
    scorer: str
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
    sentence_thresholds: Mapping[str, Iterable[float]],
    max_sentences: Iterable[int | None],
    budgets: Iterable[tuple[int | None, BudgetStrategy]],
) -> list[CompressionRow]:
    """Answerable questions only; the empty rule is disabled (see empty_report)."""
    answerable = [r for r in records if r.answerable]
    caps, budgets = list(max_sentences), list(budgets)
    rows = []
    for name, scorer in itertools.product(answerable[0].by_config, _scorers(records, sentence_thresholds)):
        cands = [r.by_config[name] for r in answerable]
        views = [view(c, scorer) for c in cands]
        for threshold, cap, (budget, strategy) in itertools.product(
            sentence_thresholds[scorer], caps, budgets
        ):
            kept_answer, kept_tokens, fractions = [], [], []
            for c, v in zip(cands, views):
                chosen = select_sentences(v.scores, threshold, NO_EMPTY, 1, cap)
                chosen = [chosen[i] for i in fit_budget(v.tokens[chosen].tolist(), budget, strategy)]
                tokens = int(v.tokens[chosen].sum())
                kept_answer.append(bool(v.has_answer[chosen].any()))
                kept_tokens.append(tokens)
                fractions.append(tokens / c.original_tokens if c.original_tokens else 0.0)
            retrieved = [k for k, c in zip(kept_answer, cands) if c.answer_in_retrieved]
            rows.append(
                CompressionRow(
                    chunk_config=name,
                    scorer=scorer,
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


def pareto(rows: Iterable[CompressionRow]) -> list[CompressionRow]:
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
