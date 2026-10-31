from __future__ import annotations

import numpy as np
import pytest

from context_middleware import BudgetStrategy
from evaluation.answers import contains_answer, normalize
from evaluation.candidates import Candidates, QuestionRecord
from evaluation.nq_data import Doc, Example, split, unanswerable
from evaluation.sweep import (
    NO_EMPTY,
    compression_sweep,
    empty_report,
    pareto,
    retrieval_report,
)


def test_normalize_strips_case_punctuation_and_articles() -> None:
    assert normalize("The  Wilhelm Conrad Röntgen!") == "wilhelm conrad röntgen"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Awarded to Wilhelm Conrad Röntgen, of Germany", True), ("Awarded to Marie Curie", False)],
)
def test_contains_answer(text: str, expected: bool) -> None:
    assert contains_answer(text, ["Marie Skłodowska", "Wilhelm Conrad Röntgen"]) is expected


def test_empty_answer_never_matches() -> None:
    assert not contains_answer("anything", ["", "the"])


def example(qid: int, answer: str, text: str) -> Example:
    return Example(qid, f"q{qid}", (answer,), (Doc("t", text, True),))


def test_split_is_deterministic_and_disjoint() -> None:
    examples = [example(i, "x", "x") for i in range(10)]
    dev, test = split(examples)
    assert split(examples) == (dev, test)
    assert {e.qid for e in dev}.isdisjoint({e.qid for e in test})
    assert len(dev) + len(test) == 10


def test_unanswerable_uses_other_answer_free_passages() -> None:
    examples = [example(i, f"ans{i}", f"text about ans{i}") for i in range(20)]
    negatives = unanswerable(examples)
    assert negatives
    for neg in negatives:
        assert not neg.answerable
        assert not any(contains_answer(d.text, neg.answers) for d in neg.docs)
        assert not any(d.is_gold for d in neg.docs)


def cands(scores: list[float], answer_at: int | None, tokens: int = 10) -> Candidates:
    has = np.zeros(len(scores), dtype=np.bool_)
    if answer_at is not None:
        has[answer_at] = True
    return Candidates(
        original_tokens=tokens * len(scores),
        gold_retrieved=answer_at is not None,
        answer_in_retrieved=answer_at is not None,
        scores=np.array(scores, dtype=np.float32),
        tokens=np.full(len(scores), tokens, dtype=np.int32),
        has_answer=has,
    )


RECORDS = [
    QuestionRecord(0, True, {"c": cands([0.9, 0.8, 0.7], answer_at=0)}),  # answer ranked 1st
    QuestionRecord(1, True, {"c": cands([0.9, 0.8, 0.7], answer_at=2)}),  # answer ranked 3rd
    QuestionRecord(2, False, {"c": cands([0.6, 0.5], answer_at=None)}),
]


def test_retrieval_report_ranks() -> None:
    (row,) = retrieval_report(RECORDS)
    assert row.questions == 2
    assert row.answer_rank_1 == 0.5
    assert row.answer_rank_le_3 == 1.0
    assert row.median_answer_rank == 2.0


def test_empty_report_rates() -> None:
    rows = {r.empty_threshold: r for r in empty_report(RECORDS, [0.65, 0.95])}
    assert (rows[0.65].false_empty, rows[0.65].true_empty) == (0.0, 1.0)
    assert (rows[0.95].false_empty, rows[0.95].true_empty) == (1.0, 1.0)


def test_compression_sweep_uses_production_selection() -> None:
    rows = compression_sweep(
        RECORDS,
        sentence_thresholds=[0.85],
        max_sentences=[None],
        budgets=[(None, BudgetStrategy.FILL), (15, BudgetStrategy.FILL)],
    )
    no_budget, budget = rows
    # threshold 0.85 keeps only the top sentence: answer kept for q0 only.
    assert no_budget.answer_kept == 0.5
    assert no_budget.kept_fraction == pytest.approx(1 / 3)
    # a 15-token budget fits no 10+10, so still one sentence each.
    assert budget.mean_compressed_tokens == 10


def test_pareto_keeps_only_non_dominated_rows() -> None:
    rows = compression_sweep(RECORDS, [0.65, 0.75, 0.85], [None], [(None, BudgetStrategy.FILL)])
    front = pareto(rows)
    assert [r.sentence_threshold for r in front] == [0.85, 0.65]


def test_no_empty_sentinel_disables_rule() -> None:
    assert NO_EMPTY < -1e9
