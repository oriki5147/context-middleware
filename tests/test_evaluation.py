from __future__ import annotations

import numpy as np
import pytest

from context_middleware import BudgetStrategy, TextSegment
from evaluation.answers import contains_answer, normalize
from evaluation.candidates import Candidates, QuestionRecord, _retrieve
from evaluation.nq_data import Doc, Example, distractors_only, split, unanswerable
from evaluation.sweep import (
    NO_EMPTY,
    compression_sweep,
    empty_report,
    pareto,
    retrieval_report,
    SCORERS,
    view,
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


def example(qid: int, answer: str, *texts: str) -> Example:
    docs = (Doc("t", texts[0], True),) + tuple(Doc("t", t, False) for t in texts[1:])
    return Example(qid, f"q{qid}", (answer,), docs)


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
        assert neg.kind == "unrelated" and not neg.answerable
        assert not any(contains_answer(d.text, neg.answers) for d in neg.docs)
        assert not any(d.is_gold for d in neg.docs)


def test_distractors_only_drops_gold_and_skips_leaky_distractors() -> None:
    clean = example(0, "paris", "paris is the capital", "france is big", "lyon is a city")
    leaky = example(1, "paris", "paris is the capital", "paris again")
    (neg,) = distractors_only([clean, leaky])
    assert neg.qid == 0 and neg.kind == "distractors"
    assert [d.text for d in neg.docs] == ["france is big", "lyon is a city"]


def test_retrieve_fills_token_budget_best_first() -> None:
    chunks = [TextSegment(str(i), "w " * n, "d") for i, n in enumerate([700, 600, 500])]
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    got = _retrieve(chunks, scores, lambda text: len(text.split()))
    assert [c.segment_id for c in got] == ["1", "2"]  # 600 + 500 <= 1200; 700 would exceed


def cands(
    scores: list[float], answer_at: int | None, rerank: list[float] | None = None, tokens: int = 10
) -> Candidates:
    has = np.zeros(len(scores), dtype=np.bool_)
    if answer_at is not None:
        has[answer_at] = True
    return Candidates(
        original_tokens=tokens * len(scores),
        gold_retrieved=answer_at is not None,
        answer_in_retrieved=answer_at is not None,
        scores=np.array(scores, dtype=np.float32),
        rerank_scores=np.array(rerank if rerank else [np.nan] * len(scores), dtype=np.float32),
        # The reader mirrors the reranker here; scorer plumbing is shared.
        reader_scores=np.array(rerank if rerank else [np.nan] * len(scores), dtype=np.float32),
        tokens=np.full(len(scores), tokens, dtype=np.int32),
        has_answer=has,
    )


RECORDS = [
    # answer ranked 1st by both scorers
    QuestionRecord(0, "answerable", {"c": cands([0.9, 0.8, 0.7], 0, [0.9, 0.1, 0.05])}),
    # answer ranked 3rd by the bi-encoder; the reranker lifts it to 1st
    QuestionRecord(1, "answerable", {"c": cands([0.9, 0.8, 0.7], 2, [0.1, 0.2, 0.8])}),
    QuestionRecord(2, "unrelated", {"c": cands([0.6, 0.5], None, [0.01, 0.02])}),
    QuestionRecord(3, "distractors", {"c": cands([0.75, 0.5], None, [0.3, 0.2])}),
]


def test_view_restricts_rerank_to_bi_encoder_top_n() -> None:
    # Bi-encoder order is 0.9 > 0.8 > 0.7, so rerank2 sees the first two only.
    v = view(RECORDS[1].by_config["c"], "rerank2")
    assert v.scores.tolist() == pytest.approx([0.1, 0.2])
    assert v.has_answer.tolist() == [False, False]


def test_view_rejects_shortlist_beyond_cache() -> None:
    partial = cands([0.9, 0.8, 0.7], 0, [0.9, 0.1, float("nan")])
    assert len(view(partial, "rerank2").scores) == 2
    with pytest.raises(ValueError):
        view(partial, "rerank3")


def test_retrieval_report_ranks_per_scorer() -> None:
    rows = {r.scorer: r for r in retrieval_report(RECORDS)}
    assert rows["bi"].questions == 2
    assert (rows["bi"].answer_rank_1, rows["bi"].median_answer_rank) == (0.5, 2.0)
    assert (rows["rerank10"].answer_rank_1, rows["rerank10"].answer_in_candidates) == (1.0, 1.0)


def test_empty_report_rates_per_negative_kind() -> None:
    rows = {
        (r.scorer, r.empty_threshold): r
        for r in empty_report(RECORDS, {"bi": [0.65, 0.95], **{s: [0.05] for s in SCORERS[1:]}})
    }
    bi = rows[("bi", 0.65)]
    assert (bi.false_empty, bi.true_empty_unrelated, bi.true_empty_distractors) == (0.0, 1.0, 0.0)
    assert rows[("bi", 0.95)].false_empty == 1.0
    rr = rows[("rerank10", 0.05)]
    assert (rr.false_empty, rr.true_empty_unrelated, rr.true_empty_distractors) == (0.0, 1.0, 0.0)


def test_compression_sweep_uses_production_selection() -> None:
    rows = compression_sweep(
        RECORDS,
        sentence_thresholds={"bi": [0.85], "rerank10": [0.85], "rerank20": [], "rerank30": [], "rerank50": []},
        max_sentences=[None],
        budgets=[(None, BudgetStrategy.FILL), (15, BudgetStrategy.FILL)],
    )
    bi_none, bi_budget, rr_none, _ = rows
    # Top sentence only: bi keeps the answer for q0 only; rerank for both.
    assert (bi_none.scorer, bi_none.answer_kept) == ("bi", 0.5)
    assert bi_none.kept_fraction == pytest.approx(1 / 3)
    assert bi_budget.mean_compressed_tokens == 10
    assert (rr_none.scorer, rr_none.answer_kept) == ("rerank10", 1.0)


def test_pareto_keeps_only_non_dominated_rows() -> None:
    rows = compression_sweep(
        RECORDS, {"bi": [0.65, 0.75, 0.85], **{s: [] for s in SCORERS[1:]}}, [None], [(None, BudgetStrategy.FILL)]
    )
    assert [r.sentence_threshold for r in pareto(rows)] == [0.85, 0.65]


def test_no_empty_sentinel_disables_rule() -> None:
    assert NO_EMPTY < -1e9
