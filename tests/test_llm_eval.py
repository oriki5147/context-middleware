from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from context_middleware import MiddlewareConfig
from evaluation.llm_clients import Completion
from evaluation.llm_eval import (
    ANSWERABLE_CONDITIONS,
    NEGATIVE_CONDITIONS,
    NO_RESULTS,
    ResponseCache,
    answer_type,
    build_tasks,
    contexts,
    plant_counterfactual,
    question_word,
    report,
    run,
)
from test_raw import make_raw

# make_raw(): retrieval order c0, c2, c1; candidates (raw idx) 0, 1, 3, 2;
# bi scores 0.1, 0.9, 0.3, 0.5; reranker scores 0.01, 0.9, 0.05, 0.2.


def test_contexts_follow_production_selection() -> None:
    ctx = contexts(make_raw(), MiddlewareConfig())
    assert ctx["mw_bi"] == "B.\nC.\nA."  # best 3 by bi: idx 1, 2, 3
    assert ctx["mw_rerank"] == "B.\nC.\nA."  # best 3 by reranker: idx 1, 2, 3
    assert ctx["mw_rerank_nb"] == "A.\nB.\nC.\nA.\nA."  # neighbors stay within a document
    assert ctx["oracle"] == "Document [1](Title: T0) A. B. C."
    assert ctx["rag"].splitlines() == [
        "Document [1](Title: T0) A. B.",
        "Document [2](Title: T1) A. A.",
        "Document [3](Title: T0) C.",
    ]


def test_empty_rule_yields_no_results() -> None:
    ctx = contexts(make_raw(), MiddlewareConfig(empty_threshold=0.95, rerank_empty_threshold=0.95))
    assert ctx["mw_bi"] == "" and ctx["mw_rerank"] == ""


@pytest.mark.parametrize(
    ("answer", "kind"), [("1901", "year"), ("3.5", "number"), ("Wilhelm Röntgen", "name"), ("blue", "other")]
)
def test_answer_type(answer: str, kind: str) -> None:
    assert answer_type(answer) == kind


def test_counterfactual_swaps_whole_words_with_same_type_answer() -> None:
    planted = plant_counterfactual("Bob met Bobby about B.", ["Bob"], ["Alice", "1901", "bob"], seed=1)
    assert planted == ("Alice met Bobby about B.", "Alice")
    assert plant_counterfactual("no answer here", ["Bob"], ["Alice"], seed=1) is None
    assert plant_counterfactual("Bob", ["Bob"], ["1901"], seed=1) is None  # no same-type option


def test_counterfactual_distinct_substitutes_and_surname_mentions() -> None:
    context = "Werner Heisenberg and Dmitri Ivanenko proposed it. Heisenberg's papers followed."
    planted = plant_counterfactual(
        context, ["Dmitri Ivanenko", "Werner Heisenberg"], ["Niels Bohr", "Max Planck"], seed=3
    )
    assert planted is not None
    text, first = planted
    assert "Heisenberg" not in text and "Ivanenko" not in text
    assert {"Niels Bohr", "Max Planck"} <= set(re.findall(r"Niels Bohr|Max Planck", text))
    assert re.search(r"\b(Bohr|Planck)'s papers", text)  # the surname-only mention was swapped
    assert first in ("Niels Bohr", "Max Planck")


def test_question_word() -> None:
    assert question_word("who got the first nobel prize") == "who"
    assert question_word("In which year did it open") == "which"
    assert question_word("name the capital") == "other"


def test_build_tasks_conditions() -> None:
    raws = [make_raw(), make_raw(qid=8, answers=("Paris",)), make_raw(qid=9, kind="distractors", gold_doc=-1)]
    tasks = build_tasks(raws, answerable=2, negatives=1)
    by_q: dict[tuple[int, str], list[str]] = {}
    for t in tasks:
        by_q.setdefault((t.qid, t.kind), []).append(t.condition)
    # qid 7's answer "B" occurs in its context and "Paris" is a same-type substitute.
    assert by_q[(7, "answerable")] == list(ANSWERABLE_CONDITIONS)
    assert "counterfactual" not in by_q[(8, "answerable")]  # "Paris" is not in its context
    assert by_q[(9, "distractors")] == list(NEGATIVE_CONDITIONS)
    cf = next(t for t in tasks if t.condition == "counterfactual")
    assert cf.substitute == "Paris" and "Paris." in cf.prompt and "B." not in cf.context
    closed = next(t for t in tasks if t.condition == "closed_book")
    assert "Search results" not in closed.prompt
    # The open variant shares its base condition's context but not its instruction.
    get = {t.condition: t for t in tasks if t.qid == 7}
    assert get["mw_rerank_open"].context == get["mw_rerank"].context
    assert '"unknown"' in get["mw_rerank"].prompt and "own knowledge" in get["mw_rerank_open"].prompt
    assert len(get["mw_rerank_k7"].context) >= len(get["mw_rerank"].context)
    # Equal-budget retrieval: the best chunks of the plain RAG context, cut short.
    assert get["rag_eq_k3"].context.startswith("Document [1]")
    assert get["rag"].context.startswith(get["rag_eq_k3"].context)
    assert len(get["rag_eq_k3"].context) <= len(get["rag_eq_k7"].context)


class FakeClient:
    """Answers "B" when the context shows it, the planted answer if present, else unknown."""

    name = "fake"
    price_per_mtok = (1.0, 2.0)

    def __init__(self, fail_on: str = "") -> None:
        self.calls = 0
        self.fail_on = fail_on

    def complete(self, prompt: str) -> Completion:
        self.calls += 1
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("boom")
        if "Paris." in prompt:
            text = "Paris"
        elif "B." in prompt:
            text = "B"
        elif "Search results" in prompt:
            text = "unknown"
        else:
            text = "no idea"
        return Completion(text, input_tokens=100, output_tokens=5)


def test_run_resumes_without_paying_twice_and_respects_cap(tmp_path: Path) -> None:
    tasks = build_tasks([make_raw(), make_raw(qid=8, answers=("Paris",))], answerable=2, negatives=0)
    cache = ResponseCache(tmp_path / "fake.jsonl")
    client = FakeClient()
    assert run(tasks, client, cache, max_requests=3, workers=2, log=lambda _: None) == 0
    assert client.calls == 3
    run(tasks, client, ResponseCache(tmp_path / "fake.jsonl"), max_requests=100, log=lambda _: None)
    assert client.calls == len(tasks)  # reloaded cache: only the rest was sent


def test_failures_are_not_cached(tmp_path: Path) -> None:
    tasks = build_tasks([make_raw()], answerable=1, negatives=0)
    cache = ResponseCache(tmp_path / "fake.jsonl")
    failures = run(tasks, FakeClient(fail_on="Document [1](Title: T0) A. B. C."), cache, 100, log=lambda _: None)
    assert failures == 1 and len(cache.done) == len(tasks) - 1


def test_report_gain_harm_adoption_and_cost(tmp_path: Path) -> None:
    tasks = build_tasks([make_raw(), make_raw(qid=8, answers=("Paris",))], answerable=2, negatives=0)
    cache = ResponseCache(tmp_path / "fake.jsonl")
    run(tasks, FakeClient(), cache, 100, log=lambda _: None)
    rows = {r.condition: r for r in report(tasks, cache, "fake", (1.0, 2.0))}

    assert rows["closed_book"].accuracy == 0.0 and math.isnan(rows["closed_book"].gain)
    mw = rows["mw_rerank"]  # q7: context shows "B." -> correct; q8: "B." too, but its answer is Paris
    assert (mw.n, mw.accuracy, mw.gain) == (2, 0.5, 0.5)
    assert math.isnan(mw.harm)  # closed-book never correct, so nothing to harm
    assert mw.cost_usd == pytest.approx(2 * (100 * 1.0 + 5 * 2.0) / 1e6)
    cf = rows["counterfactual"]
    assert (cf.n, cf.accuracy, cf.substitute_adopted) == (1, 0.0, 1.0)


def test_no_results_placeholder_in_prompt() -> None:
    raw = make_raw()
    tasks = build_tasks([raw], 1, 0, MiddlewareConfig(empty_threshold=0.95, rerank_empty_threshold=0.95))
    mw = next(t for t in tasks if t.condition == "mw_bi")
    assert mw.context == "" and NO_RESULTS in mw.prompt


def test_build_tasks_offset_and_condition_subset() -> None:
    raws = [make_raw(), make_raw(qid=8, answers=("Paris",)), make_raw(qid=9, kind="distractors", gold_doc=-1)]
    tasks = build_tasks(raws, answerable=1, negatives=1, offset=1, only=["rag", "mw_rerank_nb"])
    assert {(t.qid, t.condition) for t in tasks} == {(8, "rag"), (8, "mw_rerank_nb")}
