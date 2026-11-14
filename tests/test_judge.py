from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.judge import (
    ABSTAIN,
    CORRECT,
    JUDGE_PREFIX,
    WRONG,
    compare,
    judge_tasks,
    parse_verdict,
    question_of,
    shortcut,
    sign_test,
    summary,
    verdicts,
)
from evaluation.llm_clients import Completion
from evaluation.llm_eval import ResponseCache, Task, build_tasks
from test_raw import make_raw


def answered(tmp_path: Path, texts: dict[str, str]) -> tuple[list[Task], ResponseCache]:
    tasks = [t for t in build_tasks([make_raw()], 1, 0, only=list(texts))]
    cache = ResponseCache(tmp_path / "answers.jsonl")
    for t in tasks:
        cache.add(t, Completion(texts[t.condition], input_tokens=100, output_tokens=3))
    return tasks, cache


def test_shortcuts_settle_bare_unknown_and_exact_answers(tmp_path: Path) -> None:
    (task,), _ = answered(tmp_path, {"rag": "x"})  # make_raw(): the answer is "B"
    assert shortcut(task, "Unknown.") == ABSTAIN
    assert shortcut(task, "The b") == CORRECT  # normalization drops case and articles
    assert shortcut(task, "It is B, I think") is None
    assert shortcut(task, "") is None


def test_judge_tasks_cover_only_what_the_shortcuts_leave(tmp_path: Path) -> None:
    tasks, cache = answered(tmp_path, {"closed_book": "unknown", "rag": "B", "mw_rerank": "Probably B or C"})
    (grading,) = judge_tasks(tasks, cache)
    assert grading.task_id == JUDGE_PREFIX + "7:answerable:mw_rerank"
    question = question_of(tasks[0])
    assert question and f"Question: {question}\n" in grading.prompt
    assert "Accepted answers (any one is enough): B\n" in grading.prompt
    assert "Answer to grade: Probably B or C\n" in grading.prompt
    assert "Search results" not in grading.prompt  # the judge sees no context


@pytest.mark.parametrize(
    ("text", "verdict"),
    [("CORRECT", CORRECT), ("Abstain.", ABSTAIN), ("WRONG - it names C", WRONG), ("maybe", None), ("", None)],
)
def test_parse_verdict(text: str, verdict: str | None) -> None:
    assert parse_verdict(text) == verdict


def test_verdicts_and_summary(tmp_path: Path) -> None:
    tasks, cache = answered(tmp_path, {"closed_book": "unknown", "rag": "B", "mw_rerank": "Probably C", "mw_bi": "hm"})
    graded = ResponseCache(tmp_path / "judge.jsonl")
    by_condition = {t.condition: t for t in judge_tasks(tasks, cache)}
    graded.add(by_condition["mw_rerank"], Completion("WRONG", 50, 1))
    result = verdicts(tasks, cache, graded)
    # mw_bi is answered but not graded yet: it is left out, not guessed.
    assert result == {"7:answerable:closed_book": ABSTAIN, "7:answerable:rag": CORRECT, "7:answerable:mw_rerank": WRONG}
    rows = {r.condition: r for r in summary(tasks, cache, result)}
    assert set(rows) == {"closed_book", "rag", "mw_rerank"}
    assert (rows["rag"].n, rows["rag"].correct, rows["rag"].mean_input_tokens) == (1, 1.0, 100.0)
    assert rows["closed_book"].abstain == 1.0 and rows["mw_rerank"].wrong == 1.0


def test_sign_test() -> None:
    assert sign_test(0, 0) == 1.0 and sign_test(5, 5) == 1.0
    assert sign_test(5, 0) == pytest.approx(2 / 32)
    assert sign_test(1, 4) == pytest.approx(2 * 6 / 32)


def test_compare_pairs_questions_and_splits_by_closed_book() -> None:
    def graded(condition: str, results: str) -> dict[str, str]:
        return {f"{q}:answerable:{condition}": CORRECT if c == "1" else WRONG for q, c in enumerate(results) if c != "-"}

    a = graded("x", "1110")
    b = graded("y", "100-")  # question 3 has no b verdict: dropped from the pairing
    closed = {**graded("closed_book", "10"), "2:answerable:closed_book": ABSTAIN}
    rows = {r.stratum: r for r in compare("x", "y", a, b, closed)}
    everything = rows["all"]
    assert (everything.n, everything.wins, everything.losses) == (3, 2, 0)
    assert everything.accuracy_a == 1.0 and everything.accuracy_b == pytest.approx(1 / 3)
    assert everything.difference == pytest.approx(2 / 3)
    assert everything.ci_low < everything.difference < everything.ci_high
    assert everything.p_value == pytest.approx(0.5)
    assert rows["unknown"].n == 2 and rows["unknown"].wins == 2  # closed-book wrong or abstained
    assert "known" not in rows  # a single question: no interval to report
    # Distractor questions are a separate kind and never mix in.
    assert compare("x", "y", a, b, closed, kind="distractors") == []
