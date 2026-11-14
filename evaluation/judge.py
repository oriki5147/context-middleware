"""Level 2 grading by an LLM judge, and paired comparisons between conditions.

Substring matching (``answers.contains_answer``) cannot tell an abstention
from a wrong answer and misses correct answers in another surface form. The
judge reads the question, the accepted answers and the model's answer and
labels it correct / abstain / wrong. Two cases need no judge: a bare
"unknown", and an answer equal to an accepted one after normalization.

A paired comparison reports the accuracy difference between two conditions
over the questions both answered, with a 95% interval from the paired
standard error and an exact sign test, overall and split by whether the
model already knew the answer closed-book.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .answers import normalize
from .llm_eval import ResponseCache, Task

CORRECT, ABSTAIN, WRONG = "correct", "abstain", "wrong"
VERDICTS = (CORRECT, ABSTAIN, WRONG)
JUDGE_PREFIX = "judge:"
JUDGE_PROMPT = (
    "You are grading a short answer to a question.\n\n"
    "Question: {question}\n"
    "Accepted answers (any one is enough): {answers}\n"
    "Answer to grade: {answer}\n\n"
    "Reply with exactly one word:\n"
    "CORRECT - the answer states one of the accepted answers or the same fact in another "
    "form (a different spelling, more or less detail), and does not also give a conflicting answer.\n"
    "ABSTAIN - the answer gives no answer: it says it does not know, or that the search results "
    "do not contain the answer.\n"
    "WRONG - anything else."
)


def shortcut(task: Task, text: str) -> str | None:
    """The verdict when it needs no judge, else None."""
    answer = normalize(text)
    if answer == "unknown":
        return ABSTAIN
    if answer and answer in {normalize(a) for a in task.answers}:
        return CORRECT
    return None


def question_of(task: Task) -> str:
    """The question text, recovered from the answer prompt's last "Question:" line."""
    return task.prompt.rsplit("Question: ", 1)[1].rsplit("\nAnswer:", 1)[0]


def judge_tasks(tasks: Sequence[Task], answers: ResponseCache) -> list[Task]:
    """One grading task per answered task the shortcuts do not settle."""
    out = []
    for t in tasks:
        record = answers.done.get(t.task_id)
        if record is None or shortcut(t, str(record["text"])) is not None:
            continue
        prompt = JUDGE_PROMPT.format(
            question=question_of(t), answers=" | ".join(t.answers), answer=str(record["text"]) or "(empty)"
        )
        out.append(Task(JUDGE_PREFIX + t.task_id, t.qid, t.kind, t.condition, prompt, t.answers))
    return out


def parse_verdict(text: str) -> str | None:
    words = normalize(text).split()
    return words[0] if words and words[0] in VERDICTS else None


def verdicts(tasks: Sequence[Task], answers: ResponseCache, judged: ResponseCache) -> dict[str, str]:
    """task_id -> verdict, for every answered task that is graded."""
    out = {}
    for t in tasks:
        record = answers.done.get(t.task_id)
        if record is None:
            continue
        verdict = shortcut(t, str(record["text"]))
        if verdict is None and (grade := judged.done.get(JUDGE_PREFIX + t.task_id)) is not None:
            verdict = parse_verdict(str(grade["text"]))
        if verdict is not None:
            out[t.task_id] = verdict
    return out


@dataclass(frozen=True, slots=True)
class VerdictRow:
    kind: str
    condition: str
    n: int
    correct: float
    abstain: float
    wrong: float
    mean_input_tokens: float


def summary(tasks: Sequence[Task], answers: ResponseCache, graded: Mapping[str, str]) -> list[VerdictRow]:
    groups: dict[tuple[str, str], list[Task]] = {}
    for t in tasks:
        if t.task_id in graded:
            groups.setdefault((t.kind, t.condition), []).append(t)
    rows = []
    for (kind, condition), items in sorted(groups.items()):
        share = {v: sum(graded[t.task_id] == v for t in items) / len(items) for v in VERDICTS}
        tokens = sum(int(str(answers.done[t.task_id]["input_tokens"])) for t in items) / len(items)
        rows.append(VerdictRow(kind, condition, len(items), share[CORRECT], share[ABSTAIN], share[WRONG], tokens))
    return rows


# ------------------------------------------------------------------ paired


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value."""
    n, m = wins + losses, min(wins, losses)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(m + 1)) / float(2**n))


@dataclass(frozen=True, slots=True)
class Paired:
    a: str
    b: str
    stratum: str  # "all", "known" (closed-book correct) or "unknown"
    n: int
    accuracy_a: float
    accuracy_b: float
    difference: float  # a - b
    ci_low: float
    ci_high: float
    wins: int  # a correct, b not
    losses: int
    p_value: float


def paired(a: str, b: str, stratum: str, pairs: Sequence[tuple[bool, bool]]) -> Paired | None:
    n = len(pairs)
    if n < 2:
        return None
    d = [int(x) - int(y) for x, y in pairs]
    mean = sum(d) / n
    se = math.sqrt(sum((v - mean) ** 2 for v in d) / (n * (n - 1)))
    wins, losses = d.count(1), d.count(-1)
    return Paired(
        a, b, stratum, n, sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n,
        mean, mean - 1.96 * se, mean + 1.96 * se, wins, losses, sign_test(wins, losses),
    )


def compare(
    a: str,
    b: str,
    graded_a: Mapping[str, str],
    graded_b: Mapping[str, str],
    closed: Mapping[str, str],
    kind: str = "answerable",
) -> list[Paired]:
    """Condition ``a`` (verdicts ``graded_a``) against ``b``, on questions graded in both.

    The two verdict maps may come from different raw caches (rerankers);
    ``closed`` holds the closed-book verdicts that define the strata.
    """
    strata: dict[str, list[tuple[bool, bool]]] = {"all": [], "known": [], "unknown": []}
    suffix = f":{kind}:{a}"
    for task_id, verdict in graded_a.items():
        if not task_id.endswith(suffix):
            continue
        stem = task_id[: -len(a)]
        other = graded_b.get(stem + b)
        if other is None:
            continue
        pair = (verdict == CORRECT, other == CORRECT)
        strata["all"].append(pair)
        known = closed.get(stem + "closed_book")
        if known is not None:
            strata["known" if known == CORRECT else "unknown"].append(pair)
    return [row for name, pairs in strata.items() if (row := paired(a, b, name, pairs)) is not None]
