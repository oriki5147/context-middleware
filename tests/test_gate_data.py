from __future__ import annotations

import random
import re
from collections.abc import Sequence
from pathlib import Path

from evaluation.gate_data import Pair, build, is_validation, question_pairs, read_jsonl, write_jsonl


def split(paragraph: str) -> Sequence[str]:
    return re.findall(r"[^.]+\.?", paragraph)


def ctx(title: str, text: str) -> dict[str, str]:
    return {"title": title, "text": text}


RECORD = {
    "question": "who discovered x-rays",
    "answers": ["Wilhelm Röntgen"],
    "positive_ctxs": [ctx("X-ray", "X-rays are radiation. They were discovered by Wilhelm Röntgen in 1895.")],
    "hard_negative_ctxs": [
        ctx("Radiation", "X-ray machines are used in hospitals. Some mention Wilhelm Röntgen."),
        ctx("Radiology", "Radiology uses X-rays. It is a medical field."),
    ],
    "negative_ctxs": [ctx("Paris", "Paris is in France.")],
}


def test_question_pairs_labels_and_noise_filter() -> None:
    pairs = question_pairs(3, RECORD, split, random.Random(0), hard=10)
    by_kind = {k: [p for p in pairs if p.kind == k] for k in ("answer", "hard", "easy")}
    assert [p.text for p in by_kind["answer"]] == ["X-ray\nThey were discovered by Wilhelm Röntgen in 1895."]
    assert all(p.label == 1 for p in by_kind["answer"]) and all(p.label == 0 for p in by_kind["hard"])
    # The non-answer sentence of the positive passage is ambiguous and left out;
    # the hard-negative sentence naming the answer is label noise and dropped.
    assert not any("radiation." in p.text for p in pairs)
    assert not any("Some mention" in p.text for p in pairs)
    assert len(by_kind["hard"]) == 3 and len(by_kind["easy"]) == 1
    assert all(p.group == 3 and p.question == RECORD["question"] for p in pairs)


def test_question_without_answer_sentence_is_skipped() -> None:
    record = {**RECORD, "answers": ["Marie Curie"]}
    assert question_pairs(0, record, split, random.Random(0)) == []


def test_build_is_deterministic_and_validation_split_is_stable() -> None:
    assert list(build([RECORD, RECORD], split)) == list(build([RECORD, RECORD], split))
    assert is_validation("who discovered x-rays") == is_validation("who discovered x-rays")
    questions = [f"question {i}" for i in range(2000)]
    share = sum(map(is_validation, questions)) / len(questions)
    assert 0.07 < share < 0.13


def test_jsonl_round_trip(tmp_path: Path) -> None:
    pairs = [Pair(1, "q", "t\ns", 1, "answer"), Pair(1, "q", "t\nn", 0, "hard")]
    assert write_jsonl(pairs, tmp_path / "p.jsonl") == 2
    assert read_jsonl(tmp_path / "p.jsonl") == pairs
