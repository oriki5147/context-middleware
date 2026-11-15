from __future__ import annotations

from typing import Any

from evaluation.trivia_data import examples


def ctx(title: str, text: str) -> dict[str, Any]:
    return {"title": title, "text": text, "score": 1.0}


def record(question: str, answers: list[str], positives: list[dict[str, Any]], hard: list[dict[str, Any]]) -> dict[str, Any]:
    return {"question": question, "answers": answers, "positive_ctxs": positives, "hard_negative_ctxs": hard}


def test_examples_take_the_first_answer_passage_and_the_closest_answer_free_ones() -> None:
    records = [
        record(
            "Which country does Alfred Brendel come from?",
            ["Austria", "Republic of Austria"],
            [ctx("Alfred Brendel", "He moved to Graz, Austria."), ctx("Graz", "Graz is in Austria.")],
            [ctx("Brendel", "A pianist."), ctx("Austria", "A country."), ctx("Piano", "Keys."), ctx("Poet", "Words.")],
        ),
        record("No answer passage?", ["x"], [], [ctx("a", "b"), ctx("c", "d"), ctx("e", "f")]),
        record("Too few others?", ["x"], [ctx("t", "x here")], [ctx("a", "b")]),
    ]
    (example,) = examples(records, total_docs=3)
    assert example.qid == 0 and example.answers == ("Austria", "Republic of Austria")
    assert [(d.title, d.is_gold) for d in example.docs] == [("Alfred Brendel", True), ("Brendel", False), ("Piano", False)]


def test_examples_come_in_a_fixed_random_order() -> None:
    records = [
        record(f"Question {i}?", [f"answer{i}"], [ctx("t", f"answer{i} here")], [ctx("a", "b"), ctx("c", "d")])
        for i in range(30)
    ]
    first, again = examples(records, total_docs=3), examples(records, total_docs=3)
    assert first == again and [e.qid for e in first] == list(range(30))
    assert [e.question for e in first] != [r["question"] for r in records]
