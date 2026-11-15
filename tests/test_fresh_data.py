from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.fresh_data import Lexical, build_examples, load, parse_written, passages, save, write_tasks


def sentence(tag: str, n: int) -> str:
    return " ".join([tag] * (n - 1)) + " end."


def test_passages_keep_sentences_whole_and_stop_at_headings_and_references() -> None:
    text = (
        f"{sentence('a', 60)} {sentence('b', 60)} {sentence('c', 50)}\n\n"
        f"== History ==\n{sentence('d', 45)}\n\n"
        f"=== Short ===\n{sentence('e', 10)}\n\n"
        f"== References ==\n{sentence('f', 200)}"
    )
    result = passages(text)
    # 120 words close the first passage; the 50-word rest and the 45-word section stand alone;
    # a 10-word section is too short, and nothing after "References" is kept.
    assert [len(p.split()) for p in result] == [120, 50, 45]
    assert result[0].startswith("a a") and result[0].endswith("b end.")
    assert not any("==" in p or "f f" in p or "e e" in p for p in result)


def test_parse_written_enforces_the_question_rules() -> None:
    passage = "Storm Adam was named by the Spanish State Meteorological Agency on 3 October 2026."

    def reply(question: str, answer: str) -> str:
        return f'Here it is: {{"question": "{question}", "answer": "{answer}"}}'

    good = reply("Which agency named Storm Adam in October 2026?", "Spanish State Meteorological Agency")
    assert parse_written(good, passage) == (
        "Which agency named Storm Adam in October 2026?", "Spanish State Meteorological Agency"
    )
    assert parse_written("SKIP", passage) is None
    assert parse_written("{not json}", passage) is None
    rejected = [
        reply("Which agency named Storm Adam in October 2026?", "AEMET"),  # not in the passage
        reply("Which agency named Storm Adam in October 2026?", "the Spanish State Meteorological Agency on 3"),  # too long
        reply("When did the Spanish State Meteorological Agency name Storm Adam?", "Spanish State Meteorological Agency"),  # leaked
        reply("Was Storm Adam named by a Spanish agency in 2026?", "Spanish"),  # yes/no
        reply("Which agency named Storm Adam? And when?", "3 October 2026"),  # two questions
        reply("Which agency does the passage say named Storm Adam?", "Spanish State Meteorological Agency"),
        reply("Storm Adam agency", "Spanish State Meteorological Agency"),  # not a question
    ]
    assert [parse_written(r, passage) for r in rejected] == [None] * len(rejected)


def test_lexical_ranks_by_shared_rare_words() -> None:
    index = Lexical(["the storm hit spain", "the election in spain", "the film festival"])
    scores = index.scores("storm in Spain?")
    assert list(np.argsort(-scores)) == [0, 1, 2]
    assert scores[2] == 0.0
    assert index.scores("zebra").sum() == 0.0


def test_build_examples_adds_the_closest_passages_without_the_answer(tmp_path: Path) -> None:
    def article(title: str, *facts: str) -> dict[str, str]:
        return {"title": title, "text": "\n\n== S ==\n".join(f"{fact} {sentence('pad', 45)}" for fact in facts)}

    articles = [
        article("Storm Adam", "Storm Adam was named by Aemet.", "Storm Adam flooded Valencia.", "Storm Adam ended Friday."),
        article("Storm Bea", "Storm Bea was named by Aemet too.", "Storm Bea flooded Porto."),
        article("Film Festival", "The festival opened at Rome."),
    ]
    tasks = write_tasks(articles, per_article=5)
    assert len(tasks) == 6 and len({t.task_id for t in tasks}) == 6
    first = next(t for t in tasks if "named by Aemet." in t.context)
    written = [
        (first, '{"question": "Which agency named Storm Adam in 2026?", "answer": "Aemet"}'),
        (first, '{"question": "which agency named storm adam in 2026?", "answer": "Aemet"}'),  # duplicate
        (tasks[-1], "SKIP"),
    ]
    (example,) = build_examples(articles, written, total_docs=4)
    assert example.question == "Which agency named Storm Adam in 2026?" and example.answers == ("Aemet",)
    assert example.docs[0].is_gold and example.docs[0].text == first.context
    others = example.docs[1:]
    assert len(others) == 3 and not any(d.is_gold or "Aemet" in d.text for d in others)
    # The same storm's other passages rank above the unrelated festival.
    assert [d.title for d in others[:2]] == ["Storm Adam", "Storm Adam"]
    # Too few answer-free passages: the question is dropped, not padded.
    assert build_examples(articles, written, total_docs=6) == []

    save([example], tmp_path / "q.jsonl")
    assert load(tmp_path / "q.jsonl") == [example]
