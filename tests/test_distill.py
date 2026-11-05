from __future__ import annotations

from pathlib import Path

from evaluation.distill import Shortlist, custom_id, read_jsonl, write_jsonl
from evaluation.rank_judge import format_prompt


def make_shortlist() -> Shortlist:
    return Shortlist(
        group=4,
        question="who discovered x-rays",
        answers=["Wilhelm Röntgen"],
        validation=False,
        titles=["Radiology", "X-ray"],
        texts=["Radiology uses X-rays.", "Wilhelm Röntgen discovered X-rays."],
        has_answer=[False, True],
        bi=[0.71, 0.69],
    )


def test_prompt_numbers_sentences_in_shown_order() -> None:
    item = make_shortlist()
    assert item.prompt() == format_prompt(item.question, [("Radiology", "Radiology uses X-rays."), ("X-ray", "Wilhelm Röntgen discovered X-rays.")])
    assert "[1] (Radiology) Radiology uses X-rays.\n[2] (X-ray) Wilhelm Röntgen" in item.prompt()
    assert custom_id(item) == "q4"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    items = [make_shortlist()]
    assert write_jsonl(items, tmp_path / "s.jsonl") == 1
    assert read_jsonl(tmp_path / "s.jsonl") == items
