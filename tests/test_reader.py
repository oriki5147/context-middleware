from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from context_middleware import TextSegment
from evaluation.reader import ONNXExtractiveReader


@dataclass
class FakeEncoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]
    sequence_ids: list[int | None]


class FakeTokenizer:
    """<s> query </s> passage </s>; each passage word is one token whose id is its value."""

    def encode_batch(self, input: list[tuple[str, str]]) -> list[FakeEncoding]:
        out = []
        for _, passage in input:
            words = [int(w) for w in passage.split()]
            ids = [0, 0, 0] + words + [0]
            seq: list[int | None] = [None, 0, None] + [1] * len(words) + [None]
            out.append(FakeEncoding(ids, [1] * len(ids), [0] * len(ids), seq))
        width = max(len(e.ids) for e in out)
        for e in out:
            pad = width - len(e.ids)
            e.ids += [0] * pad
            e.attention_mask += [0] * pad
            e.type_ids += [0] * pad
            e.sequence_ids += [None] * pad
        return out


class FakeSession:
    """start = token id, end = 10 - token id; the no-answer position 0 scores 5 + 5."""

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        ids = input_feed["input_ids"].astype(np.float32)
        start, end = ids.copy(), 10.0 - ids
        start[:, 0] = end[:, 0] = 5.0
        return [start, end]


def seg(text: str) -> TextSegment:
    return TextSegment(text, text, "doc")


# Passage "9 0 0 1": start = [9, 0, 0, 1], end = [1, 10, 10, 9].
# Best span overall: start 9 + end 10 (two tokens) = 19; no-answer = 10.


def test_margin_is_best_span_minus_no_answer() -> None:
    reader = ONNXExtractiveReader(FakeSession(), FakeTokenizer())
    assert reader.score("q", [seg("9 0 0 1")]) == pytest.approx([9.0])


def test_span_length_limit_and_order() -> None:
    reader = ONNXExtractiveReader(FakeSession(), FakeTokenizer(), max_answer_tokens=1)
    # Single-token spans only: best is 9 + 1, 0 + 10 or 1 + 9 = 10 -> margin 0.
    assert reader.score("q", [seg("9 0 0 1")]) == pytest.approx([0.0])
    # An end before the start is never allowed: "0 9" -> best 0 + 10 = 10, not 9 + 10.
    assert ONNXExtractiveReader(FakeSession(), FakeTokenizer()).score("q", [seg("0 9")]) == pytest.approx([0.0])


def test_batches_and_empty_input() -> None:
    reader = ONNXExtractiveReader(FakeSession(), FakeTokenizer(), batch_size=1)
    assert reader.score("q", []).shape == (0,)
    assert reader.score("q", [seg("9 0"), seg("0 9")]) == pytest.approx([9.0, 0.0])
