from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from context_middleware import ONNXCrossEncoderReranker, TextSegment


@dataclass
class FakeEncoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class FakePairTokenizer:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def encode_batch(self, input: list[tuple[str, str]]) -> list[FakeEncoding]:
        self.pairs.extend(input)
        # Encode the passage length as the single token id.
        return [FakeEncoding([len(p)], [1], [0]) for _, p in input]


class FakeSession:
    """Logit = token id - 5, so a 5-char passage scores sigmoid(0) = 0.5."""

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.inputs = ("input_ids", "attention_mask", "token_type_ids")

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=n) for n in self.inputs]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        assert "token_type_ids" in input_feed
        self.batches.append(len(input_feed["input_ids"]))
        return [(input_feed["input_ids"] - 5).astype(np.float32)]


def test_scores_are_sigmoid_of_logits_and_title_prefixed() -> None:
    tokenizer = FakePairTokenizer()
    reranker = ONNXCrossEncoderReranker(FakeSession(), tokenizer)
    scores = reranker.score("q", [TextSegment("1", "abcde", "d"), TextSegment("2", "x", "d", title="T")])
    assert scores == pytest.approx([0.5, 1 / (1 + np.exp(2))])  # "T\nx" -> 3 chars -> logit -2
    assert tokenizer.pairs == [("q", "abcde"), ("q", "T\nx")]


def test_each_pair_runs_alone_by_default_and_workers_keep_the_order() -> None:
    session = FakeSession()
    segments = [TextSegment(str(i), "x" * i, "d") for i in range(1, 8)]
    alone = ONNXCrossEncoderReranker(session, FakePairTokenizer()).score("q", segments)
    assert session.batches == [1] * 7  # a score never depends on batch-mates
    parallel = ONNXCrossEncoderReranker(FakeSession(), FakePairTokenizer(), workers=4).score("q", segments)
    assert parallel.tolist() == alone.tolist()
    assert np.all(np.diff(alone) > 0)  # longer passage -> higher fake logit, in input order


def test_invalid_sizes_raise() -> None:
    with pytest.raises(ValueError):
        ONNXCrossEncoderReranker(FakeSession(), FakePairTokenizer(), batch_size=0)
    with pytest.raises(ValueError):
        ONNXCrossEncoderReranker(FakeSession(), FakePairTokenizer(), workers=0)


def test_batches_pairs_and_handles_empty_input() -> None:
    session = FakeSession()
    reranker = ONNXCrossEncoderReranker(session, FakePairTokenizer(), batch_size=2)
    assert reranker.score("q", []).shape == (0,)
    reranker.score("q", [TextSegment(str(i), "abc", "d") for i in range(5)])
    assert session.batches == [2, 2, 1]
