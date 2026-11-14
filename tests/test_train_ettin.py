import random

import pytest

from evaluation.train_distill import Example
from evaluation.train_ettin import sample_list, set_target


def test_set_target_spreads_the_weight_evenly_over_the_picks() -> None:
    assert set_target(3, 5) == [pytest.approx(1 / 3)] * 3 + [0.0, 0.0]
    assert set_target(1, 3) == [1.0, 0.0, 0.0]


def test_sample_list_keeps_the_picks_first_and_adds_unpicked_sentences() -> None:
    example = Example("q", [f"s{i}" for i in range(8)], [5, 2], [False] * 8)
    chosen = sample_list(example, 6, random.Random(0))
    assert chosen[:2] == [5, 2]
    assert len(chosen) == len(set(chosen)) == 6
    # A short shortlist gives every sentence once.
    assert sorted(sample_list(example, 20, random.Random(0))) == list(range(8))
