from __future__ import annotations

import random

import numpy as np
import pytest

from evaluation.train_distill import Example, sample_list, sentence_text, val_metrics


def test_sentence_text_matches_production_input_and_appends_neighbors() -> None:
    assert sentence_text("X-ray", "It was found in 1895.") == "X-ray\nIt was found in 1895."
    assert sentence_text("X-ray", "It was found in 1895.", "Röntgen studied rays.", "", context=True) == (
        "X-ray\nIt was found in 1895.\nContext: Röntgen studied rays."
    )
    assert sentence_text("", "Alone.", "", "", context=True) == "Alone."


def test_sample_list_keeps_picks_first_with_normalized_target() -> None:
    ex = Example("q", [f"s{i}" for i in range(20)], [4, 9], [False] * 20)
    chosen, target = sample_list(ex, 10, random.Random(0))
    assert chosen[:2] == [4, 9] and len(chosen) == 10 and len(set(chosen)) == 10
    assert target[:2] == pytest.approx([0.6 / 0.85, 0.25 / 0.85]) and sum(target) == pytest.approx(1.0)


def test_val_metrics() -> None:
    ex = Example("q", ["a", "b", "c", "d"], [1, 0, 2], [False, True, False, False])
    m = val_metrics([ex], [np.array([0.1, 0.9, 0.2, 0.0], dtype=np.float32)])
    assert m == {"teacher_top1": 1.0, "answer_rank1": 1.0, "answer_top3": 1.0}
