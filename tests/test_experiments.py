from __future__ import annotations

import numpy as np
import pytest

from evaluation.experiments import BM25, neighbors, rewrite_question, rrf, smooth, tokenize, zsum
from test_raw import make_raw


def test_tokenize_drops_stopwords_and_punctuation() -> None:
    assert tokenize("Who got the first Nobel Prize, in 1901?") == ["got", "first", "nobel", "prize", "1901"]


@pytest.mark.parametrize(
    ("question", "rewritten"),
    [
        ("who got the first nobel prize in physics", "got the first nobel prize in physics"),
        ("When was the Eiffel Tower built?", "the Eiffel Tower built"),
        ("how many states are in the us", "states are in the us"),
        ("name the first president", "name the first president"),
        ("who?", "who?"),
    ],
)
def test_rewrite_question(question: str, rewritten: str) -> None:
    assert rewrite_question(question) == rewritten


def test_bm25_prefers_rare_matching_terms() -> None:
    docs = [["nobel", "prize", "physics"], ["prize", "money"], ["roentgen", "physics", "nobel"]] * 3
    bm25 = BM25(docs)
    scores = bm25.scores(["roentgen", "prize"], docs[:3])
    assert scores.argmax() == 2  # rare "roentgen" outweighs common "prize"
    assert bm25.scores(["unknown"], docs[:1]).tolist() == [0.0]


def test_rrf_rewards_agreement() -> None:
    a = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fused = rrf(a, b)
    assert fused[1] < fused[0] == pytest.approx(fused[2])  # 1st+3rd beats 2nd+2nd under k=60


def test_zsum_weights_second_signal() -> None:
    a = np.array([0.0, 1.0], dtype=np.float32)
    b = np.array([10.0, 0.0], dtype=np.float32)
    assert zsum(a, b, 0.3).argmax() == 1
    assert zsum(a, b, 3.0).argmax() == 0


def test_neighbors_stay_within_document() -> None:
    raw = make_raw()  # sentences: doc 0 -> 0, 1, 2; doc 1 -> 3, 4
    assert neighbors(raw, 2, "both") == [1]
    assert neighbors(raw, 3, "both") == [4]
    assert neighbors(raw, 1, "prev") == [0] and neighbors(raw, 1, "next") == [2]


def test_smooth_adds_neighbor_mean() -> None:
    raw = make_raw()
    scores = np.array([1.0, 0.0, 1.0, 0.0, 2.0], dtype=np.float32)
    assert smooth(raw, scores, 0.5).tolist() == pytest.approx([1.0, 0.5, 1.0, 1.0, 2.0])
