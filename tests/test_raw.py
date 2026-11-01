from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from evaluation.raw import RawQuestion, derive

f32 = np.float32


def make_raw(**overrides: object) -> RawQuestion:
    """Doc 0 (gold) has chunks c0, c1; doc 1 has chunk c2; every chunk is 10 tokens.

    Dense chunk scores with q = [1, 0]: c0 = 1.0, c1 = 0.5, c2 = 0.8.
    Sentences: c0 -> "A." "B.", c1 -> "C.", c2 -> "A." "A." (a repeat within doc 1).
    """
    raw = RawQuestion(
        qid=7,
        kind="answerable",
        question="q",
        answers=("B",),
        gold_doc=0,
        query_embedding=np.array([1, 0], dtype=f32),
        chunk_doc=np.array([0, 0, 1], dtype=np.int32),
        chunk_text=["A. B.", "C.", "A. A."],
        chunk_title=["T0", "T0", "T1"],
        chunk_tokens=np.array([10, 10, 10], dtype=np.int32),
        chunk_embedding=np.array([[1, 0], [0.5, 0], [0.8, 0]], dtype=f32),
        sent_chunk=np.array([0, 0, 1, 2, 2], dtype=np.int32),
        sent_doc=np.array([0, 0, 0, 1, 1], dtype=np.int32),
        sent_pos=np.array([0, 1, 2, 0, 1], dtype=np.int32),
        sent_text=["A.", "B.", "C.", "A.", "A."],
        sent_tokens=np.array([2, 2, 2, 2, 2], dtype=np.int32),
        sent_has_answer=np.array([False, True, False, False, False]),
        sent_embedding=np.array([[0.1, 0], [0.9, 0], [0.5, 0], [0.3, 0], [0.3, 0]], dtype=f32),
        sent_rerank=np.array([0.01, 0.9, 0.2, 0.05, 0.05], dtype=f32),
    )
    return replace(raw, **overrides)  # type: ignore[arg-type]


def test_derive_retrieves_best_chunks_within_token_budget() -> None:
    c = derive(make_raw(), retrieval_tokens=20)  # c0 (1.0) and c2 (0.8) fit; c1 doesn't
    assert c.original_tokens == 20
    assert c.gold_retrieved and c.answer_in_retrieved
    # Candidates follow compressor.split: by chunk rank, the repeat in doc 1 dropped.
    assert c.scores.tolist() == pytest.approx([0.1, 0.9, 0.3])
    assert c.rerank_scores.tolist() == pytest.approx([0.01, 0.9, 0.05])
    assert c.has_answer.tolist() == [False, True, False]
    assert c.reader_scores is None


def test_derive_always_retrieves_at_least_one_chunk() -> None:
    assert derive(make_raw(), retrieval_tokens=5).original_tokens == 10


def test_score_overrides_change_retrieval_and_ranking() -> None:
    c = derive(
        make_raw(),
        chunk_scores=np.array([0.0, 0.0, 1.0], dtype=f32),  # e.g. a BM25 fusion favoring doc 1
        sentence_scores=np.arange(5, dtype=f32),
        retrieval_tokens=10,
    )
    assert not c.gold_retrieved and not c.answer_in_retrieved
    assert c.scores.tolist() == [3.0]


def test_unanswerable_kinds_have_no_answer_flags() -> None:
    c = derive(make_raw(kind="distractors", gold_doc=-1), retrieval_tokens=30)
    assert not c.gold_retrieved and not c.answer_in_retrieved
