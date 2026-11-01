from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from context_middleware import SelectiveSentenceCompressor, TextSegment


class FakeEncoder:
    """1-d embeddings: a sentence's score is looked up by its text (query = Q = [1])."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.embedded: list[TextSegment] = []

    def embed_segments(self, segments: Sequence[TextSegment]) -> NDArray[np.float32]:
        self.embedded.extend(segments)
        return np.array([[self.scores.get(s.content, 0.0)] for s in segments], dtype=np.float32)


Q = np.ones(1, dtype=np.float32)


def period_splitter(paragraph: str) -> Sequence[str]:
    return re.findall(r"[^.]+(?:\.|$)\s*", paragraph)


SCORES = {"A.": 0.9, "B.": 0.7, "C.": 0.5, "D.": 0.2}


def make(**kwargs: Any) -> SelectiveSentenceCompressor:
    """Threshold-driven by default here, to test the threshold rules in isolation."""
    kwargs.setdefault("sentence_splitter", period_splitter)
    kwargs.setdefault("sentence_threshold", 0.65)
    kwargs.setdefault("empty_threshold", 0.45)
    kwargs.setdefault("max_sentences", None)
    return SelectiveSentenceCompressor(FakeEncoder(SCORES), **kwargs)


def chunk(segment_id: str, content: str, source_doc: str = "doc", title: str = "") -> TextSegment:
    return TextSegment(segment_id, content, source_doc, title=title)


CHUNKS = [chunk("doc#0", "C. A. D."), chunk("doc#1", "B.")]


def test_keeps_sentences_above_threshold_best_first() -> None:
    kept = make().compress(Q, CHUNKS)
    assert [s.content for s in kept] == ["A.", "B."]
    assert [s.score for s in kept] == pytest.approx([0.9, 0.7])
    assert [s.segment_id for s in kept] == ["doc#0:1", "doc#1:0"]


def test_default_selection_is_cap_driven() -> None:
    # Production defaults: no sentence threshold, keep the best 3.
    compressor = SelectiveSentenceCompressor(FakeEncoder(SCORES), sentence_splitter=period_splitter)
    assert [s.content for s in compressor.compress(Q, CHUNKS)] == ["A.", "B.", "C."]


def test_default_empty_threshold_applies() -> None:
    low = FakeEncoder({"A.": 0.6, "B.": 0.5})
    compressor = SelectiveSentenceCompressor(low, sentence_splitter=period_splitter)
    assert compressor.compress(Q, [chunk("doc#0", "A. B.")]) == []  # 0.6 <= 0.64


def test_empty_when_best_sentence_at_or_below_empty_threshold() -> None:
    assert make(empty_threshold=0.9, sentence_threshold=0.95).compress(Q, CHUNKS) == []


def test_between_thresholds_keeps_top_min_sentences() -> None:
    compressor = make(sentence_threshold=0.95, empty_threshold=0.45, min_sentences=2)
    assert [s.content for s in compressor.compress(Q, CHUNKS)] == ["A.", "B."]


def test_min_sentences_zero_allows_empty_between_thresholds() -> None:
    assert make(sentence_threshold=0.95, min_sentences=0).compress(Q, CHUNKS) == []


def test_max_sentences_caps_output() -> None:
    kept = make(sentence_threshold=0.1, empty_threshold=0.0, max_sentences=3).compress(Q, CHUNKS)
    assert [s.content for s in kept] == ["A.", "B.", "C."]


def test_overlap_duplicates_scored_once_first_occurrence_wins() -> None:
    encoder = FakeEncoder(SCORES)
    compressor = SelectiveSentenceCompressor(encoder, sentence_splitter=period_splitter)
    overlapping = [chunk("doc#1", "B. A."), chunk("doc#0", "A. C."), chunk("x#0", "A.", "other")]

    kept = compressor.compress(Q, overlapping)

    assert [s.content for s in encoder.embedded] == ["B.", "A.", "C.", "A."]  # per-doc dedupe
    assert [(s.segment_id, s.content) for s in kept] == [
        ("doc#1:1", "A."),
        ("x#0:0", "A."),
        ("doc#1:0", "B."),
    ]


def test_sentences_carry_chunk_title_and_stay_verbatim() -> None:
    encoder = FakeEncoder(SCORES)
    compressor = SelectiveSentenceCompressor(encoder, sentence_splitter=period_splitter)
    kept = compressor.compress(Q, [chunk("doc#0", "  A.\n\nB.  ", title="Doc Title")])
    assert {s.title for s in encoder.embedded} == {"Doc Title"}
    assert [s.content for s in kept] == ["A.", "B."]


def test_no_chunks_returns_empty() -> None:
    assert make().compress(Q, []) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_sentences": -1},
        {"max_sentences": 0},
        {"min_sentences": 3, "max_sentences": 2},
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make(**kwargs)


class FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[str] = []

    def score(self, query: str, segments: Sequence[TextSegment]) -> NDArray[np.float32]:
        self.seen.extend(s.content for s in segments)
        return np.array([self.scores[s.content] for s in segments], dtype=np.float32)


def test_reranker_rescores_bi_encoder_shortlist_only() -> None:
    # Bi-encoder order A > B > C > D; reranker prefers C, and D is never shortlisted.
    reranker = FakeReranker({"A.": 0.2, "B.": 0.1, "C.": 0.9})
    compressor = make(reranker=reranker, rerank_top_n=3, sentence_threshold=0.5, empty_threshold=0.05)
    kept = compressor.compress(Q, CHUNKS, "q")
    assert sorted(reranker.seen) == ["A.", "B.", "C."]
    assert [s.content for s in kept] == ["C."]
    assert kept[0].score == pytest.approx(0.9)


def test_reranker_scores_drive_the_empty_rule() -> None:
    reranker = FakeReranker({"A.": 0.01, "B.": 0.02, "C.": 0.03, "D.": 0.0})
    compressor = make(reranker=reranker, sentence_threshold=0.5, empty_threshold=0.05)
    assert compressor.compress(Q, CHUNKS, "q") == []


def test_reranker_requires_query_text() -> None:
    with pytest.raises(ValueError):
        make(reranker=FakeReranker({})).compress(Q, CHUNKS)
