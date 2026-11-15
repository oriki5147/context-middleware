from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
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
    # Production defaults: no sentence threshold, keep the best 5 (here all 4, best first).
    compressor = SelectiveSentenceCompressor(FakeEncoder(SCORES), sentence_splitter=period_splitter)
    assert (compressor.max_sentences, compressor.sentence_threshold) == (5, None)
    assert [s.content for s in compressor.compress(Q, CHUNKS)] == ["A.", "B.", "C.", "D."]


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
    compressor = SelectiveSentenceCompressor(encoder, sentence_splitter=period_splitter, max_sentences=3)
    overlapping = [chunk("doc#1", "B. A."), chunk("doc#0", "A. C."), chunk("x#0", "A.", "other")]

    kept = compressor.compress(Q, overlapping)

    # Overlap repeats are dropped per document; the identical (title, text) from
    # another document reuses the same embedding but is still kept separately.
    assert [s.content for s in encoder.embedded] == ["B.", "A.", "C."]
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


def test_reranker_scores_have_their_own_empty_threshold() -> None:
    reranker = FakeReranker({"A.": 0.01, "B.": 0.02, "C.": 0.03, "D.": 0.0})
    compressor = make(reranker=reranker, sentence_threshold=0.5, rerank_empty_threshold=0.05)
    assert compressor.compress(Q, CHUNKS, "q") == []
    assert reranker.seen


def test_bi_encoder_empty_rule_runs_before_the_reranker() -> None:
    # Best bi-encoder score is 0.9: off-topic by that rule, so nothing is reranked.
    reranker = FakeReranker({"A.": 0.9, "B.": 0.9, "C.": 0.9, "D.": 0.9})
    compressor = make(reranker=reranker, empty_threshold=0.95)
    assert compressor.compress(Q, CHUNKS, "q") == []
    assert reranker.seen == []


def test_rerank_context_shows_the_reranker_document_neighbors() -> None:
    # Documents: "doc" = C. A. D. B. and "other" = E.
    chunks = [*CHUNKS, chunk("other#0", "E.", source_doc="other")]
    in_context = {
        "C.\nContext: A.": 0.1, "A.\nContext: C. ... D.": 0.9, "D.\nContext: A. ... B.": 0.2,
        "B.\nContext: D.": 0.3, "E.": 0.4,
    }
    reranker = FakeReranker(in_context)
    compressor = make(reranker=reranker, sentence_threshold=None, max_sentences=2, empty_threshold=-1.0)
    compressor.precompute(chunks)
    kept = compressor.compress(Q, chunks[::-1], "q")  # retrieval order is not document order
    assert sorted(reranker.seen) == sorted(in_context)
    assert [s.content for s in kept] == ["A.", "E."]  # the output stays the sentence alone


def test_rerank_context_needs_precompute_and_can_be_turned_off() -> None:
    plain = {"A.": 0.9, "B.": 0.1, "C.": 0.1, "D.": 0.1}
    reranker = FakeReranker(plain)
    make(reranker=reranker).compress(Q, CHUNKS, "q")  # nothing ingested: no neighbors known
    off = make(reranker=reranker, rerank_context=False)
    off.precompute(CHUNKS)
    off.compress(Q, CHUNKS, "q")
    assert sorted(reranker.seen) == sorted([*plain, *plain])


def test_second_reranker_rescores_only_the_first_rerankers_head() -> None:
    # First reranker: C > A > B > D. The second sees C, A, B only and prefers B.
    first = FakeReranker({"A.": 0.6, "B.": 0.5, "C.": 0.9, "D.": 0.4})
    second = FakeReranker({"A.": 0.2, "B.": 0.8, "C.": 0.3})
    compressor = make(reranker=first, second_reranker=second, second_top_n=3, sentence_threshold=None, max_sentences=2)
    kept = compressor.compress(Q, CHUNKS, "q")
    assert sorted(second.seen) == ["A.", "B.", "C."]
    assert [s.content for s in kept] == ["B.", "C."]
    assert [s.score for s in kept] == pytest.approx([0.8, 0.3])


def test_second_reranker_does_not_overrule_the_empty_rule_or_run_without_a_first() -> None:
    first = FakeReranker({"A.": 0.01, "B.": 0.02, "C.": 0.03, "D.": 0.0})
    second = FakeReranker({"A.": 0.9, "B.": 0.9, "C.": 0.9, "D.": 0.9})
    compressor = make(reranker=first, second_reranker=second, rerank_empty_threshold=0.05)
    assert compressor.compress(Q, CHUNKS, "q") == []
    assert second.seen == []
    with pytest.raises(ValueError):
        make(second_reranker=second)
    # Its scores never empty the context: that was the first reranker's call.
    low = FakeReranker({"A.": 0.001, "B.": 0.0, "C.": 0.0, "D.": 0.0})
    kept = make(reranker=FakeReranker({"A.": 0.9, "B.": 0.8, "C.": 0.7, "D.": 0.6}), second_reranker=low,
                sentence_threshold=None, max_sentences=1).compress(Q, CHUNKS, "q")
    assert [s.content for s in kept] == ["A."]


def test_reranker_requires_query_text() -> None:
    with pytest.raises(ValueError):
        make(reranker=FakeReranker({})).compress(Q, CHUNKS)


def test_precompute_moves_embedding_out_of_the_query_path() -> None:
    encoder = FakeEncoder(SCORES)
    compressor = SelectiveSentenceCompressor(encoder, sentence_splitter=period_splitter, max_sentences=3)
    compressor.precompute(CHUNKS)
    embedded_at_ingest = len(encoder.embedded)
    assert embedded_at_ingest == 4

    kept = compressor.compress(Q, CHUNKS)

    assert len(encoder.embedded) == embedded_at_ingest  # nothing encoded at query time
    cold = SelectiveSentenceCompressor(FakeEncoder(SCORES), sentence_splitter=period_splitter, max_sentences=3)
    # Identical results with or without the cache, except that only ingested
    # sentences know their position in the document (C. A. D. B.).
    assert [replace(s, position=None) for s in kept] == cold.compress(Q, CHUNKS)
    assert [(s.content, s.position) for s in kept] == [("A.", 1), ("B.", 3), ("C.", 0)]


def test_cache_misses_are_encoded_once_then_reused() -> None:
    encoder = FakeEncoder(SCORES)
    compressor = SelectiveSentenceCompressor(encoder, sentence_splitter=period_splitter)
    compressor.precompute(CHUNKS[:1])
    compressor.compress(Q, CHUNKS)  # "B." was not precomputed
    compressor.compress(Q, CHUNKS)
    assert [s.content for s in encoder.embedded].count("B.") == 1
    compressor.clear_cache()
    compressor.compress(Q, CHUNKS)
    assert [s.content for s in encoder.embedded].count("B.") == 2


def test_same_sentence_under_different_titles_is_embedded_separately() -> None:
    encoder = FakeEncoder(SCORES)
    compressor = SelectiveSentenceCompressor(encoder, sentence_splitter=period_splitter)
    compressor.precompute([chunk("a#0", "A.", "a", title="T1"), chunk("b#0", "A.", "b", title="T2")])
    assert [s.title for s in encoder.embedded] == ["T1", "T2"]
