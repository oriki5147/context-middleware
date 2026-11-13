import pytest

from context_middleware import ContextLayout, ContextReorderer, ReorderStrategy, TextSegment
from context_middleware.reorderer import render


def seg(segment_id: str, score: float) -> TextSegment:
    return TextSegment(segment_id, segment_id, "doc", score=score)


# Deliberately not in score order.
SEGMENTS = [seg("r3", 0.7), seg("r1", 0.9), seg("r5", 0.5), seg("r2", 0.8), seg("r4", 0.6)]


def ids(segments: list[TextSegment]) -> list[str]:
    return [s.segment_id for s in segments]


def test_best_first_is_default_and_sorts_descending() -> None:
    assert ids(ContextReorderer().reorder(SEGMENTS)) == ["r1", "r2", "r3", "r4", "r5"]


def test_u_shape_puts_best_at_edges_and_weakest_in_middle() -> None:
    ordered = ContextReorderer(ReorderStrategy.U_SHAPE).reorder(SEGMENTS)
    assert ids(ordered) == ["r1", "r3", "r5", "r4", "r2"]


def test_ties_keep_input_order() -> None:
    tied = [seg("a", 0.5), seg("b", 0.5), seg("c", 0.9)]
    assert ids(ContextReorderer().reorder(tied)) == ["c", "a", "b"]


@pytest.mark.parametrize("strategy", list(ReorderStrategy))
def test_does_not_mutate_input_and_handles_small_inputs(strategy: ReorderStrategy) -> None:
    reorderer = ContextReorderer(strategy)
    original = list(SEGMENTS)
    reorderer.reorder(SEGMENTS)
    assert SEGMENTS == original
    assert reorderer.reorder([]) == []
    assert ids(reorderer.reorder([seg("only", 0.1)])) == ["only"]


def sentence(content: str, doc: str, position: int | None, title: str = "") -> TextSegment:
    return TextSegment(content, content, doc, title=title, position=position)


def test_plain_layout_is_one_sentence_per_line() -> None:
    segments = [sentence("B.", "d1", 1), sentence("A.", "d0", 0)]
    assert render(segments, ContextLayout.PLAIN) == "B.\nA."


def test_titled_layout_groups_by_document_in_document_order() -> None:
    segments = [
        sentence("E.", "d1", 9, "Second"), sentence("C.", "d0", 2, "First"),
        sentence("A.", "d0", 0, "First"), sentence("D.", "d0", 3, "First"),
    ]
    # The first sentence's document leads; adjacent sentences join, a gap is marked.
    assert render(segments).splitlines() == ["Document [1](Title: Second) E.", "Document [2](Title: First) A. ... C. D."]


def test_titled_layout_without_titles_or_positions() -> None:
    # Not ingested: the given (reordered) order is kept, and adjacency is unknown.
    segments = [sentence("B.", "d0", None), sentence("A.", "d0", None)]
    assert render(segments, ContextLayout.TITLED) == "Document [1] B. ... A."
    assert render([]) == ""
