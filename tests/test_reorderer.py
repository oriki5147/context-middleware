import pytest

from context_middleware import ContextReorderer, ReorderStrategy, TextSegment


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
