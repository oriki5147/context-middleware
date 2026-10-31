"""Phase 4: U-shape (sandwich) reordering to counter "lost in the middle"."""

from __future__ import annotations

from collections.abc import Sequence

from .models import TextSegment


class UShapeReorderer:
    def reorder(self, segments: Sequence[TextSegment]) -> list[TextSegment]:
        """Place the highest-scoring segments at the start and end, weakest in the middle."""
        raise NotImplementedError
