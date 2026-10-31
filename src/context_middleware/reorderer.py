"""Phase 4: order selected sentences to counter "lost in the middle".

BEST_FIRST (default) sorts by descending score, as LongLLMLingua's document
reordering does (Eq. 4) and as Lost in the Middle recommends ("pushing relevant
information closer to the start"). U_SHAPE alternates ranks between the start
and the end so the weakest land in the middle; no paper we follow evaluates
it, so it is kept as an option to measure, not a default.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from .models import TextSegment


class ReorderStrategy(str, Enum):
    BEST_FIRST = "best_first"
    U_SHAPE = "u_shape"


class ContextReorderer:
    def __init__(self, strategy: ReorderStrategy = ReorderStrategy.BEST_FIRST) -> None:
        self.strategy = strategy

    def reorder(self, segments: Sequence[TextSegment]) -> list[TextSegment]:
        """Return a new list ordered by ``strategy``; ties keep their input order."""
        ranked = sorted(segments, key=lambda s: s.score, reverse=True)
        if self.strategy is ReorderStrategy.BEST_FIRST:
            return ranked
        # Ranks 1, 3, 5, ... fill from the start; ranks 2, 4, 6, ... from the end.
        return ranked[0::2] + ranked[1::2][::-1]
