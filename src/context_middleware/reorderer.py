"""Phase 4: order selected sentences to counter "lost in the middle", and lay
them out as the context text.

BEST_FIRST (default) sorts by descending score, as LongLLMLingua's document
reordering does (Eq. 4) and as Lost in the Middle recommends ("pushing relevant
information closer to the start"). U_SHAPE alternates ranks between the start
and the end so the weakest land in the middle; no paper we follow evaluates
it, so it is kept as an option to measure, not a default.

The TITLED layout (default) groups the sentences by document under its title,
in document order, like retrieval results. LLMs read bare sentences worse than
contiguous chunks holding the same answer; on NQ dev the titles raised answer
accuracy by 3-4 points for Claude Haiku and 8 for a 4B Qwen (3 or 7 sentences).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from .models import TextSegment


class ReorderStrategy(str, Enum):
    BEST_FIRST = "best_first"
    U_SHAPE = "u_shape"


class ContextLayout(str, Enum):
    PLAIN = "plain"  # one sentence per line, in the reordered order
    TITLED = "titled"  # one line per document: its title, then its sentences


GAP = " ... "  # between two sentences of a document that are not adjacent


def render(segments: Sequence[TextSegment], layout: ContextLayout = ContextLayout.TITLED) -> str:
    """The context text for ``segments`` (already reordered).

    TITLED keeps the documents in the order their first sentence appears, so
    with BEST_FIRST the best sentence's document leads. Within a document,
    sentences follow document order when their ``position`` is known (given
    order otherwise), with ``GAP`` wherever they are not adjacent.
    """
    if layout is ContextLayout.PLAIN:
        return "\n".join(s.content for s in segments)
    by_doc: dict[str, list[TextSegment]] = {}
    for s in segments:
        by_doc.setdefault(s.source_doc, []).append(s)
    lines = []
    for n, members in enumerate(by_doc.values(), 1):
        if all(s.position is not None for s in members):
            members = sorted(members, key=lambda s: s.position or 0)
        body = members[0].content
        for before, s in zip(members, members[1:]):
            adjacent = before.position is not None and s.position == before.position + 1
            body += (" " if adjacent else GAP) + s.content
        header = f"Document [{n}](Title: {members[0].title})" if members[0].title else f"Document [{n}]"
        lines.append(f"{header} {body}")
    return "\n".join(lines)


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
