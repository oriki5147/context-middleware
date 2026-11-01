"""Phase 1: document -> Markdown -> sentence-aligned, near-fixed-length chunks.

Chunks are measured in tokens of the retriever's own tokenizer, so
``chunk_size_tokens`` is exactly what the encoder sees. Whole sentences are
packed into each chunk up to that budget, and the overlap is made of whole
sentences carried over from the previous chunk, so the compressor (Phase 3)
never sees a sentence fragment. Only a sentence longer than the budget is
split, at word boundaries. Chunk content is always a verbatim slice of the
Markdown.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Protocol

from .models import TextSegment
from .sentences import SentenceSplitter, pysbd_splitter, sentence_spans

_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)


class OffsetEncodingLike(Protocol):
    @property
    def offsets(self) -> list[tuple[int, int]]: ...


class OffsetTokenizerLike(Protocol):
    """Structural subset of ``tokenizers.Tokenizer``. Must not truncate."""

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> OffsetEncodingLike: ...


class _Unit(NamedTuple):
    """A sentence (or a piece of an over-long one): char span and token count."""

    start: int
    end: int
    tokens: int


class MarkItDownPreprocessor:
    def __init__(
        self,
        tokenizer: OffsetTokenizerLike,
        chunk_size_tokens: int = 64,
        chunk_overlap_tokens: int = 0,
        sentence_splitter: SentenceSplitter | None = None,
    ) -> None:
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if not 0 <= chunk_overlap_tokens < chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be in [0, chunk_size_tokens)")
        self._tokenizer = tokenizer
        self._split = sentence_splitter or pysbd_splitter()
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        chunk_size_tokens: int = 64,
        chunk_overlap_tokens: int = 0,
        sentence_splitter: SentenceSplitter | None = None,
    ) -> MarkItDownPreprocessor:
        """Load ``<model_dir>/tokenizer.json``: a separate instance from the retriever's,
        since that one truncates to the encoder's max length."""
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return cls(tokenizer, chunk_size_tokens, chunk_overlap_tokens, sentence_splitter)

    def convert(self, path: str | Path) -> str:
        """Convert a PDF/DOCX/XLSX/... file to Markdown via MarkItDown."""
        from markitdown import MarkItDown

        text: str = MarkItDown().convert(str(path)).text_content
        return text

    def chunk(self, markdown: str, source_doc: str, title: str = "") -> list[TextSegment]:
        """Pack whole sentences into chunks of at most ``chunk_size_tokens``."""
        units = self._units(markdown)
        return [
            TextSegment(
                segment_id=f"{source_doc}#{i}",
                content=markdown[units[first].start : units[last - 1].end],
                source_doc=source_doc,
                title=title,
            )
            for i, (first, last) in enumerate(self._pack(units))
        ]

    def count_tokens(self, text: str) -> int:
        """Encoder tokens in ``text`` (no special tokens, no truncation)."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).offsets)

    def process(self, path: str | Path) -> list[TextSegment]:
        markdown = self.convert(path)
        return self.chunk(markdown, source_doc=str(path), title=_title(markdown, Path(path)))

    # ---------------------------------------------------------------- internals

    def _units(self, text: str) -> list[_Unit]:
        """Sentence spans with token counts; over-long sentences cut at word starts."""
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        units: list[_Unit] = []
        tok = 0  # first token not yet assigned to a unit
        # Every token is assigned to a unit, so text the splitter dropped
        # mid-paragraph is absorbed by the following sentence's unit.
        for _, end in sentence_spans(text, self._split):
            first = tok
            while tok < len(offsets) and offsets[tok][0] < end:
                tok += 1
            if tok == first:
                continue
            for a, b in _word_aligned_pieces(offsets, first, tok, self.chunk_size_tokens):
                units.append(_Unit(offsets[a][0], offsets[b - 1][1], b - a))
        return units

    def _pack(self, units: Sequence[_Unit]) -> list[tuple[int, int]]:
        """Unit index ranges ``[first, last)``: greedy fill, whole-unit overlap."""
        size, overlap = self.chunk_size_tokens, self.chunk_overlap_tokens
        ranges: list[tuple[int, int]] = []
        first = 0
        while first < len(units):
            last, total = first, 0
            while last < len(units) and total + units[last].tokens <= size:
                total += units[last].tokens
                last += 1
            ranges.append((first, last))
            if last == len(units):
                break
            # Carry over trailing units until they cover ``overlap`` tokens,
            # without carrying the whole chunk (which would make no progress).
            nxt, carried = last, 0
            while nxt - 1 > first and carried < overlap:
                nxt -= 1
                carried += units[nxt].tokens
            # Keep the next chunk within budget even after the carried units.
            while nxt < last and carried + units[last].tokens > size:
                carried -= units[nxt].tokens
                nxt += 1
            first = nxt
        return ranges


def _word_aligned_pieces(
    offsets: Sequence[tuple[int, int]], first: int, last: int, size: int
) -> list[tuple[int, int]]:
    """Split tokens ``[first, last)`` into consecutive pieces of at most ``size``
    tokens, cutting only before a word start unless a single word exceeds ``size``."""
    pieces: list[tuple[int, int]] = []
    start = first
    while start < last:
        end = min(start + size, last)
        while end < last and end > start + 1 and offsets[end][0] == offsets[end - 1][1]:
            end -= 1
        if end < last and offsets[end][0] == offsets[end - 1][1]:
            end = min(start + size, last)  # one word longer than the budget: hard cut
        pieces.append((start, end))
        start = end
    return pieces


def _title(markdown: str, path: Path) -> str:
    match = _H1.search(markdown)
    return match.group(1) if match else path.stem
