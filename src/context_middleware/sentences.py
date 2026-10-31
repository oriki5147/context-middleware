"""Sentence segmentation shared by chunking (Phase 1) and compression (Phase 3).

Both phases must agree on sentence boundaries: the chunker packs whole
sentences, and the compressor re-splits chunks into exactly those sentences.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

# A paragraph is a run of text without a blank line.
_PARAGRAPH = re.compile(r"(?:[^\n]|\n(?![ \t]*\n))+")

# Splits a paragraph into sentences; each must appear verbatim, in order, in it.
SentenceSplitter = Callable[[str], Sequence[str]]


def pysbd_splitter() -> SentenceSplitter:
    import pysbd

    segmenter = pysbd.Segmenter(language="en", clean=False)

    def split(paragraph: str) -> Sequence[str]:
        sentences: list[str] = segmenter.segment(paragraph)
        return sentences

    return split


def sentence_spans(text: str, split: SentenceSplitter) -> list[tuple[int, int]]:
    """Character spans of the sentences in ``text``, whitespace-trimmed, in order.

    Paragraphs are split first, so a heading without punctuation never merges
    into the next sentence. Trailing text the splitter dropped or altered
    becomes its own span; text dropped mid-paragraph is left between spans.
    """
    spans: list[tuple[int, int]] = []
    for paragraph in _PARAGRAPH.finditer(text):
        base, body = paragraph.start(), paragraph.group()
        cursor = 0
        for sentence in split(body):
            stripped = sentence.strip()
            found = body.find(stripped, cursor) if stripped else -1
            if found < 0:
                continue
            spans.append((base + found, base + found + len(stripped)))
            cursor = found + len(stripped)
        if tail := body[cursor:].strip():
            found = body.find(tail, cursor)
            spans.append((base + found, base + found + len(tail)))
    return spans
