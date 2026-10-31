from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from context_middleware import MarkItDownPreprocessor

BGE_DIR = Path(__file__).parent.parent / "models" / "bge-small-en-v1.5"


@dataclass
class FakeEncoding:
    offsets: list[tuple[int, int]]


class FakeTokenizer:
    """Whitespace words, each split into word pieces of at most 4 characters."""

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> FakeEncoding:
        offsets = []
        for match in re.finditer(r"\S+", sequence):
            for start in range(match.start(), match.end(), 4):
                offsets.append((start, min(start + 4, match.end())))
        return FakeEncoding(offsets)


def period_splitter(paragraph: str) -> Sequence[str]:
    return re.findall(r"[^.]+(?:\.|$)\s*", paragraph)


def make(size: int, overlap: int) -> MarkItDownPreprocessor:
    return MarkItDownPreprocessor(FakeTokenizer(), size, overlap, period_splitter)


def token_count(text: str) -> int:
    return len(FakeTokenizer().encode(text).offsets)


# Sentences of 2-4 tokens ("longword" = 2 tokens).
SENTENCES = [f"s{i} {'longword ' * (i % 3)}end." for i in range(30)]
TEXT = " ".join(SENTENCES)


def sentences_of(chunk: str) -> list[str]:
    return [s.strip() for s in period_splitter(chunk)]


def test_chunks_are_verbatim_whole_sentences_within_size() -> None:
    chunks = make(10, 3).chunk(TEXT, "doc")
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content in TEXT
        assert token_count(chunk.content) <= 10
        assert set(sentences_of(chunk.content)) <= set(SENTENCES)


def test_every_sentence_appears_whole_in_some_chunk() -> None:
    covered = {s for c in make(10, 3).chunk(TEXT, "doc") for s in sentences_of(c.content)}
    assert covered == set(SENTENCES)


def test_zero_overlap_partitions_the_text() -> None:
    chunks = make(10, 0).chunk(TEXT, "doc")
    assert " ".join(c.content for c in chunks) == TEXT


def test_overlap_carries_whole_sentences_covering_overlap_tokens() -> None:
    chunks = make(10, 3).chunk(TEXT, "doc")
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_s, next_s = sentences_of(prev.content), sentences_of(nxt.content)
        shared = next(k for k in range(len(prev_s), 0, -1) if prev_s[-k:] == next_s[:k])
        assert shared < len(prev_s)  # always makes progress
        # Overlap reaches 3 tokens unless that would mean carrying the whole chunk.
        assert token_count(" ".join(prev_s[-shared:])) >= 3 or shared == len(prev_s) - 1


def test_sentence_longer_than_budget_split_at_word_starts() -> None:
    long_sentence = " ".join(["word"] * 12) + "."
    chunks = make(5, 0).chunk(long_sentence, "doc")
    assert " ".join(c.content for c in chunks) == long_sentence
    assert all(token_count(c.content) <= 5 for c in chunks)


def test_word_longer_than_budget_is_hard_cut() -> None:
    chunks = make(2, 0).chunk("a" * 20, "doc")
    assert "".join(c.content for c in chunks) == "a" * 20
    assert all(token_count(c.content) <= 2 for c in chunks)


def test_paragraphs_split_even_without_punctuation() -> None:
    chunks = make(3, 0).chunk("# Heading\n\nbody text.", "doc")
    assert [c.content for c in chunks] == ["# Heading", "body text."]


def test_text_dropped_by_splitter_is_still_covered() -> None:
    def lossy(paragraph: str) -> Sequence[str]:
        return ["first."]  # drops "second part"

    pre = MarkItDownPreprocessor(FakeTokenizer(), 50, 0, lossy)
    assert [c.content for c in pre.chunk("first. second part", "doc")] == ["first. second part"]


def test_segment_metadata() -> None:
    chunks = make(10, 3).chunk(TEXT, "doc.pdf", title="Title")
    assert [c.segment_id for c in chunks[:2]] == ["doc.pdf#0", "doc.pdf#1"]
    assert all(c.source_doc == "doc.pdf" and c.title == "Title" for c in chunks)


def test_empty_text_yields_no_chunks() -> None:
    assert make(10, 3).chunk("   \n\n ", "doc") == []


@pytest.mark.parametrize(
    ("markdown", "title"),
    [("intro\n# Real Title #\n## Section\nbody", "Real Title"), ("no heading here", "report")],
)
def test_process_title_from_h1_or_filename(
    monkeypatch: pytest.MonkeyPatch, markdown: str, title: str
) -> None:
    pre = make(10, 3)
    monkeypatch.setattr(pre, "convert", lambda path: markdown)
    chunks = pre.process(Path("dir/report.docx"))
    assert chunks[0].title == title
    assert chunks[0].content in markdown


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        make(size, overlap)


@pytest.mark.skipif(not (BGE_DIR / "tokenizer.json").exists(), reason="bge tokenizer not downloaded")
def test_real_bge_tokenizer_and_pysbd() -> None:
    from tokenizers import Tokenizer

    pre = MarkItDownPreprocessor.from_pretrained(BGE_DIR, 120, 30)
    sentences = [
        f"Dr. Smith measured {i}.5 ms latency on the U.S. cluster, e.g. with BM25."
        for i in range(60)
    ]
    text = " ".join(sentences)
    chunks = pre.chunk(text, "doc")

    tokenizer = Tokenizer.from_file(str(BGE_DIR / "tokenizer.json"))
    lengths = [len(tokenizer.encode(c.content, add_special_tokens=False).ids) for c in chunks]
    assert max(lengths) <= 120
    assert min(lengths[:-1]) >= 90  # whole-sentence packing loses at most ~one sentence
    for chunk in chunks:
        assert chunk.content.startswith("Dr. Smith") and chunk.content.endswith("BM25.")
