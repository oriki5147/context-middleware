"""End-to-end tests: real pipeline classes over the bag-of-words fake encoder.

With mean pooling + normalization, query "cat" scores the sentence
"cat cat cat" at ~0.894 and any cat-free sentence at ~0.224.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_retriever import FakeSession, FakeTokenizer

from context_middleware import (
    BudgetStrategy,
    CPUMiddlewareEngine,
    ContextLayout,
    ContextReorderer,
    MarkItDownPreprocessor,
    MiddlewareConfig,
    ONNXBiEncoderRetriever,
    Pooling,
    SelectiveSentenceCompressor,
    TextSegment,
)


@dataclass
class OffsetEncoding:
    offsets: list[tuple[int, int]]


class WhitespaceTokenizer:
    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> OffsetEncoding:
        return OffsetEncoding([m.span() for m in re.finditer(r"\S+", sequence)])


def line_splitter(paragraph: str) -> Sequence[str]:
    return re.findall(r"[^\n]+\n?", paragraph)


CHUNKS = [
    TextSegment("a#0", "cat cat cat\nfish fish fish", "a"),
    TextSegment("b#0", "dog dog dog\nbird bird bird", "b"),
]


def make_engine(**config: Any) -> tuple[CPUMiddlewareEngine, FakeTokenizer]:
    """Threshold-driven (0.65 / 0.45, no cap) so the fake scores separate cleanly."""
    tokenizer = FakeTokenizer()
    retriever = ONNXBiEncoderRetriever(FakeSession(), tokenizer, pooling=Pooling.MEAN)
    config.setdefault("sentence_threshold", 0.65)
    config.setdefault("empty_threshold", 0.45)
    config.setdefault("max_sentences", None)
    config.setdefault("layout", ContextLayout.PLAIN)  # these tests read the bare sentences
    cfg = MiddlewareConfig(**config)
    engine = CPUMiddlewareEngine(
        preprocessor=MarkItDownPreprocessor(WhitespaceTokenizer(), sentence_splitter=line_splitter),
        retriever=retriever,
        compressor=SelectiveSentenceCompressor(
            retriever,
            sentence_threshold=cfg.sentence_threshold,
            empty_threshold=cfg.empty_threshold,
            min_sentences=cfg.min_sentences,
            max_sentences=cfg.max_sentences,
            sentence_splitter=line_splitter,
        ),
        reorderer=ContextReorderer(cfg.reorder_strategy),
        config=cfg,
    )
    engine.retriever.add(CHUNKS)
    return engine, tokenizer


def test_optimize_keeps_only_relevant_sentences() -> None:
    engine, _ = make_engine()
    result = engine.optimize("cat")
    assert result.compressed_prompt == "cat cat cat"
    assert not result.is_empty
    assert result.original_token_count == 12  # both retrieved chunks
    assert result.compressed_token_count == 3
    assert result.compression_ratio == pytest.approx(0.25)


def test_query_is_embedded_once() -> None:
    engine, tokenizer = make_engine()
    tokenizer.seen.clear()
    engine.optimize("cat")
    assert tokenizer.seen.count("cat") == 1


def test_irrelevant_query_returns_empty_context() -> None:
    engine, _ = make_engine()
    result = engine.optimize("unknown words")  # embeds as [CLS] only: best score ~0.316
    assert result.is_empty
    assert result.compressed_prompt == ""
    assert result.original_token_count == 12


def test_token_budget_fills_best_first_and_skips_what_does_not_fit() -> None:
    engine, _ = make_engine(sentence_threshold=0.2, empty_threshold=0.1, max_context_tokens=7)
    result = engine.optimize("cat")
    assert result.compressed_prompt == "cat cat cat\nfish fish fish"
    assert result.compressed_token_count == 6


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [(BudgetStrategy.FILL, ["s1", "s3", "s4"]), (BudgetStrategy.STOP, ["s1"])],
)
def test_budget_strategy_on_sentence_that_does_not_fit(
    strategy: BudgetStrategy, expected: list[str]
) -> None:
    engine, _ = make_engine(max_context_tokens=50, budget_strategy=strategy)
    engine.count_tokens = lambda text: {"s1": 20, "s2": 40, "s3": 15, "s4": 10}[text]
    ranked = [TextSegment(i, i, "doc") for i in ["s1", "s2", "s3", "s4"]]
    assert [s.content for s in engine._fit_budget(ranked)] == expected


def test_budget_smaller_than_any_sentence_returns_empty() -> None:
    engine, _ = make_engine(max_context_tokens=2)
    assert engine.optimize("cat").is_empty


def test_empty_index_returns_empty_result() -> None:
    engine, _ = make_engine()
    engine.retriever.clear()
    result = engine.optimize("cat")
    assert result.is_empty and result.original_token_count == 0


def test_custom_token_counter_is_used_for_reporting() -> None:
    engine, _ = make_engine()
    engine.count_tokens = len  # e.g. a target-LLM tokenizer; here: characters
    result = engine.optimize("cat")
    assert result.compressed_token_count == len("cat cat cat")


def test_titled_layout_groups_ingested_sentences_under_their_document() -> None:
    engine, _ = make_engine(sentence_threshold=0.5, empty_threshold=0.1, layout=ContextLayout.TITLED)
    engine.retriever.clear()
    engine.add_chunks(
        [
            TextSegment("a#0", "cat cat cat\nfish fish fish\nbird bird bird", "a", title="Pets"),
            TextSegment("a#1", "cat cat dog", "a", title="Pets"),
            TextSegment("b#0", "cat fish dog", "b"),
        ]
    )
    result = engine.optimize("cat")
    # Document a leads (it holds the best sentence); a gap marker stands where
    # its two cat-free sentences were dropped.
    assert result.compressed_prompt.splitlines() == [
        "Document [1](Title: Pets) cat cat cat ... cat cat dog",
        "Document [2] cat fish dog",
    ]
    assert result.compressed_token_count == engine.count_tokens(result.compressed_prompt)  # headers included


def test_ingest_chunks_and_indexes_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = make_engine()
    engine.retriever.clear()
    monkeypatch.setattr(engine.preprocessor, "convert", lambda path: "# Pets\n\ncat cat cat")
    engine.ingest([Path("pets.docx")])
    assert len(engine.retriever) == 1  # heading + sentence fit one chunk
    assert engine.retriever.retrieve("cat")[0].title == "Pets"
