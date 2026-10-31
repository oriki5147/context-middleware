from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from context_middleware import ONNXBiEncoderRetriever, Pooling, TextSegment

VOCAB = {"[PAD]": 0, "[CLS]": 1, "cat": 2, "dog": 3, "fish": 4, "bird": 5}
DIM = len(VOCAB)


@dataclass
class FakeEncoding:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


@dataclass
class FakeTokenizer:
    """Whitespace tokenizer over VOCAB, padded to max_len like a real fast tokenizer."""

    max_len: int = 8
    seen: list[str] = field(default_factory=list)

    def encode_batch(self, input: list[str]) -> list[FakeEncoding]:
        self.seen.extend(input)
        out = []
        for text in input:
            ids = [VOCAB["[CLS]"]] + [VOCAB[w] for w in text.lower().split() if w in VOCAB]
            ids = ids[: self.max_len]
            pad = self.max_len - len(ids)
            out.append(FakeEncoding(ids + [0] * pad, [1] * len(ids) + [0] * pad, [0] * self.max_len))
        return out


class FakeSession:
    """Emits a one-hot hidden state per token, so mean pooling = bag of words."""

    def __init__(self, input_names: tuple[str, ...] = ("input_ids", "attention_mask")) -> None:
        self._input_names = input_names
        self.feeds: list[dict[str, Any]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=n) for n in self._input_names]

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        self.feeds.append(input_feed)
        return [np.eye(DIM, dtype=np.float32)[input_feed["input_ids"]]]


def make_retriever(**kwargs: Any) -> ONNXBiEncoderRetriever:
    kwargs.setdefault("pooling", Pooling.MEAN)
    return ONNXBiEncoderRetriever(kwargs.pop("session", FakeSession()), FakeTokenizer(), **kwargs)


SEGMENTS = [
    TextSegment("a", "fish bird", "doc1"),
    TextSegment("b", "cat dog", "doc1"),
    TextSegment("c", "cat cat cat", "doc2"),
]


def test_encode_returns_unit_norm_rows() -> None:
    matrix = make_retriever().encode(["cat", "dog fish", "bird"])
    assert matrix.shape == (3, DIM)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-6)


def test_retrieve_ranks_by_cosine_similarity() -> None:
    retriever = make_retriever()
    retriever.add(SEGMENTS)

    results = retriever.retrieve("cat", top_k=10)

    assert [r.segment_id for r in results] == ["c", "b", "a"]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(4 / np.sqrt(20))
    assert all(r.embedding is not None and r.embedding.shape == (DIM,) for r in results)


def test_retrieve_respects_top_k_and_does_not_mutate_index() -> None:
    retriever = make_retriever()
    retriever.add(SEGMENTS)

    assert len(retriever.retrieve("cat", top_k=2)) == 2
    assert all(s.score == 0.0 and s.embedding is None for s in SEGMENTS)


def test_add_appends_to_index() -> None:
    retriever = make_retriever()
    retriever.add(SEGMENTS[:1])
    retriever.add(SEGMENTS[1:])
    assert len(retriever) == 3
    assert retriever.retrieve("cat", top_k=1)[0].segment_id == "c"


def test_empty_index_returns_nothing() -> None:
    assert make_retriever().retrieve("cat") == []


def test_invalid_top_k_raises() -> None:
    with pytest.raises(ValueError):
        make_retriever().retrieve("cat", top_k=0)


def test_batches_inputs() -> None:
    session = FakeSession()
    make_retriever(session=session, batch_size=2).encode(["cat"] * 5)
    assert [len(f["input_ids"]) for f in session.feeds] == [2, 2, 1]


def test_token_type_ids_only_sent_when_model_declares_them() -> None:
    plain = FakeSession()
    make_retriever(session=plain).encode(["cat"])
    assert "token_type_ids" not in plain.feeds[0]

    bert = FakeSession(("input_ids", "attention_mask", "token_type_ids"))
    make_retriever(session=bert).encode(["cat"])
    assert "token_type_ids" in bert.feeds[0]


def test_cls_pooling_uses_first_token() -> None:
    vector = make_retriever(pooling=Pooling.CLS).encode(["dog"])[0]
    np.testing.assert_allclose(vector, np.eye(DIM, dtype=np.float32)[VOCAB["[CLS]"]])


def test_query_prefix_applied_to_queries_only() -> None:
    tokenizer = FakeTokenizer()
    retriever = ONNXBiEncoderRetriever(FakeSession(), tokenizer, query_prefix="query: ")
    retriever.add(SEGMENTS[:1])
    retriever.retrieve("cat")
    assert tokenizer.seen == ["fish bird", "query: cat"]


def test_title_prepended_to_passage_embedding_only() -> None:
    tokenizer = FakeTokenizer()
    retriever = ONNXBiEncoderRetriever(FakeSession(), tokenizer)
    retriever.add([TextSegment("a", "fish", "doc", title="Birds")])
    result = retriever.retrieve("cat")[0]
    assert tokenizer.seen == ["Birds\nfish", "cat"]
    assert result.content == "fish"


def test_pooled_2d_output_passes_through() -> None:
    class PooledSession(FakeSession):
        def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
            return [np.full((len(input_feed["input_ids"]), 4), 2.0, dtype=np.float32)]

    matrix = make_retriever(session=PooledSession()).encode(["cat"])
    np.testing.assert_allclose(matrix, [[0.5, 0.5, 0.5, 0.5]])


def test_normalize_false_keeps_raw_dot_product() -> None:
    retriever = make_retriever(normalize=False)
    expected = np.zeros(DIM, dtype=np.float32)
    expected[[VOCAB["[CLS]"], VOCAB["cat"]]] = 0.5
    np.testing.assert_allclose(retriever.encode(["cat"])[0], expected)
    retriever.add(SEGMENTS)
    # Query "cat" -> (CLS + cat) / 2; "cat cat cat" -> (CLS + 3 cat) / 4.
    assert retriever.retrieve("cat", top_k=1)[0].score == pytest.approx(0.25 * 0.5 + 0.75 * 0.5)
