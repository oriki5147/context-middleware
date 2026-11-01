"""Experimental second-stage scorer: extractive QA reader (evaluation only).

Evaluated on NQ dev and NOT adopted: it ranked answer sentences worse than the
cross-encoder reranker and did not separate on-topic unanswerable questions
(see evaluation results "dev_reader"). Kept here so the result is reproducible.

A SQuAD 2.0 reader reads (query, sentence) together and predicts both the best
answer span and a "no answer" score (start + end logits at position 0). Its
score here is the margin ``best_span - no_answer``: positive when the sentence
appears to contain the answer, strongly negative when it is on-topic but
answer-free -- the signal a relevance scorer (bi-encoder or reranker) lacks.

Only the score is used; the predicted span is never emitted, so compression
stays purely extractive. It satisfies the compressor's ``Reranker`` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from context_middleware import TextSegment
from context_middleware.reranker import load_pair_model
from context_middleware.retriever import EncodingLike, SessionLike, passage_text


class SpanEncodingLike(EncodingLike, Protocol):
    @property
    def sequence_ids(self) -> list[int | None]: ...


class SpanTokenizerLike(Protocol):
    """Structural subset of ``tokenizers.Tokenizer`` encoding (query, passage) pairs."""

    def encode_batch(self, input: list[tuple[str, str]]) -> Sequence[SpanEncodingLike]: ...


class ONNXExtractiveReader:
    def __init__(
        self,
        session: SessionLike,
        tokenizer: SpanTokenizerLike,
        *,
        max_answer_tokens: int = 30,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0 or max_answer_tokens <= 0:
            raise ValueError("batch_size and max_answer_tokens must be positive")
        self._session = session
        self._tokenizer = tokenizer
        self._max_answer_tokens = max_answer_tokens
        self._batch_size = batch_size
        self._input_names = frozenset(node.name for node in session.get_inputs())

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        model_file: str = "model_int8.onnx",
        max_length: int = 384,
        intra_op_num_threads: int = 0,
        batch_size: int = 32,
    ) -> ONNXExtractiveReader:
        session, tokenizer = load_pair_model(model_dir, model_file, max_length, intra_op_num_threads)
        return cls(session, tokenizer, batch_size=batch_size)

    def score(self, query: str, segments: Sequence[TextSegment]) -> NDArray[np.float32]:
        """Answer margin of each (title-prefixed) segment: best span minus no-answer."""
        pairs = [(query, passage_text(s)) for s in segments]
        if not pairs:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(
            [
                self._margins(pairs[start : start + self._batch_size])
                for start in range(0, len(pairs), self._batch_size)
            ]
        )

    def _margins(self, pairs: list[tuple[str, str]]) -> NDArray[np.float32]:
        encodings = self._tokenizer.encode_batch(pairs)
        feed: dict[str, Any] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        start_logits, end_logits = (np.asarray(x, dtype=np.float32) for x in self._session.run(None, feed)[:2])

        margins = np.empty(len(pairs), dtype=np.float32)
        for i, encoding in enumerate(encodings):
            context = np.flatnonzero(np.array(encoding.sequence_ids, dtype=object) == 1)
            no_answer = start_logits[i, 0] + end_logits[i, 0]
            if len(context) == 0:
                margins[i] = -np.inf
                continue
            start, end = start_logits[i, context], end_logits[i, context]
            # Best start + end with 0 <= end - start < max_answer_tokens.
            span = start[:, None] + end[None, :]
            offset = np.subtract.outer(np.arange(len(context)), np.arange(len(context)))
            valid = (offset <= 0) & (offset > -self._max_answer_tokens)
            margins[i] = float(span[valid].max() - no_answer)
        return margins
