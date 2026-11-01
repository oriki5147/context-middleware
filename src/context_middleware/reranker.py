"""Optional Phase 3 stage: cross-encoder reranking of candidate sentences.

A bi-encoder embeds query and sentence separately, so it measures topical
closeness; a cross-encoder reads them together and scores whether the
sentence answers the query. It cannot be precomputed and costs one forward
pass per (query, sentence) pair, so it only reranks a bi-encoder shortlist.

Scores are sigmoid(logit) in (0, 1); thresholds on them are model-specific
and must be calibrated like the bi-encoder ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .models import TextSegment
from .retriever import EncodingLike, SessionLike, passage_text


class PairTokenizerLike(Protocol):
    """Structural subset of ``tokenizers.Tokenizer`` encoding (query, passage) pairs."""

    def encode_batch(self, input: list[tuple[str, str]]) -> Sequence[EncodingLike]: ...


def load_pair_model(
    model_dir: str | Path, model_file: str, max_length: int, intra_op_num_threads: int
) -> tuple[Any, Any]:
    """CPU ONNX session + padded, truncating tokenizer for (query, passage) models."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(model_dir)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = intra_op_num_threads
    session = ort.InferenceSession(
        str(model_dir / model_file), sess_options=options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=max_length)
    # BERT vocabularies pad with [PAD], RoBERTa ones with <pad>.
    pad_token = next((t for t in ("[PAD]", "<pad>") if tokenizer.token_to_id(t) is not None), "[PAD]")
    pad_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token=pad_token)
    return session, tokenizer


class ONNXCrossEncoderReranker:
    def __init__(
        self, session: SessionLike, tokenizer: PairTokenizerLike, *, batch_size: int = 32
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._session = session
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._input_names = frozenset(node.name for node in session.get_inputs())

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        model_file: str = "model_int8.onnx",
        max_length: int = 512,
        intra_op_num_threads: int = 0,
        batch_size: int = 32,
    ) -> ONNXCrossEncoderReranker:
        session, tokenizer = load_pair_model(model_dir, model_file, max_length, intra_op_num_threads)
        return cls(session, tokenizer, batch_size=batch_size)

    def score(self, query: str, segments: Sequence[TextSegment]) -> NDArray[np.float32]:
        """Relevance of each (title-prefixed) segment to ``query``, in (0, 1)."""
        pairs = [(query, passage_text(s)) for s in segments]
        if not pairs:
            return np.empty(0, dtype=np.float32)
        logits = np.concatenate(
            [
                self._logits(pairs[start : start + self._batch_size])
                for start in range(0, len(pairs), self._batch_size)
            ]
        )
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def _logits(self, pairs: list[tuple[str, str]]) -> NDArray[np.float32]:
        encodings = self._tokenizer.encode_batch(pairs)
        feed: dict[str, Any] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        output = np.asarray(self._session.run(None, feed)[0], dtype=np.float32)
        return output.reshape(len(pairs))
