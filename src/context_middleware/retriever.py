"""Phase 2: Dense Passage Retrieval with an INT8-quantized ONNX bi-encoder.

Passages and queries are embedded independently by the same encoder (DPR
dual-encoder paradigm) and scored by inner product with a single matrix-vector
product (MIPS). Whether embeddings are L2-normalized first (making the score a
cosine similarity) must match how the encoder was trained: bge-* and
sentence-transformers models use cosine, DPR / Contriever use raw dot product.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .models import Embedding, TextSegment

# bge-*-en-v1.5 models expect this instruction on queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Pooling(str, Enum):
    CLS = "cls"  # bge-* family
    MEAN = "mean"  # sentence-transformers family, e.g. all-MiniLM-L6-v2


class EncodingLike(Protocol):
    @property
    def ids(self) -> list[int]: ...
    @property
    def attention_mask(self) -> list[int]: ...
    @property
    def type_ids(self) -> list[int]: ...


class TokenizerLike(Protocol):
    """Structural subset of ``tokenizers.Tokenizer``. Must pad to equal lengths."""

    def encode_batch(self, input: list[str]) -> Sequence[EncodingLike]: ...


class NodeArgLike(Protocol):
    @property
    def name(self) -> str: ...


class SessionLike(Protocol):
    """Structural subset of ``onnxruntime.InferenceSession``."""

    def get_inputs(self) -> Sequence[NodeArgLike]: ...
    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> Sequence[Any]: ...


class ONNXBiEncoderRetriever:
    """In-memory dense retriever backed by an ONNX Runtime CPU session."""

    def __init__(
        self,
        session: SessionLike,
        tokenizer: TokenizerLike,
        *,
        pooling: Pooling = Pooling.CLS,
        normalize: bool = True,
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._session = session
        self._tokenizer = tokenizer
        self._pooling = pooling
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._input_names = frozenset(node.name for node in session.get_inputs())
        self._segments: list[TextSegment] = []
        self._matrix: NDArray[np.float32] | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        model_file: str = "model_int8.onnx",
        max_length: int = 512,
        intra_op_num_threads: int = 0,
        pooling: Pooling = Pooling.CLS,
        normalize: bool = True,
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> ONNXBiEncoderRetriever:
        """Load ``<model_dir>/<model_file>`` and ``<model_dir>/tokenizer.json``.

        ``intra_op_num_threads=0`` lets ONNX Runtime pick the physical core count.
        """
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = intra_op_num_threads
        session = ort.InferenceSession(
            str(model_dir / model_file),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=max_length)
        pad_token = "[PAD]"
        pad_id = tokenizer.token_to_id(pad_token)
        tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token=pad_token)

        return cls(
            session,
            tokenizer,
            pooling=pooling,
            normalize=normalize,
            query_prefix=query_prefix,
            batch_size=batch_size,
        )

    def __len__(self) -> int:
        return len(self._segments)

    # ------------------------------------------------------------------ encoding

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed passages into an ``(n, dim)`` float32 matrix (unit-norm rows if ``normalize``)."""
        if not texts:
            dim = 0 if self._matrix is None else self._matrix.shape[1]
            return np.empty((0, dim), dtype=np.float32)
        batches = [
            self._encode_batch(list(texts[start : start + self._batch_size]))
            for start in range(0, len(texts), self._batch_size)
        ]
        return np.vstack(batches)

    def encode_query(self, query: str) -> Embedding:
        """Embed a single query (with the configured prefix) into a ``(dim,)`` vector."""
        vector: Embedding = self.encode([self._query_prefix + query])[0]
        return vector

    def _encode_batch(self, texts: list[str]) -> NDArray[np.float32]:
        encodings = self._tokenizer.encode_batch(texts)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed: dict[str, NDArray[np.int64]] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        output = np.asarray(self._session.run(None, feed)[0], dtype=np.float32)
        if output.ndim == 3:  # last_hidden_state: (batch, seq, dim)
            pooled = self._pool(output, attention_mask)
        elif output.ndim == 2:  # model already exports a pooled sentence embedding
            pooled = output
        else:
            raise ValueError(f"unexpected encoder output shape {output.shape}")
        return _l2_normalize(pooled) if self._normalize else pooled

    def _pool(
        self, hidden: NDArray[np.float32], attention_mask: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        if self._pooling is Pooling.CLS:
            return hidden[:, 0, :]
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return (summed / counts).astype(np.float32)

    # ----------------------------------------------------------------- indexing

    def embed_segments(self, segments: Sequence[TextSegment]) -> NDArray[np.float32]:
        """Embed segments as passages: title-prefixed when they carry a title."""
        return self.encode([_passage_text(s) for s in segments])

    def add(self, segments: Sequence[TextSegment]) -> None:
        """Embed ``segments`` and append them to the in-memory index."""
        if not segments:
            return
        embeddings = self.embed_segments(segments)
        self._matrix = embeddings if self._matrix is None else np.vstack([self._matrix, embeddings])
        self._segments.extend(segments)

    def clear(self) -> None:
        self._segments = []
        self._matrix = None

    def retrieve(self, query: str, top_k: int = 10) -> list[TextSegment]:
        """Return up to ``top_k`` segments ranked by inner product with ``query``.

        Results are new ``TextSegment`` objects carrying ``score`` and their
        ``embedding``; the indexed segments are not mutated.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self._matrix is None:
            return []

        scores = self._matrix @ self.encode_query(query)
        k = min(top_k, len(scores))
        # O(n) partial selection, then sort only the k winners.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            replace(self._segments[i], embedding=self._matrix[i], score=float(scores[i]))
            for i in top
        ]


def _passage_text(segment: TextSegment) -> str:
    return f"{segment.title}\n{segment.content}" if segment.title else segment.content


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
