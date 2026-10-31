"""Orchestrates the four phases into a single CPU-only middleware call."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .compressor import SelectiveSentenceCompressor
from .config import MiddlewareConfig
from .models import CompressionResult
from .preprocessor import MarkItDownPreprocessor
from .reorderer import UShapeReorderer
from .retriever import ONNXBiEncoderRetriever


class CPUMiddlewareEngine:
    def __init__(
        self,
        preprocessor: MarkItDownPreprocessor,
        retriever: ONNXBiEncoderRetriever,
        compressor: SelectiveSentenceCompressor,
        reorderer: UShapeReorderer,
        config: MiddlewareConfig | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.retriever = retriever
        self.compressor = compressor
        self.reorderer = reorderer
        self.config = config or MiddlewareConfig()

    def ingest(self, paths: Iterable[str | Path]) -> None:
        for path in paths:
            self.retriever.add(self.preprocessor.process(path))

    def optimize(self, query: str) -> CompressionResult:
        """Retrieve -> compress -> reorder, returning the context for the target LLM."""
        raise NotImplementedError
