"""CPU-first, extractive context optimization middleware for RAG."""

from .compressor import SelectiveSentenceCompressor
from .config import BudgetStrategy, MiddlewareConfig
from .engine import CPUMiddlewareEngine
from .models import CompressionResult, Embedding, TextSegment
from .preprocessor import MarkItDownPreprocessor
from .reorderer import ContextReorderer, ReorderStrategy
from .reranker import ONNXCrossEncoderReranker
from .retriever import BGE_QUERY_PREFIX, ONNXBiEncoderRetriever, Pooling

__all__ = [
    "BGE_QUERY_PREFIX",
    "BudgetStrategy",
    "CPUMiddlewareEngine",
    "CompressionResult",
    "ContextReorderer",
    "Embedding",
    "MarkItDownPreprocessor",
    "MiddlewareConfig",
    "ONNXBiEncoderRetriever",
    "ONNXCrossEncoderReranker",
    "Pooling",
    "ReorderStrategy",
    "SelectiveSentenceCompressor",
    "TextSegment",
]
