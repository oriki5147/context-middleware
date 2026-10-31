"""CPU-first, extractive context optimization middleware for RAG."""

from .compressor import SelectiveSentenceCompressor
from .config import MiddlewareConfig
from .engine import CPUMiddlewareEngine
from .models import CompressionResult, Embedding, TextSegment
from .preprocessor import MarkItDownPreprocessor
from .reorderer import UShapeReorderer
from .retriever import BGE_QUERY_PREFIX, ONNXBiEncoderRetriever, Pooling

__all__ = [
    "BGE_QUERY_PREFIX",
    "CPUMiddlewareEngine",
    "CompressionResult",
    "Embedding",
    "MarkItDownPreprocessor",
    "MiddlewareConfig",
    "ONNXBiEncoderRetriever",
    "Pooling",
    "SelectiveSentenceCompressor",
    "TextSegment",
    "UShapeReorderer",
]
