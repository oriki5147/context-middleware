"""Orchestrates the four phases into a single CPU-only middleware call."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .compressor import SelectiveSentenceCompressor
from .config import BudgetStrategy, MiddlewareConfig
from .models import CompressionResult, TextSegment
from .preprocessor import MarkItDownPreprocessor
from .reorderer import ContextReorderer
from .retriever import ONNXBiEncoderRetriever, Pooling

# Counts tokens the way the target LLM would; the default is the encoder's
# tokenizer, a model-agnostic proxy. Pass e.g. a tiktoken-based counter to
# report exact numbers for a specific LLM.
TokenCounter = Callable[[str], int]

SENTENCE_SEPARATOR = "\n"


class CPUMiddlewareEngine:
    def __init__(
        self,
        preprocessor: MarkItDownPreprocessor,
        retriever: ONNXBiEncoderRetriever,
        compressor: SelectiveSentenceCompressor,
        reorderer: ContextReorderer,
        config: MiddlewareConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.retriever = retriever
        self.compressor = compressor
        self.reorderer = reorderer
        self.config = config or MiddlewareConfig()
        self.count_tokens = token_counter or preprocessor.count_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        config: MiddlewareConfig | None = None,
        *,
        model_file: str = "model_int8.onnx",
        pooling: Pooling = Pooling.CLS,
        normalize: bool = True,
        query_prefix: str = "",
        token_counter: TokenCounter | None = None,
    ) -> CPUMiddlewareEngine:
        """Build every phase from one config and one encoder directory."""
        config = config or MiddlewareConfig()
        retriever = ONNXBiEncoderRetriever.from_pretrained(
            model_dir,
            model_file=model_file,
            pooling=pooling,
            normalize=normalize,
            query_prefix=query_prefix,
        )
        return cls(
            preprocessor=MarkItDownPreprocessor.from_pretrained(
                model_dir, config.chunk_size_tokens, config.chunk_overlap_tokens
            ),
            retriever=retriever,
            compressor=SelectiveSentenceCompressor(
                retriever,
                sentence_threshold=config.sentence_threshold,
                empty_threshold=config.empty_threshold,
                min_sentences=config.min_sentences,
                max_sentences=config.max_sentences,
            ),
            reorderer=ContextReorderer(config.reorder_strategy),
            config=config,
            token_counter=token_counter,
        )

    def ingest(self, paths: Iterable[str | Path]) -> None:
        for path in paths:
            self.retriever.add(self.preprocessor.process(path))

    def optimize(self, query: str) -> CompressionResult:
        """Retrieve -> compress -> budget -> reorder, returning the LLM context.

        ``original_token_count`` is the size of the top-K chunks, i.e. what
        would have been sent without compression.
        """
        query_embedding = self.retriever.encode_query(query)  # embedded once
        chunks = self.retriever.search(query_embedding, self.config.top_k)
        original = self.count_tokens("\n\n".join(c.content for c in chunks))

        sentences = self._fit_budget(self.compressor.compress(query_embedding, chunks))
        if not sentences:
            return CompressionResult.empty(original)

        prompt = SENTENCE_SEPARATOR.join(s.content for s in self.reorderer.reorder(sentences))
        return CompressionResult.from_counts(prompt, original, self.count_tokens(prompt))

    def _fit_budget(self, ranked: list[TextSegment]) -> list[TextSegment]:
        """Keep best-first sentences whose tokens fit ``max_context_tokens``.

        On the first sentence that does not fit, ``budget_strategy`` decides:
        FILL skips it and keeps trying lower-ranked ones, STOP ends there.
        Separators are not counted.
        """
        budget = self.config.max_context_tokens
        if budget is None:
            return ranked
        kept: list[TextSegment] = []
        for sentence in ranked:
            cost = self.count_tokens(sentence.content)
            if cost <= budget:
                kept.append(sentence)
                budget -= cost
            elif self.config.budget_strategy is BudgetStrategy.STOP:
                break
        return kept
