"""Orchestrates the four phases into a single CPU-only middleware call."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from .compressor import SelectiveSentenceCompressor
from .config import BudgetStrategy, MiddlewareConfig
from .models import CompressionResult, TextSegment
from .preprocessor import MarkItDownPreprocessor
from .reorderer import ContextReorderer
from .reranker import ONNXCrossEncoderReranker
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
        intra_op_num_threads: int = 0,
        reranker_dir: str | Path | None = None,
        token_counter: TokenCounter | None = None,
    ) -> CPUMiddlewareEngine:
        """Build every phase from one config and one encoder directory.

        ``reranker_dir`` (a cross-encoder with ``model_int8.onnx`` and
        ``tokenizer.json``) enables the optional reranking stage.
        """
        config = config or MiddlewareConfig()
        reranker = (
            None
            if reranker_dir is None
            else ONNXCrossEncoderReranker.from_pretrained(
                reranker_dir, intra_op_num_threads=intra_op_num_threads
            )
        )
        retriever = ONNXBiEncoderRetriever.from_pretrained(
            model_dir,
            model_file=model_file,
            intra_op_num_threads=intra_op_num_threads,
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
                reranker=reranker,
                rerank_top_n=config.rerank_top_n,
                rerank_empty_threshold=config.rerank_empty_threshold,
                rerank_context=config.rerank_context,
            ),
            reorderer=ContextReorderer(config.reorder_strategy),
            config=config,
            token_counter=token_counter,
        )

    def ingest(self, paths: Iterable[str | Path]) -> None:
        for path in paths:
            self.add_chunks(self.preprocessor.process(path))

    def add_chunks(self, chunks: Sequence[TextSegment]) -> None:
        """Index chunks and precompute their sentence embeddings (query-time savings)."""
        self.retriever.add(chunks)
        self.compressor.precompute(chunks)

    def optimize(self, query: str) -> CompressionResult:
        """Retrieve -> compress -> budget -> reorder, returning the LLM context.

        ``original_token_count`` is the size of the top-K chunks, i.e. what
        would have been sent without compression.
        """
        query_embedding = self.retriever.encode_query(query)  # embedded once
        chunks = self.retriever.search(query_embedding, self.config.top_k)
        original = self.count_tokens("\n\n".join(c.content for c in chunks))

        sentences = self._fit_budget(self.compressor.compress(query_embedding, chunks, query))
        if not sentences:
            return CompressionResult.empty(original)

        prompt = SENTENCE_SEPARATOR.join(s.content for s in self.reorderer.reorder(sentences))
        return CompressionResult.from_counts(prompt, original, self.count_tokens(prompt))

    def _fit_budget(self, ranked: list[TextSegment]) -> list[TextSegment]:
        costs = [self.count_tokens(s.content) for s in ranked]
        kept = fit_budget(costs, self.config.max_context_tokens, self.config.budget_strategy)
        return [ranked[i] for i in kept]


def fit_budget(
    costs: Sequence[int], budget: int | None, strategy: BudgetStrategy = BudgetStrategy.FILL
) -> list[int]:
    """Indices of best-first items (``costs`` in rank order) that fit ``budget``.

    On the first item that does not fit, ``strategy`` decides: FILL skips it
    and keeps trying lower-ranked ones, STOP ends there. Separators are not
    counted. A pure function so the evaluation harness sweeps the same rule.
    """
    if budget is None:
        return list(range(len(costs)))
    kept: list[int] = []
    for i, cost in enumerate(costs):
        if cost <= budget:
            kept.append(i)
            budget -= cost
        elif strategy is BudgetStrategy.STOP:
            break
    return kept
