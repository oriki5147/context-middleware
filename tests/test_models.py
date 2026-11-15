import numpy as np
import pytest

from context_middleware import CompressionResult, MiddlewareConfig, TextSegment


def test_segments_with_embeddings_compare_without_error() -> None:
    a = TextSegment("1", "text", "doc", embedding=np.ones(3, dtype=np.float32))
    b = TextSegment("1", "text", "doc", embedding=np.zeros(3, dtype=np.float32))
    assert a == b


def test_from_counts_computes_kept_fraction() -> None:
    result = CompressionResult.from_counts("ctx", original_token_count=1000, compressed_token_count=50)
    assert result.compression_ratio == pytest.approx(0.05)
    assert not result.is_empty


def test_empty_result_has_zero_tokens() -> None:
    result = CompressionResult.empty(original_token_count=1200)
    assert result.compressed_prompt == ""
    assert result.compressed_token_count == 0
    assert result.is_empty


def test_empty_result_rejects_content() -> None:
    with pytest.raises(ValueError):
        CompressionResult("text", 10, 1, 0.1, is_empty=True)


def test_config_defaults_follow_evaluation() -> None:
    config = MiddlewareConfig()
    assert (config.chunk_size_tokens, config.chunk_overlap_tokens) == (64, 0)
    assert (config.max_sentences, config.sentence_threshold) == (5, None)


@pytest.mark.parametrize(
    "kwargs", [{"max_sentences": 1, "min_sentences": 2}, {"chunk_overlap_tokens": 64}]
)
def test_config_rejects_inconsistent_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        MiddlewareConfig(**kwargs)  # type: ignore[arg-type]
