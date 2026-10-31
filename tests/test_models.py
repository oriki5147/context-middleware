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


def test_config_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError):
        MiddlewareConfig(sentence_threshold=0.4, empty_threshold=0.5)
