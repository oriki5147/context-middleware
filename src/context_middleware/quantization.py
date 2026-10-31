"""Offline INT8 dynamic quantization of an exported FP32 ONNX bi-encoder.

Typical setup (export needs ``optimum``, which is a one-off dev tool, not a
runtime dependency):

    optimum-cli export onnx --model BAAI/bge-small-en-v1.5 models/bge-small
    python -m context_middleware.quantization models/bge-small/model.onnx \\
        models/bge-small/model_int8.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def quantize_int8(fp32_model: str | Path, int8_model: str | Path) -> Path:
    """Quantize weights to INT8 (activations quantized dynamically at runtime)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = Path(int8_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_model),
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fp32_model", type=Path)
    parser.add_argument("int8_model", type=Path)
    args = parser.parse_args()
    print(quantize_int8(args.fp32_model, args.int8_model))


if __name__ == "__main__":
    main()
