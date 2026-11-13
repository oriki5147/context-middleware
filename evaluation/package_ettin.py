"""Package an Ettin reranker (Apache-2.0) as one ONNX file the middleware can load.

The published ONNX files hold the encoder only; the scoring head lives in
separate safetensors modules (CLS pooling -> Dense + GELU -> LayerNorm ->
Dense). This appends the head to the encoder graph, for the FP32 file and for
the published dynamically quantized one, and lays the result out like the
other model dirs (model.onnx, model_int8.onnx, tokenizer.json).

The models are trained on unbounded teacher scores (about -12 to 22), whose
sigmoid saturates in float32, so the packaged logit is scaled by ``LOGIT_SCALE``:
the order is untouched and the score stays usable as a probability-like value.

    python -m evaluation.package_ettin cross-encoder/ettin-reranker-68m-v1 models/ettin-reranker-68m-v1
"""

from __future__ import annotations

import argparse
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

LOGIT_SCALE = 0.25
FILES = {  # published path -> packaged name
    "onnx/model.onnx": "model.onnx",
    "onnx/model_quint8_avx2.onnx": "model_int8.onnx",
}
HEAD = ("2_Dense/model.safetensors", "3_LayerNorm/model.safetensors", "4_Dense/model.safetensors")


def download(repo: str, source: Path) -> None:
    for name in (*FILES, *HEAD, "tokenizer.json"):
        target = source / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{name}", target)


def add_head(encoder: Path, source: Path, out: Path) -> Path:
    """``encoder`` (output last_hidden_state) plus the scoring head, output ``logits`` of shape (batch,)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from safetensors.numpy import load_file

    dense = load_file(str(source / HEAD[0]))["linear.weight"].astype(np.float32)
    norm = load_file(str(source / HEAD[1]))
    final = load_file(str(source / HEAD[2]))
    model = onnx.load(str(encoder))
    graph = model.graph
    (hidden,) = [o.name for o in graph.output]

    def const(name: str, value: np.ndarray) -> str:
        graph.initializer.append(numpy_helper.from_array(value, f"head.{name}"))
        return f"head.{name}"

    def node(op: str, inputs: list[str], name: str, **attrs: Any) -> str:
        graph.node.append(helper.make_node(op, inputs, [f"head.{name}"], name=f"head.{name}", **attrs))
        return f"head.{name}"

    cls = node("Gather", [hidden, const("first", np.array(0, dtype=np.int64))], "cls", axis=1)
    x = node("MatMul", [cls, const("dense", np.ascontiguousarray(dense.T))], "dense_out")
    # GELU(x) = x/2 * (1 + erf(x / sqrt(2))); the Gelu operator needs opset 20.
    erf = node("Erf", [node("Div", [x, const("sqrt2", np.array(math.sqrt(2.0), dtype=np.float32))], "scaled")], "erf")
    gelu = node("Mul", [node("Mul", [x, node("Add", [erf, const("one", np.array(1.0, dtype=np.float32))], "erf1")], "x_erf"),
                        const("half", np.array(0.5, dtype=np.float32))], "gelu")
    normed = node(
        "LayerNormalization",
        [gelu, const("gamma", norm["norm.weight"].astype(np.float32)), const("beta", norm["norm.bias"].astype(np.float32))],
        "norm", axis=-1, epsilon=1e-5,
    )
    weight = final["linear.weight"].astype(np.float32) * LOGIT_SCALE  # (1, hidden)
    bias = final["linear.bias"].astype(np.float32) * LOGIT_SCALE
    score = node("Add", [node("MatMul", [normed, const("out", np.ascontiguousarray(weight.T))], "out_mm"), const("bias", bias)], "score")
    graph.node.append(helper.make_node("Squeeze", [score, const("last", np.array([-1], dtype=np.int64))], ["logits"], name="head.logits"))
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size"]))
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    return out


def main() -> None:
    import shutil

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="Hugging Face repo, e.g. cross-encoder/ettin-reranker-68m-v1")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    source = args.out / "hf"
    download(args.repo, source)
    for published, packaged in FILES.items():
        print(f"saved {add_head(source / published, source, args.out / packaged)}")
    shutil.copy(source / "tokenizer.json", args.out / "tokenizer.json")


if __name__ == "__main__":
    main()
