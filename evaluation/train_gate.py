"""Fine-tune the MiniLM cross-encoder into an answerability gate (offline, CPU).

Starts from the production reranker's weights (cross-encoder/ms-marco-
MiniLM-L6-v2, saved under models/.../hf) and trains it with binary
cross-entropy on gate_data pairs. Needs torch + transformers, which the
product never imports: the result is exported to ONNX and quantized like the
other models.

    python -m evaluation.train_gate --epochs 1
    python -m evaluation.train_gate --export-only   # then ONNX + INT8
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .gate_data import Pair, read_jsonl

BASE = Path("models/ms-marco-MiniLM-L6-v2/hf")
OUT = Path("models/answer-gate-MiniLM-L6")
DATA = Path("data/gate")
MAX_LENGTH = 128


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(a random positive outscores a random negative), ties counted half."""
    scores = np.concatenate([positive, negative])
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="stable")
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):  # average ranks of ties
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(pairs: Sequence[Pair], scores: NDArray[np.float32]) -> dict[str, float]:
    """Sentence-level separation, and the question-level "answer beats every hard negative"."""
    by_kind: dict[str, list[float]] = defaultdict(list)
    best: dict[int, dict[str, float]] = defaultdict(lambda: {"answer": -math.inf, "hard": -math.inf})
    for pair, score in zip(pairs, scores.tolist()):
        by_kind[pair.kind].append(score)
        if pair.kind in ("answer", "hard"):
            best[pair.group][pair.kind] = max(best[pair.group][pair.kind], score)
    wins = [b["answer"] > b["hard"] for b in best.values() if b["hard"] > -math.inf]
    return {
        "auc_hard": auc(by_kind["answer"], by_kind["hard"]),
        "auc_easy": auc(by_kind["answer"], by_kind["easy"]),
        "answer_beats_hard": float(np.mean(wins)),
    }


def export(hf_dir: Path = OUT / "hf", out_dir: Path = OUT) -> Path:
    """FP32 ONNX + INT8 copy + tokenizer.json, laid out like the other model dirs."""
    import shutil

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from context_middleware.quantization import quantize_int8

    # FP32 even when the checkpoint is stored in FP16: INT8 quantization needs FP32 scales.
    # Eager attention: the SDPA path exports mask-building ops that made the INT8
    # MiniLM ~1.7x slower on CPU (49 vs 28 ms per 20 sentences).
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    sample = dict(tokenizer(["a question"], ["a sentence"], return_tensors="pt"))
    # XLM-R style models (e.g. bge-reranker) take no token_type_ids.
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    out_dir.mkdir(parents=True, exist_ok=True)
    axes = {n: {0: "batch", 1: "sequence"} for n in names}
    torch.onnx.export(
        model, (), str(out_dir / "model.onnx"), kwargs={n: sample[n] for n in names},
        input_names=names, output_names=["logits"], dynamic_axes={**axes, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    shutil.copy(hf_dir / "tokenizer.json", out_dir / "tokenizer.json")
    return quantize_int8(out_dir / "model.onnx", out_dir / "model_int8.onnx")


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--export-only", action="store_true", help="convert a trained model to ONNX and stop")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true", help="score the base model and stop")
    args = parser.parse_args()
    if args.export_only:
        print(f"saved {export()}")
        return

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE)
    train, val = read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "val.jsonl")

    def encode(batch: Sequence[Pair]) -> dict[str, torch.Tensor]:
        return dict(
            tokenizer(
                [p.question for p in batch], [p.text for p in batch],
                truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt",
            )
        )

    def score(pairs: Sequence[Pair]) -> NDArray[np.float32]:
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(pairs), 128):
                out.append(model(**encode(pairs[start : start + 128])).logits.squeeze(-1).numpy())
        return np.concatenate(out).astype(np.float32)

    def report(tag: str) -> None:
        t = time.time()
        m = metrics(val, score(val))
        print(f"[{tag}] " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()) + f"  ({time.time() - t:.0f}s)", flush=True)

    report("base")
    if args.eval_only:
        return

    steps = int(len(train) * args.epochs / args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup)))
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(0)
    order: list[Pair] = []
    t, running = time.time(), 0.0
    for step in range(steps):
        if len(order) < args.batch_size:
            order = train[:]
            rng.shuffle(order)
        batch, order = order[: args.batch_size], order[args.batch_size :]
        model.train()
        logits = model(**encode(batch)).logits.squeeze(-1)
        loss = loss_fn(logits, torch.tensor([float(p.label) for p in batch]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running = 0.98 * running + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % 100 == 0:
            rate = (step + 1) / (time.time() - t)
            print(f"step {step + 1}/{steps}  loss {running:.3f}  {rate:.2f} steps/s  eta {(steps - step - 1) / rate / 60:.0f} min", flush=True)
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            report(f"step {step + 1}")
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT / "hf")
    tokenizer.save_pretrained(OUT / "hf")
    print(f"saved to {OUT / 'hf'}")


if __name__ == "__main__":
    main()
