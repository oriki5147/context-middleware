"""Fine-tune an Ettin reranker on the LLM teacher's picks (offline, CPU).

Same data and validation as train_distill, with two differences. The model
is the published Ettin cross-encoder (ModernBERT encoder, then CLS -> Dense +
GELU -> LayerNorm -> Dense), loaded from the files package_ettin downloads.
And the teacher's picks are a set, not an order: the teacher repeats its own
first pick only about half the time, so the listwise target is uniform over
the picks and the pointwise BCE labels every pick 1.

Checkpoints every ``--save-every`` steps; ``--resume`` continues from the last.

    python -m evaluation.train_ettin cross-encoder/ettin-reranker-68m-v1 \
        --data data/distill data/distill/train --out models/distill-ettin-68m
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import distill
from .package_ettin import HEAD
from .train_distill import Example, load_examples, val_metrics

WEIGHTS = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


def download(repo: str, source: Path) -> None:
    for name in (*WEIGHTS, *HEAD):
        target = source / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{name}", target)


def set_target(n_picks: int, size: int) -> list[float]:
    """Listwise target over a sampled list whose first ``n_picks`` entries are the picks."""
    return [1.0 / n_picks] * n_picks + [0.0] * (size - n_picks)


def sample_list(example: Example, size: int, rng: random.Random) -> list[int]:
    """The picks, then random unpicked sentences."""
    others = [i for i in range(len(example.texts)) if i not in example.picks]
    return example.picks + rng.sample(others, min(len(others), size - len(example.picks)))


def build(source: Path) -> Any:
    """The Ettin cross-encoder as one torch module returning a logit per pair."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel

    class Scorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Eager attention, as in train_gate.export: the SDPA path exports slower graphs.
            self.encoder = AutoModel.from_pretrained(source, dtype=torch.float32, attn_implementation="eager")
            hidden = self.encoder.config.hidden_size
            self.dense = torch.nn.Linear(hidden, hidden, bias=False)
            self.norm = torch.nn.LayerNorm(hidden, eps=1e-5)
            self.out = torch.nn.Linear(hidden, 1)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            scores: torch.Tensor = self.out(self.norm(torch.nn.functional.gelu(self.dense(cls)))).squeeze(-1)
            return scores

    model = Scorer()
    model.dense.load_state_dict({"weight": load_file(source / HEAD[0])["linear.weight"]})
    norm = load_file(source / HEAD[1])
    model.norm.load_state_dict({"weight": norm["norm.weight"], "bias": norm["norm.bias"]})
    final = load_file(source / HEAD[2])
    model.out.load_state_dict({"weight": final["linear.weight"], "bias": final["linear.bias"]})
    return model


def export(model: Any, tokenizer: Any, source: Path, out: Path) -> Path:
    """FP32 ONNX + INT8 copy + tokenizer.json, laid out like the other model dirs."""
    import torch

    from context_middleware.quantization import quantize_int8

    model.eval()
    sample = tokenizer(["a question"], ["a sentence"], return_tensors="pt")
    names = ["input_ids", "attention_mask"]
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (), str(out / "model.onnx"), kwargs={n: sample[n] for n in names},
        input_names=names, output_names=["logits"],
        dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    shutil.copy(source / "tokenizer.json", out / "tokenizer.json")
    return quantize_int8(out / "model.onnx", out / "model_int8.onnx")


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("repo", help="Hugging Face repo, e.g. cross-encoder/ettin-reranker-68m-v1")
    parser.add_argument("--data", type=Path, nargs="+", default=[distill.OUT], help="shortlist dirs; validation from the first")
    parser.add_argument("--teacher", default="haiku", help="reads <dir>/teacher_<name>.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", action="store_true", help="train on sentence + passage neighbors")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--questions-per-step", type=int, default=6)
    parser.add_argument("--list-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--val-limit", type=int, default=None, help="validate on the first N shortlists only")
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (timing runs); skips the export")
    parser.add_argument("--export-only", action="store_true", help="export the untrained model, to check the ONNX path")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = args.out / "hf"
    download(args.repo, source)
    max_length = 256 if args.context else 128
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = build(source)
    if args.export_only:
        print(f"saved {export(model, tokenizer, source, args.out)}", flush=True)
        return
    train, val = load_examples(args.data, args.teacher, args.context)
    val = val[: args.val_limit]
    print(f"{len(train)} training / {len(val)} validation shortlists ({'context' if args.context else 'sentence'})", flush=True)

    def logits(question: str, texts: Sequence[str]) -> torch.Tensor:
        enc = tokenizer([question] * len(texts), list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        scores: torch.Tensor = model(enc["input_ids"], enc["attention_mask"])
        return scores

    def report(tag: str) -> None:
        model.eval()
        t = time.time()
        with torch.no_grad():
            scores = [logits(ex.question, ex.texts).numpy() for ex in val]
        m = val_metrics(val, scores)
        print(f"[{tag}] " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()) + f"  ({time.time() - t:.0f}s)", flush=True)

    steps = int(len(train) * args.epochs / args.questions_per_step)
    rng = random.Random(0)
    schedule: list[int] = []  # fixed for the whole run, so a resumed run sees the same batches
    while len(schedule) < steps * args.questions_per_step:
        order = list(range(len(train)))
        rng.shuffle(order)
        schedule.extend(order)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup)))
    )
    checkpoint = args.out / "checkpoint.pt"
    start = 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["step"]
        print(f"resumed at step {start}", flush=True)
    else:
        report("base")

    bce = torch.nn.BCEWithLogitsLoss()
    t, running = time.time(), 0.0
    last = min(steps, args.max_steps or steps)
    for step in range(start, last):
        model.train()
        batch = schedule[step * args.questions_per_step : (step + 1) * args.questions_per_step]
        step_rng = random.Random(step)
        loss = torch.zeros(())
        for k in batch:
            ex = train[k]
            chosen = sample_list(ex, args.list_size, step_rng)
            target = torch.tensor(set_target(len(ex.picks), len(chosen)))
            z = logits(ex.question, [ex.texts[i] for i in chosen])
            listwise = -(target * torch.log_softmax(z, dim=0)).sum()
            loss = loss + (listwise + args.bce_weight * bce(z, (target > 0).float())) / len(batch)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running = 0.98 * running + 0.02 * loss.item() if step > start else loss.item()
        done = step + 1
        if done % 50 == 0:
            rate = (done - start) / (time.time() - t)
            print(f"step {done}/{steps}  loss {running:.3f}  {rate:.2f} steps/s  eta {(steps - done) / rate / 60:.0f} min", flush=True)
        if done % args.save_every == 0 and done < steps:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": done}, checkpoint)
        if done == steps // 2:
            report(f"step {done}")
    if last < steps:
        return
    report("final")
    print(f"saved {export(model, tokenizer, source, args.out)}", flush=True)
    checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
