"""Distill the LLM teacher's sentence rankings into the MiniLM reranker (offline, CPU).

Each training example is one DPR shortlist: the teacher's picks (best first)
plus a sample of the sentences it did not pick. The loss is listwise (softmax
cross-entropy against a target putting 0.6 / 0.25 / 0.15 on picks 1 / 2 / 3)
plus a pointwise BCE (pick 1 -> 1, unpicked -> 0) that keeps the sigmoid
scores usable for the empty-context threshold. ``--context`` trains on the
sentence followed by its passage neighbors, the format raw.rescore(context=True)
produces for NQ.

Checkpoints every ``--save-every`` steps; ``--resume`` continues from the last.

    python -m evaluation.train_distill [--context] [--resume]
    python -m evaluation.train_distill --data data/distill data/distill/train --out models/distill-big-MiniLM-L6

Validation shortlists come from the first ``--data`` dir only (DPR dev, as
before); every shortlist of the other dirs is used for training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from context_middleware.compressor import with_neighbors

from . import distill
from .rank_judge import parse_ranking
from .train_gate import BASE, export

PICK_WEIGHTS = (0.6, 0.25, 0.15)


def sentence_text(title: str, text: str, prev: str = "", nxt: str = "", context: bool = False) -> str:
    """Reranker input: the production "title\\nsentence", plus neighbors when ``context``."""
    if context:
        text = with_neighbors(text, prev, nxt)
    return f"{title}\n{text}" if title else text


@dataclass(frozen=True, slots=True)
class Example:
    question: str
    texts: list[str]  # model inputs, shortlist order
    picks: list[int]  # teacher's picks, best first
    has_answer: list[bool]


def load_dir(data: Path, teacher: str, context: bool) -> list[tuple[bool, Example]]:
    """(validation flag, example) for each shortlist the teacher ranked."""
    shortlists = {distill.custom_id(s): s for s in distill.read_jsonl(data / "shortlists.jsonl")}
    around: dict[int, dict[str, list[str]]] = {}
    if context:
        with (data / "context.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                around[row["group"]] = row
    out: list[tuple[bool, Example]] = []
    with (data / f"teacher_{teacher}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            item = shortlists.get(record["custom_id"])
            if item is None or record["type"] != "succeeded":
                continue
            picks = parse_ranking(str(record["text"]), len(item.texts))[:3]
            if not picks:
                continue
            ctx = around.get(item.group, {"prev": [""] * len(item.texts), "next": [""] * len(item.texts)})
            texts = [
                sentence_text(t, s, p, n, context)
                for t, s, p, n in zip(item.titles, item.texts, ctx["prev"], ctx["next"])
            ]
            out.append((item.validation, Example(item.question, texts, picks, item.has_answer)))
    return out


def load_examples(dirs: Sequence[Path], teacher: str, context: bool) -> tuple[list[Example], list[Example]]:
    train: list[Example] = []
    val: list[Example] = []
    first: set[str] = set()
    for n, data in enumerate(dirs):
        for is_val, ex in load_dir(data, teacher, context):
            if n == 0:
                first.add(ex.question)
            elif ex.question in first:  # DPR train repeats a few dev questions
                continue
            (val if is_val and n == 0 else train).append(ex)
    return train, val


def sample_list(example: Example, size: int, rng: random.Random) -> tuple[list[int], list[float]]:
    """The picks plus random unpicked sentences, and the listwise target over them."""
    others = [i for i in range(len(example.texts)) if i not in example.picks]
    chosen = example.picks + rng.sample(others, min(len(others), size - len(example.picks)))
    weights = list(PICK_WEIGHTS[: len(example.picks)])
    target = [w / sum(weights) for w in weights] + [0.0] * (len(chosen) - len(example.picks))
    return chosen, target


def val_metrics(examples: Sequence[Example], scores: Sequence[NDArray[np.float32]]) -> dict[str, float]:
    agree, rank1, top3 = [], [], []
    for ex, s in zip(examples, scores):
        order = np.argsort(-s, kind="stable")
        agree.append(int(order[0]) == ex.picks[0])
        if any(ex.has_answer):
            rank1.append(ex.has_answer[int(order[0])])
            top3.append(any(ex.has_answer[int(i)] for i in order[:3]))
    return {"teacher_top1": float(np.mean(agree)), "answer_rank1": float(np.mean(rank1)), "answer_top3": float(np.mean(top3))}


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", default=[distill.OUT], help="shortlist dirs; validation from the first")
    parser.add_argument("--teacher", default="haiku", help="reads <dir>/teacher_<name>.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="model dir (default by --context)")
    parser.add_argument("--context", action="store_true", help="train on sentence + passage neighbors")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--questions-per-step", type=int, default=6)
    parser.add_argument("--list-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = args.out or Path("models") / ("distill-context-MiniLM-L6" if args.context else "distill-MiniLM-L6")
    max_length = 256 if args.context else 128
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE)
    train, val = load_examples(args.data, args.teacher, args.context)
    print(f"{len(train)} training / {len(val)} validation shortlists ({'context' if args.context else 'sentence'})", flush=True)

    def logits(question: str | Sequence[str], texts: Sequence[str]) -> torch.Tensor:
        questions = [question] * len(texts) if isinstance(question, str) else list(question)
        enc = tokenizer(questions, list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        scores: torch.Tensor = model(**enc).logits.squeeze(-1)
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
    checkpoint = out / "checkpoint.pt"
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
    for step in range(start, steps):
        model.train()
        batch = schedule[step * args.questions_per_step : (step + 1) * args.questions_per_step]
        step_rng = random.Random(step)
        loss = torch.zeros(())
        for k in batch:
            ex = train[k]
            chosen, target = sample_list(ex, args.list_size, step_rng)
            z = logits(ex.question, [ex.texts[i] for i in chosen])
            listwise = -(torch.tensor(target) * torch.log_softmax(z, dim=0)).sum()
            n_picks = len(ex.picks)
            point_idx = [0] + list(range(n_picks, len(chosen)))  # pick 1 and the unpicked
            point_lab = torch.tensor([1.0] + [0.0] * (len(chosen) - n_picks))
            loss = loss + (listwise + args.bce_weight * bce(z[point_idx], point_lab)) / len(batch)
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
            out.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": done}, checkpoint)
        if done == steps // 2:
            report(f"step {done}")
    report("final")
    model.save_pretrained(out / "hf")
    tokenizer.save_pretrained(out / "hf")
    print(f"saved {export(out / 'hf', out)}", flush=True)
    checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
