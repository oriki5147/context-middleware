"""Usage:
    python -m evaluation build  --split dev [--limit N] [--negatives N]
    python -m evaluation report --split dev

``build`` runs the encoder once (slow) and caches candidates to data/eval/;
``report`` sweeps selection rules over the cache (fast) and writes CSVs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from context_middleware import BudgetStrategy

from . import candidates, nq_data, sweep

OUT = Path("data/eval")
MODEL = Path("models/bge-small-en-v1.5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "report"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="answerable questions")
    parser.add_argument("--negatives", type=int, default=None, help="unanswerable questions")
    args = parser.parse_args()
    cache = OUT / f"{args.split}.pkl"

    if args.command == "build":
        dev, test = nq_data.split(nq_data.load())
        examples = (dev if args.split == "dev" else test)[: args.limit]
        negatives = nq_data.unanswerable(examples)[: args.negatives]
        print(f"building {len(examples)} answerable + {len(negatives)} unanswerable ({args.split})")
        candidates.save(candidates.build(examples + negatives, MODEL), cache)
        print(f"saved {cache}")
    else:
        report(candidates.load(cache), OUT / "results" / args.split)


def report(records: Sequence[candidates.QuestionRecord], out: Path) -> None:
    n_pos = sum(r.answerable for r in records)
    print(f"{n_pos} answerable + {len(records) - n_pos} unanswerable questions\n")

    print("== Retrieval & ranking (answerable) ==")
    retrieval = sweep.retrieval_report(records)
    print(f"{'config':8} {'gold@K':>7} {'ans@K':>7} {'rank1':>7} {'rank<=3':>8} {'med.rank':>9} {'cands':>6} {'orig.tok':>9}")
    for r in retrieval:
        print(
            f"{r.chunk_config:8} {r.gold_retrieved:7.1%} {r.answer_in_retrieved:7.1%} "
            f"{r.answer_rank_1:7.1%} {r.answer_rank_le_3:8.1%} {r.median_answer_rank:9.1f} "
            f"{r.median_candidates:6.0f} {r.mean_original_tokens:9.0f}"
        )
    sweep.write_csv(retrieval, out / "retrieval.csv")

    empty = sweep.empty_report(records, np.round(np.arange(0.50, 0.801, 0.02), 3))
    sweep.write_csv(empty, out / "empty.csv")
    if n_pos < len(records):
        print("\n== Empty-context threshold (best sentence score <= t) ==")
        print(f"{'config':8} {'t':>5} {'false-empty':>12} {'true-empty':>11}")
        for e in empty:
            if e.chunk_config in ("doc", "120") and round(e.empty_threshold * 100) % 4 == 2:
                print(f"{e.chunk_config:8} {e.empty_threshold:5.2f} {e.false_empty:12.1%} {e.true_empty:11.1%}")

    rows = sweep.compression_sweep(
        records,
        sentence_thresholds=np.round(np.arange(0.55, 0.851, 0.025), 3),
        max_sentences=[None, 1, 2, 3, 5, 8],
        budgets=[(None, BudgetStrategy.FILL)]
        + [(b, s) for b in (50, 100, 200) for s in BudgetStrategy],
    )
    sweep.write_csv(rows, out / "compression.csv")
    print("\n== Compression: Pareto front (answer kept vs. tokens kept), all configs ==")
    _print_rows(sweep.pareto(rows))
    for target in (0.05, 0.10, 0.20):
        best = max((r for r in rows if r.kept_fraction <= target), key=lambda r: r.answer_kept, default=None)
        if best:
            print(f"\nBest at <= {target:.0%} tokens kept:")
            _print_rows([best])
    print(f"\nCSV files in {out}")


def _print_rows(rows: Sequence[sweep.CompressionRow]) -> None:
    print(f"{'config':8} {'thresh':>6} {'max_s':>5} {'budget':>6} {'strat':>5} {'ans.kept':>9} {'|retr':>7} {'kept%':>7} {'tokens':>7}")
    for r in rows:
        print(
            f"{r.chunk_config:8} {r.sentence_threshold:6.3f} {str(r.max_sentences):>5} "
            f"{str(r.max_context_tokens):>6} {r.budget_strategy:>5} {r.answer_kept:9.1%} "
            f"{r.answer_kept_given_retrieved:7.1%} {r.kept_fraction:7.1%} {r.mean_compressed_tokens:7.1f}"
        )


if __name__ == "__main__":
    main()
