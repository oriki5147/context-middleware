"""Usage:
    python -m evaluation build  --split dev [--limit N] [--negatives N]
    python -m evaluation report --split dev

``build`` runs the models once (slow) and caches candidates to data/eval/;
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
RERANKER = Path("models/ms-marco-MiniLM-L6-v2")

_BI_SENTENCE = [round(float(t), 3) for t in np.arange(0.55, 0.851, 0.025)]
_RERANK_SENTENCE = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
_BI_EMPTY = [round(float(t), 3) for t in np.arange(0.50, 0.801, 0.02)]
_RERANK_EMPTY = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
SENTENCE_THRESHOLDS = {s: _BI_SENTENCE if s == "bi" else _RERANK_SENTENCE for s in sweep.SCORERS}
EMPTY_THRESHOLDS = {s: _BI_EMPTY if s == "bi" else _RERANK_EMPTY for s in sweep.SCORERS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "report"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="answerable questions")
    parser.add_argument("--negatives", type=int, default=None, help="per negative kind")
    args = parser.parse_args()
    cache = OUT / f"{args.split}.pkl"

    if args.command == "build":
        dev, test = nq_data.split(nq_data.load())
        examples = (dev if args.split == "dev" else test)[: args.limit]
        negatives = (
            nq_data.unanswerable(examples)[: args.negatives]
            + nq_data.distractors_only(examples)[: args.negatives]
        )
        print(f"building {len(examples)} answerable + {len(negatives)} unanswerable ({args.split})")
        candidates.save(candidates.build(examples + negatives, MODEL, RERANKER), cache)
        print(f"saved {cache}")
    else:
        report(candidates.load(cache), OUT / "results" / args.split)


def report(records: Sequence[candidates.QuestionRecord], out: Path) -> None:
    kinds = {k: sum(r.kind == k for r in records) for k in ("answerable", *sweep.NEGATIVE_KINDS)}
    print(", ".join(f"{n} {k}" for k, n in kinds.items()), "\n")

    print("== Retrieval & ranking (answerable) ==")
    retrieval = sweep.retrieval_report(records)
    print(
        f"{'config':8} {'scorer':6} {'gold':>6} {'ans.retr':>8} {'ans.cand':>8} "
        f"{'rank1':>6} {'rank<=3':>7} {'med.rank':>8} {'cands':>5} {'orig.tok':>8}"
    )
    for r in retrieval:
        print(
            f"{r.chunk_config:8} {r.scorer:6} {r.gold_retrieved:6.1%} {r.answer_in_retrieved:8.1%} "
            f"{r.answer_in_candidates:8.1%} {r.answer_rank_1:6.1%} {r.answer_rank_le_3:7.1%} "
            f"{r.median_answer_rank:8.1f} {r.median_candidates:5.0f} {r.mean_original_tokens:8.0f}"
        )
    sweep.write_csv(retrieval, out / "retrieval.csv")

    empty = sweep.empty_report(records, EMPTY_THRESHOLDS)
    sweep.write_csv(empty, out / "empty.csv")
    print("\n== Empty context: best score <= t (config 'doc') ==")
    print(f"{'scorer':6} {'t':>7} {'false-empty':>11} {'true-empty(unrel)':>17} {'true-empty(distr)':>17}")
    for e in empty:
        if e.chunk_config == "doc":
            print(
                f"{e.scorer:6} {e.empty_threshold:7.4f} {e.false_empty:11.1%} "
                f"{e.true_empty_unrelated:17.1%} {e.true_empty_distractors:17.1%}"
            )

    rows = sweep.compression_sweep(
        records,
        sentence_thresholds=SENTENCE_THRESHOLDS,
        max_sentences=[None, 1, 2, 3, 5, 8],
        budgets=[(None, BudgetStrategy.FILL)]
        + [(b, s) for b in (50, 100, 200) for s in BudgetStrategy],
    )
    sweep.write_csv(rows, out / "compression.csv")
    sweep.write_csv(
        [r for s in sweep.SCORERS for r in sweep.pareto(x for x in rows if x.scorer == s)],
        out / "pareto.csv",
    )
    print("\n== Compression: best answer kept within a token budget (all chunk configs) ==")
    _print_header()
    for target in (0.03, 0.05, 0.10, 0.20, 1.0):
        for scorer in sweep.SCORERS:
            fits = [r for r in rows if r.scorer == scorer and r.kept_fraction <= target]
            if fits:
                print(f"<={target:4.0%} ", end="")
                _print_row(max(fits, key=lambda r: r.answer_kept))
    print(f"\nCSV files in {out}")


def _print_header() -> None:
    print(
        f"{'kept<=':6} {'config':8} {'scorer':6} {'thresh':>6} {'max_s':>5} {'budget':>6} "
        f"{'strat':>5} {'ans.kept':>8} {'|retr':>6} {'kept%':>6} {'tokens':>6}"
    )


def _print_row(r: sweep.CompressionRow) -> None:
    print(
        f"{r.chunk_config:8} {r.scorer:6} {r.sentence_threshold:6.3f} {str(r.max_sentences):>5} "
        f"{str(r.max_context_tokens):>6} {r.budget_strategy:>5} {r.answer_kept:8.1%} "
        f"{r.answer_kept_given_retrieved:6.1%} {r.kept_fraction:6.1%} {r.mean_compressed_tokens:6.1f}"
    )


if __name__ == "__main__":
    main()
