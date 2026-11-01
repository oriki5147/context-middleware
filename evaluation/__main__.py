"""Usage:
    python -m evaluation build  --split dev [--limit N] [--negatives N] [--configs 64 ...] [--tag T]
    python -m evaluation report --split dev [--tag T]
    python -m evaluation build-raw  --split dev [--limit N] [--negatives N] [--tag T]
    python -m evaluation report-raw --split dev [--tag T]

``build`` runs the models once (slow) and caches candidates to data/eval/;
``report`` sweeps selection rules over the cache (fast) and writes CSVs.
The ``-raw`` variants cache raw material instead (see ``evaluation.raw``) and
derive the production pipeline from it, so new experiments need no re-encoding.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from context_middleware import BudgetStrategy

from . import candidates, nq_data, raw, sweep

OUT = Path("data/eval")
MODEL = Path("models/bge-small-en-v1.5")
RERANKER = Path("models/ms-marco-MiniLM-L6-v2")
READER = Path("models/tinyroberta-squad2")

_BI_SENTENCE = [round(float(t), 3) for t in np.arange(0.55, 0.851, 0.025)]
_RERANK_SENTENCE = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
# Reader scores are answer margins (span minus no-answer logits), unbounded.
_READER_SENTENCE = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0]
_BI_EMPTY = [round(float(t), 3) for t in np.arange(0.50, 0.801, 0.02)]
_RERANK_EMPTY = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
_READER_EMPTY = [-12.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
_GRIDS = {
    "bi": (_BI_SENTENCE, _BI_EMPTY),
    "rerank": (_RERANK_SENTENCE, _RERANK_EMPTY),
    "reader": (_READER_SENTENCE, _READER_EMPTY),
}
SENTENCE_THRESHOLDS = {s: _GRIDS[s.rstrip("0123456789")][0] for s in sweep.SCORERS}
EMPTY_THRESHOLDS = {s: _GRIDS[s.rstrip("0123456789")][1] for s in sweep.SCORERS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "report", "build-raw", "report-raw", "experiments"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="answerable questions")
    parser.add_argument("--negatives", type=int, default=None, help="per negative kind")
    parser.add_argument("--configs", nargs="*", default=None, help="chunk config names")
    parser.add_argument("--tag", default="", help="suffix for the cache/results name")
    args = parser.parse_args()
    name = args.split + (f"_{args.tag}" if args.tag else "")
    cache = OUT / f"{name}.pkl"
    raw_cache = OUT / f"{name}_raw.pkl"

    if args.command in ("build", "build-raw"):
        dev, test = nq_data.split(nq_data.load())
        examples = (dev if args.split == "dev" else test)[: args.limit]
        negatives = (
            nq_data.unanswerable(examples)[: args.negatives]
            + nq_data.distractors_only(examples)[: args.negatives]
        )
    if args.command == "build-raw":
        print(f"building raw material: {len(examples)} answerable + {len(negatives)} unanswerable ({args.split})")
        raw.save(raw.build(examples + negatives, MODEL, RERANKER), raw_cache)
        print(f"saved {raw_cache}")
    elif args.command == "experiments":
        run_experiments(raw.load(raw_cache), OUT / "results" / f"{name}_experiments")
    elif args.command == "report-raw":
        raws = raw.load(raw_cache)
        records = [candidates.QuestionRecord(r.qid, r.kind, {"64": raw.derive(r)}) for r in raws]
        report(records, OUT / "results" / f"{name}_raw")
    elif args.command == "build":
        configs = [c for c in candidates.CHUNK_CONFIGS if not args.configs or c.name in args.configs]
        print(
            f"building {len(examples)} answerable + {len(negatives)} unanswerable ({args.split}), "
            f"configs {[c.name for c in configs]}"
        )
        records = candidates.build(examples + negatives, MODEL, RERANKER, READER, configs)
        candidates.save(records, cache)
        print(f"saved {cache}")
    else:
        report(candidates.load(cache), OUT / "results" / name)


def report(records: Sequence[candidates.QuestionRecord], out: Path) -> None:
    kinds = {k: sum(r.kind == k for r in records) for k in ("answerable", *sweep.NEGATIVE_KINDS)}
    print(", ".join(f"{n} {k}" for k, n in kinds.items()), "\n")
    scorers = sweep.available_scorers(records)

    print("== Retrieval & ranking (answerable) ==")
    retrieval = sweep.retrieval_report(records)
    print(
        f"{'config':8} {'scorer':8} {'gold':>6} {'ans.retr':>8} {'ans.cand':>8} "
        f"{'rank1':>6} {'rank<=3':>7} {'med.rank':>8} {'cands':>5} {'orig.tok':>8}"
    )
    for r in retrieval:
        print(
            f"{r.chunk_config:8} {r.scorer:8} {r.gold_retrieved:6.1%} {r.answer_in_retrieved:8.1%} "
            f"{r.answer_in_candidates:8.1%} {r.answer_rank_1:6.1%} {r.answer_rank_le_3:7.1%} "
            f"{r.median_answer_rank:8.1f} {r.median_candidates:5.0f} {r.mean_original_tokens:8.0f}"
        )
    sweep.write_csv(retrieval, out / "retrieval.csv")

    empty = sweep.empty_report(records, EMPTY_THRESHOLDS)
    sweep.write_csv(empty, out / "empty.csv")
    shown = "doc" if "doc" in records[0].by_config else next(iter(records[0].by_config))
    print(f"\n== Empty context: best score <= t (config '{shown}') ==")
    print(f"{'scorer':8} {'t':>8} {'false-empty':>11} {'true-empty(unrel)':>17} {'true-empty(distr)':>17}")
    for e in empty:
        if e.chunk_config == shown:
            print(
                f"{e.scorer:8} {e.empty_threshold:8.4f} {e.false_empty:11.1%} "
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
        [r for s in scorers for r in sweep.pareto(x for x in rows if x.scorer == s)],
        out / "pareto.csv",
    )
    print("\n== Compression: best answer kept within a token budget (all chunk configs) ==")
    _print_header()
    for target in (0.03, 0.05, 0.10, 0.20, 1.0):
        for scorer in scorers:
            fits = [r for r in rows if r.scorer == scorer and r.kept_fraction <= target]
            if fits:
                print(f"<={target:4.0%} ", end="")
                _print_row(max(fits, key=lambda r: r.answer_kept))
    print(f"\nCSV files in {out}")


def run_experiments(raws: Sequence[raw.RawQuestion], out: Path) -> None:
    from context_middleware import BGE_QUERY_PREFIX, ONNXBiEncoderRetriever

    from . import experiments

    encoder = ONNXBiEncoderRetriever.from_pretrained(
        MODEL, query_prefix=BGE_QUERY_PREFIX, intra_op_num_threads=8
    )
    answerable = [r for r in raws if r.kind == "answerable"]
    print(f"{len(answerable)} answerable questions\n")

    print("== Scoring variants (production chunking, 1,200 retrieved tokens) ==")
    rows = experiments.evaluate_variants(
        answerable, experiments.standard_variants(answerable, encoder.encode_query)
    )
    print(
        f"{'variant':26} {'ans.retr':>8} | {'bi r1':>6} {'bi r<=3':>7} | {'rr r1':>6} {'rr r<=3':>7} | "
        f"{'bi k1':>6} {'bi k3':>6} {'rr k1':>6} {'rr k3':>6} | {'tok k3':>6}"
    )
    for v in rows:
        print(
            f"{v.variant:26} {v.answer_in_retrieved:8.1%} | {v.bi_rank1:6.1%} {v.bi_top3:7.1%} | "
            f"{v.rerank_rank1:6.1%} {v.rerank_top3:7.1%} | {v.bi_kept_1:6.1%} {v.bi_kept_3:6.1%} "
            f"{v.rerank_kept_1:6.1%} {v.rerank_kept_3:6.1%} | {v.tokens_3:6.1f}"
        )
    sweep.write_csv(rows, out / "variants.csv")

    print("\n== Output expansion: best k sentences plus document neighbors ==")
    expansion = experiments.evaluate_expansion(answerable, [1, 2, 3, 5], ["none", "prev", "next", "both"])
    print(f"{'scorer':9} {'k':>2} {'window':6} {'ans.kept':>8} {'tokens':>7} {'kept%':>6}")
    for e in expansion:
        print(f"{e.scorer:9} {e.top_k:2d} {e.window:6} {e.answer_kept:8.1%} {e.mean_tokens:7.1f} {e.kept_fraction:6.1%}")
    sweep.write_csv(expansion, out / "expansion.csv")
    print(f"\nCSV files in {out}")


def _print_header() -> None:
    print(
        f"{'kept<=':6} {'config':8} {'scorer':8} {'thresh':>6} {'max_s':>5} {'budget':>6} "
        f"{'strat':>5} {'ans.kept':>8} {'|retr':>6} {'kept%':>6} {'tokens':>6}"
    )


def _print_row(r: sweep.CompressionRow) -> None:
    print(
        f"{r.chunk_config:8} {r.scorer:8} {r.sentence_threshold:6.3f} {str(r.max_sentences):>5} "
        f"{str(r.max_context_tokens):>6} {r.budget_strategy:>5} {r.answer_kept:8.1%} "
        f"{r.answer_kept_given_retrieved:6.1%} {r.kept_fraction:6.1%} {r.mean_compressed_tokens:6.1f}"
    )


if __name__ == "__main__":
    main()
