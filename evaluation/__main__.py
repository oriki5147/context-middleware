"""Usage:
    python -m evaluation build  --split dev [--limit N] [--negatives N] [--configs 64 ...] [--tag T]
    python -m evaluation report --split dev [--tag T]
    python -m evaluation build-raw  --split dev [--limit N] [--negatives N] [--tag T]
    python -m evaluation report-raw --split dev [--tag T]
    python -m evaluation rescore-raw --split dev --reranker DIR --tag T [--context] [--source-tag S] [--second]
    python -m evaluation rank-judge --split dev [--limit N] [--models haiku,sonnet] [--confirm]
    python -m evaluation llm        --split dev [--tag T] [--models haiku] [--conditions a,b] [--confirm]
    python -m evaluation llm-judge  --split dev [--tag T] [--models haiku] [--judge haiku] [--confirm]
    python -m evaluation llm-paired --split dev [--tag T] [--b-tag U] --contrast a,b [--contrast c,d]

``build`` runs the models once (slow) and caches candidates to data/eval/;
``report`` sweeps selection rules over the cache (fast) and writes CSVs.
The ``-raw`` variants cache raw material instead (see ``evaluation.raw``) and
derive the production pipeline from it, so new experiments need no re-encoding.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from context_middleware import BudgetStrategy, MiddlewareConfig

from . import candidates, nq_data, raw, sweep

OUT = Path("data/eval")
MODEL = Path("models/bge-small-en-v1.5")
RERANKER = Path("models/ms-marco-MiniLM-L6-v2")
READER = Path("models/tinyroberta-squad2")
# Claude counts about 1.4 tokens for every 4 characters of these prompts
# (measured on 2,300 cached Haiku requests); cost estimates use it.
CLAUDE_TOKEN_RATIO = 1.4

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
    parser.add_argument(
        "command", choices=["build", "report", "build-raw", "report-raw", "experiments", "llm", "llm-report", "llm-judge", "llm-paired", "rescore-raw", "rank-judge"]
    )
    parser.add_argument("--models", default="haiku,sonnet,qwen", help="llm: comma-separated")
    parser.add_argument("--qwen-model", default="qwen3.5:4b", help="llm: Ollama model tag")
    parser.add_argument("--max-requests", type=int, default=1000, help="llm: per model, per run")
    parser.add_argument("--offset", type=int, default=0, help="llm: skip the first N questions of each kind")
    parser.add_argument("--conditions", default="", help="llm: comma-separated subset of conditions")
    parser.add_argument("--confirm", action="store_true", help="llm: actually send (default: dry run)")
    parser.add_argument("--judge", default="haiku", help="llm-judge, llm-paired: the grading model")
    parser.add_argument("--contrast", action="append", default=[], help="llm-paired: 'a,b' conditions (a from --tag, b from --b-tag)")
    parser.add_argument("--shortlist", type=int, default=None, help="llm*: sentences the reranker sees (default: production); answers go to <name>_sl<N>")
    parser.add_argument("--b-tag", default=None, help="llm-paired: raw cache tag of the second condition (default: --tag)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="answerable questions")
    parser.add_argument("--negatives", type=int, default=None, help="per negative kind")
    parser.add_argument("--configs", nargs="*", default=None, help="chunk config names")
    parser.add_argument("--tag", default="", help="suffix for the cache/results name")
    parser.add_argument("--reranker", type=Path, default=None, help="rescore-raw: cross-encoder model dir")
    parser.add_argument("--context", action="store_true", help="rescore-raw: score sentences with their neighbors")
    parser.add_argument("--source-tag", default="", help="rescore-raw: tag of the cache to rescore (default: untagged)")
    parser.add_argument("--second", action="store_true", help="rescore-raw: --reranker is a second stage over the source's best sentences")
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
    elif args.command == "rescore-raw":
        from context_middleware import ONNXCrossEncoderReranker

        if args.reranker is None or not args.tag:
            parser.error("rescore-raw needs --reranker and --tag (the rescored cache's name)")
        source = OUT / (args.split + (f"_{args.source_tag}" if args.source_tag else "") + "_raw.pkl")
        reranker = ONNXCrossEncoderReranker.from_pretrained(args.reranker, intra_op_num_threads=8)
        if args.second:
            config = replace(MiddlewareConfig(), second_rerank_context=args.context)
            raw.save(raw.cascade(raw.load(source), reranker, config), raw_cache)
        else:
            raw.save(raw.rescore(raw.load(source), reranker, context=args.context), raw_cache)
        print(f"saved {raw_cache}")
    elif args.command == "rank-judge":
        run_rank_judge(args, raw.load(raw_cache), OUT / "llm" / name)
    elif args.command in ("llm", "llm-report"):
        run_llm(args, raw.load(raw_cache), _llm_dir(name, args.shortlist))
    elif args.command == "llm-judge":
        run_llm_judge(args, raw.load(raw_cache), _llm_dir(name, args.shortlist))
    elif args.command == "llm-paired":
        b_name = name if args.b_tag is None else args.split + (f"_{args.b_tag}" if args.b_tag else "")
        run_llm_paired(args, name, b_name)
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


def run_llm(args: argparse.Namespace, raws: Sequence[raw.RawQuestion], out: Path) -> None:
    from . import llm_clients, llm_eval

    tasks = _llm_tasks(args, raws, args.shortlist)
    per_condition: dict[str, int] = {}
    for t in tasks:
        per_condition[f"{t.kind}/{t.condition}"] = per_condition.get(f"{t.kind}/{t.condition}", 0) + 1
    print(f"{len(tasks)} prompts: " + ", ".join(f"{k} {v}" for k, v in per_condition.items()))

    def client(model: str) -> llm_clients.LLMClient:
        if model == "qwen":
            return llm_clients.OllamaClient(args.qwen_model)
        return llm_clients.ClaudeClient(model)

    for model in args.models.split(","):
        cache = llm_eval.ResponseCache(out / f"{model}.jsonl")
        if args.command == "llm":
            prices = (0.0, 0.0) if model == "qwen" else llm_clients.CLAUDE_MODELS[model].price_per_mtok
            pending = [t for t in tasks if t.task_id not in cache.done][: args.max_requests]
            tokens_in = int(CLAUDE_TOKEN_RATIO * sum(llm_eval.estimate_tokens(t.prompt) for t in pending))
            cost = (tokens_in * prices[0] + 20 * len(pending) * prices[1]) / 1e6
            print(f"\n[{model}] {len(pending)} to send, ~{tokens_in:,} input tokens, est. ${cost:.2f}")
            if not args.confirm:
                continue
            failures = llm_eval.run(tasks, client(model), cache, args.max_requests, workers=1 if model == "qwen" else 4)
            print(f"[{model}] done, {failures} failures (not cached; rerun to retry)")
        else:
            prices = (0.0, 0.0) if model == "qwen" else llm_clients.CLAUDE_MODELS[model].price_per_mtok
            rows = llm_eval.report(tasks, cache, model, prices)
            if not rows:
                print(f"\n[{model}] no responses yet")
                continue
            print(f"\n== {model} ==")
            print(
                f"{'kind/condition':28} {'n':>4} {'acc':>6} {'unknown':>7} {'in.tok':>6} {'cost$':>6} "
                f"{'gain':>6} {'harm':>6} {'cf.adopt':>8}"
            )
            for r in rows:
                print(
                    f"{r.kind + '/' + r.condition:28} {r.n:4d} {r.accuracy:6.1%} {r.unknown_rate:7.1%} "
                    f"{r.mean_input_tokens:6.0f} {r.cost_usd:6.3f} {r.gain:6.1%} {r.harm:6.1%} "
                    f"{r.substitute_adopted:8.1%}"
                )
            sweep.write_csv(rows, out / f"{model}_report.csv")
    if args.command == "llm" and not args.confirm:
        print("\nDry run only: nothing was sent. Add --confirm to send.")


def _llm_dir(name: str, shortlist: int | None) -> Path:
    """Where a raw cache's answers live; a non-default reranker shortlist gets its own directory."""
    return OUT / "llm" / (name if shortlist is None else f"{name}_sl{shortlist}")


def _llm_tasks(args: argparse.Namespace, raws: Sequence[raw.RawQuestion], shortlist: int | None) -> list[Any]:
    from . import llm_eval

    only = [c for c in args.conditions.split(",") if c] or None
    config = MiddlewareConfig() if shortlist is None else replace(MiddlewareConfig(), rerank_top_n=shortlist)
    return llm_eval.build_tasks(raws, args.limit or 100, args.negatives or 50, config, offset=args.offset, only=only)


def run_llm_judge(args: argparse.Namespace, raws: Sequence[raw.RawQuestion], out: Path) -> None:
    """Grade the cached answers of each model with the judge (dry run unless --confirm)."""
    from . import judge, llm_clients, llm_eval

    tasks = _llm_tasks(args, raws, args.shortlist)
    prices = llm_clients.CLAUDE_MODELS[args.judge].price_per_mtok
    for model in args.models.split(","):
        answers = llm_eval.ResponseCache(out / f"{model}.jsonl")
        graded = llm_eval.ResponseCache(out / f"{model}_judge_{args.judge}.jsonl")
        grading = judge.judge_tasks(tasks, answers)
        pending = [t for t in grading if t.task_id not in graded.done][: args.max_requests]
        tokens_in = int(CLAUDE_TOKEN_RATIO * sum(llm_eval.estimate_tokens(t.prompt) for t in pending))
        cost = (tokens_in * prices[0] + 5 * len(pending) * prices[1]) / 1e6
        answered = sum(t.task_id in answers.done for t in tasks)
        print(
            f"\n[{model}] {answered} answers, {answered - len(grading)} settled without the judge, "
            f"{len(pending)} to grade, ~{tokens_in:,} input tokens, est. ${cost:.2f}"
        )
        if args.confirm and pending:
            failures = llm_eval.run(grading, llm_clients.ClaudeClient(args.judge, max_tokens=16), graded, args.max_requests)
            print(f"[{model}] done, {failures} failures (not cached; rerun to retry)")
        rows = judge.summary(tasks, answers, judge.verdicts(tasks, answers, graded))
        if rows:
            print(f"{'kind/condition':30} {'n':>5} {'correct':>8} {'abstain':>8} {'wrong':>7} {'in.tok':>7}")
            for r in rows:
                print(
                    f"{r.kind + '/' + r.condition:30} {r.n:5d} {r.correct:8.1%} {r.abstain:8.1%} "
                    f"{r.wrong:7.1%} {r.mean_input_tokens:7.0f}"
                )
            sweep.write_csv(rows, out / f"{model}_judged.csv")
    if not args.confirm:
        print("\nDry run only: nothing was sent. Add --confirm to send.")


def run_llm_paired(args: argparse.Namespace, a_name: str, b_name: str) -> None:
    """Judge-graded accuracy of condition a (from --tag, with --shortlist) against b (from --b-tag), paired by question."""
    from . import judge, llm_eval

    def graded(name: str, model: str, shortlist: int | None) -> dict[str, str]:
        out = _llm_dir(name, shortlist)
        tasks = _llm_tasks(args, raw.load(OUT / f"{name}_raw.pkl"), shortlist)
        return judge.verdicts(
            tasks, llm_eval.ResponseCache(out / f"{model}.jsonl"),
            llm_eval.ResponseCache(out / f"{model}_judge_{args.judge}.jsonl"),
        )

    for model in args.models.split(","):
        first = graded(a_name, model, args.shortlist)
        second = first if b_name == a_name and args.shortlist is None else graded(b_name, model, None)
        print(f"\n== {model}: {a_name} vs {b_name} ==")
        print(f"{'a':16} {'b':16} {'stratum':8} {'n':>5} {'acc a':>6} {'acc b':>6} {'a-b':>6} {'95% interval':>15} {'W/L':>9} {'p':>6}")
        for contrast in args.contrast:
            a, b = contrast.split(",")
            for r in judge.compare(a, b, first, second, {**second, **first}):
                print(
                    f"{r.a:16} {r.b:16} {r.stratum:8} {r.n:5d} {r.accuracy_a:6.1%} {r.accuracy_b:6.1%} "
                    f"{r.difference:+6.1%} [{r.ci_low:+6.1%},{r.ci_high:+6.1%}] {r.wins:4d}/{r.losses:<4d} {r.p_value:6.3f}"
                )


def run_rank_judge(args: argparse.Namespace, raws: Sequence[raw.RawQuestion], out: Path) -> None:
    from . import llm_clients, llm_eval, rank_judge

    tasks = rank_judge.build_tasks(raws, args.limit or 300)
    models = [m for m in args.models.split(",") if m in llm_clients.CLAUDE_MODELS]
    judges = {m: llm_eval.ResponseCache(out / f"rank_judge_{m}.jsonl") for m in models}
    for model, cache in judges.items():
        pending = [t for t in tasks if t.task_id not in cache.done][: args.max_requests]
        prices = llm_clients.CLAUDE_MODELS[model].price_per_mtok
        # Real Claude token counts ran ~2x the chars/4 estimate in the Level 2 pilot.
        tokens_in = 2 * sum(llm_eval.estimate_tokens(t.prompt) for t in pending)
        cost = (tokens_in * prices[0] + 20 * len(pending) * prices[1]) / 1e6
        print(f"[{model}] {len(pending)} to send, ~{tokens_in:,} input tokens, est. ${cost:.2f}")
        if args.confirm and pending:
            failures = llm_eval.run(tasks, llm_clients.ClaudeClient(model), cache, args.max_requests)
            print(f"[{model}] done, {failures} failures (not cached; rerun to retry)")
    if not args.confirm:
        print("Dry run only: nothing was sent. Add --confirm to send.")
        return
    rescored = {"rerank": raws}
    gate_cache = OUT / f"{args.split}_gate_raw.pkl"
    if gate_cache.exists():
        rescored["gate"] = raw.load(gate_cache)
    closed = {m: llm_eval.ResponseCache(out / f"{m}.jsonl") for m in models}
    rows = rank_judge.report(raws, rescored, judges, closed, tasks)
    print(f"\n{'subset':28} {'scorer':8} {'n':>4} {'rank1':>6} {'top3':>6}")
    for r in rows:
        print(f"{r.subset:28} {r.scorer:8} {r.n:4d} {r.rank1:6.1%} {r.top3:6.1%}")
    sweep.write_csv(rows, out / "rank_judge_report.csv")


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
