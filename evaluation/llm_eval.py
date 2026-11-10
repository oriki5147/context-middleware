"""Level 2: does the compressed context let a real LLM answer correctly?

Conditions per answerable question (prompts built from the raw cache with
the production selection rules and defaults):
  closed_book    no context: what the model already knows
  oracle         the gold passage only: the ceiling
  rag            uncompressed retrieval (~1,200 tokens): today's alternative
  mw_bi          middleware, bi-encoder, best 3 sentences
  mw_rerank      middleware, reranker over the bi top-20, best 3 sentences
  mw_rerank_nb   mw_rerank plus each sentence's document neighbors
  mw_rerank_k5/7 mw_rerank keeping the best 5 / 7 sentences (less compression)
  counterfactual mw_rerank with the answer string swapped for a same-type
                 wrong answer (evaluation-only probe of context reliance)
  *_open         the same context under OPEN_PROMPT, which lets the model fall
                 back on its own knowledge (diagnostic: separates "the answer
                 was cut" from "the model refused to answer")
Distractor-only questions (no answer in any passage) run closed_book, rag and
the mw_rerank variants to measure answers leaking from memory and wrong
answers induced by on-topic context.

Accuracy follows Lost in the Middle: a normalized gold answer occurs in the
output. Per-question comparison with closed_book separates what the context
contributed (gain) from what it destroyed (harm).
"""

from __future__ import annotations

import json
import random
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from context_middleware import MiddlewareConfig
from context_middleware.compressor import select_sentences

from .answers import contains_answer, normalize
from .experiments import neighbors
from .llm_clients import Completion, LLMClient
from .raw import RawQuestion, candidate_indices

CONTEXT_PROMPT = (
    "Write a high-quality answer for the given question using only the provided search "
    "results (some of which might be irrelevant). Answer with a short phrase. If the search "
    'results do not contain the answer, reply "unknown".\n\n'
    "Search results:\n{context}\n\nQuestion: {question}\nAnswer:"
)
OPEN_PROMPT = (
    "Write a high-quality answer for the given question. Use the provided search results "
    "(some of which might be irrelevant) when they contain the answer; otherwise answer from "
    "your own knowledge. Answer with a short phrase.\n\n"
    "Search results:\n{context}\n\nQuestion: {question}\nAnswer:"
)
CLOSED_BOOK_PROMPT = "Answer the question with a short phrase.\n\nQuestion: {question}\nAnswer:"
NO_RESULTS = "(no search results)"
OPEN_SUFFIX = "_open"
MORE_SENTENCES = {"mw_rerank_k5": 5, "mw_rerank_k7": 7}

ANSWERABLE_CONDITIONS = (
    "closed_book", "oracle", "rag", "mw_bi", "mw_rerank", "mw_rerank_nb", "counterfactual",
    "mw_rerank_k5", "mw_rerank_k7", "rag_open", "mw_rerank_open", "mw_rerank_k7_open",
)
NEGATIVE_CONDITIONS = (
    "closed_book", "rag", "mw_rerank", "mw_rerank_k7", "rag_open", "mw_rerank_open", "mw_rerank_k7_open",
    "mw_rerank_nb",
)


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str  # "<qid>:<kind>:<condition>"
    qid: int
    kind: str
    condition: str
    prompt: str
    answers: tuple[str, ...]
    context: str = ""  # what was given as search results ("" for closed_book)
    substitute: str = ""  # counterfactual only: the planted wrong answer


# --------------------------------------------------------------- contexts


def _middleware_sentences(raw: RawQuestion, use_reranker: bool, config: MiddlewareConfig) -> list[int]:
    """Raw sentence indices the production compressor would keep, best first."""
    idx, _, _ = candidate_indices(raw)
    bi = raw.sent_embedding[idx] @ raw.query_embedding
    if use_reranker:
        shortlist = idx[np.argsort(-bi, kind="stable")[: config.rerank_top_n]]
        scores, pool, empty = raw.sent_rerank[shortlist], shortlist, config.rerank_empty_threshold
    else:
        scores, pool, empty = bi, idx, config.empty_threshold
    kept = select_sentences(scores, config.sentence_threshold, empty, config.min_sentences, config.max_sentences)
    return [int(pool[i]) for i in kept]


def _with_neighbors(raw: RawQuestion, chosen: list[int]) -> list[int]:
    """Each chosen sentence surrounded by its document neighbors, in rank order."""
    out: list[int] = []
    for i in chosen:
        for j in sorted({i, *neighbors(raw, i, "both")}):
            if j not in out:
                out.append(j)
    return out


def contexts(raw: RawQuestion, config: MiddlewareConfig) -> dict[str, str]:
    """Search-results text for every context condition (empty string = none)."""
    _, retrieved, _ = candidate_indices(raw)
    rag = "\n".join(
        f"Document [{n}](Title: {raw.chunk_title[c]}) {raw.chunk_text[c]}" for n, c in enumerate(retrieved, 1)
    )
    out = {
        "rag": rag,
        "mw_bi": "\n".join(raw.sent_text[i] for i in _middleware_sentences(raw, False, config)),
    }
    rerank = _middleware_sentences(raw, True, config)
    out["mw_rerank"] = "\n".join(raw.sent_text[i] for i in rerank)
    out["mw_rerank_nb"] = "\n".join(raw.sent_text[i] for i in _with_neighbors(raw, rerank))
    for condition, k in MORE_SENTENCES.items():
        wider = replace(config, max_sentences=k)
        out[condition] = "\n".join(raw.sent_text[i] for i in _middleware_sentences(raw, True, wider))
    if raw.gold_doc >= 0:
        gold = [c for c in range(len(raw.chunk_text)) if raw.chunk_doc[c] == raw.gold_doc]
        title = raw.chunk_title[gold[0]] if gold else ""
        out["oracle"] = f"Document [1](Title: {title}) " + " ".join(raw.chunk_text[c] for c in gold)
    return out


# --------------------------------------------------------- counterfactual


def answer_type(answer: str) -> str:
    a = answer.strip()
    if re.fullmatch(r"\d{4}", a):
        return "year"
    if re.fullmatch(r"[\d.,\s%$]+", a):
        return "number"
    if a[:1].isupper():
        return "name"
    return "other"


def _literal(replacement: str) -> str:
    """Escape a replacement string so re.sub inserts it verbatim."""
    return replacement.replace("\\", "\\\\")


def question_word(question: str) -> str:
    """Coarse answer-type cue: the question's wh-word ("who" answers are mostly people)."""
    words = question.lower().split()
    return next((w for w in words[:3] if w in ("who", "when", "where", "what", "which", "how")), "other")


def plant_counterfactual(
    context: str, answers: Sequence[str], pool: Sequence[str], seed: int
) -> tuple[str, str] | None:
    """Replace each gold answer in ``context`` with a distinct same-type wrong answer.

    ``pool`` should hold answers to questions of the same kind (see
    ``question_word``). For multi-word names, a standalone mention of the last
    word ("Heisenberg's") is swapped too, so the original does not leak.
    Returns (new_context, first_substitute), or None if no answer occurs.
    """
    def pattern(a: str) -> str:
        # Whole-word match, so answer "B" never rewrites the "b" inside "about".
        return (r"\b" if a[0].isalnum() else "") + re.escape(a) + (r"\b" if a[-1].isalnum() else "")

    found = [a for a in sorted(answers, key=len, reverse=True) if a and re.search(pattern(a), context, re.I)]
    if not found:
        return None
    kind = answer_type(found[0])
    banned = {normalize(a) for a in answers}
    options = sorted({p for p in pool if answer_type(p) == kind and normalize(p) not in banned})
    if len(options) < len(found):
        return None
    substitutes = random.Random(seed).sample(options, len(found))
    planted = context
    for a, s in zip(found, substitutes):
        planted = re.sub(pattern(a), _literal(s), planted, flags=re.I)
    for a, s in zip(found, substitutes):
        last, new_last = a.split()[-1], s.split()[-1]
        if kind == "name" and len(a.split()) > 1 and len(last) >= 4:
            planted = re.sub(pattern(last), _literal(new_last), planted, flags=re.I)
    return planted, substitutes[0]


# ------------------------------------------------------------------ tasks


def build_tasks(
    raws: Sequence[RawQuestion],
    answerable: int,
    negatives: int,
    config: MiddlewareConfig | None = None,
    *,
    offset: int = 0,
    only: Sequence[str] | None = None,
) -> list[Task]:
    """Tasks for questions ``offset .. offset+answerable`` (and likewise for the
    distractor-only negatives), restricted to the ``only`` conditions if given."""
    config = config or MiddlewareConfig()
    pos = [r for r in raws if r.kind == "answerable"][offset : offset + answerable]
    neg = [r for r in raws if r.kind == "distractors"][offset : offset + negatives]
    pools: dict[str, list[str]] = defaultdict(list)
    for r in raws:
        if r.kind == "answerable" and r.answers:
            pools[question_word(r.question)].append(r.answers[0])
    tasks = []
    for raw in pos + neg:
        ctx = contexts(raw, config)
        conditions = ANSWERABLE_CONDITIONS if raw.kind == "answerable" else NEGATIVE_CONDITIONS
        for condition in conditions:
            if only is not None and condition not in only:
                continue
            substitute = ""
            if condition == "closed_book":
                context, prompt = "", CLOSED_BOOK_PROMPT.format(question=raw.question)
            else:
                if condition == "counterfactual":
                    pool = pools[question_word(raw.question)]
                    planted = plant_counterfactual(ctx["mw_rerank"], raw.answers, pool, raw.qid)
                    if planted is None:
                        continue  # the compressed context holds no answer to swap
                    context, substitute = planted
                else:
                    context = ctx[condition.removesuffix(OPEN_SUFFIX)]
                template = OPEN_PROMPT if condition.endswith(OPEN_SUFFIX) else CONTEXT_PROMPT
                prompt = template.format(context=context or NO_RESULTS, question=raw.question)
            tasks.append(
                Task(f"{raw.qid}:{raw.kind}:{condition}", raw.qid, raw.kind, condition, prompt, raw.answers, context, substitute)
            )
    return tasks


def estimate_tokens(text: str) -> int:
    """Rough English estimate (~4 characters per token); real counts come from the API."""
    return max(1, len(text) // 4)


@dataclass(frozen=True, slots=True)
class Estimate:
    tasks: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def estimate(tasks: Sequence[Task], client: LLMClient, output_tokens_per_task: int = 20) -> Estimate:
    tokens_in = sum(estimate_tokens(t.prompt) for t in tasks)
    tokens_out = output_tokens_per_task * len(tasks)
    price_in, price_out = client.price_per_mtok
    return Estimate(len(tasks), tokens_in, tokens_out, (tokens_in * price_in + tokens_out * price_out) / 1e6)


# ----------------------------------------------------------------- runner


@dataclass
class ResponseCache:
    """Append-only JSONL of completions, so an interrupted run never pays twice."""

    path: Path
    done: dict[str, dict[str, object]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self.done[record["task_id"]] = record

    def add(self, task: Task, completion: Completion) -> None:
        record = {"task_id": task.task_id, **asdict(completion)}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.done[task.task_id] = record


def run(
    tasks: Sequence[Task],
    client: LLMClient,
    cache: ResponseCache,
    max_requests: int,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> int:
    """Complete tasks missing from the cache, at most ``max_requests`` of them."""
    todo = [t for t in tasks if t.task_id not in cache.done][:max_requests]
    log(f"{client.name}: {len(tasks) - len(todo)} cached, sending {len(todo)}")
    failures = 0
    counter = {"n": 0}

    def one(task: Task) -> None:
        nonlocal failures
        try:
            cache.add(task, client.complete(task.prompt))
        except Exception as error:  # noqa: BLE001 - keep going; failures are not cached
            failures += 1
            log(f"  failed {task.task_id}: {type(error).__name__}: {error}")
        counter["n"] += 1
        if counter["n"] % 50 == 0:
            log(f"  {counter['n']}/{len(todo)}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, todo))
    return failures


# ----------------------------------------------------------------- report


@dataclass(frozen=True, slots=True)
class ConditionRow:
    model: str
    kind: str
    condition: str
    n: int
    accuracy: float  # gold answer in output (leakage, for distractor-only)
    unknown_rate: float
    mean_input_tokens: float
    cost_usd: float
    gain: float  # P(correct | closed-book wrong)
    harm: float  # P(wrong | closed-book correct)
    substitute_adopted: float  # counterfactual only


def _correct(task: Task, record: dict[str, object]) -> bool:
    return contains_answer(str(record["text"]), task.answers)


def _unknown(record: dict[str, object]) -> bool:
    return "unknown" in normalize(str(record["text"]))


def report(tasks: Sequence[Task], cache: ResponseCache, model: str, price_per_mtok: tuple[float, float]) -> list[ConditionRow]:
    answered = [(t, cache.done[t.task_id]) for t in tasks if t.task_id in cache.done]
    closed = {(t.qid, t.kind): _correct(t, r) for t, r in answered if t.condition == "closed_book"}
    groups: dict[tuple[str, str], list[tuple[Task, dict[str, object]]]] = defaultdict(list)
    for t, r in answered:
        groups[(t.kind, t.condition)].append((t, r))
    rows = []
    for (kind, condition), items in sorted(groups.items()):
        correct = [_correct(t, r) for t, r in items]
        known = [closed.get((t.qid, t.kind)) for t, _ in items]
        gained = [c for c, k in zip(correct, known) if k is False]
        kept = [c for c, k in zip(correct, known) if k is True]
        tokens_in = [int(str(r["input_tokens"])) for _, r in items]
        tokens_out = [int(str(r["output_tokens"])) for _, r in items]
        rows.append(
            ConditionRow(
                model=model,
                kind=kind,
                condition=condition,
                n=len(items),
                accuracy=float(np.mean(correct)),
                unknown_rate=float(np.mean([_unknown(r) for _, r in items])),
                mean_input_tokens=float(np.mean(tokens_in)),
                cost_usd=(sum(tokens_in) * price_per_mtok[0] + sum(tokens_out) * price_per_mtok[1]) / 1e6,
                gain=float(np.mean(gained)) if gained and condition != "closed_book" else float("nan"),
                harm=float(1 - np.mean(kept)) if kept and condition != "closed_book" else float("nan"),
                substitute_adopted=(
                    float(np.mean([contains_answer(str(r["text"]), [t.substitute]) for t, r in items]))
                    if condition == "counterfactual" else float("nan")
                ),
            )
        )
    return rows
