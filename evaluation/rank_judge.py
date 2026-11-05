"""Ranking ceiling: can an LLM pick the answer sentence out of the shortlist?

The production compressor shortlists the bi-encoder's top ``rerank_top_n``
sentences and lets the cross-encoder pick the best few; that pick is the
pipeline's main loss (the answer survives top-3 in about half the questions).
Here an LLM sees the same shortlist, shuffled (no position hint), and names
the 3 sentences most likely to hold the answer. Its rank-1 / top-3 hit rates,
next to the bi-encoder's, the reranker's and the trained gate's on the same
shortlists, bound what a better sentence scorer could gain.

Caveat: an LLM may recognize the answer from memory rather than read it, so
hits are also split by whether the model answered the question closed-book.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .llm_eval import ResponseCache, Task, _correct
from .raw import RawQuestion, candidate_indices

PROMPT = (
    "Below are numbered sentences from search results. Which sentences most likely contain the "
    "answer to the question? Reply with the numbers of the 3 best sentences, best first, separated "
    "by commas, and nothing else.\n\nQuestion: {question}\n\nSentences:\n{sentences}\n\nBest 3:"
)
CONDITION = "rank_judge"
TOP_K = 3


def shortlist(raw: RawQuestion, top_n: int = 20) -> NDArray[np.int64]:
    """Raw sentence indices the reranker scores: the bi-encoder's best ``top_n``, best first."""
    idx, _, _ = candidate_indices(raw)
    bi = raw.sent_embedding[idx] @ raw.query_embedding
    return idx[np.argsort(-bi, kind="stable")[:top_n]]


def shown_order(raw: RawQuestion, top_n: int = 20) -> list[int]:
    """The shortlist in the (seeded, per-question) order the LLM sees it."""
    items = [int(i) for i in shortlist(raw, top_n)]
    return random.Random(raw.qid).sample(items, len(items))


def format_prompt(question: str, sentences: Sequence[tuple[str, str]]) -> str:
    """The judge prompt over (title, sentence) pairs, numbered from 1 in the given order."""
    lines = "\n".join(f"[{n}] ({title}) {text}" for n, (title, text) in enumerate(sentences, 1))
    return PROMPT.format(question=question, sentences=lines)


def build_tasks(raws: Sequence[RawQuestion], limit: int, top_n: int = 20) -> list[Task]:
    tasks = []
    for raw in [r for r in raws if r.kind == "answerable"][:limit]:
        shown = [(raw.chunk_title[raw.sent_chunk[i]], raw.sent_text[i]) for i in shown_order(raw, top_n)]
        prompt = format_prompt(raw.question, shown)
        tasks.append(Task(f"{raw.qid}:{raw.kind}:{CONDITION}", raw.qid, raw.kind, CONDITION, prompt, raw.answers))
    return tasks


def parse_ranking(text: str, n: int) -> list[int]:
    """0-based positions named in a reply like "4, 12, 7", in order, deduplicated, in range."""
    out: list[int] = []
    for number in re.findall(r"\d+", text):
        position = int(number) - 1
        if 0 <= position < n and position not in out:
            out.append(position)
    return out


@dataclass(frozen=True, slots=True)
class RankRow:
    scorer: str
    subset: str
    n: int
    rank1: float  # the first pick holds the answer
    top3: float  # one of the first 3 picks does


def _hits(raw: RawQuestion, picks: Sequence[int]) -> tuple[bool, bool]:
    has = raw.sent_has_answer
    return bool(picks and has[picks[0]]), any(bool(has[i]) for i in picks[:TOP_K])


def report(
    raws: Sequence[RawQuestion],
    rescored: Mapping[str, Sequence[RawQuestion]],
    judges: Mapping[str, ResponseCache],
    closed_book: Mapping[str, ResponseCache],
    tasks: Sequence[Task],
    top_n: int = 20,
) -> list[RankRow]:
    """Hit rates of every scorer on the questions all judges answered.

    ``rescored`` maps a scorer name to a copy of ``raws`` whose ``sent_rerank``
    holds that scorer's sentence scores ("rerank" for the production model).
    """
    by_qid = {r.qid: r for r in raws if r.kind == "answerable"}
    answered = [t for t in tasks if all(t.task_id in j.done for j in judges.values())]
    picks: dict[str, dict[int, list[int]]] = {"bi": {}}
    for t in answered:
        picks["bi"][t.qid] = [int(i) for i in shortlist(by_qid[t.qid], top_n)]
    for name, copies in rescored.items():
        scores = {r.qid: r for r in copies if r.kind == "answerable"}
        picks[name] = {}
        for t in answered:
            short = shortlist(by_qid[t.qid], top_n)
            picks[name][t.qid] = [int(i) for i in short[np.argsort(-scores[t.qid].sent_rerank[short], kind="stable")]]
    for name, cache in judges.items():
        picks[name] = {}
        for t in answered:
            order = shown_order(by_qid[t.qid], top_n)
            picks[name][t.qid] = [order[p] for p in parse_ranking(str(cache.done[t.task_id]["text"]), len(order))]

    def answer_shortlisted(qid: int) -> bool:
        return bool(by_qid[qid].sent_has_answer[shortlist(by_qid[qid], top_n)].any())

    subsets: dict[str, list[int]] = {
        "all": [t.qid for t in answered],
        "answer in shortlist": [t.qid for t in answered if answer_shortlisted(t.qid)],
    }
    for name, cache in closed_book.items():
        known = {}
        for qid in subsets["answer in shortlist"]:
            key = f"{qid}:answerable:closed_book"
            if key in cache.done:
                task = Task(key, qid, "answerable", "closed_book", "", by_qid[qid].answers)
                known[qid] = _correct(task, cache.done[key])
        subsets[f"in shortlist, {name} knew it"] = [q for q, k in known.items() if k]
        subsets[f"in shortlist, {name} did not"] = [q for q, k in known.items() if not k]

    rows = []
    for subset, qids in subsets.items():
        if not qids:
            continue
        for scorer, chosen in picks.items():
            hits = [_hits(by_qid[q], chosen[q]) for q in qids]
            rows.append(RankRow(scorer, subset, len(qids), float(np.mean([h[0] for h in hits])), float(np.mean([h[1] for h in hits]))))
    return rows
