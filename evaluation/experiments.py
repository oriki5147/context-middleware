"""Cheap experiments over the raw-material cache (no corpus re-encoding).

1. Output expansion: add each selected sentence's neighbor(s) in its document.
2. Neighbor score smoothing: s'_i = s_i + alpha * mean(neighbor scores).
3. Hybrid lexical + dense: BM25 fused with dense scores (RRF or z-score sum),
   for chunk retrieval and for sentence ranking.
4. Query rewriting: drop the leading wh-word / auxiliary before encoding.

Each scoring variant is turned into production-equivalent ``Candidates`` via
``raw.derive`` and measured with the same sweep code as everything else.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from context_middleware import BudgetStrategy

from . import sweep
from .candidates import Candidates, QuestionRecord
from .raw import RawQuestion, candidate_indices, derive

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of in on at to for by with from as and or is are was were be been "
    "who what when where which why how whom whose did does do has have had that this it its".split()
)
_WH_PREFIX = re.compile(
    r"^\s*(who|whom|whose|what|when|where|which|why|how many|how much|how long|how)\b\s*"
    r"(is|are|was|were|did|does|do|has|have|had|will|can)?\b\s*",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def rewrite_question(question: str) -> str:
    """Heuristic declarative-ish form: drop the leading wh-word and auxiliary."""
    rewritten = _WH_PREFIX.sub("", question).strip().rstrip("?").strip()
    return rewritten or question


# --------------------------------------------------------------------- BM25


class BM25:
    """Okapi BM25 with corpus statistics from every text it was fitted on."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(documents)
        self.avgdl = sum(len(d) for d in documents) / max(self.n, 1)
        df: Counter[str] = Counter()
        for d in documents:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: Sequence[str], documents: Sequence[Sequence[str]]) -> NDArray[np.float32]:
        terms = [t for t in set(query) if t in self.idf]
        out = np.zeros(len(documents), dtype=np.float32)
        for i, d in enumerate(documents):
            tf = Counter(d)
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if tf[t])
        return out


def rrf(*score_arrays: NDArray[np.float32], k: int = 60) -> NDArray[np.float32]:
    """Reciprocal rank fusion of score arrays over the same items."""
    fused = np.zeros(len(score_arrays[0]), dtype=np.float64)
    for scores in score_arrays:
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
        fused += 1.0 / (k + 1 + ranks)
    return fused.astype(np.float32)


def zsum(a: NDArray[np.float32], b: NDArray[np.float32], weight: float) -> NDArray[np.float32]:
    def z(x: NDArray[np.float32]) -> NDArray[np.float64]:
        std = float(x.std())
        if std == 0:
            return np.zeros(len(x), dtype=np.float64)
        return np.asarray((x - x.mean()) / std, dtype=np.float64)

    return (z(a) + weight * z(b)).astype(np.float32)


# ----------------------------------------------------------------- neighbors


def neighbors(raw: RawQuestion, i: int, window: str) -> list[int]:
    """Raw indices of sentence i's neighbors in its document ("prev", "next", "both")."""
    out = []
    if window in ("prev", "both") and i > 0 and raw.sent_doc[i - 1] == raw.sent_doc[i]:
        out.append(i - 1)
    if window in ("next", "both") and i + 1 < len(raw.sent_text) and raw.sent_doc[i + 1] == raw.sent_doc[i]:
        out.append(i + 1)
    return out


def smooth(raw: RawQuestion, scores: NDArray[np.float32], alpha: float) -> NDArray[np.float32]:
    out = scores.astype(np.float64).copy()
    for i in range(len(scores)):
        nb = neighbors(raw, i, "both")
        if nb:
            out[i] += alpha * float(np.mean(scores[nb]))
    return out.astype(np.float32)


# ----------------------------------------------------------------- variants

# A variant maps a raw question to Candidates-producing score overrides.
Overrides = dict[str, NDArray[np.float32]]
Variant = Callable[[RawQuestion], Overrides]


@dataclass(frozen=True, slots=True)
class VariantRow:
    variant: str
    answer_in_retrieved: float
    bi_rank1: float
    bi_top3: float
    rerank_rank1: float
    rerank_top3: float
    bi_kept_1: float  # answer kept with the best 1 / 3 sentences
    bi_kept_3: float
    rerank_kept_1: float
    rerank_kept_3: float
    tokens_3: float  # mean compressed tokens with 3 sentences (bi)


def evaluate_variants(raws: Sequence[RawQuestion], variants: dict[str, Variant]) -> list[VariantRow]:
    rows = []
    answerable = [r for r in raws if r.kind == "answerable"]
    for name, variant in variants.items():
        records = [QuestionRecord(r.qid, r.kind, {"64": _derive(r, variant(r))}) for r in answerable]
        ranks = {row.scorer: row for row in sweep.retrieval_report(records)}
        comp = sweep.compression_sweep(
            records,
            sentence_thresholds={"bi": [sweep.NO_EMPTY], "rerank20": [sweep.NO_EMPTY]},
            max_sentences=[1, 3],
            budgets=[(None, BudgetStrategy.FILL)],
        )
        kept = {(c.scorer, c.max_sentences): c for c in comp}
        rows.append(
            VariantRow(
                variant=name,
                answer_in_retrieved=ranks["bi"].answer_in_retrieved,
                bi_rank1=ranks["bi"].answer_rank_1,
                bi_top3=ranks["bi"].answer_rank_le_3,
                rerank_rank1=ranks["rerank20"].answer_rank_1,
                rerank_top3=ranks["rerank20"].answer_rank_le_3,
                bi_kept_1=kept[("bi", 1)].answer_kept,
                bi_kept_3=kept[("bi", 3)].answer_kept,
                rerank_kept_1=kept[("rerank20", 1)].answer_kept,
                rerank_kept_3=kept[("rerank20", 3)].answer_kept,
                tokens_3=kept[("bi", 3)].mean_compressed_tokens,
            )
        )
    return rows


def _derive(r: RawQuestion, overrides: Overrides) -> Candidates:
    return derive(
        r,
        chunk_scores=overrides.get("chunk_scores"),
        sentence_scores=overrides.get("sentence_scores"),
        rerank_scores=overrides.get("rerank_scores"),
    )


# ------------------------------------------------------- output expansion


@dataclass(frozen=True, slots=True)
class ExpansionRow:
    scorer: str
    top_k: int
    window: str
    answer_kept: float
    mean_tokens: float
    kept_fraction: float


def evaluate_expansion(
    raws: Sequence[RawQuestion], top_ks: Sequence[int], windows: Sequence[str], rerank_top_n: int = 20
) -> list[ExpansionRow]:
    rows = []
    answerable = [r for r in raws if r.kind == "answerable"]
    for scorer in ("bi", "rerank20"):
        ranked: list[tuple[RawQuestion, list[int], int]] = []
        for r in answerable:
            idx, _, used = candidate_indices(r)
            bi = r.sent_embedding[idx] @ r.query_embedding
            order = idx[np.argsort(-bi, kind="stable")]
            if scorer == "rerank20":
                short = order[:rerank_top_n]
                order = short[np.argsort(-r.sent_rerank[short], kind="stable")]
            ranked.append((r, [int(i) for i in order], used))
        for k in top_ks:
            for window in windows:
                kept, tokens, fractions = [], [], []
                for r, best, used in ranked:
                    chosen = set(best[:k])
                    if window != "none":
                        chosen |= {n for i in best[:k] for n in neighbors(r, i, window)}
                    sel = sorted(chosen)
                    t = int(r.sent_tokens[sel].sum())
                    kept.append(bool(r.sent_has_answer[sel].any()))
                    tokens.append(t)
                    fractions.append(t / used if used else 0.0)
                rows.append(
                    ExpansionRow(scorer, k, window, float(np.mean(kept)), float(np.mean(tokens)), float(np.mean(fractions)))
                )
    return rows


# ----------------------------------------------------------- variant factory


def standard_variants(
    raws: Sequence[RawQuestion], encode_query: Callable[[str], NDArray[np.float32]] | None = None
) -> dict[str, Variant]:
    sent_tokens = {id(r): [tokenize(f"{r.chunk_title[int(r.sent_chunk[i])]} {t}") for i, t in enumerate(r.sent_text)] for r in raws}
    chunk_tokens = {id(r): [tokenize(f"{title} {t}") for title, t in zip(r.chunk_title, r.chunk_text)] for r in raws}
    sent_bm25 = BM25([d for r in raws for d in sent_tokens[id(r)]])
    chunk_bm25 = BM25([d for r in raws for d in chunk_tokens[id(r)]])

    def lexical(r: RawQuestion) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        q = tokenize(r.question)
        return chunk_bm25.scores(q, chunk_tokens[id(r)]), sent_bm25.scores(q, sent_tokens[id(r)])

    def dense(r: RawQuestion) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return r.chunk_embedding @ r.query_embedding, r.sent_embedding @ r.query_embedding

    def hybrid(fuse: Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.float32]], rerank: bool) -> Variant:
        def variant(r: RawQuestion) -> Overrides:
            (dc, ds), (lc, ls) = dense(r), lexical(r)
            out = {"chunk_scores": fuse(dc, lc), "sentence_scores": fuse(ds, ls)}
            if rerank:
                out["rerank_scores"] = fuse(r.sent_rerank, ls)
            return out

        return variant

    def smoothed(alpha: float) -> Variant:
        def variant(r: RawQuestion) -> Overrides:
            return {
                "sentence_scores": smooth(r, r.sent_embedding @ r.query_embedding, alpha),
                "rerank_scores": smooth(r, r.sent_rerank, alpha),
            }

        return variant

    variants: dict[str, Variant] = {
        "baseline": lambda r: {},
        "smooth a=0.25": smoothed(0.25),
        "smooth a=0.5": smoothed(0.5),
        "bm25 only": lambda r: dict(zip(("chunk_scores", "sentence_scores"), lexical(r))),
        "hybrid rrf": hybrid(lambda a, b: rrf(a, b), rerank=False),
        "hybrid rrf (+rerank)": hybrid(lambda a, b: rrf(a, b), rerank=True),
        "hybrid z w=0.3": hybrid(lambda a, b: zsum(a, b, 0.3), rerank=False),
        "hybrid z w=0.3 (+rerank)": hybrid(lambda a, b: zsum(a, b, 0.3), rerank=True),
    }
    if encode_query is not None:
        rewritten = {id(r): encode_query(rewrite_question(r.question)) for r in raws}

        def rewrite(r: RawQuestion) -> Overrides:
            q = rewritten[id(r)]
            return {"chunk_scores": r.chunk_embedding @ q, "sentence_scores": r.sent_embedding @ q}

        variants["query rewrite"] = rewrite
    return variants
