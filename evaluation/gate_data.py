"""Training pairs for the answerability gate, built from DPR's NQ retriever data.

A pair is (question, title-prefixed sentence, label). Label 1: a sentence of a
positive passage that contains a gold answer. Label 0: a sentence of a BM25
hard-negative passage (on-topic, no answer: the case the off-the-shelf
reranker cannot reject), plus a few from random negative passages. Other
sentences of positive passages are left out, since whether they "hold the
answer" is ambiguous. Splits are by question, so no question is in both.
"""

from __future__ import annotations

import gzip
import json
import random
import urllib.request
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from context_middleware.sentences import SentenceSplitter, sentence_spans

from .answers import contains_answer

URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"
DEFAULT_PATH = Path("data/dpr/biencoder-nq-dev.json.gz")


@dataclass(frozen=True, slots=True)
class Pair:
    group: int  # question index: splits and per-question metrics key on it
    question: str
    text: str  # "title\nsentence", as the production reranker sees it
    label: int
    kind: str  # "answer", "hard" or "easy"


def download(path: Path = DEFAULT_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path


def load(path: Path = DEFAULT_PATH) -> list[dict[str, Any]]:
    with gzip.open(download(path), "rt", encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    return records


def _sentences(ctx: dict[str, Any], split: SentenceSplitter) -> list[str]:
    text = str(ctx["text"])
    return [f"{ctx['title']}\n{text[a:b]}" for a, b in sentence_spans(text, split)]


def question_pairs(
    group: int,
    record: dict[str, Any],
    split: SentenceSplitter,
    rng: random.Random,
    *,
    positives: int = 2,
    hard: int = 8,
    easy: int = 2,
    passages: int = 5,
) -> list[Pair]:
    """Up to ``positives`` answer sentences and ``hard`` + ``easy`` negatives."""
    question, answers = str(record["question"]), [str(a) for a in record["answers"]]

    def pool(key: str) -> list[str]:
        return [s for ctx in record[key][:passages] for s in _sentences(ctx, split)]

    answer = [s for s in pool("positive_ctxs") if contains_answer(s, answers)]
    # A hard negative mentioning the answer is label noise (0.1% in DPR): skip it.
    hard_pool = [s for s in pool("hard_negative_ctxs") if not contains_answer(s, answers)]
    easy_pool = [s for s in pool("negative_ctxs") if not contains_answer(s, answers)]
    if not answer:
        return []

    def sample(items: list[str], k: int) -> list[str]:
        return rng.sample(items, min(k, len(items)))

    return (
        [Pair(group, question, s, 1, "answer") for s in sample(answer, positives)]
        + [Pair(group, question, s, 0, "hard") for s in sample(hard_pool, hard)]
        + [Pair(group, question, s, 0, "easy") for s in sample(easy_pool, easy)]
    )


def is_validation(question: str, fraction: float = 0.1) -> bool:
    """Deterministic question-level split, stable across runs and machines."""
    return zlib.crc32(question.encode("utf-8")) % 1000 < fraction * 1000


def build(records: Sequence[dict[str, Any]], split: SentenceSplitter, seed: int = 0) -> Iterator[Pair]:
    rng = random.Random(seed)
    for group, record in enumerate(records):
        yield from question_pairs(group, record, split, rng)


def write_jsonl(pairs: Iterable[Pair], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Pair]:
    with path.open(encoding="utf-8") as f:
        return [Pair(**json.loads(line)) for line in f]
