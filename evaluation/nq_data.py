"""Download, load and split the Lost in the Middle NQ multi-document data."""

from __future__ import annotations

import gzip
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .answers import contains_answer

# Gold passage first; our pipeline re-ranks, so the input order does not matter.
URL = (
    "https://github.com/nelson-liu/lost-in-the-middle/raw/main/qa_data/"
    "20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz"
)
DEFAULT_PATH = Path("data/nq/nq-open-20_total_documents_gold_at_0.jsonl.gz")


@dataclass(frozen=True, slots=True)
class Doc:
    title: str
    text: str
    is_gold: bool


@dataclass(frozen=True, slots=True)
class Example:
    qid: int
    question: str
    answers: tuple[str, ...]
    docs: tuple[Doc, ...]
    # "answerable", or a synthetic negative whose passages hold no answer:
    # "unrelated" (another question's passages, off-topic) or "distractors"
    # (its own 19 distractors without the gold passage, on-topic and harder).
    kind: str = "answerable"

    @property
    def answerable(self) -> bool:
        return self.kind == "answerable"


def download(path: Path = DEFAULT_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path


def load(path: Path = DEFAULT_PATH) -> list[Example]:
    examples = []
    with gzip.open(download(path), "rt", encoding="utf-8") as f:
        for qid, line in enumerate(f):
            row = json.loads(line)
            docs = tuple(Doc(c["title"], c["text"], bool(c["isgold"])) for c in row["ctxs"])
            examples.append(Example(qid, row["question"], tuple(row["answers"]), docs))
    return examples


def split(examples: list[Example], seed: int = 0) -> tuple[list[Example], list[Example]]:
    """Fixed random dev/test halves: calibrate on dev, report once on test."""
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


def unanswerable(examples: list[Example], seed: int = 1) -> list[Example]:
    """Pair each question with another question's passages (answer-free only).

    Easy negatives (off-topic); they should yield an empty context.
    """
    rng = random.Random(seed)
    out = []
    for ex in examples:
        other = examples[rng.randrange(len(examples))]
        if other.qid == ex.qid or any(contains_answer(d.text, ex.answers) for d in other.docs):
            continue
        docs = tuple(Doc(d.title, d.text, False) for d in other.docs)
        out.append(Example(ex.qid, ex.question, ex.answers, docs, kind="unrelated"))
    return out


def distractors_only(examples: list[Example]) -> list[Example]:
    """Each question with its own distractors, gold passage removed.

    Hard negatives: on-topic passages retrieved for this very question that
    still lack the answer (questions whose distractors mention it are skipped).
    """
    out = []
    for ex in examples:
        docs = tuple(d for d in ex.docs if not d.is_gold)
        if not any(contains_answer(d.text, ex.answers) for d in docs):
            out.append(Example(ex.qid, ex.question, ex.answers, docs, kind="distractors"))
    return out
