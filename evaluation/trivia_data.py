"""TriviaQA questions in the shape of the NQ data, as a second question set.

Nothing in the pipeline was tuned on TriviaQA, and its questions were written
by trivia authors who never saw the passages, so it checks whether the NQ
results hold on another kind of question. The source is DPR's retriever file
for the TriviaQA dev questions: every question comes with Wikipedia passages
that contain an answer and with the BM25-closest passages that do not. Each
``Example`` is the first answer passage plus the closest ``TOTAL_DOCS - 1``
answer-free ones, like the NQ data's gold passage and its 19 distractors.

The gold passage here only *contains* an answer string (DPR matched it by
BM25); nobody checked that it answers the question, unlike NQ's gold passage.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import gate_data
from .answers import contains_answer
from .nq_data import Doc, Example

URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz"
DEFAULT_PATH = Path("data/dpr/biencoder-trivia-dev.json.gz")
TOTAL_DOCS = 20


def examples(records: Sequence[dict[str, Any]], total_docs: int = TOTAL_DOCS, seed: int = 0) -> list[Example]:
    """One ``Example`` per record that has an answer passage and enough
    answer-free ones, in a fixed random order (so a ``--limit`` is a sample)."""
    out: list[tuple[str, tuple[str, ...], tuple[Doc, ...]]] = []
    for record in records:
        answers = tuple(str(a) for a in record["answers"])
        if not record["positive_ctxs"]:
            continue
        gold = record["positive_ctxs"][0]
        docs = [Doc(str(gold["title"]), str(gold["text"]), True)]
        for ctx in record["hard_negative_ctxs"]:
            if len(docs) >= total_docs:
                break
            if not contains_answer(f"{ctx['title']} {ctx['text']}", answers):
                docs.append(Doc(str(ctx["title"]), str(ctx["text"]), False))
        if len(docs) == total_docs:
            out.append((str(record["question"]), answers, tuple(docs)))
    random.Random(seed).shuffle(out)
    return [Example(qid, question, answers, docs) for qid, (question, answers, docs) in enumerate(out)]


def load(path: Path = DEFAULT_PATH) -> list[Example]:
    return examples(gate_data.load(gate_data.download(path, URL)))
