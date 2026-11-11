"""Teacher labels for distilling an LLM's sentence ranking into the reranker.

``build`` turns DPR NQ questions into the same kind of shortlist the
production compressor hands its reranker: each question's positive and BM25
hard-negative passages are chunked and split by the production code, and the
bi-encoder's top ``top_n`` sentences are kept, shuffled. ``submit`` sends one
rank_judge prompt per shortlist through the Message Batches API (half price,
processed server-side: the machine may sleep until ``fetch``), and ``fetch``
stores the replies next to the shortlists.

    python -m evaluation.distill build
    python -m evaluation.distill context   # neighbors of each shortlisted sentence
    python -m evaluation.distill submit [--model haiku] [--confirm]
    python -m evaluation.distill status | fetch

``--source train`` does the same for DPR's NQ train questions (streamed,
first ``--limit``), under data/distill/train/; training reads both.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from context_middleware import (
    BGE_QUERY_PREFIX,
    MarkItDownPreprocessor,
    ONNXBiEncoderRetriever,
    SelectiveSentenceCompressor,
)

from . import gate_data
from .answers import contains_answer
from .llm_eval import estimate_tokens
from .rank_judge import format_prompt

MODEL = Path("models/bge-small-en-v1.5")
OUT = Path("data/distill")
POSITIVES, HARD = 3, 12  # DPR passages per question: gold, then BM25 hard negatives
SOURCES = {
    "dev": (gate_data.DEFAULT_PATH, gate_data.URL, OUT),
    "train": (gate_data.TRAIN_PATH, gate_data.TRAIN_URL, OUT / "train"),
}


@dataclass(frozen=True, slots=True)
class Shortlist:
    group: int  # DPR question index
    question: str
    answers: list[str]
    validation: bool  # gate_data's question-level split
    titles: list[str]  # all per-sentence lists are in the shown (shuffled) order
    texts: list[str]
    has_answer: list[bool]  # string match, for analysis only: the teacher sets labels
    bi: list[float]

    def prompt(self) -> str:
        return format_prompt(self.question, list(zip(self.titles, self.texts)))


def build(
    records: Sequence[dict[str, Any]],
    encoder: ONNXBiEncoderRetriever,
    preprocessor: MarkItDownPreprocessor,
    splitter: SelectiveSentenceCompressor,
    *,
    positives: int = POSITIVES,
    hard: int = HARD,
    top_n: int = 20,
    log: Any = print,
) -> Iterator[Shortlist]:
    for group, record in enumerate(records):
        question, answers = str(record["question"]), [str(a) for a in record["answers"]]
        passages = record["positive_ctxs"][:positives] + record["hard_negative_ctxs"][:hard]
        chunks = [
            chunk
            for j, ctx in enumerate(passages)
            for chunk in preprocessor.chunk(str(ctx["text"]), f"d{j}", title=str(ctx["title"]))
        ]
        sentences = splitter.split(chunks)
        if len(sentences) < 2:
            continue
        bi = encoder.embed_segments(sentences) @ encoder.encode_query(question)
        best = [int(i) for i in np.argsort(-bi, kind="stable")[:top_n]]
        shown = random.Random(group).sample(best, len(best))
        yield Shortlist(
            group,
            question,
            answers,
            gate_data.is_validation(question),
            [sentences[i].title for i in shown],
            [sentences[i].content for i in shown],
            [contains_answer(sentences[i].content, answers) for i in shown],
            [round(float(bi[i]), 5) for i in shown],
        )
        if (group + 1) % 500 == 0:
            log(f"  {group + 1}/{len(records)}")


def neighbor_context(
    records: Sequence[dict[str, Any]],
    shortlists: Sequence[Shortlist],
    preprocessor: MarkItDownPreprocessor,
    splitter: SelectiveSentenceCompressor,
    *,
    positives: int = POSITIVES,
    hard: int = HARD,
    log: Any = print,
) -> Iterator[dict[str, Any]]:
    """Each shortlisted sentence's previous and next sentence in its passage
    ("" at a passage edge), re-split exactly as ``build`` split them."""
    for n, item in enumerate(shortlists, 1):
        record = records[item.group]
        passages = record["positive_ctxs"][:positives] + record["hard_negative_ctxs"][:hard]
        chunks = [
            chunk
            for j, ctx in enumerate(passages)
            for chunk in preprocessor.chunk(str(ctx["text"]), f"d{j}", title=str(ctx["title"]))
        ]
        sentences = splitter.split(chunks)
        around: dict[tuple[str, str], tuple[str, str]] = {}
        for i, s in enumerate(sentences):
            prev = sentences[i - 1].content if i > 0 and sentences[i - 1].source_doc == s.source_doc else ""
            nxt = sentences[i + 1].content if i + 1 < len(sentences) and sentences[i + 1].source_doc == s.source_doc else ""
            around.setdefault((s.title, s.content), (prev, nxt))
        pairs = [around.get(key, ("", "")) for key in zip(item.titles, item.texts)]
        yield {"group": item.group, "prev": [p for p, _ in pairs], "next": [x for _, x in pairs]}
        if n % 500 == 0:
            log(f"  {n}/{len(shortlists)}")


def write_jsonl(items: Iterable[Shortlist], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[Shortlist]:
    with path.open(encoding="utf-8") as f:
        return [Shortlist(**json.loads(line)) for line in f]


def load_records(source: str, limit: int | None = None) -> list[dict[str, Any]]:
    path, url, _ = SOURCES[source]
    records = gate_data.iter_records(gate_data.download(path, url), positives=POSITIVES, hard=HARD)
    return list(islice(records, limit))


def custom_id(item: Shortlist) -> str:
    return f"q{item.group}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "context", "submit", "status", "fetch"])
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--source", choices=sorted(SOURCES), default="dev", help="DPR question file")
    parser.add_argument("--limit", type=int, default=None, help="build/submit: first N questions")
    parser.add_argument("--confirm", action="store_true", help="submit: actually send (default: dry run)")
    args = parser.parse_args()
    out_dir = SOURCES[args.source][2]
    shortlists = out_dir / "shortlists.jsonl"
    state = out_dir / f"batch_{args.model}.json"

    if args.command == "build":
        encoder = ONNXBiEncoderRetriever.from_pretrained(MODEL, query_prefix=BGE_QUERY_PREFIX, intra_op_num_threads=8)
        preprocessor = MarkItDownPreprocessor.from_pretrained(MODEL, 64, 0)
        records = load_records(args.source, args.limit)
        n = write_jsonl(build(records, encoder, preprocessor, SelectiveSentenceCompressor(encoder)), shortlists)
        print(f"saved {n} shortlists to {shortlists}")
        return
    if args.command == "context":
        encoder = ONNXBiEncoderRetriever.from_pretrained(MODEL, query_prefix=BGE_QUERY_PREFIX, intra_op_num_threads=8)
        preprocessor = MarkItDownPreprocessor.from_pretrained(MODEL, 64, 0)
        out = out_dir / "context.jsonl"
        items = read_jsonl(shortlists)
        records = load_records(args.source, max(i.group for i in items) + 1)
        rows = neighbor_context(records, items, preprocessor, SelectiveSentenceCompressor(encoder))
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved {out}")
        return

    from . import llm_clients

    spec = llm_clients.CLAUDE_MODELS[args.model]
    import anthropic

    if args.command == "submit":
        if state.exists():
            raise SystemExit(f"{state} exists: a batch was already submitted (use status/fetch)")
        items = read_jsonl(shortlists)[: args.limit]
        # Real Claude token counts ran ~2x the chars/4 estimate; batches cost half.
        tokens = 2 * sum(estimate_tokens(i.prompt()) for i in items)
        cost = 0.5 * (tokens * spec.price_per_mtok[0] + 20 * len(items) * spec.price_per_mtok[1]) / 1e6
        print(f"{len(items)} requests, ~{tokens:,} input tokens, est. ${cost:.2f} ({args.model}, batch)")
        if not args.confirm:
            print("Dry run only: nothing was sent. Add --confirm to send.")
            return
        from anthropic.types import ThinkingConfigParam

        thinking: ThinkingConfigParam = (
            {"type": "disabled"} if spec.thinking_off == "disabled" else {"type": "between_tools"}
        )
        batch = anthropic.Anthropic().messages.batches.create(
            requests=[
                {
                    "custom_id": custom_id(i),
                    "params": {
                        "model": spec.model_id,
                        "max_tokens": 256,
                        "thinking": thinking,
                        "output_config": {"effort": "low"},
                        "messages": [{"role": "user", "content": i.prompt()}],
                    },
                }
                for i in items
            ]
        )
        state.write_text(json.dumps({"id": batch.id, "requests": len(items)}), encoding="utf-8")
        print(f"submitted batch {batch.id}; saved to {state}. The machine may sleep now.")
        return

    batch_id = json.loads(state.read_text(encoding="utf-8"))["id"]
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    print(f"batch {batch_id}: {batch.processing_status}, {batch.request_counts}")
    if args.command == "fetch":
        if batch.processing_status != "ended":
            raise SystemExit("not finished yet; try again later")
        out = out_dir / f"teacher_{args.model}.jsonl"
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for result in client.messages.batches.results(batch_id):
                record: dict[str, object] = {"custom_id": result.custom_id, "type": result.result.type}
                if result.result.type == "succeeded":
                    message = result.result.message
                    record["text"] = "".join(b.text for b in message.content if b.type == "text").strip()
                    record["input_tokens"] = message.usage.input_tokens
                    record["output_tokens"] = message.usage.output_tokens
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
        print(f"saved {n} results to {out}")


if __name__ == "__main__":
    main()
