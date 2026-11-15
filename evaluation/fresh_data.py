"""A question set the models cannot have memorized.

NQ questions are largely known to a strong model without any context, which
hides what the context contributes. Here the documents are English Wikipedia
articles about events of the last few months (after the models' training),
and each question is written by an LLM from one ~100-word passage, with a
short answer copied from it. Every question then gets that gold passage plus
the lexically closest other passages that lack the answer, in the same shape
as the NQ data (``nq_data.Example``), so the rest of the evaluation runs
unchanged with ``--split fresh``.

Questions written from a passage repeat its wording, which makes finding the
passage far easier than it is for real user questions. ``rewrite`` has the
LLM reword each question away from its passage; ``build --rewritten`` uses
those. This set complements NQ, it does not replace it.

    python -m evaluation.fresh_data crawl --months "October 2026" "September 2026" [--limit 800]
    python -m evaluation.fresh_data write [--per-article 3] [--confirm]
    python -m evaluation.fresh_data rewrite [--confirm]
    python -m evaluation.fresh_data build [--rewritten]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .answers import contains_answer, normalize
from .llm_eval import ResponseCache, Task
from .nq_data import Doc, Example

DIR = Path("data/fresh")
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "research-script/0.1"
PASSAGE_WORDS = 100
TOTAL_DOCS = 20  # like the NQ data: the gold passage and 19 others

_ENDING = re.compile(
    r"^==+\s*(See also|References|External links|Notes|Further reading|Sources|Bibliography)\s*==+\s*$", re.M | re.I
)
_HEADING = re.compile(r"^==+.*==+\s*$", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_YES_NO = frozenset("is are was were did does do has have had can could will would should".split())

WRITE_PROMPT = (
    'Below is a passage from the Wikipedia article "{title}".\n\n'
    "Passage: {passage}\n\n"
    "Write one question, in the style of a web search query, that this passage answers.\n"
    "- The question must stand alone: name the event, person or place in full, and never say "
    '"the passage", "the article" or "this event".\n'
    "- Ask about what actually happened in this event (a result, a name, a number, a date, a "
    "place), not about background that was known long before it.\n"
    "- Ask one thing only, and not a yes/no question.\n"
    "- The answer must be the shortest phrase that answers it, copied exactly from the passage "
    '(for example "17", not "all 17 occupants"; "Monaco", not "at the Grimaldi Forum in Monaco"), '
    "at most 4 words, and the only correct answer to the question.\n"
    "- The answer must not appear in the question.\n\n"
    'Reply with JSON only: {{"question": "...", "answer": "..."}}. If the passage has no such fact, '
    "reply SKIP."
)

REWRITE_PROMPT = (
    "Question: {question}\nAnswer: {answer}\n\n"
    "The question was written by someone who was looking at this passage:\n{passage}\n\n"
    "Rewrite the question the way a person who has never seen the passage would type it into a "
    "search engine.\n"
    "- Keep only the names, dates and numbers needed to identify what is asked about.\n"
    "- Replace every other word borrowed from the passage with different wording: synonyms, a "
    "more general term, another sentence structure.\n"
    "- It must still ask one thing, have the same answer, and not contain the answer.\n\n"
    'Reply with JSON only: {{"question": "..."}}.'
)


# ------------------------------------------------------------------ crawl


def _get(pause: float, **params: Any) -> dict[str, Any]:
    """One polite API request: a pause before it, and a wait when the server says so."""
    url = API + "?" + urllib.parse.urlencode({"format": "json", "formatversion": 2, "maxlag": 5, **params})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        time.sleep(pause)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data: dict[str, Any] = json.loads(response.read())
                return data
        except urllib.error.HTTPError as error:
            wait = int(error.headers.get("Retry-After") or 30 * (attempt + 1))
            print(f"  HTTP {error.code}, waiting {wait}s", flush=True)
            time.sleep(min(wait, 300))
        except OSError as error:
            print(f"  {type(error).__name__}, retrying", flush=True)
            time.sleep(10)
    raise SystemExit("the server keeps refusing; rerun later (progress is saved)")


def _members(category: str, depth: int, seen: set[str], pages: dict[str, str], limit: int, pause: float) -> None:
    if category in seen or len(pages) >= limit:
        return
    seen.add(category)
    cont: dict[str, Any] = {}
    while True:
        data = _get(pause, action="query", list="categorymembers", cmtitle=category, cmlimit=500, **cont)
        for m in data.get("query", {}).get("categorymembers", []):
            if m["ns"] == 0:
                pages.setdefault(m["title"], category)
            elif m["ns"] == 14 and depth > 0:
                _members(m["title"], depth - 1, seen, pages, limit, pause)
        cont = data.get("continue", {})
        if not cont or len(pages) >= limit:
            return


def crawl(months: Sequence[str], out_dir: Path, limit: int, pause: float = 1.0, min_chars: int = 1500) -> None:
    """Articles under ``Category:<month>`` (three levels down) as title + plain text. Resumable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    titles_path, articles_path = out_dir / "titles.json", out_dir / "articles.jsonl"
    pages: dict[str, str]
    if titles_path.exists():
        pages = json.loads(titles_path.read_text(encoding="utf-8"))
    else:
        pages = {}
        for month in months:
            _members(f"Category:{month}", 3, set(), pages, limit * 5, pause)
            print(f"{month}: {len(pages)} titles so far", flush=True)
        titles_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    done = {a["title"] for a in load_articles(articles_path)} if articles_path.exists() else set()
    kept = len(done)
    # A fixed shuffle spreads the articles over the categories.
    titles = sorted(t for t in pages if not t.startswith(("Deaths in", "List of")))
    random.Random(0).shuffle(titles)
    with articles_path.open("a", encoding="utf-8") as f:
        for title in titles:
            if kept >= limit:
                break
            if title in done:
                continue
            data = _get(pause, action="query", prop="extracts", explaintext=1, titles=title)
            text = (data.get("query", {}).get("pages") or [{}])[0].get("extract", "")
            if len(text) < min_chars:
                continue
            f.write(json.dumps({"title": title, "category": pages[title], "text": text}, ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
            if kept % 50 == 0:
                print(f"  articles: {kept}", flush=True)
    print(f"{kept} articles in {articles_path}")


def load_articles(path: Path = DIR / "articles.jsonl") -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------- passages


def passages(text: str, words: int = PASSAGE_WORDS, min_words: int = 40) -> list[str]:
    """The article body as passages of about ``words`` words: whole sentences,
    never across a section heading, headings and the reference sections dropped."""
    out = []
    for section in _HEADING.split(_ENDING.split(text)[0]):
        current: list[str] = []
        count = 0
        for sentence in _SENTENCE_END.split(" ".join(section.split())):
            current.append(sentence)
            count += len(sentence.split())
            if count >= words:
                out.append(" ".join(current))
                current, count = [], 0
        if count >= min_words:
            out.append(" ".join(current))
    return out


# -------------------------------------------------------------- questions


def write_tasks(articles: Sequence[dict[str, str]], per_article: int, seed: int = 0) -> list[Task]:
    """One question-writing task for each of up to ``per_article`` random passages of every article."""
    rng = random.Random(seed)
    tasks = []
    for a, article in enumerate(articles):
        options = passages(article["text"])
        for p in sorted(rng.sample(range(len(options)), min(per_article, len(options)))):
            prompt = WRITE_PROMPT.format(title=article["title"], passage=options[p])
            tasks.append(Task(f"write:{article['title']}:{p}", a, "write", "write", prompt, (), options[p]))
    return tasks


def parse_written(text: str, passage: str) -> tuple[str, str] | None:
    """(question, answer) from the writer's reply, or None if it skipped or broke a rule."""
    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return None
    try:
        item = json.loads(match.group(0))
        question, answer = " ".join(str(item["question"]).split()), " ".join(str(item["answer"]).split())
    except (ValueError, KeyError, TypeError):
        return None
    if (
        not normalize(answer)
        or len(answer.split()) > 4
        or not contains_answer(passage, [answer])
        or not _question_ok(question, answer)
    ):
        return None
    return question, answer


def _question_ok(question: str, answer: str) -> bool:
    words = question.lower().split()
    return (
        not contains_answer(question, [answer])
        and question.endswith("?")
        and question.count("?") == 1
        and 4 <= len(words) <= 30
        and words[0] not in _YES_NO
        and not re.search(r"\b(passage|article)\b", question, re.I)
    )


def rewrite_tasks(written: Sequence[tuple[Task, str]]) -> list[Task]:
    """One rewording task for every accepted question; the task id matches its writing task."""
    tasks = []
    for task, reply in written:
        parsed = parse_written(reply, task.context)
        if parsed is not None:
            prompt = REWRITE_PROMPT.format(question=parsed[0], answer=parsed[1], passage=task.context)
            tasks.append(Task("re" + task.task_id, task.qid, "rewrite", "rewrite", prompt, (parsed[1],), task.context))
    return tasks


def apply_rewrites(written: Sequence[tuple[Task, str]], rewrites: dict[str, str]) -> list[tuple[Task, str]]:
    """``written`` with each question replaced by its rewording (``rewrites``:
    rewording task id -> the rewriter's reply). A question without a rewording
    that keeps the question rules is left out."""
    out = []
    for task, reply in written:
        parsed = parse_written(reply, task.context)
        match = re.search(r"\{.*\}", rewrites.get("re" + task.task_id, ""), re.S)
        if parsed is None or match is None:
            continue
        try:
            question = " ".join(str(json.loads(match.group(0))["question"]).split())
        except (ValueError, KeyError, TypeError):
            continue
        if _question_ok(question, parsed[1]):
            out.append((task, json.dumps({"question": question, "answer": parsed[1]})))
    return out


# ------------------------------------------------------------ distractors


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Lexical:
    """Okapi BM25 over a fixed set of texts, scored through an inverted index."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        counts = [Counter(_tokens(t)) for t in texts]
        length = np.array([sum(c.values()) for c in counts], dtype=np.float64)
        norm = k1 * (1 - b + b * length / max(1.0, float(length.mean()) if len(texts) else 1.0))
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, count in enumerate(counts):
            for term, tf in count.items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        self._n = len(texts)
        self._weights: dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]] = {}
        for term, (ids, tfs) in postings.items():
            index = np.array(ids, dtype=np.int64)
            freq = np.array(tfs, dtype=np.float64)
            idf = np.log(1 + (self._n - len(ids) + 0.5) / (len(ids) + 0.5))
            self._weights[term] = (index, idf * freq * (k1 + 1) / (freq + norm[index]))

    def scores(self, query: str) -> NDArray[np.float64]:
        out = np.zeros(self._n, dtype=np.float64)
        for term in set(_tokens(query)):
            if term in self._weights:
                index, weight = self._weights[term]
                out[index] += weight
        return out


def build_examples(
    articles: Sequence[dict[str, str]], written: Sequence[tuple[Task, str]], total_docs: int = TOTAL_DOCS
) -> list[Example]:
    """An ``Example`` per accepted question: its gold passage first, then the
    passages closest to the question (BM25 over every passage of every article)
    that do not contain the answer. ``written`` pairs each writing task with the
    writer's reply."""
    pool = [(article["title"], p) for article in articles for p in passages(article["text"])]
    index = Lexical([f"{title} {text}" for title, text in pool])
    examples: list[Example] = []
    seen: set[str] = set()
    for task, reply in written:
        parsed = parse_written(reply, task.context)
        if parsed is None or normalize(parsed[0]) in seen:
            continue
        question, answer = parsed
        title = articles[task.qid]["title"]
        docs = [Doc(title, task.context, True)]
        for i in np.argsort(-index.scores(question), kind="stable"):
            if len(docs) >= total_docs:
                break
            other_title, text = pool[int(i)]
            if text != task.context and not contains_answer(f"{other_title} {text}", [answer]):
                docs.append(Doc(other_title, text, False))
        if len(docs) < total_docs:
            continue
        seen.add(normalize(question))
        examples.append(Example(len(examples), question, (answer,), tuple(docs)))
    return examples


def save(examples: Sequence[Example], path: Path = DIR / "questions.jsonl") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = {
                "question": ex.question, "answers": list(ex.answers),
                "ctxs": [{"title": d.title, "text": d.text, "isgold": d.is_gold} for d in ex.docs],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load(path: Path = DIR / "questions.jsonl") -> list[Example]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for qid, line in enumerate(f):
            row = json.loads(line)
            docs = tuple(Doc(c["title"], c["text"], bool(c["isgold"])) for c in row["ctxs"])
            examples.append(Example(qid, row["question"], tuple(row["answers"]), docs))
    return examples


# ------------------------------------------------------------------- main


def main() -> None:
    from . import llm_clients, llm_eval
    from .__main__ import CLAUDE_TOKEN_RATIO

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["crawl", "write", "rewrite", "build"])
    parser.add_argument("--months", nargs="*", default=[], help='crawl: e.g. "October 2026"')
    parser.add_argument("--limit", type=int, default=800, help="crawl: articles to keep")
    parser.add_argument("--per-article", type=int, default=3, help="write: passages per article")
    parser.add_argument("--model", default="haiku", help="write: the question writer")
    parser.add_argument("--confirm", action="store_true", help="write / rewrite: actually send (default: dry run)")
    parser.add_argument("--rewritten", action="store_true", help="build: use the reworded questions")
    args = parser.parse_args()

    if args.command == "crawl":
        crawl(args.months, DIR, args.limit)
        return
    articles = load_articles()
    tasks = write_tasks(articles, args.per_article)
    cache = ResponseCache(DIR / f"written_{args.model}.jsonl")
    written = [(t, str(cache.done[t.task_id]["text"])) for t in tasks if t.task_id in cache.done]
    reworded = ResponseCache(DIR / f"rewritten_{args.model}.jsonl")
    if args.command == "build":
        if args.rewritten:
            written = apply_rewrites(written, {k: str(v["text"]) for k, v in reworded.done.items()})
        examples = build_examples(articles, written)
        save(examples)
        print(f"{len(written)} replies -> {len(examples)} questions in {DIR / 'questions.jsonl'}")
        return
    if args.command == "rewrite":
        tasks, cache = rewrite_tasks(written), reworded
    pending = [t for t in tasks if t.task_id not in cache.done]
    prices = llm_clients.CLAUDE_MODELS[args.model].price_per_mtok
    tokens_in = int(CLAUDE_TOKEN_RATIO * sum(llm_eval.estimate_tokens(t.prompt) for t in pending))
    cost = (tokens_in * prices[0] + 40 * len(pending) * prices[1]) / 1e6
    print(f"{len(articles)} articles, {len(tasks)} requests, {len(pending)} to send, ~{tokens_in:,} input tokens, est. ${cost:.2f}")
    if not args.confirm:
        print("Dry run only: nothing was sent. Add --confirm to send.")
        return
    failures = llm_eval.run(tasks, llm_clients.ClaudeClient(args.model), cache, len(tasks))
    print(f"done, {failures} failures (not cached; rerun to retry)")

if __name__ == "__main__":
    main()
