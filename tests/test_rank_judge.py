from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation.llm_clients import Completion
from evaluation.llm_eval import ResponseCache
from evaluation.rank_judge import build_tasks, parse_ranking, report, shortlist, shown_order
from test_raw import make_raw


def test_parse_ranking() -> None:
    assert parse_ranking("4, 12, 7", 20) == [3, 11, 6]
    assert parse_ranking("[2], [2], 30, 1", 20) == [1, 0]  # duplicates and out-of-range dropped
    assert parse_ranking("none of them", 20) == []


def test_shown_order_is_a_seeded_shuffle_of_the_shortlist() -> None:
    r = make_raw()
    assert sorted(shown_order(r)) == sorted(int(i) for i in shortlist(r))
    assert shown_order(r) == shown_order(r)


def test_report_scores_judge_picks_through_the_shuffle(tmp_path: Path) -> None:
    r = make_raw()
    (task,) = build_tasks([r], limit=1)
    answer = int(np.flatnonzero(r.sent_has_answer)[0])
    order = shown_order(r)
    cache = ResponseCache(tmp_path / "judge.jsonl")
    cache.add(task, Completion(str(order.index(answer) + 1), 10, 2))
    rows = report([r], {}, {"judge": cache}, {}, [task])
    judge = next(row for row in rows if row.scorer == "judge" and row.subset == "all")
    assert judge.rank1 == 1.0 and judge.top3 == 1.0
    # A rescored copy ranking the answer sentence last misses at rank 1.
    worst = replace(r, sent_rerank=np.where(r.sent_has_answer, -1.0, 1.0).astype(np.float32))
    rows = report([r], {"flip": [worst]}, {"judge": cache}, {}, [task])
    flip = next(row for row in rows if row.scorer == "flip" and row.subset == "all")
    assert flip.rank1 == 0.0
