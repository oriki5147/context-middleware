"""Answer matching, following Lost in the Middle / SQuAD normalization:
lowercase, strip punctuation and articles, collapse whitespace, then test
whether any gold answer occurs as a substring."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def contains_answer(text: str, answers: Sequence[str]) -> bool:
    haystack = normalize(text)
    return any((needle := normalize(a)) and needle in haystack for a in answers)
