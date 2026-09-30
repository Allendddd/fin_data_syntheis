from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .util import token_set


class QuestionIndex:
    """Exact Jaccard gate with an inverted-token candidate index."""

    def __init__(self, questions: Iterable[str]):
        self._tokens: list[frozenset[str]] = []
        self._inverted: dict[str, list[int]] = defaultdict(list)
        self._empty: list[int] = []
        for question in questions:
            tokens = frozenset(token_set(question))
            index = len(self._tokens)
            self._tokens.append(tokens)
            if not tokens:
                self._empty.append(index)
            for token in tokens:
                self._inverted[token].append(index)

    def overlaps(self, question: str, threshold: float) -> bool:
        if threshold <= 0.0:
            return bool(self._tokens)
        query = frozenset(token_set(question))
        if not query:
            return bool(self._empty) and threshold <= 1.0
        candidates: set[int] = set()
        for token in query:
            candidates.update(self._inverted.get(token, ()))
        for index in candidates:
            candidate = self._tokens[index]
            score = len(query & candidate) / len(query | candidate)
            if score >= threshold:
                return True
        return False

    def __len__(self) -> int:
        return len(self._tokens)
