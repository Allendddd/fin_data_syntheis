from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping
from mint_synth.synthesis.replay import answer_matches, replay_graph


def _answer_ok(task: Mapping[str, Any], proposed: str, tolerance: str) -> bool:
    facts = {fact["fact_id"]: fact for fact in task.get("evidence", [])}
    replay = replay_graph(task["evidence_graph"], facts)
    if not answer_matches(task["answer"], replay, tolerance=tolerance):
        return False
    expected = str(task["answer"].get("value", "")).strip()
    expected_text = str(task["answer"].get("text", "")).strip()
    if isinstance(task["answer"].get("value"), bool):
        english = ("true", "yes", "supported", "correct") if task["answer"]["value"] else ("false", "no", "unsupported", "incorrect")
        chinese = ("是", "成立") if task["answer"]["value"] else ("否", "不成立")
        text = proposed.casefold()
        if task["answer"]["value"] and ("不是" in text or "不成立" in text or re.search(r"\bnot\s+(?:supported|correct|true)\b", text)):
            return False
        return any(re.search(r"\b" + word + r"\b", text) for word in english) or any(word in text for word in chinese)
    try:
        target = Decimal(expected)
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", proposed)
        return bool(numbers) and abs(Decimal(numbers[-1]) - target) <= Decimal(tolerance) * max(Decimal(1), abs(target))
    except (InvalidOperation, ValueError):
        return bool(expected_text) and expected_text.casefold() in proposed.casefold()
