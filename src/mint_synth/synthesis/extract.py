from __future__ import annotations

import re
from typing import Any

from mint_synth.inference.llm import JsonLLM
from mint_synth.inference.prompts import FACT_SYSTEM, fact_user
from mint_synth.core.util import canonical_decimal, normalize_text, stable_id


ALLOWED_FACT_TYPES = {"reported", "definition", "qualitative", "event", "date"}
_NUMBER_RE = re.compile(r"\(?[-+]?\d[\d,]*(?:\.\d+)?%?\)?")


def _normalize_scale(value: Any) -> str | None:
    if value is None:
        return None
    folded = normalize_text(value)
    aliases = {
        "one": "ones",
        "unit": "ones",
        "units": "ones",
        "thousands": "thousand",
        "millions": "million",
        "billions": "billion",
        "trillions": "trillion",
    }
    return aliases.get(folded, folded or None)


def _normalize_unit(value: Any) -> str | None:
    if value is None:
        return None
    folded = normalize_text(value)
    if folded in {"%", "percentage", "percentage points", "pct"}:
        return "percent" if folded != "percentage points" else "percentage_points"
    return str(value).strip() or None


class FactExtractor:
    def __init__(
        self,
        llm: JsonLLM,
        *,
        max_facts_per_chunk: int = 30,
        max_entity_chars: int = 256,
        max_metric_chars: int = 256,
        max_period_chars: int = 64,
    ):
        self.llm = llm
        self.max_facts_per_chunk = max_facts_per_chunk
        self.max_entity_chars = max_entity_chars
        self.max_metric_chars = max_metric_chars
        self.max_period_chars = max_period_chars

    def extract(self, chunk: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        response = self.llm.complete_json(
            system=FACT_SYSTEM,
            user=fact_user(chunk),
            temperature=0.0,
            max_tokens=5000,
        )
        raw_facts = response.get("facts") if isinstance(response, dict) else None
        if not isinstance(raw_facts, list):
            return [], [self._rejection(chunk, "invalid_response", "response.facts is not a list")]
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for candidate in raw_facts[: self.max_facts_per_chunk]:
            fact, reason = self._localize(
                chunk,
                candidate,
                max_entity_chars=self.max_entity_chars,
                max_metric_chars=self.max_metric_chars,
                max_period_chars=self.max_period_chars,
            )
            if fact is None:
                rejected.append(self._rejection(chunk, "fact_rejected", reason, candidate))
            else:
                accepted.append(fact)
        return accepted, rejected

    @staticmethod
    def _rejection(
        chunk: dict[str, Any],
        stage: str,
        reason: str,
        candidate: Any = None,
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "source_id": chunk.get("source_id"),
            "chunk_id": chunk.get("chunk_id"),
            "reason": reason,
            "candidate": candidate,
        }

    @staticmethod
    def _localize(
        chunk: dict[str, Any],
        candidate: Any,
        *,
        max_entity_chars: int = 256,
        max_metric_chars: int = 256,
        max_period_chars: int = 64,
    ) -> tuple[dict[str, Any] | None, str]:
        if not isinstance(candidate, dict):
            return None, "candidate is not an object"
        quote = candidate.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            return None, "missing verbatim quote"
        local_start = chunk["text"].find(quote)
        if local_start < 0:
            return None, "quote is not an exact substring of the chunk"
        entity = candidate.get("entity")
        metric = candidate.get("metric")
        if not isinstance(entity, str) or not entity.strip():
            return None, "missing entity"
        if not isinstance(metric, str) or not metric.strip():
            return None, "missing metric"
        if len(entity.strip()) > max_entity_chars:
            return None, "entity exceeds extraction.max_entity_chars"
        if len(metric.strip()) > max_metric_chars:
            return None, "metric exceeds extraction.max_metric_chars"
        period = candidate.get("period")
        if period is not None and (not isinstance(period, str) or len(period.strip()) > max_period_chars):
            return None, "period must be a bounded string or null"
        value_text = candidate.get("value_text")
        if value_text is not None and not isinstance(value_text, (str, bool)):
            value_text = str(value_text)
        value_num = canonical_decimal(candidate.get("value_num"))
        if value_num is None and value_text in (None, ""):
            return None, "fact has neither a numeric nor textual value"
        if value_num is not None:
            quote_numbers = {
                normalized
                for token in _NUMBER_RE.findall(quote)
                if (normalized := canonical_decimal(token)) is not None
            }
            if value_num not in quote_numbers:
                return None, "value_num is not explicitly present in the evidence quote"
        if isinstance(value_text, str) and normalize_text(value_text) not in normalize_text(quote):
            return None, "value_text is not a source-faithful substring of the evidence quote"
        fact_type = str(candidate.get("fact_type", "reported")).lower()
        if fact_type not in ALLOWED_FACT_TYPES:
            return None, f"unsupported fact_type {fact_type!r}"
        chunk_start = int(chunk["locator"]["char_start"])
        global_start = chunk_start + local_start
        global_end = global_start + len(quote)
        fact_id = stable_id(
            "fact",
            chunk["source_id"],
            global_start,
            global_end,
            normalize_text(entity),
            normalize_text(period),
            normalize_text(metric),
            value_num if value_num is not None else normalize_text(value_text),
            _normalize_unit(candidate.get("unit")),
            _normalize_scale(candidate.get("scale")),
            normalize_text(candidate.get("currency")),
        )
        return (
            {
                "fact_id": fact_id,
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "entity": entity.strip(),
                "period": period.strip() if isinstance(period, str) else None,
                "metric": metric.strip(),
                "value_text": value_text,
                "value_num": value_num,
                "unit": _normalize_unit(candidate.get("unit")),
                "scale": _normalize_scale(candidate.get("scale")),
                "currency": str(candidate["currency"]).strip().upper() if candidate.get("currency") else None,
                "fact_type": fact_type,
                "quote": quote,
                "locator": {
                    **chunk.get("locator", {}),
                    "char_start": global_start,
                    "char_end": global_end,
                },
            },
            "",
        )


def deduplicate_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for fact in facts:
        current = chosen.get(fact["fact_id"])
        if current is None or len(str(fact.get("quote", ""))) < len(str(current.get("quote", ""))):
            chosen[fact["fact_id"]] = fact
    return sorted(
        chosen.values(),
        key=lambda item: (
            str(item.get("source_id")),
            int(item.get("locator", {}).get("char_start", 0)),
            str(item.get("fact_id")),
        ),
    )
