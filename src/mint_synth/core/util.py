from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    payload = "\x1f".join(
        json.dumps(part, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if not isinstance(part, str)
        else part
        for part in parts
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_question(value: str) -> str:
    value = normalize_text(value)
    return re.sub(r"[^\w\u4e00-\u9fff]+", " ", value).strip()


def token_set(value: str) -> set[str]:
    normalized = normalize_question(value)
    latin = re.findall(r"[a-z0-9_]+", normalized)
    han = re.findall(r"[\u4e00-\u9fff]", normalized)
    return set(latin + han)


def jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def canonical_decimal(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raw = str(value).strip().replace(",", "")
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1]
    raw = re.sub(r"^[\$€£¥￥]", "", raw).strip()
    raw = raw.rstrip("%").strip()
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if negative:
        number = -number
    if number == 0:
        return "0"
    rendered = format(number.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return canonical_decimal(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value

