from __future__ import annotations

import bisect
import re
from typing import Any

from .util import stable_id


_BOUNDARY_RE = re.compile(r"(?:\r?\n){2,}|(?<=[。！？.!?])(?:[ \t]+|\r?\n)|\r?\n")


def _semantic_boundaries(text: str) -> list[int]:
    points = {0, len(text)}
    for match in _BOUNDARY_RE.finditer(text):
        points.add(match.end())
    return sorted(points)


def semantic_chunks(
    source: dict[str, Any],
    *,
    max_chars: int = 6000,
    overlap_chars: int = 400,
    min_chars: int = 600,
) -> list[dict[str, Any]]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < max_chars")
    if min_chars <= 0 or min_chars > max_chars:
        raise ValueError("min_chars must satisfy 0 < min_chars <= max_chars")

    text = source["text"]
    boundaries = _semantic_boundaries(text)
    chunks: list[dict[str, Any]] = []
    cursor = 0
    ordinal = 0
    while cursor < len(text):
        upper = min(len(text), cursor + max_chars)
        lower = min(upper, cursor + min_chars)
        index = bisect.bisect_right(boundaries, upper) - 1
        end = boundaries[index] if index >= 0 and boundaries[index] >= lower else upper
        if end <= cursor:
            end = upper

        raw_start, raw_end = cursor, end
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        while raw_end > raw_start and text[raw_end - 1].isspace():
            raw_end -= 1
        if raw_end > raw_start:
            chunk_text = text[raw_start:raw_end]
            chunk_id = stable_id("chk", source["source_id"], raw_start, raw_end, chunk_text)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": source["source_id"],
                    "ordinal": ordinal,
                    "text": chunk_text,
                    "locator": {
                        **source.get("locator", {}),
                        "char_start": raw_start,
                        "char_end": raw_end,
                    },
                    "source_metadata": source.get("metadata", {}),
                }
            )
            ordinal += 1

        if end >= len(text):
            break
        desired = max(cursor + 1, end - overlap_chars)
        next_index = bisect.bisect_left(boundaries, desired)
        next_cursor = boundaries[next_index] if next_index < len(boundaries) and boundaries[next_index] < end else desired
        cursor = max(cursor + 1, min(next_cursor, end))
    return chunks


def chunk_sources(
    sources: list[dict[str, Any]],
    **options: Any,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in sources:
        result.extend(semantic_chunks(source, **options))
    return result

