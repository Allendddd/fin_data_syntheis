from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from mint_synth.core.util import stable_id


class JsonlError(ValueError):
    pass


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise JsonlError(f"{source}:{line_number}: each JSONL row must be an object")
            yield item


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temp_name = handle.name
    try:
        with handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return count


def load_sources(
    path: str | Path,
    *,
    text_field: str = "text",
    id_field: str | None = None,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    for index, row in enumerate(read_jsonl(path), 1):
        if max_records is not None and len(sources) >= max_records:
            break
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise JsonlError(f"{path}:{index}: field {text_field!r} must be a non-empty string")
        external_id = row.get(id_field) if id_field else row.get("id")
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_sha256 in seen_content:
            continue
        seen_content.add(content_sha256)
        identity = str(external_id) if external_id is not None else content_sha256
        source_id = stable_id("src", identity, content_sha256)
        metadata = {key: value for key, value in row.items() if key != text_field}
        sources.append(
            {
                "source_id": source_id,
                "external_id": external_id,
                "content_sha256": content_sha256,
                "text": text,
                "metadata": metadata,
                "locator": {"jsonl_line": index, "text_field": text_field},
            }
        )
    return sources
