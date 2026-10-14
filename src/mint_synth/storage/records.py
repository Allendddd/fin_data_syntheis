from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from mint_synth.storage.catalog import InputSegment, file_fingerprint
from mint_synth.storage.parquetio import quarantine_row
from mint_synth.core.util import stable_id


def _field(record: Mapping[str, Any], name: str) -> Any:
    value: Any = record
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def verify_segment_file(segment: InputSegment, *, fingerprint_mode: str = "metadata") -> None:
    path = Path(segment.source_uri)
    try:
        stat = path.stat()
    except OSError as exc:
        raise RuntimeError(f"input disappeared after planning: {path}: {exc}") from exc
    if stat.st_size != segment.file_size or stat.st_mtime_ns != segment.mtime_ns:
        raise RuntimeError(f"input changed after planning: {path}")
    if fingerprint_mode == "sha256":
        fingerprint, _ = file_fingerprint(path, "sha256")
        if fingerprint != segment.file_fingerprint:
            raise RuntimeError(f"input content hash changed after planning: {path}")


def _discard_to_newline(stream: Any, *, block_size: int = 1024 * 1024) -> None:
    while True:
        block = stream.readline(block_size)
        if not block or block.endswith(b"\n"):
            return


def _jsonl_objects(
    segment: InputSegment,
    *,
    max_line_bytes: int,
) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any], str | None]]:
    path = Path(segment.source_uri)
    start = int(segment.byte_start or 0)
    end = int(segment.byte_end or 0)
    if end <= start:
        return
    with path.open("rb") as stream:
        if start > 0:
            stream.seek(start - 1)
            preceding = stream.read(1)
            if preceding not in {b"\n", b"\r"}:
                _discard_to_newline(stream)
        else:
            stream.seek(0)
        while True:
            line_start = stream.tell()
            if line_start >= end:
                break
            raw = stream.readline(max_line_bytes + 1)
            if not raw:
                break
            if len(raw) > max_line_bytes:
                if not raw.endswith(b"\n"):
                    _discard_to_newline(stream)
                locator = {"byte_start": line_start, "byte_end": stream.tell()}
                yield None, locator, f"JSONL line exceeds input.max_jsonl_line_bytes={max_line_bytes}"
                continue
            locator = {"byte_start": line_start, "byte_end": stream.tell()}
            if not raw.strip():
                continue
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                yield None, locator, f"invalid UTF-8: {exc}"
                continue
            if line_start == 0:
                decoded = decoded.removeprefix("\ufeff")
            try:
                value = json.loads(decoded)
            except json.JSONDecodeError as exc:
                yield None, locator, f"invalid JSON: {exc}"
                continue
            if not isinstance(value, dict):
                yield None, locator, "JSONL row is not an object"
                continue
            yield value, locator, None


def _parquet_objects(
    segment: InputSegment,
    *,
    fields: list[str],
    batch_rows: int,
) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any], str | None]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("pyarrow is required to read Parquet input") from exc
    parquet = pq.ParquetFile(segment.source_uri)
    available = set(parquet.schema_arrow.names)
    top_level = sorted({field.split(".", 1)[0] for field in fields if field.split(".", 1)[0] in available})
    row_index = 0
    for batch in parquet.iter_batches(
        batch_size=batch_rows,
        row_groups=[int(segment.row_group or 0)],
        columns=top_level,
    ):
        for value in batch.to_pylist():
            locator = {"row_group": segment.row_group, "row_index": row_index}
            row_index += 1
            if not isinstance(value, dict):
                yield None, locator, "Parquet row is not an object"
            else:
                yield value, locator, None


def iter_segment_objects(
    segment: InputSegment,
    input_config: Mapping[str, Any],
    *,
    batch_rows: int,
) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any], str | None]]:
    if segment.source_format == "jsonl":
        yield from _jsonl_objects(
            segment,
            max_line_bytes=int(input_config.get("max_jsonl_line_bytes", 64 * 1024 * 1024)),
        )
        return
    fields = [str(input_config["text_field"])]
    if input_config.get("id_field"):
        fields.append(str(input_config["id_field"]))
    fields.extend(str(value) for value in input_config.get("metadata_fields", []))
    yield from _parquet_objects(segment, fields=fields, batch_rows=batch_rows)


def normalize_object(
    value: Mapping[str, Any] | None,
    error: str | None,
    *,
    segment: InputSegment,
    locator: Mapping[str, Any],
    input_config: Mapping[str, Any],
    dedup_buckets: int,
    shuffle_partitions: int,
    external_id_partitions: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if error or value is None:
        return None, quarantine_row(
            "normalize", "malformed_record", error or "missing record",
            source_uri=segment.source_uri, source_locator=locator,
        )
    text_field = str(input_config["text_field"])
    text = _field(value, text_field)
    if not isinstance(text, str) or not text.strip():
        return None, quarantine_row(
            "normalize", "invalid_text", f"field {text_field!r} must be a non-empty string",
            source_uri=segment.source_uri, source_locator=locator,
            raw_excerpt=str(text) if text is not None else None,
        )
    if len(text) > int(input_config.get("max_text_chars", 20_000_000)):
        return None, quarantine_row(
            "normalize",
            "text_too_large",
            f"field {text_field!r} exceeds input.max_text_chars",
            source_uri=segment.source_uri,
            source_locator=locator,
            raw_excerpt=text[:2000],
        )
    id_field = input_config.get("id_field")
    external_raw = _field(value, str(id_field)) if id_field else None
    if id_field and external_raw is None:
        return None, quarantine_row(
            "normalize",
            "missing_external_id",
            f"configured field {id_field!r} is missing or null",
            source_uri=segment.source_uri,
            source_locator=locator,
        )
    if external_raw is not None and (isinstance(external_raw, bool) or not isinstance(external_raw, (str, int))):
        return None, quarantine_row(
            "normalize",
            "invalid_external_id",
            f"field {id_field!r} must be a string or integer when present",
            source_uri=segment.source_uri,
            source_locator=locator,
            raw_excerpt=str(external_raw),
        )
    external_id = str(external_raw).strip() if external_raw is not None else None
    if external_id == "":
        external_id = None
    if id_field and external_id is None:
        return None, quarantine_row(
            "normalize",
            "missing_external_id",
            f"configured field {id_field!r} must not be empty",
            source_uri=segment.source_uri,
            source_locator=locator,
        )
    metadata = {
        field: _field(value, str(field))
        for field in input_config.get("metadata_fields", [])
        if _field(value, str(field)) is not None
    }
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    record_id = (
        stable_id("rec", external_id, length=32)
        if external_id is not None
        else stable_id("rec", segment.source_uri, locator, content_hash, length=32)
    )
    dedup_bucket = int(content_hash[:16], 16) % dedup_buckets
    shuffle_partition = dedup_bucket % shuffle_partitions
    id_partition_count = external_id_partitions or shuffle_partitions
    external_id_partition = (
        int(hashlib.sha256(external_id.encode("utf-8")).hexdigest()[:16], 16) % id_partition_count
        if external_id is not None
        else -1
    )
    source_locator = {"segment_id": segment.segment_id, **locator}
    return (
        {
            "record_id": record_id,
            "content_hash": content_hash,
            "text": text,
            "source_uri": segment.source_uri,
            "source_format": segment.source_format,
            "source_locator_json": json.dumps(source_locator, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "external_id": external_id,
            "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str),
            "dedup_bucket": dedup_bucket,
            "shuffle_partition": shuffle_partition,
            "external_id_partition": external_id_partition,
        },
        None,
    )
