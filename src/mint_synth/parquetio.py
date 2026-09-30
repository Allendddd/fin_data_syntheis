from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .util import canonical_decimal


class ParquetDependencyError(RuntimeError):
    pass


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on deployment environment
        raise ParquetDependencyError("pyarrow is required for v2 data stages") from exc
    return pa, pq


def normalized_schema():
    pa, _ = _arrow()
    return pa.schema(
        [
            ("record_id", pa.string()),
            ("content_hash", pa.string()),
            ("text", pa.string()),
            ("source_uri", pa.string()),
            ("source_format", pa.string()),
            ("source_locator_json", pa.string()),
            ("external_id", pa.string()),
            ("metadata_json", pa.string()),
            ("dedup_bucket", pa.int32()),
            ("shuffle_partition", pa.int32()),
            ("external_id_partition", pa.int32()),
        ]
    )


def external_id_schema():
    pa, _ = _arrow()
    return pa.schema(
        [
            ("external_id", pa.string()),
            ("content_hash", pa.string()),
            ("external_id_partition", pa.int32()),
            ("shuffle_partition", pa.int32()),
            ("record_id", pa.string()),
            ("source_uri", pa.string()),
            ("source_locator_json", pa.string()),
            ("raw_excerpt", pa.string()),
        ]
    )


def envelope_schema():
    pa, _ = _arrow()
    return pa.schema(
        [
            ("object_id", pa.string()),
            ("record_id", pa.string()),
            ("chunk_id", pa.string()),
            ("source_id", pa.string()),
            ("entity_surface", pa.string()),
            ("entity_block", pa.string()),
            ("canonical_entity_id", pa.string()),
            ("bucket_id", pa.int32()),
            ("task_type", pa.string()),
            ("source_uri", pa.string()),
            ("period", pa.string()),
            ("metric", pa.string()),
            ("value_num", pa.string()),
            ("value_text", pa.string()),
            ("value_bool", pa.bool_()),
            ("unit", pa.string()),
            ("scale", pa.string()),
            ("currency", pa.string()),
            ("question", pa.string()),
            ("formula_id", pa.string()),
            ("formula_version", pa.int32()),
            ("graph_id", pa.string()),
            ("quality_accepted", pa.bool_()),
            ("payload_json", pa.string()),
        ]
    )


def quarantine_schema():
    pa, _ = _arrow()
    return pa.schema(
        [
            ("stage", pa.string()),
            ("error_code", pa.string()),
            ("error_message", pa.string()),
            ("source_uri", pa.string()),
            ("source_locator_json", pa.string()),
            ("record_id", pa.string()),
            ("raw_excerpt", pa.string()),
            ("payload_json", pa.string()),
        ]
    )


class BufferedParquetWriter:
    def __init__(
        self,
        path: str | Path,
        schema: Any,
        *,
        compression: str = "zstd",
        batch_rows: int = 8192,
        row_group_rows: int = 50000,
    ):
        _, pq = _arrow()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.batch_rows = batch_rows
        self.row_group_rows = row_group_rows
        self._buffer: list[dict[str, Any]] = []
        self._writer = pq.ParquetWriter(
            self.path,
            schema,
            compression=None if compression == "none" else compression,
        )
        self.rows_written = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self._buffer.append(dict(row))
        if len(self._buffer) >= self.batch_rows:
            self.flush()

    def extend(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.write(row)

    def flush(self) -> None:
        if not self._buffer:
            return
        pa, _ = _arrow()
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table, row_group_size=self.row_group_rows)
        self.rows_written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        if self._writer is None:
            return
        self.flush()
        self._writer.close()
        self._writer = None

    def __enter__(self) -> "BufferedParquetWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class PartitionedWriter:
    """Lazily opens one deterministic Parquet file per physical shuffle partition."""

    def __init__(
        self,
        root: str | Path,
        schema: Any,
        *,
        prefix: str = "part",
        compression: str = "zstd",
        batch_rows: int = 8192,
        row_group_rows: int = 50000,
    ):
        self.root = Path(root)
        self.schema = schema
        self.prefix = prefix
        self.compression = compression
        self.batch_rows = batch_rows
        self.row_group_rows = row_group_rows
        self.writers: dict[int, BufferedParquetWriter] = {}

    def write(self, partition: int, row: Mapping[str, Any]) -> None:
        writer = self.writers.get(partition)
        if writer is None:
            writer = BufferedParquetWriter(
                self.root / f"{self.prefix}-{partition:06d}.parquet",
                self.schema,
                compression=self.compression,
                batch_rows=self.batch_rows,
                row_group_rows=self.row_group_rows,
            )
            self.writers[partition] = writer
        writer.write(row)

    @property
    def rows_written(self) -> int:
        return sum(writer.rows_written + len(writer._buffer) for writer in self.writers.values())

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()

    def __enter__(self) -> "PartitionedWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def envelope_row(
    payload: Mapping[str, Any],
    *,
    object_id: str | None = None,
    record_id: str | None = None,
    chunk_id: str | None = None,
    source_id: str | None = None,
    entity_surface: str | None = None,
    entity_block: str | None = None,
    canonical_entity_id: str | None = None,
    bucket_id: int | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    answer = payload.get("answer") if isinstance(payload.get("answer"), Mapping) else {}
    verification = payload.get("verification") if isinstance(payload.get("verification"), Mapping) else {}
    raw_value = payload.get("value_num")
    if raw_value is None and isinstance(answer, Mapping):
        raw_value = answer.get("value")
    numeric_value = canonical_decimal(raw_value)
    display_value = payload.get("value_text")
    if display_value is None and isinstance(answer, Mapping):
        display_value = answer.get("text")
    formula_version_value = payload.get("formula_version")
    return {
        "object_id": object_id,
        "record_id": record_id,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "entity_surface": entity_surface,
        "entity_block": entity_block,
        "canonical_entity_id": canonical_entity_id,
        "bucket_id": bucket_id,
        "task_type": task_type,
        "source_uri": str(payload["source_uri"]) if payload.get("source_uri") is not None else None,
        "period": str(payload["period"]) if payload.get("period") is not None else None,
        "metric": str(payload["metric"]) if payload.get("metric") is not None else None,
        "value_num": numeric_value,
        "value_text": str(display_value) if display_value is not None else None,
        "value_bool": raw_value if isinstance(raw_value, bool) else None,
        "unit": str(payload.get("unit") or answer.get("unit")) if (payload.get("unit") or answer.get("unit")) is not None else None,
        "scale": str(payload.get("scale") or answer.get("scale")) if (payload.get("scale") or answer.get("scale")) is not None else None,
        "currency": str(payload.get("currency") or answer.get("currency")) if (payload.get("currency") or answer.get("currency")) is not None else None,
        "question": str(payload["question"]) if payload.get("question") is not None else None,
        "formula_id": str(payload["formula_id"]) if payload.get("formula_id") is not None else None,
        "formula_version": int(formula_version_value) if formula_version_value is not None else None,
        "graph_id": str(payload["graph_id"]) if payload.get("graph_id") is not None else None,
        "quality_accepted": bool(verification["accepted"]) if "accepted" in verification else None,
        "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def quarantine_row(
    stage: str,
    error_code: str,
    error_message: str,
    *,
    source_uri: str | None = None,
    source_locator: Mapping[str, Any] | None = None,
    record_id: str | None = None,
    raw_excerpt: str | None = None,
    payload: Any = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_code": error_code,
        "error_message": error_message[:4000],
        "source_uri": source_uri,
        "source_locator_json": json.dumps(source_locator, ensure_ascii=False, sort_keys=True) if source_locator else None,
        "record_id": record_id,
        "raw_excerpt": raw_excerpt[:2000] if raw_excerpt else None,
        "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) if payload is not None else None,
    }


def iter_rows(path: str | Path, *, batch_rows: int = 8192, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    _, pq = _arrow()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        yield from batch.to_pylist()


def iter_payloads(path: str | Path, *, batch_rows: int = 8192) -> Iterator[dict[str, Any]]:
    for row in iter_rows(path, batch_rows=batch_rows, columns=["payload_json"]):
        payload = json.loads(row["payload_json"])
        if isinstance(payload, dict):
            yield payload


def write_empty(path: str | Path, schema: Any, *, compression: str = "zstd") -> None:
    with BufferedParquetWriter(path, schema, compression=compression):
        pass


def parquet_num_rows(path: str | Path) -> int:
    _, pq = _arrow()
    return int(pq.ParquetFile(path).metadata.num_rows)
