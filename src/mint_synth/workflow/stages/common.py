from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from mint_synth.core.run_state import RunLayout
from mint_synth.inference.runtime import client_for_role
from mint_synth.storage.parquetio import BufferedParquetWriter, envelope_schema, iter_payloads, quarantine_schema

StageHandler = Callable[[RunLayout, Mapping[str, Any], Mapping[str, Any], int, Path], dict[str, Any]]


def _compression(config: Mapping[str, Any]) -> str:
    return str(config["output"]["compression"])


def _writer_options(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compression": _compression(config),
        "batch_rows": int(config["reader"]["batch_rows"]),
        "row_group_rows": int(config["output"]["parquet_row_group_rows"]),
    }


def _client_stats(client: Any) -> dict[str, int]:
    stats = getattr(client, "stats", None)
    return stats() if callable(stats) else {}


def _role_client(
    layout: RunLayout,
    config: Mapping[str, Any],
    stage: str,
    shard_id: int,
    role: str,
):
    cache_path = layout.root / "cache" / "llm" / stage / f"{shard_id:06d}-{role}.sqlite3"
    return client_for_role(config, role, cache_path=cache_path)


def _envelope_writer(path: Path, config: Mapping[str, Any]) -> BufferedParquetWriter:
    return BufferedParquetWriter(path, envelope_schema(), **_writer_options(config))


def _quarantine_writer(path: Path, config: Mapping[str, Any]) -> BufferedParquetWriter:
    return BufferedParquetWriter(path, quarantine_schema(), **_writer_options(config))


def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''").replace("\\", "/") + "'"


def _sql_paths(paths: Sequence[Path]) -> str:
    return "[" + ",".join(_sql_string(path) for path in paths) + "]"


def _duckdb(layout: RunLayout, config: Mapping[str, Any], stage: str, shard_id: int):
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("duckdb is required for shuffle and dedup stages") from exc
    scratch = Path(config["run"]["work_dir"]) / layout.root.name / "duckdb" / f"{stage}-{shard_id:06d}"
    scratch.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET temp_directory={_sql_string(scratch)}")
    connection.execute(f"SET memory_limit={_sql_string(str(config['reader']['duckdb_memory_limit']))}")
    connection.execute(f"SET threads={int(config['reader']['duckdb_threads'])}")
    return connection


def _copy_options(config: Mapping[str, Any]) -> str:
    compression = _compression(config)
    codec = "UNCOMPRESSED" if compression == "none" else compression.upper()
    return f"FORMAT PARQUET, COMPRESSION {codec}, ROW_GROUP_SIZE {int(config['output']['parquet_row_group_rows'])}"


def _grouped_payloads(
    path: Path,
    key: str,
    batch_rows: int,
    *,
    max_group_rows: int | None = None,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    values = iter_payloads(path, batch_rows=batch_rows)
    for group_key, group in itertools.groupby(values, key=lambda item: str(item.get(key, ""))):
        collected: list[dict[str, Any]] = []
        for value in group:
            collected.append(value)
            if max_group_rows is not None and len(collected) > max_group_rows:
                raise RuntimeError(
                    f"group {key}={group_key!r} exceeds configured maximum of {max_group_rows} rows"
                )
        yield group_key, collected
