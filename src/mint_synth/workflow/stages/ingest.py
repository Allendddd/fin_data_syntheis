from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import RunLayout
from mint_synth.storage.catalog import read_assignment
from mint_synth.storage.parquetio import BufferedParquetWriter, external_id_schema, normalized_schema, parquet_num_rows, quarantine_schema, write_empty
from mint_synth.storage.records import iter_segment_objects, normalize_object, verify_segment_file
from mint_synth.workflow.planner import stage_by_name

from .common import _compression, _copy_options, _duckdb, _quarantine_writer, _sql_paths, _sql_string, _writer_options


def stage_normalize(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_id: int,
    output: Path,
) -> dict[str, Any]:
    del plan
    assignment = read_assignment(layout.assignment("normalize", shard_id))
    input_count = accepted = invalid = 0
    raw_path = output / "records-unsorted.parquet"
    verified_files: set[str] = set()
    with BufferedParquetWriter(
        raw_path, normalized_schema(), **_writer_options(config)
    ) as data_writer, _quarantine_writer(output / "quarantine.parquet", config) as rejected:
        for segment in assignment:
            if segment.source_uri not in verified_files:
                verify_segment_file(
                    segment,
                    fingerprint_mode=str(config["run"]["fingerprint_mode"]),
                )
                verified_files.add(segment.source_uri)
            for value, locator, error in iter_segment_objects(
                segment,
                config["input"],
                batch_rows=int(config["reader"]["batch_rows"]),
            ):
                input_count += 1
                row, rejection = normalize_object(
                    value,
                    error,
                    segment=segment,
                    locator=locator,
                    input_config=config["input"],
                    dedup_buckets=int(config["reader"]["dedup_buckets"]),
                    shuffle_partitions=int(config["reader"]["shuffle_partitions"]),
                    external_id_partitions=int(config["reader"]["external_id_partitions"]),
                )
                if row is not None:
                    data_writer.write(row)
                    accepted += 1
                elif rejection is not None:
                    rejected.write(rejection)
                    invalid += 1
                    if config["input"]["invalid_record_policy"] == "fail":
                        raise ValueError(rejection["error_message"])
    connection = _duckdb(layout, config, "normalize", shard_id)
    try:
        connection.execute(
            f"COPY (SELECT * FROM read_parquet({_sql_string(raw_path)}) "
            "ORDER BY shuffle_partition, dedup_bucket, content_hash, source_uri, source_locator_json) "
            f"TO {_sql_string(output / 'data.parquet')} ({_copy_options(config)})"
        )
        connection.execute(
            f"COPY (SELECT external_id, content_hash, external_id_partition, shuffle_partition, "
            f"record_id, source_uri, source_locator_json, substr(text, 1, 2000) AS raw_excerpt "
            f"FROM read_parquet({_sql_string(raw_path)}) "
            "WHERE external_id IS NOT NULL "
            "ORDER BY external_id_partition, external_id, content_hash, source_uri, source_locator_json) "
            f"TO {_sql_string(output / 'external-ids.parquet')} ({_copy_options(config)})"
        )
    finally:
        connection.close()
    raw_path.unlink(missing_ok=True)
    ratio = invalid / input_count if input_count else 0.0
    return {
        "input_records": input_count,
        "accepted_records": accepted,
        "invalid_records": invalid,
        "invalid_ratio": ratio,
        "input_segments": len(assignment),
    }


def stage_id_validation(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_id: int,
    output: Path,
) -> dict[str, Any]:
    """Globally reject external IDs that map to more than one content hash."""
    normalize_count = int(stage_by_name(plan, "normalize")["shard_count"])
    paths = [layout.shard("normalize", value) / "external-ids.parquet" for value in range(normalize_count)]
    paths = [path for path in paths if path.is_file()]
    conflicts_path = output / "conflicts.parquet"
    quarantine_path = output / "quarantine.parquet"
    if not paths:
        write_empty(conflicts_path, external_id_schema(), compression=_compression(config))
        write_empty(quarantine_path, quarantine_schema(), compression=_compression(config))
        return {"input_records": 0, "conflicting_ids": 0, "conflicting_records": 0, "conflict_ratio": 0.0}

    connection = _duckdb(layout, config, "id_validation", shard_id)
    try:
        connection.execute(
            f"CREATE VIEW source_rows AS SELECT * FROM read_parquet({_sql_paths(paths)}) "
            f"WHERE external_id_partition = {shard_id}"
        )
        input_count = int(connection.execute("SELECT count(*) FROM source_rows").fetchone()[0])
        connection.execute(
            """
            CREATE TEMP VIEW colliding_ids AS
            SELECT external_id
            FROM source_rows
            GROUP BY external_id
            HAVING count(DISTINCT content_hash) > 1
            """
        )
        conflicting_ids = int(connection.execute("SELECT count(*) FROM colliding_ids").fetchone()[0])
        conflicting_records = int(
            connection.execute(
                "SELECT count(*) FROM source_rows WHERE external_id IN (SELECT external_id FROM colliding_ids)"
            ).fetchone()[0]
        )
        connection.execute(
            f"""
            COPY (
              SELECT *
              FROM source_rows
              WHERE external_id IN (SELECT external_id FROM colliding_ids)
              ORDER BY shuffle_partition, external_id, content_hash, source_uri, source_locator_json
            ) TO {_sql_string(conflicts_path)} ({_copy_options(config)})
            """
        )
        connection.execute(
            f"""
            COPY (
              SELECT
                'id_validation' AS stage,
                'external_id_content_conflict' AS error_code,
                'external id maps to multiple content hashes: ' || external_id AS error_message,
                source_uri,
                source_locator_json,
                record_id,
                raw_excerpt,
                to_json(struct_pack(external_id := external_id, content_hash := content_hash)) AS payload_json
              FROM source_rows
              WHERE external_id IN (SELECT external_id FROM colliding_ids)
              ORDER BY external_id, source_uri, source_locator_json
            ) TO {_sql_string(quarantine_path)} ({_copy_options(config)})
            """
        )
    finally:
        connection.close()
    if conflicting_records and config["input"]["invalid_record_policy"] == "fail":
        raise ValueError(
            f"external-id validation found {conflicting_records} records across {conflicting_ids} conflicting ids"
        )
    ratio = conflicting_records / input_count if input_count else 0.0
    return {
        "input_records": input_count,
        "conflicting_ids": conflicting_ids,
        "conflicting_records": conflicting_records,
        "conflict_ratio": ratio,
    }


def stage_dedup(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_id: int,
    output: Path,
) -> dict[str, Any]:
    normalize_count = int(stage_by_name(plan, "normalize")["shard_count"])
    paths = [layout.shard("normalize", value) / "data.parquet" for value in range(normalize_count)]
    paths = [path for path in paths if path.is_file()]
    validation_count = int(stage_by_name(plan, "id_validation")["shard_count"])
    conflict_paths = [layout.shard("id_validation", value) / "conflicts.parquet" for value in range(validation_count)]
    conflict_paths = [path for path in conflict_paths if path.is_file()]
    data_path = output / "data.parquet"
    quarantine_path = output / "quarantine.parquet"
    if not paths:
        write_empty(data_path, normalized_schema(), compression=_compression(config))
        write_empty(quarantine_path, quarantine_schema(), compression=_compression(config))
        return {"input_records": 0, "output_records": 0, "exact_duplicates": 0, "id_collisions": 0}

    connection = _duckdb(layout, config, "dedup", shard_id)
    try:
        connection.execute(
            f"CREATE VIEW source_rows AS SELECT * FROM read_parquet({_sql_paths(paths)}) "
            f"WHERE shuffle_partition = {shard_id}"
        )
        input_count = int(connection.execute("SELECT count(*) FROM source_rows").fetchone()[0])
        if conflict_paths:
            connection.execute(
                f"CREATE TEMP VIEW colliding_ids AS SELECT DISTINCT external_id "
                f"FROM read_parquet({_sql_paths(conflict_paths)}) WHERE shuffle_partition = {shard_id}"
            )
        else:
            connection.execute("CREATE TEMP TABLE colliding_ids(external_id VARCHAR)")
        collision_count = int(
            connection.execute(
                "SELECT count(*) FROM source_rows WHERE external_id IN (SELECT external_id FROM colliding_ids)"
            ).fetchone()[0]
        )
        columns = (
            "record_id, content_hash, text, source_uri, source_format, source_locator_json, "
            "external_id, metadata_json, dedup_bucket, shuffle_partition, external_id_partition"
        )
        destination = _sql_string(data_path)
        connection.execute(
            f"""
            COPY (
              SELECT {columns}
              FROM (
                SELECT *, row_number() OVER (
                  PARTITION BY content_hash ORDER BY source_uri, source_locator_json, record_id
                ) AS dedup_rank
                FROM source_rows
                WHERE external_id IS NULL OR external_id NOT IN (SELECT external_id FROM colliding_ids)
              )
              WHERE dedup_rank = 1
              ORDER BY content_hash, source_uri, source_locator_json
            ) TO {destination} ({_copy_options(config)})
            """
        )
        output_count = parquet_num_rows(data_path)
    finally:
        connection.close()
    write_empty(quarantine_path, quarantine_schema(), compression=_compression(config))
    return {
        "input_records": input_count,
        "output_records": output_count,
        "exact_duplicates": input_count - collision_count - output_count,
        "id_collisions": collision_count,
    }
