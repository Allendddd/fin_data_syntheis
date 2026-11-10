from __future__ import annotations

from collections import defaultdict
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.provenance import sha256_file
from mint_synth.core.run_state import RunLayout, atomic_json, read_json, utc_now
from mint_synth.storage.parquetio import iter_payloads, parquet_num_rows
from mint_synth.workflow.planner import stage_by_name


def _success_metrics(layout: RunLayout, stage: str, count: int) -> list[dict[str, Any]]:
    return [read_json(layout.success(stage, shard_id)).get("metrics", {}) for shard_id in range(count)]


def _enforce_global_quality(layout: RunLayout, config: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stage in plan["stages"]:
        name = stage["name"]
        metrics = _success_metrics(layout, name, int(stage["shard_count"])) if name != "export" else []
        if metrics:
            summary[name] = metrics
    normalize = summary.get("normalize", [])
    total = sum(int(item.get("input_records", 0)) for item in normalize)
    invalid = sum(int(item.get("invalid_records", 0)) for item in normalize)
    if total and invalid / total > float(config["quality"]["max_invalid_record_ratio"]):
        raise RuntimeError("global invalid record ratio exceeds configured limit")
    id_validation = summary.get("id_validation", [])
    id_records = sum(int(item.get("input_records", 0)) for item in id_validation)
    id_conflicts = sum(int(item.get("conflicting_records", 0)) for item in id_validation)
    if id_records and id_conflicts / id_records > float(config["quality"]["max_external_id_conflict_ratio"]):
        raise RuntimeError("global external-id conflict ratio exceeds configured limit")
    llm_denominators = {
        "facts": "chunks",
        "atomic": "candidates",
        "entity_resolution": "llm_requests",
        "graph_planning": "llm_requests",
        "query_realization": "candidates",
    }
    for name, denominator_key in llm_denominators.items():
        values = summary.get(name, [])
        numerator = sum(int(item.get("terminal_errors", 0)) for item in values)
        denominator = sum(int(item.get(denominator_key, 0)) for item in values)
        if denominator and numerator / denominator > float(config["quality"]["max_llm_terminal_error_ratio"]):
            raise RuntimeError(f"global {name} LLM terminal error ratio exceeds configured limit")
    validations = summary.get("validation", [])
    validation_total = sum(int(item.get("input_tasks", 0)) for item in validations)
    validation_failed = sum(int(item.get("rejected", 0)) for item in validations)
    if validation_total and validation_failed / validation_total > float(config["quality"]["max_validation_failure_ratio"]):
        raise RuntimeError("global validation failure ratio exceeds configured limit")
    validated_tasks = validation_total - validation_failed
    if validated_tasks < int(config["quality"]["min_validated_tasks"]):
        raise RuntimeError(
            f"validated task count {validated_tasks} is below quality.min_validated_tasks"
        )
    return {
        "input_records": total,
        "invalid_records": invalid,
        "external_id_records": id_records,
        "external_id_conflicts": id_conflicts,
        "validated_tasks": validation_total,
        "accepted_tasks": validated_tasks,
        "rejected_tasks": validation_failed,
        "failed_shards": 0,
    }


def stage_export(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_id: int,
    output: Path,
) -> dict[str, Any]:
    del shard_id
    quality = _enforce_global_quality(layout, config, plan)
    if config.get("version") == 3 and config["training"]["enabled"]:
        prepared_metrics = _success_metrics(layout, "sft_prepare", int(stage_by_name(plan, "sft_prepare")["shard_count"]))
        for family, enabled_key, minimum_key in (
            ("atomic", "atomic_enabled", "min_atomic_samples"),
            ("long", "long_enabled", "min_long_samples"),
        ):
            available = sum(int(metric.get(f"{family}_train", 0)) for metric in prepared_metrics)
            required = int(config["training"][minimum_key])
            if config["training"][enabled_key] and available < required:
                raise RuntimeError(f"SFT publication blocked: {family} has {available} accepted training samples; requires {required}")
    validation_count = int(stage_by_name(plan, "validation")["shard_count"])
    paths = [layout.shard("validation", value) / "tasks.parquet" for value in range(validation_count)]
    paths = [path for path in paths if path.is_file() and parquet_num_rows(path) > 0]
    formats = set(config["output"]["export_formats"])
    exported = 0
    if "parquet" in formats:
        dataset_root = output / "dataset"
        dataset_root.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(paths):
            destination = dataset_root / f"part-{index:06d}.parquet"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        if "jsonl" not in formats:
            exported = sum(parquet_num_rows(path) for path in paths)
    jsonl_stream = (output / "dataset.jsonl").open("w", encoding="utf-8", newline="\n") if "jsonl" in formats else None
    try:
        if jsonl_stream is not None:
            for path in paths:
                for task in iter_payloads(path, batch_rows=int(config["reader"]["batch_rows"])):
                    jsonl_stream.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
                    exported += 1
    finally:
        if jsonl_stream is not None:
            jsonl_stream.flush()
            os.fsync(jsonl_stream.fileno())
            jsonl_stream.close()
    sft_counts: dict[str, int] = defaultdict(int)
    if config.get("version") == 3 and config["training"]["enabled"]:
        root = output / "sft"
        root.mkdir()
        prepared_count = int(stage_by_name(plan, "sft_prepare")["shard_count"])
        streams = {
            (family, split): (root / f"{family}-{split}.jsonl").open("w", encoding="utf-8", newline="\n")
            for family in ("atomic", "long") for split in ("train", "val")
        }
        sidecar = (root / "audit-sidecar.jsonl").open("w", encoding="utf-8", newline="\n")
        trajectories = (root / "trajectory-index.jsonl").open("w", encoding="utf-8", newline="\n")
        try:
            for prepared_shard in range(prepared_count):
                prepared = layout.shard("sft_prepare", prepared_shard)
                for family in ("atomic", "long"):
                    with (prepared / f"{family}.jsonl").open("r", encoding="utf-8") as source:
                        for line in source:
                            sample = json.loads(line)
                            split = sample.pop("_split")
                            streams[(family, split)].write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                            sft_counts[family] += 1
                            sft_counts[f"{family}_{split}"] += 1
                shard_prefix = f"traces/{prepared_shard:06d}/"
                with (prepared / "sidecar.jsonl").open("r", encoding="utf-8") as source:
                    for line in source:
                        row = json.loads(line)
                        for candidate in row.get("teacher_audit", {}).get("candidates", []):
                            if candidate.get("trace"):
                                candidate["trace"] = shard_prefix + candidate["trace"].removeprefix("traces/")
                        sidecar.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if (prepared / "traces").is_dir():
                    shutil.copytree(prepared / "traces", root / "traces" / f"{prepared_shard:06d}")
                if (prepared / "trajectories.jsonl").is_file():
                    with (prepared / "trajectories.jsonl").open("r", encoding="utf-8") as source:
                        for line in source:
                            row = json.loads(line)
                            row["trace"] = shard_prefix + row["trace"].removeprefix("traces/")
                            trajectories.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        finally:
            for stream in [*streams.values(), sidecar, trajectories]:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
        for family, enabled_key, minimum_key in (
            ("atomic", "atomic_enabled", "min_atomic_samples"),
            ("long", "long_enabled", "min_long_samples"),
        ):
            required = max(1, int(config["training"][minimum_key]))
            available = sft_counts[f"{family}_train"]
            if config["training"][enabled_key] and available < required:
                raise RuntimeError(f"SFT publication blocked: {family} has {available} accepted training samples; requires {required}")
        atomic_json(root / "manifest.json", {"format": "ms-swift messages", "counts": dict(sft_counts), "tool_observations_loss": False, "audit_sidecar": "audit-sidecar.jsonl", "trajectory_index": "trajectory-index.jsonl"})
    query_counts: dict[str, int] = {}
    if config.get("version") == 3 and config["query_export"]["enabled"]:
        query_source = layout.shard("query_export", 0) / "queries"
        query_counts = dict(read_json(query_source / "manifest.json")["counts"])
        shutil.copytree(query_source, output / "queries")
    atomic_json(output / "quality_summary.json", quality)
    artifact_inventory: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        if path.name not in {"dataset.jsonl", "quality_summary.json"} and relative.parts[0] not in {"dataset", "sft", "queries"}:
            continue
        artifact: dict[str, Any] = {"path": relative.as_posix(), "bytes": path.stat().st_size}
        if config["output"].get("artifact_checksums", True):
            artifact["sha256"] = sha256_file(path)
        artifact_inventory.append(artifact)
    atomic_json(
        output / "artifact_manifest.json",
        {"created_at": utc_now(), "artifacts": artifact_inventory},
    )
    return {
        **quality, "exported_tasks": exported, "formats": sorted(formats),
        "sft": dict(sft_counts), "queries": query_counts,
    }
