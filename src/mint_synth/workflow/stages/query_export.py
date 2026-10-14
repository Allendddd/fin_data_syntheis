"""Publish validated long-horizon objectives separately from hidden verifier data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint_synth.core.provenance import sha256_file
from mint_synth.core.run_state import RunLayout, atomic_json
from mint_synth.harness.registry import load_tool_registry
from mint_synth.sft.selection import matches_filters, split_for_task
from mint_synth.storage.parquetio import iter_payloads, parquet_num_rows
from mint_synth.synthesis.entity import entity_bucket
from mint_synth.workflow.planner import stage_by_name


def export_queries(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    task_paths: Sequence[Path],
    output: Path,
) -> dict[str, int]:
    """Write query-only inputs and a separate, evaluator-only task contract."""
    root = output / "queries"
    root.mkdir()
    streams = {
        split: (root / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
        for split in ("train", "val")
    }
    hidden = (root / "hidden-verifier.jsonl").open("w", encoding="utf-8", newline="\n")
    counts = {"train": 0, "val": 0}
    snapshots: set[Path] = set()
    try:
        for path in task_paths:
            for task in iter_payloads(path, batch_rows=int(config["reader"]["batch_rows"])):
                if task.get("task_type") != "long_horizon" or not matches_filters(task, config):
                    continue
                query = str(task.get("question") or "").strip()
                task_id = str(task.get("task_id") or "")
                entity_id = str(task.get("canonical_entity_id") or "")
                if not query or not task_id or not entity_id:
                    raise ValueError("validated long-horizon task lacks a query, task ID, or entity ID")
                split = split_for_task(task_id, float(config["training"]["split_validation_fraction"]))
                bucket = entity_bucket(entity_id, int(config["dag"]["entity_buckets"]))
                snapshot = layout.shard("retrieval_index", bucket) / "snapshot.sqlite3"
                if not snapshot.is_file():
                    raise RuntimeError(f"query snapshot index is missing: {snapshot}")
                snapshots.add(snapshot)
                line_index = counts[split]
                streams[split].write(json.dumps({"query": query}, ensure_ascii=False, separators=(",", ":")) + "\n")
                hidden.write(json.dumps({
                    "query_id": task_id,
                    "split": split,
                    "line_index": line_index,
                    "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                    "snapshot_index": snapshot.relative_to(layout.root).as_posix(),
                    "task": task,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[split] += 1
    finally:
        for stream in [*streams.values(), hidden]:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    required = int(config["query_export"]["min_train_queries"])
    if counts["train"] < required:
        raise RuntimeError(
            f"query publication blocked: {counts['train']} training queries; requires {required}"
        )
    atomic_json(root / "manifest.json", {
        "format": "query-only JSONL",
        "source_run_fingerprint": plan["run_fingerprint"],
        "cutoff": config["run"]["cutoff"],
        "available_at_field": config["input"]["available_at_field"],
        "toolset_fingerprint": load_tool_registry(config.get("harness", {}).get("tool_registry")).fingerprint,
        "counts": counts,
        "hidden_verifier": "hidden-verifier.jsonl",
        "model_visible_fields": ["query"],
        "artifacts": {
            name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
            for name in ("train.jsonl", "val.jsonl", "hidden-verifier.jsonl")
        },
        "snapshot_artifacts": {
            snapshot.relative_to(layout.root).as_posix(): {
                "bytes": snapshot.stat().st_size, "sha256": sha256_file(snapshot),
            }
            for snapshot in sorted(snapshots)
        },
    })
    return counts


def stage_query_export(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_id: int,
    output: Path,
) -> dict[str, int]:
    del shard_id
    count = int(stage_by_name(plan, "validation")["shard_count"])
    paths = [layout.shard("validation", index) / "tasks.parquet" for index in range(count)]
    paths = [path for path in paths if path.is_file() and parquet_num_rows(path) > 0]
    return export_queries(layout, config, plan, paths, output)
