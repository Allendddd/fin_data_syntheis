"""Prepare supervised fine tuning data from audited tasks and teacher rollouts."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import RunLayout
from mint_synth.harness.registry import load_tool_registry
from mint_synth.sft.formatting import atomic_messages
from mint_synth.sft.rollout import long_rollout
from mint_synth.sft.selection import financial_topic, selected, split_for_task
from mint_synth.storage.parquetio import iter_payloads
from mint_synth.synthesis.entity import entity_bucket

from .common import _client_stats, _role_client


def stage_sft_prepare(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_id: int,
    output: Path,
) -> dict[str, Any]:
    del plan
    task_path = layout.shard("validation", shard_id) / "tasks.parquet"
    teacher = (
        _role_client(layout, config, "sft_prepare", shard_id, str(config["training"]["teacher_role"]))
        if config["training"]["long_enabled"] else None
    )
    registry = load_tool_registry(config.get("harness", {}).get("tool_registry")) if teacher is not None else None
    counts: dict[str, int] = defaultdict(int)
    streams = {
        name: (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")
        for name in ("atomic", "long", "sidecar", "rejected")
    }
    try:
        for task in iter_payloads(task_path, batch_rows=int(config["reader"]["batch_rows"])):
            counts["input_tasks"] += 1
            if not selected(task, config):
                counts["filtered"] += 1
                continue
            task_id = str(task["task_id"])
            family = "atomic" if task["task_type"] == "atomic" else "long"
            try:
                if family == "atomic":
                    sample = atomic_messages(task)
                    reason = "accepted"
                else:
                    sample = None
                    reason = "no rollout accepted"
                    assert teacher is not None
                    bucket = entity_bucket(str(task.get("canonical_entity_id") or ""), int(config["dag"]["entity_buckets"]))
                    index_path = layout.shard("retrieval_index", bucket) / "snapshot.sqlite3"
                    for _ in range(int(config["training"]["rollouts_per_task"])):
                        sample, reason = long_rollout(task, teacher, index_path, config, registry)
                        counts["rollouts"] += 1
                        if sample is not None:
                            break
                if sample is None:
                    counts["rejected"] += 1
                    streams["rejected"].write(json.dumps({"task_id": task_id, "reason": reason}, ensure_ascii=False) + "\n")
                    continue
                sample["messages"][0]["loss"] = False
                sample["messages"][1]["loss"] = False
                split = split_for_task(task_id, float(config["training"]["split_validation_fraction"]))
                sample["_split"] = split
                streams[family].write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                sidecar = {
                    "task_id": task_id, "task_type": task["task_type"], "split": split,
                    "capability": task.get("capability"), "financial_topic": financial_topic(task),
                    "objective_type": task.get("task_spec", {}).get("objective_type"),
                    "formula_id": task.get("formula_id"), "seed_id": task.get("seed", {}).get("seed_id"),
                    "evidence_ids": task.get("evidence_ids"), "source_pack": task.get("source_pack"),
                    "answer": task.get("answer"), "cutoff": task.get("cutoff"),
                    "evidence_graph": task.get("evidence_graph"),
                    "verification": task.get("verification"), "semantic_audit": task.get("semantic_audit"),
                }
                streams["sidecar"].write(json.dumps(sidecar, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[f"{family}_accepted"] += 1
                counts[f"{family}_{split}"] += 1
            except Exception as exc:
                counts["rejected"] += 1
                streams["rejected"].write(json.dumps({"task_id": task_id, "reason": str(exc)}, ensure_ascii=False) + "\n")
    finally:
        for stream in streams.values():
            stream.close()
    return {**counts, "llm": {"teacher": _client_stats(teacher)}}
