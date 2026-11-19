"""Resample frozen validated tasks without modifying or re-running synthesis."""

from __future__ import annotations

from pathlib import Path
import uuid

from mint_synth.core.config_v2 import load_config, dump_config
from mint_synth.core.provenance import json_fingerprint, sha256_file
from mint_synth.core.run_state import RunLayout, atomic_json, read_json, shard_complete
from mint_synth.workflow.planner import stage_by_name, verify_plan_fingerprint


def collect_frozen_tasks(run_dir, config_path, output_dir):
    from mint_synth.workflow.stages.sft_data import stage_sft_prepare
    source = RunLayout(run_dir)
    plan = read_json(source.plan)
    if not verify_plan_fingerprint(plan):
        raise ValueError("source run plan failed its immutable fingerprint check")
    config = load_config(config_path)
    original = load_config(source.root / "resolved_config.yaml")
    if config.get("version") != 3:
        raise ValueError("teacher collection requires version 3 configuration")
    for path in (("run", "cutoff"), ("training", "split_validation_fraction"), ("quality", "answer_tolerance")):
        if config[path[0]][path[1]] != original[path[0]][path[1]]:
            raise ValueError("resampling cannot change frozen task/audit contract: " + ".".join(path))
    count = int(stage_by_name(plan, "validation")["shard_count"])
    artifacts = []
    for shard in range(count):
        signature = plan.get("stage_signatures", {}).get("validation", [None] * count)[shard]
        if not shard_complete(source, "validation", shard, signature):
            raise ValueError(f"source validated-task shard is missing or corrupt: {shard}")
        path = source.shard("validation", shard) / "tasks.parquet"
        artifacts.append({"shard": shard, "path": str(path), "sha256": sha256_file(path)})
    destination = Path(output_dir).resolve()
    if destination == source.root or destination in source.root.parents or destination.exists() or destination.is_relative_to(source.root / "stages"):
        raise ValueError("collection output must be a new directory, not the source run or its ancestor")
    resource_identity = None
    if config.get("harness", {}).get("retrieval"):
        from mint_synth.harness.resources import resource_fingerprint, validate_retrieval
        validate_retrieval(config["harness"]["retrieval"])
        resource_identity = resource_fingerprint(config["harness"]["retrieval"])
    destination.mkdir(parents=True, mode=0o700)
    class CollectionLayout(RunLayout):
        def shard(self, stage, shard_id):
            if stage in {"validation", "retrieval_index"}:
                return source.shard(stage, shard_id)
            return super().shard(stage, shard_id)
    layout = CollectionLayout(destination)
    identity = {"version": 1, "collection_id": uuid.uuid4().hex, "source_run": str(source.root),
                "source_plan_fingerprint": plan["plan_fingerprint"], "tasks": artifacts,
                "task_fingerprint": plan.get("task_fingerprint"),
                "sampling_config": {"llm": config["llm"], "training": config["training"], "harness": config.get("harness", {})}}
    from mint_synth.agents.base import search_backend_contract
    identity["search_backend"] = search_backend_contract(config)
    if resource_identity:
        identity["resource_fingerprint"] = resource_identity
    identity["sampling_fingerprint"] = json_fingerprint(identity)
    atomic_json(destination / "collection.json", identity)
    dump_config(config, destination / "resolved_config.yaml")
    metrics = []
    try:
        for shard in range(count):
            output = layout.shard("sft_prepare", shard)
            output.mkdir(parents=True, mode=0o700)
            metrics.append(stage_sft_prepare(layout, config, identity, shard, output))
        if any(sha256_file(row["path"]) != row["sha256"] for row in artifacts):
            raise ValueError("frozen tasks changed while teacher collection was running")
        for family in ("atomic", "long"):
            if config["training"][family + "_enabled"]:
                accepted = sum(int(m.get(family + "_train", 0)) for m in metrics)
                if accepted < int(config["training"]["min_" + family + "_samples"]):
                    raise ValueError(f"collection publication gate failed: insufficient {family} training samples")
        atomic_json(destination / "result.json", {"completed": True, "metrics": metrics})
    except Exception as exc:
        atomic_json(destination / "result.json", {"completed": False, "error": str(exc), "metrics": metrics})
        raise
    finally:
        for client in getattr(layout, "_role_clients", []):
            close = getattr(client, "close", None)
            if callable(close):
                close()
    return {"output_dir": str(destination), "sampling_fingerprint": identity["sampling_fingerprint"], "metrics": metrics}
