"""Prepare supervised fine tuning data from audited tasks and teacher rollouts."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import RunLayout
from mint_synth.agents.base import settings_for_role
from mint_synth.agents.native_capture import NATIVE_FORMATS
from mint_synth.harness.registry import load_tool_registry
from mint_synth.inference.runtime import bounded_map
from mint_synth.sft.atomic_teacher import sample_atomic
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
    training = config["training"]
    atomic_roles = training.get("atomic_teacher_roles", ["constructor"]) if training["atomic_enabled"] else []
    long_roles = (training.get("long_teacher_roles") or [training["teacher_role"]]) if training["long_enabled"] else []
    agent_settings = {role: settings_for_role(config, role) for role in long_roles}
    json_roles = [*atomic_roles, *(role for role in long_roles if agent_settings[role][1].name == "native")]
    clients = {role: _role_client(layout, config, "sft_prepare", shard_id, role) for role in dict.fromkeys(json_roles)}
    needs_finance = any(not settings.native_tools for _model, settings in agent_settings.values())
    registry = load_tool_registry(config.get("harness", {}).get("tool_registry")) if needs_finance else None
    if long_roles:
        traces = output / "traces"
        traces.mkdir(mode=0o700, exist_ok=True)
        if os.name == "posix":
            traces.chmod(0o700)
    counts: dict[str, int] = defaultdict(int)
    streams = {
        name: (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")
        for name in ("atomic", "long", *NATIVE_FORMATS.values(), "sidecar", "rejected", "trajectories")
    }

    def selected_tasks():
        for task in iter_payloads(task_path, batch_rows=int(config["reader"]["batch_rows"])):
            counts["input_tasks"] += 1
            if selected(task, config):
                yield task
            else:
                counts["filtered"] += 1

    def prepare(task):
        task_id = str(task["task_id"])
        family = "atomic" if task["task_type"] == "atomic" else "long"
        teacher_audit = {"candidates": [], "selected": None}
        try:
            if family == "atomic":
                sample, teacher_audit, reason = sample_atomic(
                    task, {role: clients[role] for role in atomic_roles},
                    attempts=int(training.get("atomic_samples_per_task", 2)),
                    policy=str(training.get("atomic_selection_policy", "concise")),
                    fallback=bool(training.get("atomic_audit_fallback", False)),
                    tolerance=str(config["quality"]["answer_tolerance"]),
                )
            else:
                sample = None
                reason = "no rollout accepted"
                bucket = entity_bucket(str(task.get("canonical_entity_id") or ""), int(config["dag"]["entity_buckets"]))
                index_path = layout.shard("retrieval_index", bucket) / "snapshot.sqlite3"
                for attempt in range(int(training["rollouts_per_task"])):
                    role = long_roles[attempt % len(long_roles)]
                    model, runtime = agent_settings[role]
                    sample_key = f"long:{task_id}:{role}:{attempt + 1}"
                    trace_ref = "traces/" + hashlib.sha256(sample_key.encode()).hexdigest()[:32]
                    candidate = {"role": role, "attempt": attempt + 1, "runtime": runtime.name,
                                 "model": model.model, "trace": trace_ref, "accepted": False, "reason": "rollout failed"}
                    if runtime.name == "claude_code":
                        candidate["claude_code_mode"] = runtime.claude_code.mode
                    candidate["tool_mode"] = runtime.tool_mode
                    teacher_audit["candidates"].append(candidate)
                    sample, reason = long_rollout(
                        task, clients.get(role), index_path, config, registry, sample_key=sample_key,
                        model=model, runtime=runtime, trace_dir=output / trace_ref,
                    )
                    candidate.update(accepted=sample is not None, reason=reason)
                    if sample is not None:
                        candidate["trajectory_id"] = sample["trajectory_id"]
                        candidate["sample_format"] = sample.get("sample_format", "ms-swift messages")
                    if sample is not None:
                        teacher_audit["selected"] = {"source": "teacher", "role": role,
                                                     "runtime": runtime.name,
                                                     "attempt": attempt + 1}
                        break
            return task, family, sample, teacher_audit, reason
        except Exception as exc:
            return task, family, None, teacher_audit, str(exc)

    try:
        workers = int(config.get("llm", {}).get("concurrency_per_worker", 1))
        for task, family, sample, teacher_audit, reason in bounded_map(
            prepare, selected_tasks(), workers=workers,
        ):
            task_id = str(task["task_id"])
            if family == "atomic":
                counts["atomic_candidates"] += len(teacher_audit["candidates"])
            else:
                counts["rollouts"] += len(teacher_audit["candidates"])
                for candidate in teacher_audit["candidates"]:
                    streams["trajectories"].write(json.dumps({"task_id": task_id, **candidate}, ensure_ascii=False) + "\n")
            try:
                if sample is None:
                    counts["rejected"] += 1
                    streams["rejected"].write(json.dumps({"task_id": task_id, "reason": reason,
                                                          "teacher_audit": teacher_audit}, ensure_ascii=False) + "\n")
                    continue
                split = split_for_task(task_id, float(config["training"]["split_validation_fraction"]))
                native_family = NATIVE_FORMATS.get(sample.get("sample_format"))
                if native_family:
                    sample_path = Path(sample["sample_file"]).resolve()
                    if not sample_path.is_relative_to((output / "traces").resolve()):
                        raise ValueError("native sample file must be inside this shard's trace directory")
                    target = streams[native_family]
                    offset, requests = target.tell(), 0
                    try:
                        with sample_path.open("r", encoding="utf-8") as source:
                            for line in source:
                                row = json.loads(line)
                                if row.get("trajectory_id") != sample["trajectory_id"] or row.get("sample_format") != sample["sample_format"]:
                                    raise ValueError("native sample binding mismatch")
                                row["_split"] = split
                                target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                                requests += 1
                        if requests != sample["sample_count"]:
                            raise ValueError("native sample count mismatch")
                    except Exception:
                        target.seek(offset)
                        target.truncate()
                        raise
                    counts[native_family.replace("-", "_") + "_requests"] += requests
                else:
                    sample["messages"][0]["loss"] = False
                    sample["messages"][1]["loss"] = False
                    sample["_split"] = split
                    streams[family].write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                sidecar = {
                    "task_id": task_id, "task_type": task["task_type"], "split": split,
                    "trajectory_id": sample.get("trajectory_id"),
                    "sample_format": sample.get("sample_format", "ms-swift messages"),
                    "capability": task.get("capability"), "financial_topic": financial_topic(task),
                    "objective_type": task.get("task_spec", {}).get("objective_type"),
                    "formula_id": task.get("formula_id"), "seed_id": task.get("seed", {}).get("seed_id"),
                    "evidence_ids": task.get("evidence_ids"), "source_pack": task.get("source_pack"),
                    "answer": task.get("answer"), "cutoff": task.get("cutoff"),
                    "evidence_graph": task.get("evidence_graph"),
                    "verification": task.get("verification"), "semantic_audit": task.get("semantic_audit"),
                    "teacher_audit": teacher_audit,
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
    return {**counts, "llm": {role: _client_stats(client) for role, client in clients.items()}}
