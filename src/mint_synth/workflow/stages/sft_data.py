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
from mint_synth.sft.atomic_teacher import sample_atomic, sample_reference
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
    atomic_verifier = training.get("atomic_verifier_role", "auditor")
    atomic_judge = training.get("atomic_judge_role", "auditor")
    long_roles = (training.get("long_teacher_roles") or [training["teacher_role"]]) if training["long_enabled"] else []
    long_verifier = training.get("long_verifier_role", "auditor")
    agent_settings = {role: settings_for_role(config, role) for role in long_roles}
    json_roles = [*atomic_roles, *([atomic_verifier, atomic_judge] if atomic_roles else []),
                  *([long_verifier] if long_roles else []),
                  *(role for role in long_roles if agent_settings[role][1].name == "native")]
    clients = {role: _role_client(layout, config, "sft_prepare", shard_id, role) for role in dict.fromkeys(json_roles)}
    needs_finance = any(not settings.native_tools for _model, settings in agent_settings.values())
    retrieval = config.get("harness", {}).get("retrieval")
    if long_roles and retrieval:
        from mint_synth.harness.resources import validate_retrieval
        validate_retrieval(retrieval)
    registry = load_tool_registry(config.get("harness", {}).get("tool_registry"), retrieval=retrieval) if needs_finance else None
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
                    policy=str(training.get("atomic_selection_policy", "judge")),
                    fallback=bool(training.get("atomic_audit_fallback", False)),
                    tolerance=str(config["quality"]["answer_tolerance"]),
                    verifier=clients[atomic_verifier], judge=clients[atomic_judge],
                    reasoning_format=training.get("reasoning_format", "plain"),
                )
                reference = None
                reference_record = {"source": "audited_reference", "accepted": False, "reason": "reference disabled"}
                fraction = int(hashlib.sha256(("reference:" + task_id).encode()).hexdigest()[:16], 16) / 2**64
                if training.get("atomic_include_reference", True) and fraction < float(training.get("atomic_reference_fraction", 1)):
                    reference, reference_record = sample_reference(
                        task, clients[atomic_verifier], tolerance=str(config["quality"]["answer_tolerance"]))
                elif training.get("atomic_include_reference", True):
                    reference_record["reason"] = "reference excluded by configured fraction"
                teacher_audit["reference"] = reference_record
                if sample is None and reference is not None:
                    sample = reference
                    teacher_audit["exported_source"] = "audited_reference"
                elif sample is not None:
                    teacher_audit["exported_source"] = (teacher_audit.get("selected") or {}).get("source", "teacher")
                    if reference is not None and teacher_audit["exported_source"] == "teacher":
                        sample["_reference_sample"] = reference
            else:
                sample = None
                reason = "no rollout accepted"
                bucket = entity_bucket(str(task.get("canonical_entity_id") or ""), int(config["dag"]["entity_buckets"]))
                index_path = None if retrieval else layout.shard("retrieval_index", bucket) / "snapshot.sqlite3"
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
                        verifier=clients[long_verifier],
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
                counts["atomic_candidates"] += sum(row.get("source") == "teacher" for row in teacher_audit["candidates"])
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
                    reference_sample = sample.pop("_reference_sample", None)
                    sample["messages"][0]["loss"] = False
                    sample["messages"][1]["loss"] = False
                    sample["_split"] = split
                    if family == "atomic":
                        sample["sample_id"] = hashlib.sha256((task_id + ":" + teacher_audit["exported_source"]).encode()).hexdigest()
                        teacher_audit["sample_id"] = sample["sample_id"]
                    streams[family].write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if family == "atomic":
                        counts["atomic_" + teacher_audit["exported_source"]] += 1
                    if reference_sample is not None:
                        reference_sample["_split"] = split
                        reference_sample["sample_id"] = hashlib.sha256((task_id + ":audited_reference").encode()).hexdigest()
                        streams[family].write(json.dumps(reference_sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                        reference_audit = {"source": "audited_reference", "task_id": task_id, "split": split,
                                           "sample_id": reference_sample["sample_id"], "task_type": task["task_type"],
                                           "reference": teacher_audit["reference"]}
                        streams["sidecar"].write(json.dumps(reference_audit, ensure_ascii=False) + "\n")
                        counts["atomic_audited_reference"] += 1
                        counts["atomic_accepted"] += 1
                        counts[f"atomic_{split}"] += 1
                sidecar = {
                    "task_id": task_id, "task_type": task["task_type"], "split": split,
                    "sample_id": sample.get("sample_id"), "source": teacher_audit.get("exported_source", "teacher"),
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
    return {**counts, "retrieval_mode": retrieval.get("mode") if retrieval else "legacy_snapshot",
            "llm": {role: _client_stats(client) for role, client in clients.items()}}
