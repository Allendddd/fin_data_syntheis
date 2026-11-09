from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

from mint_synth.core.config_v2 import config_fingerprint, load_config
from mint_synth.core.provenance import auxiliary_fingerprints, package_fingerprint, sha256_file
from mint_synth.core.run_state import RunLayout, ShardTransaction, atomic_json, log_event, quarantine_incomplete_shard, read_json, record_failure, shard_complete, stage_signature, utc_now
from mint_synth.workflow.planner import expected_stage_signature, stage_by_name, verify_plan_fingerprint
from mint_synth.workflow.stages.publication import _success_metrics
from mint_synth.workflow.stages.registry import HANDLERS


def _upstream_signatures(layout: RunLayout, plan: Mapping[str, Any], stage: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for dependency in stage.get("dependencies", []):
        dependency_stage = stage_by_name(plan, dependency)
        for shard_id in range(int(dependency_stage["shard_count"])):
            expected = expected_stage_signature(plan, dependency, shard_id)
            if not shard_complete(layout, dependency, shard_id, expected):
                raise RuntimeError(
                    "upstream shard is incomplete, has a signature mismatch, or has damaged artifacts: "
                    f"{dependency}/{shard_id}"
                )
        result[dependency] = str(plan["stage_aggregate_signatures"][dependency])
    return result


def _enforce_upstream_quality(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    dependencies: Iterable[str],
) -> None:
    for dependency in dependencies:
        count = int(stage_by_name(plan, dependency)["shard_count"])
        metrics = _success_metrics(layout, dependency, count)
        if dependency == "normalize":
            total = sum(int(value.get("input_records", 0)) for value in metrics)
            invalid = sum(int(value.get("invalid_records", 0)) for value in metrics)
            ratio = invalid / total if total else 0.0
            if ratio > float(config["quality"]["max_invalid_record_ratio"]):
                raise RuntimeError(f"global invalid record ratio {ratio:.6f} exceeds configured limit")
        elif dependency == "id_validation":
            total = sum(int(value.get("input_records", 0)) for value in metrics)
            failed = sum(int(value.get("conflicting_records", 0)) for value in metrics)
            ratio = failed / total if total else 0.0
            if ratio > float(config["quality"]["max_external_id_conflict_ratio"]):
                raise RuntimeError(
                    f"global external-id conflict ratio {ratio:.6f} exceeds configured limit"
                )
        elif dependency in {"facts", "atomic", "entity_resolution", "graph_planning", "query_realization"}:
            denominator_key = {
                "facts": "chunks",
                "atomic": "candidates",
                "entity_resolution": "llm_requests",
                "graph_planning": "llm_requests",
                "query_realization": "candidates",
            }[dependency]
            total = sum(int(value.get(denominator_key, 0)) for value in metrics)
            failed = sum(int(value.get("terminal_errors", 0)) for value in metrics)
            ratio = failed / total if total else 0.0
            if ratio > float(config["quality"]["max_llm_terminal_error_ratio"]):
                raise RuntimeError(
                    f"global {dependency} LLM terminal error ratio {ratio:.6f} exceeds configured limit"
                )
        elif dependency == "validation":
            total = sum(int(value.get("input_tasks", 0)) for value in metrics)
            failed = sum(int(value.get("rejected", 0)) for value in metrics)
            ratio = failed / total if total else 0.0
            if ratio > float(config["quality"]["max_validation_failure_ratio"]):
                raise RuntimeError(f"global validation failure ratio {ratio:.6f} exceeds configured limit")


def _publish_export_markers(
    layout: RunLayout,
    plan: Mapping[str, Any],
    export_shard: Path,
    metrics: Mapping[str, Any],
) -> None:
    inventory_path = export_shard / "artifact_manifest.json"
    if not inventory_path.is_file():
        raise RuntimeError("committed export shard is missing artifact_manifest.json")
    inventory = read_json(inventory_path)
    if not isinstance(inventory, Mapping) or not isinstance(inventory.get("artifacts"), list):
        raise RuntimeError("committed export artifact manifest is malformed")
    artifacts: list[dict[str, Any]] = []
    for stored in inventory.get("artifacts", []):
        if not isinstance(stored, Mapping):
            raise RuntimeError("committed export artifact manifest contains a malformed entry")
        relative = Path(str(stored.get("path", "")))
        path = (export_shard / relative).resolve()
        if not path.is_relative_to(export_shard) or not path.is_file():
            raise RuntimeError(f"export artifact is missing or unsafe: {relative}")
        if path.stat().st_size != int(stored.get("bytes", -1)):
            raise RuntimeError(f"export artifact size changed after commit: {relative}")
        expected_hash = stored.get("sha256")
        if expected_hash and sha256_file(path) != expected_hash:
            raise RuntimeError(f"export artifact checksum changed after commit: {relative}")
        artifacts.append(
            {
                **dict(stored),
                "path": path.relative_to(layout.root).as_posix(),
            }
        )
    published_at = inventory.get("created_at")
    if not isinstance(published_at, str) or not published_at:
        existing_path = layout.root / "_PUBLISHED.json"
        try:
            existing = read_json(existing_path) if existing_path.is_file() else {}
        except (OSError, ValueError):
            existing = {}
        published_at = (
            existing.get("published_at")
            if isinstance(existing, Mapping)
            and existing.get("run_fingerprint") == plan["run_fingerprint"]
            else None
        )
    if not isinstance(published_at, str) or not published_at:
        published_at = utc_now()
    publication = {
        "run_id": plan["run_id"],
        "run_fingerprint": plan["run_fingerprint"],
        "published_at": published_at,
        "export_shard": str(export_shard),
        "artifacts": artifacts,
        "metrics": dict(metrics),
    }
    atomic_json(
        layout.root / "exports" / "manifest.json",
        {
            **publication,
            "quality_summary": str(export_shard / "quality_summary.json"),
        },
    )
    atomic_json(layout.root / "_PUBLISHED.json", publication)


def run_worker(run_dir: str | Path, stage_name: str, shard_id: int) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    config = load_config(layout.config)
    plan = read_json(layout.plan)
    if not verify_plan_fingerprint(plan):
        raise RuntimeError("plan fingerprint mismatch")
    if config_fingerprint(config) != plan.get("config_fingerprint"):
        raise RuntimeError("resolved configuration differs from the immutable plan")
    if package_fingerprint() != plan.get("code_fingerprint"):
        raise RuntimeError("worker package differs from the code fingerprint frozen by the plan")
    if auxiliary_fingerprints(config) != plan.get("auxiliary_fingerprints"):
        raise RuntimeError("formula, alias, seed, or evaluation file changed after planning")
    stage = stage_by_name(plan, stage_name)
    if shard_id < 0 or shard_id >= int(stage["shard_count"]):
        raise ValueError(f"shard id {shard_id} is outside stage range")
    assignment: Any = None
    if stage_name == "normalize":
        assignment = read_json(layout.assignment("normalize", shard_id))
    upstream = _upstream_signatures(layout, plan, stage)
    _enforce_upstream_quality(layout, config, plan, stage.get("dependencies", []))
    signature = stage_signature(
        run_fingerprint=str(plan["run_fingerprint"]),
        stage=stage_name,
        shard_id=shard_id,
        assignment=assignment,
        upstream=upstream,
    )
    planned_signature = expected_stage_signature(plan, stage_name, shard_id)
    if signature != planned_signature:
        raise RuntimeError("assignment or stage signature differs from the immutable plan")
    shard_existed_before_check = layout.shard(stage_name, shard_id).exists()
    if shard_complete(layout, stage_name, shard_id, signature):
        marker = read_json(layout.success(stage_name, shard_id))
        if stage_name == "export":
            _publish_export_markers(
                layout,
                plan,
                layout.shard(stage_name, shard_id),
                marker.get("metrics", {}),
            )
        return {"status": "reused", **marker}
    log_options = {
        "minimum_level": str(config["logging"]["level"]),
        "json_format": bool(config["logging"]["json"]),
    }
    quarantined = (
        quarantine_incomplete_shard(layout, stage_name, shard_id, signature)
        if shard_existed_before_check
        else None
    )
    if quarantined is not None:
        log_event(
            layout,
            stage_name,
            shard_id,
            "quarantined_incomplete_shard",
            level="WARNING",
            previous_shard=str(quarantined),
            **log_options,
        )
    log_event(layout, stage_name, shard_id, "started", signature=signature, **log_options)
    handler = HANDLERS[stage_name]
    try:
        with ShardTransaction(layout, stage_name, shard_id, signature) as transaction:
            processing_started = perf_counter()
            metrics = handler(layout, config, plan, shard_id, transaction.path)
            metrics["processing_seconds"] = round(perf_counter() - processing_started, 3)
            final = transaction.commit(metrics)
        if stage_name == "export":
            _publish_export_markers(layout, plan, final, metrics)
        log_event(layout, stage_name, shard_id, "completed", metrics=metrics, **log_options)
        return {"status": "completed", "stage": stage_name, "shard_id": shard_id, "metrics": metrics}
    except BaseException as exc:
        failure = record_failure(layout, stage_name, shard_id, signature, exc)
        log_event(
            layout,
            stage_name,
            shard_id,
            "failed",
            level="ERROR",
            error_type=type(exc).__name__,
            error=str(exc),
            failure=str(failure),
            **log_options,
        )
        raise
    finally:
        for client in layout.__dict__.pop("_role_clients", []):
            close = getattr(client, "close", None)
            if callable(close):
                close()
