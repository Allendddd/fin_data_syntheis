from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mint_synth.storage.catalog import build_catalog, catalog_fingerprint, pack_segments, write_manifest
from mint_synth.core.config_v2 import config_fingerprint, dump_config, load_config, materialize_defaults
from mint_synth.harness.registry import load_tool_registry
from mint_synth.synthesis.dag import load_formula_registry
from mint_synth.synthesis.entity import load_aliases
from mint_synth.synthesis.long_horizon import load_evaluation_questions, load_seed_bank
from mint_synth.core.provenance import (
    auxiliary_fingerprints,
    deployment_image_fingerprint,
    json_fingerprint,
    package_fingerprint,
    sha256_file,
)
from mint_synth.core.run_state import RunLayout, atomic_json, read_json, shard_complete, stage_signature, utc_now


STAGE_ORDER = (
    "normalize",
    "id_validation",
    "dedup",
    "facts",
    "atomic",
    "entity_resolution",
    "repartition",
    "graph_planning",
    "query_realization",
    "validation",
    "corpus_index",
    "retrieval_index",
    "query_export",
    "sft_prepare",
    "export",
)


def _run_id(name: str, fingerprint: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe}-{stamp}-{fingerprint[:8]}-{uuid.uuid4().hex[:6]}"


def _stage_plan(config: Mapping[str, Any], normalize_count: int) -> list[dict[str, Any]]:
    stages = config["stages"]
    shuffle = int(config["reader"]["shuffle_partitions"])
    external_id_partitions = int(config["reader"]["external_id_partitions"])
    entity_resolution_buckets = int(config["entity_resolution"]["buckets"])
    entity_buckets = int(config["dag"]["entity_buckets"])
    active: list[dict[str, Any]] = []

    def add(name: str, count: int, dependencies: list[str]) -> None:
        active.append({"name": name, "shard_count": count, "dependencies": dependencies})

    add("normalize", normalize_count, [])
    add("id_validation", external_id_partitions, ["normalize"])
    add("dedup", shuffle, ["normalize", "id_validation"])
    if stages.get("facts"):
        add("facts", shuffle, ["dedup"])
    if stages.get("atomic") and config["atomic"].get("enabled", True):
        add("atomic", shuffle, ["facts"])
    if stages.get("entity_resolution"):
        add("entity_resolution", entity_resolution_buckets, ["facts"])
        add("repartition", entity_buckets, ["entity_resolution"])
    if stages.get("graph_planning"):
        add("graph_planning", entity_buckets, ["repartition"])
    if stages.get("query_realization"):
        add("query_realization", entity_buckets, ["graph_planning"])
    if stages.get("validation"):
        dependencies: list[str] = []
        counts: list[int] = []
        if any(stage["name"] == "atomic" for stage in active):
            dependencies.append("atomic")
            counts.append(shuffle)
        if any(stage["name"] == "query_realization" for stage in active):
            dependencies.append("query_realization")
            counts.append(entity_buckets)
        add("validation", max(counts or [1]), dependencies)
    if config.get("version") == 3:
        query_export = config.get("query_export", {}).get("enabled", True)
        sft_enabled = config["training"]["enabled"]
        long_sft = sft_enabled and config["training"]["long_enabled"]
        if query_export or long_sft:
            add("corpus_index", 1, ["facts"])
            add("retrieval_index", entity_buckets, ["corpus_index", "repartition"])
        if query_export:
            add("query_export", 1, ["validation", "retrieval_index"])
        if sft_enabled:
            dependencies = ["validation", "retrieval_index"] if long_sft else ["validation"]
            add("sft_prepare", next(int(stage["shard_count"]) for stage in active if stage["name"] == "validation"), dependencies)
    if stages.get("export"):
        dependencies = ["validation"]
        if config.get("version") == 3:
            if config["training"]["enabled"]:
                dependencies.append("sft_prepare")
            if config.get("query_export", {}).get("enabled", True):
                dependencies.append("query_export")
        add("export", 1, dependencies)
    return active


def _planned_signatures(
    run_fingerprint: str,
    stages: list[dict[str, Any]],
    assignments: list[list[Any]],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    signatures: dict[str, list[str]] = {}
    aggregates: dict[str, str] = {}
    for stage in stages:
        name = str(stage["name"])
        upstream = {dependency: aggregates[dependency] for dependency in stage.get("dependencies", [])}
        stage_values: list[str] = []
        for shard_id in range(int(stage["shard_count"])):
            assignment = (
                [segment.to_dict() for segment in assignments[shard_id]]
                if name == "normalize"
                else None
            )
            stage_values.append(
                stage_signature(
                    run_fingerprint=run_fingerprint,
                    stage=name,
                    shard_id=shard_id,
                    assignment=assignment,
                    upstream=upstream,
                )
            )
        signatures[name] = stage_values
        aggregates[name] = json_fingerprint(stage_values)
    return signatures, aggregates


def _validate_auxiliary_inputs(config: Mapping[str, Any]) -> dict[str, int]:
    formulas = load_formula_registry(str(config["dag"]["formula_registry"]))
    allowed = set(config["dag"]["allowed_operations"])
    enabled_formulas = [formula for formula in formulas if set(formula.operations).issubset(allowed)]
    if config["stages"].get("graph_planning") and not enabled_formulas:
        raise ValueError("no formula registry entry is enabled by dag.allowed_operations")

    aliases = load_aliases(config["entity_resolution"].get("alias_file"))
    seeds = load_seed_bank(config["query_realization"].get("seed_file"))
    objectives = {"comparison", "change", "ratio", "aggregation", "verification"}
    seed_ids: set[str] = set()
    for index, seed in enumerate(seeds, 1):
        seed_id = str(seed.get("seed_id", ""))
        if not seed_id or seed_id in seed_ids:
            raise ValueError(f"seed bank contains an empty or duplicate seed_id: {seed_id!r}")
        seed_ids.add(seed_id)
        template = seed.get("task_spec_template")
        if not isinstance(template, dict):
            raise ValueError(f"seed {seed.get('seed_id', index)!r} requires task_spec_template")
        if template.get("objective_type") not in objectives:
            raise ValueError(f"seed {seed.get('seed_id', index)!r} has an unsupported objective_type")
        for field in ("reasoning_form", "requested_output"):
            if not isinstance(template.get(field), str) or not template[field].strip():
                raise ValueError(f"seed {seed.get('seed_id', index)!r} requires non-empty {field}")
        roles = template.get("required_fact_roles")
        if not isinstance(roles, list) or not roles or not all(isinstance(value, str) and value.strip() for value in roles):
            raise ValueError(f"seed {seed.get('seed_id', index)!r} requires non-empty required_fact_roles")
        if not isinstance(template.get("time_sensitive"), bool):
            raise ValueError(f"seed {seed.get('seed_id', index)!r} requires boolean time_sensitive")
    if config.get("version") == 3:
        training = config["training"]
        load_tool_registry(config.get("harness", {}).get("tool_registry"))
        long_requested = config.get("query_export", {}).get("enabled", True) or (
            training["enabled"] and training["long_enabled"]
        )
        missing_seeds = set(training["seed_ids"]) - seed_ids
        if missing_seeds:
            raise ValueError(f"training.seed_ids contains unknown seeds: {sorted(missing_seeds)}")
        missing_formulas = set(training["formula_ids"]) - {formula.formula_id for formula in formulas}
        if missing_formulas:
            raise ValueError(f"training.formula_ids contains unknown formulas: {sorted(missing_formulas)}")
        selected_seeds = [seed for seed in seeds if not training["seed_ids"] or seed["seed_id"] in training["seed_ids"]]
        selected_seeds = [seed for seed in selected_seeds if not training["objective_types"] or seed["task_spec_template"]["objective_type"] in training["objective_types"]]
        if long_requested and not selected_seeds:
            raise ValueError("no long-horizon seed remains after training selectors")
        objective_by_generator = {
            "temporal_pair": "change", "compatible_ratio": "ratio",
            "compatible_average": "aggregation", "compatible_verification": "verification",
            "temporal_trend": "change", "ratio_change": "change", "multi_verification": "verification",
        }
        enabled_objectives = {
            objective_by_generator.get(formula.generator)
            for formula in enabled_formulas
            if not training["formula_ids"] or formula.formula_id in training["formula_ids"]
        }
        if long_requested and not any(seed["task_spec_template"]["objective_type"] in enabled_objectives for seed in selected_seeds):
            raise ValueError("selected long-horizon seeds have no enabled matching formula")
    evaluation = load_evaluation_questions(
        config["quality"].get("evaluation_file"),
        max_questions=int(config["quality"]["max_evaluation_questions"]),
    )
    return {
        "formulas": len(formulas),
        "enabled_formulas": len(enabled_formulas),
        "aliases": len(aliases),
        "seeds": len(seeds),
        "evaluation_questions": len(evaluation),
    }


def create_plan(config_path: str | Path) -> RunLayout:
    config = load_config(config_path)
    auxiliary_validation = _validate_auxiliary_inputs(config)
    segments = build_catalog(config)
    input_fingerprint = catalog_fingerprint(segments)
    cfg_fingerprint = config_fingerprint(config)
    code_fingerprint = package_fingerprint()
    auxiliary = auxiliary_fingerprints(config)
    fingerprint_parts: dict[str, Any] = {
        "config": cfg_fingerprint,
        "input": input_fingerprint,
        "code": code_fingerprint,
        "auxiliary": auxiliary,
    }
    if config["version"] == 2 and "slurm" in config:
        fingerprint_parts["deployment_image"] = deployment_image_fingerprint(config)
    combined = config_fingerprint(fingerprint_parts)
    assignments = pack_segments(segments, int(config["reader"]["normalize_target_bytes"]))
    stage_plan = _stage_plan(config, len(assignments))
    stage_signatures, stage_aggregates = _planned_signatures(combined, stage_plan, assignments)
    run_id = _run_id(str(config["run"]["name"]), combined)
    runs_root = Path(config["output"]["root"]) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final_layout = RunLayout(runs_root / run_id)
    layout = RunLayout(runs_root / f".{run_id}.planning-{uuid.uuid4().hex[:8]}")
    layout.root.mkdir(parents=False, exist_ok=False)

    plan = {
        "version": config["version"],
        "run_id": run_id,
        "created_at": utc_now(),
        "run_fingerprint": combined,
        "config_fingerprint": cfg_fingerprint,
        "input_fingerprint": input_fingerprint,
        "code_fingerprint": code_fingerprint,
        "auxiliary_fingerprints": auxiliary,
        "auxiliary_validation": auxiliary_validation,
        "input": {
            "files": len({segment.file_id for segment in segments}),
            "segments": len(segments),
            "estimated_bytes": sum(segment.estimated_bytes for segment in segments),
            "formats": sorted({segment.source_format for segment in segments}),
        },
        "stages": stage_plan,
        "stage_signatures": stage_signatures,
        "stage_aggregate_signatures": stage_aggregates,
    }
    if "deployment_image" in fingerprint_parts:
        plan["deployment_image_fingerprint"] = fingerprint_parts["deployment_image"]
    plan["plan_fingerprint"] = json_fingerprint(plan)
    try:
        (layout.root / "assignments" / "normalize").mkdir(parents=True)
        (layout.root / "logs").mkdir(parents=True)
        (layout.root / "exports").mkdir(parents=True)
        dump_config(config, layout.config)
        write_manifest(
            layout.root / "input_manifest.parquet",
            segments,
            compression=config["output"]["compression"],
        )
        for shard_id, assignment in enumerate(assignments):
            atomic_json(layout.assignment("normalize", shard_id), [segment.to_dict() for segment in assignment])
        atomic_json(layout.plan, plan)
        atomic_json(
            layout.manifest,
            {
                "version": config["version"],
                "run_id": run_id,
                "state": "PLANNED",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "run_fingerprint": combined,
            },
        )
        if final_layout.root.exists():
            raise FileExistsError(f"run directory already exists: {final_layout.root}")
        layout.root.replace(final_layout.root)
    except BaseException:
        if layout.root.exists():
            shutil.rmtree(layout.root)
        raise
    return final_layout


def stage_by_name(plan: Mapping[str, Any], name: str) -> dict[str, Any]:
    for stage in plan.get("stages", []):
        if stage.get("name") == name:
            return dict(stage)
    raise ValueError(f"stage is not part of this run: {name}")


def verify_plan_fingerprint(plan: Mapping[str, Any]) -> bool:
    expected = plan.get("plan_fingerprint")
    material = dict(plan)
    material.pop("plan_fingerprint", None)
    return isinstance(expected, str) and expected == json_fingerprint(material)


def expected_stage_signature(plan: Mapping[str, Any], stage: str, shard_id: int) -> str:
    try:
        return str(plan["stage_signatures"][stage][shard_id])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"plan has no signature for {stage}/{shard_id}") from exc


def run_status(run_dir: str | Path) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    plan = read_json(layout.plan)
    stages: list[dict[str, Any]] = []
    overall_complete = True
    total_failures = 0
    for stage in plan["stages"]:
        name = stage["name"]
        count = int(stage["shard_count"])
        complete_ids = [
            shard_id
            for shard_id in range(count)
            if shard_complete(layout, name, shard_id, expected_stage_signature(plan, name, shard_id))
        ]
        failures_dir = layout.stage_root(name) / "failures"
        failures = len(list(failures_dir.glob("*.json"))) if failures_dir.exists() else 0
        total_failures += failures
        overall_complete = overall_complete and len(complete_ids) == count
        stages.append(
            {
                "name": name,
                "expected": count,
                "complete": len(complete_ids),
                "pending": count - len(complete_ids),
                "failure_attempts": failures,
            }
        )
    return {
        "run_id": plan["run_id"],
        "run_dir": str(layout.root),
        "plan_integrity": verify_plan_fingerprint(plan),
        "complete": overall_complete,
        "published": (layout.root / "_PUBLISHED.json").is_file(),
        "failure_attempts": total_failures,
        "stages": stages,
    }


def validate_run_structure(run_dir: str | Path) -> dict[str, Any]:
    status = run_status(run_dir)
    errors: list[str] = []
    if not status["complete"]:
        errors.append("one or more planned shards are incomplete")
    layout = RunLayout(run_dir)
    if layout.config.is_file() and layout.plan.is_file():
        config = load_config(layout.config)
        plan = read_json(layout.plan)
        if not verify_plan_fingerprint(plan):
            errors.append("plan fingerprint mismatch")
        if config_fingerprint(config) != plan.get("config_fingerprint"):
            errors.append("resolved configuration fingerprint differs from plan")
        if package_fingerprint() != plan.get("code_fingerprint"):
            errors.append("running package fingerprint differs from planned code")
        try:
            if auxiliary_fingerprints(config) != plan.get("auxiliary_fingerprints"):
                errors.append("formula, alias, seed, or evaluation file differs from plan")
            if config["version"] == 2 and "slurm" in config and deployment_image_fingerprint(config) != plan.get("deployment_image_fingerprint"):
                errors.append("deployment image differs from plan")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        for stage in plan.get("stages", []):
            name = str(stage["name"])
            for shard_id in range(int(stage["shard_count"])):
                marker_path = layout.success(name, shard_id)
                if not marker_path.is_file():
                    continue
                try:
                    marker = read_json(marker_path)
                except (OSError, ValueError) as exc:
                    errors.append(f"invalid shard marker {name}/{shard_id}: {exc}")
                    continue
                if not isinstance(marker, Mapping):
                    errors.append(f"invalid shard marker {name}/{shard_id}: expected object")
                    continue
                if marker.get("signature") != expected_stage_signature(plan, name, shard_id):
                    errors.append(f"unexpected shard signature: {name}/{shard_id}")
                    continue
                shard_root = layout.shard(name, shard_id)
                files = marker.get("files")
                if not isinstance(files, list):
                    errors.append(f"invalid shard artifact list: {name}/{shard_id}")
                    continue
                for file_info in files:
                    try:
                        relative = Path(str(file_info["path"]))
                        artifact = (shard_root / relative).resolve()
                        expected_size = int(file_info["bytes"])
                    except (KeyError, TypeError, ValueError, OSError):
                        errors.append(f"invalid shard artifact entry: {name}/{shard_id}")
                        continue
                    if (
                        not artifact.is_relative_to(shard_root)
                        or not artifact.is_file()
                        or artifact.stat().st_size != expected_size
                    ):
                        errors.append(f"missing, unsafe, or size-mismatched shard artifact: {name}/{shard_id}/{relative}")
    else:
        errors.append("run is missing resolved_config.yaml or plan.json")
    publication_path = layout.root / "_PUBLISHED.json"
    planned_stage_names = {
        str(stage.get("name")) for stage in read_json(layout.plan).get("stages", [])
    } if layout.plan.is_file() else set()
    if "export" in planned_stage_names and status["complete"] and not publication_path.is_file():
        errors.append("export completed but publication marker is missing")
    if publication_path.is_file():
        try:
            publication = read_json(publication_path)
            if not isinstance(publication, Mapping):
                raise ValueError("publication marker must be an object")
            if publication.get("run_fingerprint") != read_json(layout.plan).get("run_fingerprint"):
                errors.append("publication fingerprint differs from plan")
            artifact_entries = publication.get("artifacts")
            if not isinstance(artifact_entries, list):
                raise ValueError("publication artifacts must be a list")
            for artifact_info in artifact_entries:
                if not isinstance(artifact_info, Mapping):
                    raise ValueError("publication artifact entry must be an object")
                relative = Path(str(artifact_info.get("path", "")))
                artifact = (layout.root / relative).resolve()
                if not artifact.is_relative_to(layout.root) or not artifact.is_file():
                    errors.append(f"missing or unsafe published artifact: {relative}")
                    continue
                if artifact.stat().st_size != int(artifact_info.get("bytes", -1)):
                    errors.append(f"published artifact size mismatch: {relative}")
                    continue
                expected_hash = artifact_info.get("sha256")
                if expected_hash and sha256_file(artifact) != expected_hash:
                    errors.append(f"published artifact checksum mismatch: {relative}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"invalid publication manifest: {exc}")
    return {"valid": not errors, "errors": errors, **status}


def migrate_v1(input_path: str | Path, output_path: str | Path) -> Path:
    """Create a v2 YAML starting point from a v1 input JSONL or v1 output manifest."""
    source = Path(input_path).resolve()
    raw_input = source
    text_field = "text"
    id_field: str | None = None
    if source.is_dir() and (source / "manifest.json").is_file():
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        old = manifest.get("config", {})
        raw_input = Path(str(old.get("input_path", source))).resolve()
        text_field = str(old.get("text_field", "text"))
        id_field = old.get("id_field")
    target = Path(output_path).resolve()
    template = materialize_defaults(
        {
            "version": 2,
            "run": {"name": "migrated-v1", "work_dir": str(target.parent / "work")},
            "input": {
                "paths": [str(raw_input)], "text_field": text_field, "id_field": id_field,
            },
            "output": {"root": str(target.parent / "outputs")},
            "llm": {
                "endpoints": {
                    "inference": {
                        "base_url": "http://CHANGE_ME:8000/v1",
                        "api_key_env": "FIN_LLM_API_KEY",
                    }
                },
                "roles": {
                    role: {"endpoint": "inference", "model": f"CHANGE_ME_{role}"}
                    for role in ("extractor", "constructor", "auditor", "entity_resolver")
                },
            },
        }
    )
    dump_config(template, target)
    return target


def migrate_v2(input_path: str | Path, output_path: str | Path, *, cutoff: str | None = None) -> Path:
    """Write a v3 data-synthesis configuration; previously planned v2 runs remain readable."""
    config = load_config(input_path)
    if config["version"] != 2:
        raise ValueError("migrate-v2 requires a version 2 configuration")
    new_cutoff = cutoff or config["run"].get("cutoff")
    if not new_cutoff:
        raise ValueError("migrate-v2 requires --cutoff when the v2 configuration has no cutoff")
    config["version"] = 3
    config["run"]["cutoff"] = new_cutoff
    config["run"]["fingerprint_mode"] = "sha256"
    config.pop("slurm", None)
    metadata = config["input"]["metadata_fields"]
    if "published_at" not in metadata:
        metadata.append("published_at")
    config = materialize_defaults(config)
    target = Path(output_path).resolve()
    dump_config(config, target)
    return target
