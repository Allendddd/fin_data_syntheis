"""Compile query-only seeds once, before parallel evidence-graph construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import atomic_json
from mint_synth.core.provenance import json_fingerprint
from mint_synth.synthesis.long_horizon import DEFAULT_SEEDS, load_seed_bank, load_evaluation_questions
from mint_synth.synthesis.decontamination import QuestionIndex
from mint_synth.synthesis.seed_constraints import OBJECTIVE_BY_GENERATOR, formulas_for_seed
from mint_synth.synthesis.dag import load_formula_registry
from .common import _role_client, _client_stats


SEED_SYSTEM = (
    "Abstract this analyst research request into an entity-free financial task specification. "
    "Preserve its financial meaning, evidence requirements, temporal relations and requested output; "
    "remove original entities, specific dates and values. Select an available executable generator only "
    "when it expresses the request faithfully; otherwise return unsupported with an explanation. "
    "Reasoning descriptions and fact roles may be in the request's language. Return private JSON only."
)


def raw_seeds(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = config["query_realization"].get("seed_file")
    seeds = load_seed_bank(path)
    if path and config.get("training", {}).get("seed_fallback"):
        present = {seed["seed_id"] for seed in seeds}
        seeds.extend(dict(seed) for seed in DEFAULT_SEEDS if seed["seed_id"] not in present)
    seen, result = set(), []
    for seed in seeds:
        key = " ".join(seed["seed_query"].split()).casefold()
        if key not in seen:
            result.append(seed)
            seen.add(key)
    return result


def validate_template(template: Any, seed_id: str) -> None:
    if not isinstance(template, dict):
        raise ValueError(f"seed {seed_id!r}: parser returned no task specification")
    for key in ("objective_type", "reasoning_form", "requested_output"):
        if not isinstance(template.get(key), str) or not template[key].strip():
            raise ValueError(f"seed {seed_id!r}: missing {key}")
    roles = template.get("required_fact_roles")
    if not isinstance(roles, list) or not roles or any(not isinstance(role, str) or not role.strip() for role in roles):
        raise ValueError(f"seed {seed_id!r}: missing required_fact_roles")
    if type(template.get("time_sensitive")) is not bool:
        raise ValueError(f"seed {seed_id!r}: time_sensitive must be boolean")


def stage_seed_prepare(layout, config, plan, shard_id, output):
    del plan
    formulas = load_formula_registry(str(config["dag"]["formula_registry"]))
    allowed = set(config["dag"]["allowed_operations"])
    formulas = [f for f in formulas if set(f.operations) <= allowed]
    selected = set(config.get("training", {}).get("formula_ids", []))
    formulas = [f for f in formulas if not selected or f.formula_id in selected]
    generators = {f.generator: OBJECTIVE_BY_GENERATOR.get(f.generator) for f in formulas}
    evaluation = load_evaluation_questions(config["quality"].get("evaluation_file"))
    index = QuestionIndex(evaluation)
    seeds, rejected, client = [], [], None
    for seed in raw_seeds(config):
        if index.overlaps(seed["seed_query"], float(config["quality"]["overlap_threshold"])):
            rejected.append({"seed_id": seed["seed_id"], "reason": "seed overlaps evaluation question"})
            continue
        try:
            if not isinstance(seed.get("task_spec_template"), dict):
                if client is None:
                    client = _role_client(layout, config, "seed_prepare", shard_id, "constructor")
                response = client.complete_json(
                    system=SEED_SYSTEM,
                    user=json.dumps({"query": seed["seed_query"], "available_generators": generators,
                        "output": {"task_spec_template": {"objective_type": "...", "generator": "available generator",
                            "reasoning_form": "...", "required_fact_roles": ["..."],
                            "time_sensitive": False, "requested_output": "..."}, "unsupported": "reason if unavailable"}}, ensure_ascii=False),
                    temperature=0.0, max_tokens=1800,
                    sample_key="seed-spec-v1:" + json_fingerprint(seed),
                )
                if not isinstance(response, dict) or response.get("unsupported"):
                    raise ValueError(str(response.get("unsupported")) if isinstance(response, dict) else "invalid parser response")
                seed = {**seed, "task_spec_template": response.get("task_spec_template"), "spec_origin": "model_parsed_v1"}
                if seed["task_spec_template"].get("generator") not in generators:
                    raise ValueError("parser selected an unavailable generator")
            validate_template(seed.get("task_spec_template"), seed["seed_id"])
            formulas_for_seed(seed, formulas)
            seeds.append(seed)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            rejected.append({"seed_id": seed["seed_id"], "reason": str(exc)})
    atomic_json(output / "seeds.json", {"version": 1, "seeds": seeds, "rejected": rejected,
        "evaluation_check": "lexical_overlap" if evaluation else "not_configured"})
    if not seeds:
        raise ValueError("no usable seed remains: " + json.dumps(rejected, ensure_ascii=False))
    return {"seeds": len(seeds), "rejected": len(rejected), "evaluation_check": "lexical_overlap" if evaluation else "not_configured",
            "llm": _client_stats(client) if client is not None else {}}


def prepared_seeds(layout, config) -> list[dict[str, Any]]:
    path = layout.shard("seed_prepare", 0) / "seeds.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["seeds"]
    seeds = raw_seeds(config)
    if any(not isinstance(seed.get("task_spec_template"), dict) for seed in seeds):
        raise ValueError("query-only seeds require the seed_prepare stage before graph planning")
    return seeds  # Explicit legacy templates remain readable by older frozen plans.
