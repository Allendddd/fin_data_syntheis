from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


class ConfigError(ValueError):
    """Raised when a v2 configuration is missing, malformed, or inconsistent."""


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by dependency-free installs
        raise ConfigError("PyYAML is required; install mint-synth with its runtime dependencies") from exc
    return yaml


def _schema() -> dict[str, Any]:
    resource = resources.files("mint_synth").joinpath("resources/config-v2.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _inject_defaults(instance: Any, schema: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    """Recursively materialize JSON-Schema defaults without mutating caller data."""
    if "$ref" in schema:
        target: Any = root
        for part in str(schema["$ref"]).removeprefix("#/").split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        schema = target
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key not in instance and "default" in child_schema:
                instance[key] = copy.deepcopy(child_schema["default"])
            if key in instance:
                instance[key] = _inject_defaults(instance[key], child_schema, root)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key in instance:
                if key not in properties:
                    instance[key] = _inject_defaults(instance[key], additional, root)
    elif isinstance(instance, list) and isinstance(schema.get("items"), dict):
        instance = [_inject_defaults(value, schema["items"], root) for value in instance]
    return instance


def _validate_schema(config: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - exercised by dependency-free installs
        raise ConfigError("jsonschema is required; install mint-synth with its runtime dependencies") from exc
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    errors = sorted(validator.iter_errors(config), key=lambda error: list(error.absolute_path))
    if errors:
        rendered: list[str] = []
        for error in errors[:20]:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            rendered.append(f"{location}: {error.message}")
        suffix = f"\n... and {len(errors) - 20} more error(s)" if len(errors) > 20 else ""
        raise ConfigError("configuration validation failed:\n" + "\n".join(rendered) + suffix)


def materialize_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    schema = _schema()
    return _inject_defaults(value, schema, schema)


def validate_cross_fields(config: Mapping[str, Any]) -> None:
    chunking = config.get("chunking", {})
    if int(chunking.get("overlap_chars", 0)) >= int(chunking.get("max_chars", 1)):
        raise ConfigError("chunking.overlap_chars must be smaller than chunking.max_chars")
    if int(chunking.get("min_chars", 1)) > int(chunking.get("max_chars", 1)):
        raise ConfigError("chunking.min_chars must not exceed chunking.max_chars")

    input_config = config.get("input", {})
    if int(input_config.get("max_jsonl_line_bytes", 1)) < int(input_config.get("max_text_chars", 1)):
        raise ConfigError("input.max_jsonl_line_bytes must be at least input.max_text_chars")

    dag = config.get("dag", {})
    if int(dag.get("min_depth", 1)) > int(dag.get("max_depth", 1)):
        raise ConfigError("dag.min_depth must not exceed dag.max_depth")
    if int(dag.get("min_sources", 1)) < 2:
        raise ConfigError("dag.min_sources must be at least 2 for evidence crossover")
    if int(dag.get("min_rule_graphs_before_supplement", 0)) > int(dag.get("max_graphs_per_entity", 1)):
        raise ConfigError("dag.min_rule_graphs_before_supplement must not exceed dag.max_graphs_per_entity")

    reader = config.get("reader", {})
    if int(reader.get("shuffle_partitions", 1)) > int(reader.get("dedup_buckets", 1)):
        raise ConfigError("reader.shuffle_partitions must not exceed reader.dedup_buckets")
    if int(reader.get("dedup_buckets", 1)) % int(reader.get("shuffle_partitions", 1)) != 0:
        raise ConfigError("reader.dedup_buckets must be divisible by reader.shuffle_partitions")

    metadata_fields = set(config.get("input", {}).get("metadata_fields", []))
    identifier_fields = set(config.get("entity_resolution", {}).get("identifier_fields", []))
    missing_identifier_metadata = sorted(identifier_fields - metadata_fields)
    if missing_identifier_metadata:
        raise ConfigError(
            "entity_resolution.identifier_fields must also be listed in input.metadata_fields: "
            + ", ".join(missing_identifier_metadata)
        )

    try:
        tolerance = Decimal(str(config.get("quality", {}).get("answer_tolerance", "1e-6")))
    except InvalidOperation as exc:
        raise ConfigError("quality.answer_tolerance must be a positive finite decimal") from exc
    if not tolerance.is_finite() or tolerance <= 0:
        raise ConfigError("quality.answer_tolerance must be a positive finite decimal")

    endpoints = config.get("llm", {}).get("endpoints", {})
    for name, endpoint_config in endpoints.items():
        parsed = urlsplit(str(endpoint_config.get("base_url", "")))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigError(f"llm.endpoints.{name}.base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ConfigError(f"llm.endpoints.{name}.base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ConfigError(f"llm.endpoints.{name}.base_url must not contain a query string or fragment")
    for role, role_config in config.get("llm", {}).get("roles", {}).items():
        endpoint = role_config.get("endpoint")
        if endpoint not in endpoints:
            raise ConfigError(f"llm.roles.{role}.endpoint references unknown endpoint {endpoint!r}")
    roles = config.get("llm", {}).get("roles", {})
    if config.get("quality", {}).get("require_independent_auditor", True):
        constructor = roles.get("constructor", {})
        auditor = roles.get("auditor", {})
        constructor_identity = (
            constructor.get("endpoint"), constructor.get("model"), constructor.get("revision")
        )
        auditor_identity = (auditor.get("endpoint"), auditor.get("model"), auditor.get("revision"))
        if constructor_identity == auditor_identity:
            raise ConfigError(
                "quality.require_independent_auditor=true requires auditor and constructor to use different model identities"
            )

    stages = config.get("stages", {})
    if config.get("atomic", {}).get("enabled", True) and int(config.get("atomic", {}).get("tasks_per_chunk", 0)) < 1:
        raise ConfigError("atomic.tasks_per_chunk must be at least 1 when atomic.enabled=true")
    if not stages.get("normalize", False):
        raise ConfigError("stages.normalize must be enabled; prepared-dataset entry is not implemented")
    dependencies = {
        "facts": ("normalize",),
        "atomic": ("facts",),
        "entity_resolution": ("facts",),
        "graph_planning": ("entity_resolution",),
        "query_realization": ("graph_planning",),
        "validation": ("facts",),
        "export": ("validation",),
    }
    for stage, required in dependencies.items():
        if stages.get(stage, False):
            missing = [name for name in required if not stages.get(name, False)]
            if missing:
                raise ConfigError(f"stages.{stage} requires enabled stage(s): {', '.join(missing)}")
    if stages.get("validation"):
        has_atomic = stages.get("atomic", False) and config.get("atomic", {}).get("enabled", True)
        has_long = stages.get("query_realization", False)
        if not has_atomic and not has_long:
            raise ConfigError("stages.validation requires atomic tasks and/or query_realization")

    bind_sensitive_paths = [
        *config.get("slurm", {}).get("bind_paths", []),
        *config.get("input", {}).get("paths", []),
        config.get("run", {}).get("work_dir", ""),
        config.get("output", {}).get("root", ""),
        config.get("slurm", {}).get("apptainer_image", ""),
        config.get("entity_resolution", {}).get("alias_file", "") or "",
        config.get("dag", {}).get("formula_registry", "") or "",
        config.get("quality", {}).get("evaluation_file", "") or "",
        config.get("query_realization", {}).get("seed_file", "") or "",
    ]
    for path in bind_sensitive_paths:
        if any(character in str(path) for character in (",", "\n", "\r")):
            raise ConfigError("paths used by Apptainer binds cannot contain commas or newlines")
    for field in ("account", "partition", "qos"):
        value = config.get("slurm", {}).get(field)
        if value is not None and (
            any(character.isspace() for character in str(value))
            or str(value).startswith("-")
        ):
            raise ConfigError(f"slurm.{field} must be a single non-option token without whitespace")
    for argument in config.get("slurm", {}).get("extra_sbatch_args", []):
        if any(character in str(argument) for character in ("\n", "\r")):
            raise ConfigError("slurm.extra_sbatch_args cannot contain newlines")

def _absolute(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def resolve_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    base = Path(config_path).resolve().parent
    resolved = copy.deepcopy(config)
    resolved["run"]["work_dir"] = _absolute(resolved["run"]["work_dir"], base)
    resolved["input"]["paths"] = [_absolute(value, base) for value in resolved["input"]["paths"]]
    resolved["output"]["root"] = _absolute(resolved["output"]["root"], base)
    resolved["slurm"]["apptainer_image"] = _absolute(resolved["slurm"]["apptainer_image"], base)
    resolved["slurm"]["bind_paths"] = [_absolute(value, base) for value in resolved["slurm"]["bind_paths"]]
    alias_file = resolved.get("entity_resolution", {}).get("alias_file")
    if alias_file:
        resolved["entity_resolution"]["alias_file"] = _absolute(alias_file, base)
    formula_registry = resolved.get("dag", {}).get("formula_registry")
    if formula_registry and formula_registry != "builtin":
        resolved["dag"]["formula_registry"] = _absolute(formula_registry, base)
    evaluation_file = resolved.get("quality", {}).get("evaluation_file")
    if evaluation_file:
        resolved["quality"]["evaluation_file"] = _absolute(evaluation_file, base)
    seed_file = resolved.get("query_realization", {}).get("seed_file")
    if seed_file:
        resolved["query_realization"]["seed_file"] = _absolute(seed_file, base)
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    yaml = _require_yaml()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"cannot parse YAML {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a YAML mapping")
    schema = _schema()
    with_defaults = _inject_defaults(copy.deepcopy(raw), schema, schema)
    _validate_schema(with_defaults, schema)
    validate_cross_fields(with_defaults)
    resolved = resolve_paths(with_defaults, config_path)
    # Relative paths can inherit unsafe bind delimiters from the configuration directory.
    validate_cross_fields(resolved)
    return resolved


def redacted_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a persistable config. Secrets are represented only by environment variable names."""
    result = copy.deepcopy(dict(config))
    # Literal API keys are not part of the schema. Keep this defensive for programmatic callers.
    for endpoint in result.get("llm", {}).get("endpoints", {}).values():
        for key in list(endpoint):
            if key.casefold() in {"api_key", "token", "secret", "password"}:
                endpoint[key] = "<redacted>"
    return result


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    yaml = _require_yaml()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(redacted_config(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(redacted_config(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
