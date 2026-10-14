from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        [*root.rglob("*.py"), *root.glob("resources/*.json"), *root.glob("resources/*.yaml")],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def auxiliary_fingerprints(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = {
        "formula_registry": config.get("dag", {}).get("formula_registry"),
        "entity_aliases": config.get("entity_resolution", {}).get("alias_file"),
        "evaluation_file": config.get("quality", {}).get("evaluation_file"),
        "seed_file": config.get("query_realization", {}).get("seed_file"),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, raw in paths.items():
        if not raw or raw == "builtin":
            result[name] = {"kind": "builtin" if raw == "builtin" else "absent"}
            continue
        path = Path(str(raw))
        if not path.is_file():
            raise ValueError(f"configured {name} file does not exist: {path}")
        stat = path.stat()
        result[name] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "sha256": sha256_file(path),
        }
    if config.get("version") == 3:
        harness = config.get("harness", {})
        factory = harness.get("tool_registry")
        resources = harness.get("resource_paths", [])
        plugin: dict[str, Any] = {"factory": factory, "files": []}
        source_paths: list[Path] = []
        if factory:
            module_name, separator, function_name = str(factory).partition(":")
            if not separator or not module_name or not function_name.isidentifier():
                raise ValueError("harness.tool_registry must be package.module:register_tools")
            spec = importlib.util.find_spec(module_name)
            if spec is None or not spec.origin or not spec.origin.endswith(".py"):
                raise ValueError(f"harness tool registry source cannot be fingerprinted: {module_name}")
            origin = Path(spec.origin).resolve()
            source_paths.extend(path for path in origin.parent.rglob("*.py") if "__pycache__" not in path.parts)
        for raw in resources:
            path = Path(str(raw)).resolve()
            if path.is_file():
                source_paths.append(path)
            elif path.is_dir():
                source_paths.extend(item for item in path.rglob("*") if item.is_file() and "__pycache__" not in item.parts)
            else:
                raise ValueError(f"harness resource path does not exist: {path}")
        plugin["files"] = [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(set(source_paths))
        ]
        result["harness_tool_registry"] = plugin
    return result


def deployment_image_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the deployment fingerprint contract of legacy v2 plans."""
    slurm = config["slurm"]
    image = Path(str(slurm["apptainer_image"]))
    if not image.is_file():
        raise ValueError(f"slurm.apptainer_image does not exist or is not a file: {image}")
    stat = image.stat()
    result: dict[str, Any] = {
        "path": str(image.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if slurm.get("image_fingerprint_mode", "sha256") == "sha256":
        result["sha256"] = sha256_file(image)
    return result
