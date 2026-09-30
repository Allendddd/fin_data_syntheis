from __future__ import annotations

import hashlib
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
    root = Path(__file__).resolve().parent
    files = sorted(
        [*root.glob("*.py"), *root.glob("resources/*.json"), *root.glob("resources/*.yaml")],
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
    return result


def deployment_image_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    image = Path(str(config["slurm"]["apptainer_image"]))
    if not image.is_file():
        raise ValueError(f"slurm.apptainer_image does not exist or is not a file: {image}")
    stat = image.stat()
    result: dict[str, Any] = {
        "path": str(image.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if config["slurm"].get("image_fingerprint_mode", "sha256") == "sha256":
        result["sha256"] = sha256_file(image)
    return result
