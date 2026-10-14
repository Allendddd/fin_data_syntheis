"""Load query-only exports and private verifier cases without mixing the two views."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator, Mapping

from mint_synth.core.provenance import sha256_file
from mint_synth.harness.session import HarnessCase, case_from_task


def _query_root(run_root: Path) -> Path:
    for stage in ("export", "query_export"):
        candidate = run_root / "stages" / stage / "shards" / "000000" / "queries"
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(f"query manifest is missing under {run_root}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _next_object(stream: Any, name: str) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise ValueError(f"{name} ended before the hidden verifier")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"{name} contains a non-object row")
    return value


def iter_cases(run_dir: str | Path, *, split: str | None = None) -> Iterator[HarnessCase]:
    """Yield evaluator-side cases; never pass their ``task`` to the model."""
    if split not in (None, "train", "val"):
        raise ValueError("split must be train, val or None")
    run_root = Path(run_dir).resolve()
    query_root = _query_root(run_root)
    manifest = _read_json(query_root / "manifest.json")
    if manifest.get("model_visible_fields") != ["query"]:
        raise ValueError("query manifest does not declare a query-only model view")
    for name in ("train.jsonl", "val.jsonl", "hidden-verifier.jsonl"):
        path = query_root / name
        artifact = manifest.get("artifacts", {}).get(name, {})
        if not path.is_file() or path.stat().st_size != artifact.get("bytes") or sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"query artifact failed integrity check: {name}")
    cutoff = manifest.get("cutoff")
    available_field = manifest.get("available_at_field")
    if not cutoff or not available_field:
        import yaml

        config = yaml.safe_load((run_root / "resolved_config.yaml").read_text(encoding="utf-8"))
        cutoff = config["run"]["cutoff"]
        available_field = config["input"].get("available_at_field", "published_at")
    snapshot_artifacts = manifest.get("snapshot_artifacts", {})
    verified_snapshots: set[str] = set()
    counts = {"train": 0, "val": 0}
    with ExitStack() as stack:
        streams = {
            name: stack.enter_context((query_root / f"{name}.jsonl").open("r", encoding="utf-8"))
            for name in ("train", "val")
        }
        hidden = stack.enter_context((query_root / "hidden-verifier.jsonl").open("r", encoding="utf-8"))
        for line in hidden:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("hidden verifier contains a non-object row")
            row_split = row.get("split")
            line_index = row.get("line_index")
            if (not isinstance(row_split, str) or row_split not in streams
                    or type(line_index) is not int or line_index != counts[row_split]):
                raise ValueError("hidden verifier row has an invalid split or line index")
            query_row = _next_object(streams[row_split], f"{row_split}.jsonl")
            if set(query_row) != {"query"} or not isinstance(query_row["query"], str):
                raise ValueError("model-visible query row must contain only query")
            query = query_row["query"]
            counts[row_split] += 1
            if hashlib.sha256(query.encode("utf-8")).hexdigest() != row.get("query_sha256"):
                raise ValueError("query differs from the hidden verifier binding")
            relative = Path(str(row.get("snapshot_index") or ""))
            snapshot = (run_root / relative).resolve()
            if relative.is_absolute() or not snapshot.is_relative_to(run_root):
                raise ValueError("snapshot path escapes the run directory")
            reference = relative.as_posix()
            if reference not in verified_snapshots:
                if not snapshot.is_file():
                    raise ValueError(f"query snapshot is missing: {reference}")
                if snapshot_artifacts:
                    expected = snapshot_artifacts.get(reference, {})
                    if snapshot.stat().st_size != expected.get("bytes") or sha256_file(snapshot) != expected.get("sha256"):
                        raise ValueError(f"query snapshot failed integrity check: {reference}")
                verified_snapshots.add(reference)
            task = row.get("task")
            if not isinstance(task, Mapping):
                raise ValueError("hidden verifier task is missing")
            case = case_from_task(
                task, snapshot, cutoff=str(cutoff), available_at_field=str(available_field),
                query_id=str(row.get("query_id") or ""), query=query,
                toolset_fingerprint=manifest.get("toolset_fingerprint"),
            )
            if split is None or split == row_split:
                yield case
        for name, stream in streams.items():
            if stream.readline():
                raise ValueError(f"{name}.jsonl has extra rows")
    if counts != manifest.get("counts"):
        raise ValueError("query counts differ from the manifest")
