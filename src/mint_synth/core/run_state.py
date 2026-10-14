from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import traceback
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mint_synth import __version__


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class RunLayout:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    @property
    def config(self) -> Path:
        return self.root / "resolved_config.yaml"

    @property
    def plan(self) -> Path:
        return self.root / "plan.json"

    @property
    def manifest(self) -> Path:
        return self.root / "run_manifest.json"

    def assignment(self, stage: str, shard_id: int) -> Path:
        return self.root / "assignments" / stage / f"{shard_id:06d}.json"

    def stage_root(self, stage: str) -> Path:
        return self.root / "stages" / stage

    def shard(self, stage: str, shard_id: int) -> Path:
        return self.stage_root(stage) / "shards" / f"{shard_id:06d}"

    def success(self, stage: str, shard_id: int) -> Path:
        return self.shard(stage, shard_id) / "_SUCCESS.json"

    def logs(self, stage: str, shard_id: int) -> Path:
        return self.root / "logs" / stage / f"{shard_id:06d}.jsonl"


def stage_signature(
    *,
    run_fingerprint: str,
    stage: str,
    shard_id: int,
    assignment: Any = None,
    upstream: Any = None,
) -> str:
    material = {
        "engine_version": __version__,
        "run_fingerprint": run_fingerprint,
        "stage": stage,
        "shard_id": shard_id,
        "assignment": assignment,
        "upstream": upstream,
    }
    payload = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def shard_complete(layout: RunLayout, stage: str, shard_id: int, signature: str | None = None) -> bool:
    marker = layout.success(stage, shard_id)
    if not marker.is_file():
        return False
    try:
        value = read_json(marker)
    except (OSError, ValueError):
        return False
    if signature is not None and value.get("signature") != signature:
        return False
    if value.get("stage") != stage or value.get("shard_id") != shard_id:
        return False
    shard_root = layout.shard(stage, shard_id)
    files = value.get("files")
    if not isinstance(files, list):
        return False
    for file_info in files:
        if not isinstance(file_info, Mapping):
            return False
        try:
            artifact = (shard_root / str(file_info["path"])).resolve()
            expected_size = int(file_info["bytes"])
        except (KeyError, TypeError, ValueError, OSError):
            return False
        if not artifact.is_relative_to(shard_root) or not artifact.is_file():
            return False
        try:
            if artifact.stat().st_size != expected_size:
                return False
        except OSError:
            return False
    return True


def quarantine_incomplete_shard(
    layout: RunLayout,
    stage: str,
    shard_id: int,
    signature: str | None = None,
) -> Path | None:
    """Move a damaged/incomplete immutable shard aside so the planned shard can be rebuilt."""
    source = layout.shard(stage, shard_id)
    if not source.exists():
        return None
    if signature is not None and shard_complete(layout, stage, shard_id, signature):
        return None
    destination = (
        layout.stage_root(stage)
        / "corrupt"
        / f"{shard_id:06d}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
    except FileNotFoundError:
        return None
    return destination


class ShardTransaction(AbstractContextManager["ShardTransaction"]):
    """Same-filesystem shard commit. A completed shard is immutable."""

    def __init__(self, layout: RunLayout, stage: str, shard_id: int, signature: str):
        self.layout = layout
        self.stage = stage
        self.shard_id = shard_id
        self.signature = signature
        parent = layout.stage_root(stage) / ".tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(tempfile.mkdtemp(prefix=f"{shard_id:06d}-", dir=parent))
        self.final = layout.shard(stage, shard_id)
        self.committed = False

    @property
    def path(self) -> Path:
        return self.temporary

    def commit(self, metrics: Mapping[str, Any]) -> Path:
        atomic_json(self.temporary / "metrics.json", dict(metrics))
        files = []
        for path in sorted(self.temporary.rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(self.temporary).as_posix(), "bytes": path.stat().st_size})
        atomic_json(
            self.temporary / "_SUCCESS.json",
            {
                "stage": self.stage,
                "shard_id": self.shard_id,
                "signature": self.signature,
                "completed_at": utc_now(),
                "metrics": dict(metrics),
                "files": files,
            },
        )
        self.final.parent.mkdir(parents=True, exist_ok=True)
        if self.final.exists():
            if shard_complete(self.layout, self.stage, self.shard_id, self.signature):
                shutil.rmtree(self.temporary)
                self.committed = True
                return self.final
            raise RuntimeError(f"refusing to replace immutable shard with a different signature: {self.final}")
        try:
            os.replace(self.temporary, self.final)
        except OSError:
            # Two scheduler attempts may race. The winner's complete, matching shard is reusable.
            if shard_complete(self.layout, self.stage, self.shard_id, self.signature):
                shutil.rmtree(self.temporary)
                self.committed = True
                return self.final
            raise
        self.committed = True
        return self.final

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self.committed and self.temporary.exists():
            shutil.rmtree(self.temporary)


def record_failure(layout: RunLayout, stage: str, shard_id: int, signature: str, exc: BaseException) -> Path:
    destination = layout.stage_root(stage) / "failures" / f"{shard_id:06d}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    atomic_json(
        destination,
        {
            "stage": stage,
            "shard_id": shard_id,
            "signature": signature,
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-20000:],
        },
    )
    return destination


def log_event(
    layout: RunLayout,
    stage: str,
    shard_id: int,
    event: str,
    *,
    level: str = "INFO",
    minimum_level: str = "INFO",
    json_format: bool = True,
    **fields: Any,
) -> None:
    ranks = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    if ranks.get(level, 20) < ranks.get(minimum_level, 20):
        return
    path = layout.logs(stage, shard_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": utc_now(), "level": level, "event": event,
        "stage": stage, "shard_id": shard_id, **fields,
    }
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        if json_format:
            rendered = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        else:
            rendered = " ".join(f"{key}={json.dumps(value, ensure_ascii=False, default=str)}" for key, value in record.items())
        stream.write(rendered + "\n")
