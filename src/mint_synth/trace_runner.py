"""Timestamp an agent CLI's JSONL events alongside model API proxy records."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from mint_synth.core.run_state import atomic_json, read_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_agent_trace(command: Sequence[str], *, output_dir: str | Path, run_id: str) -> int:
    if not command:
        raise ValueError("agent command is required after --")
    if not run_id.strip():
        raise ValueError("run_id must be nonempty")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        root.chmod(0o700)
    proxy_manifest = root / "run.json"
    if proxy_manifest.exists():
        proxy_record = read_json(proxy_manifest)
        if not isinstance(proxy_record, dict) or proxy_record.get("run_id") != run_id:
            raise ValueError("run_id does not match the proxy trace in output_dir")
    metadata_path = root / "agent.json"
    if metadata_path.exists():
        raise ValueError("agent trace already exists in output_dir; use a new directory")
    metadata: dict[str, object] = {
        "format_version": 1,
        "run_id": run_id,
        "command": list(command),
        "started_at": _now(),
        "state": "running",
        "stdout_file": "agent.stdout.jsonl",
        "stderr_file": "agent.stderr.log",
        "timing_file": "agent.timing.jsonl",
    }
    atomic_json(metadata_path, metadata)

    files = {}
    try:
        for name, filename in (("stdout", "agent.stdout.jsonl"), ("stderr", "agent.stderr.log"), ("timing", "agent.timing.jsonl")):
            descriptor = os.open(root / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            files[name] = os.fdopen(descriptor, "wb")
        process = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None and process.stderr is not None
        lock = threading.Lock()
        errors: list[BaseException] = []
        started = time.monotonic_ns()

        def collect(source: object, stream_name: str, console: object) -> None:
            offset = 0
            try:
                while True:
                    line = source.readline()  # type: ignore[attr-defined]
                    if not line:
                        break
                    timestamp = _now()
                    elapsed_ms = round((time.monotonic_ns() - started) / 1_000_000, 3)
                    with lock:
                        files[stream_name].write(line)
                        files[stream_name].flush()
                        files["timing"].write((json.dumps({
                            "stream": stream_name,
                            "observed_at": timestamp,
                            "elapsed_ms": elapsed_ms,
                            "byte_offset": offset,
                            "byte_length": len(line),
                        }) + "\n").encode("utf-8"))
                        files["timing"].flush()
                    offset += len(line)
                    console.write(line)  # type: ignore[attr-defined]
                    console.flush()  # type: ignore[attr-defined]
            except BaseException as exc:
                errors.append(exc)
                process.terminate()

        stdout_thread = threading.Thread(target=collect, args=(process.stdout, "stdout", sys.stdout.buffer))
        stderr_thread = threading.Thread(target=collect, args=(process.stderr, "stderr", sys.stderr.buffer))
        stdout_thread.start()
        stderr_thread.start()
        try:
            try:
                exit_code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        finally:
            stdout_thread.join()
            stderr_thread.join()
            process.stdout.close()
            process.stderr.close()
        if errors:
            raise OSError("failed while recording agent output") from errors[0]
        for file in files.values():
            file.flush()
            os.fsync(file.fileno())
        metadata["exit_code"] = exit_code
        metadata["state"] = "complete" if exit_code == 0 else "agent_failed"
        return exit_code
    except BaseException as exc:
        metadata["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "capture_failed"
        metadata["error_type"] = type(exc).__name__
        raise
    finally:
        for file in files.values():
            file.close()
        metadata["finished_at"] = _now()
        atomic_json(metadata_path, metadata)
