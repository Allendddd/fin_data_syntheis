"""Execute the SDK's ordinary tools in a separate workspace and process.

The child owns its cwd, SDK, and tool-consent environment. No financial tool
bridge or custom final-submission tool is constructed in this mode.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from mint_synth.core.run_state import atomic_json

from .base import ModelSettings, RuntimeResult, RuntimeSettings
from .cc_workspace import finish_workspace, prepare_workspace


NATIVE_TOOLS = frozenset({"file_read", "file_write", "shell", "python_repl"})


def _private_output(path: Path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _linux_descendants(parent: int) -> list[int]:
    """Include PTY children: the official shell tool starts another session."""
    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        return []
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            parents[int(entry.name)] = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    found, pending = [], [parent]
    while pending:
        current = pending.pop()
        children = [pid for pid, ppid in parents.items() if ppid == current]
        found.extend(children)
        pending.extend(children)
    return found


def _terminate_worker(process: subprocess.Popen) -> None:
    from .cli_runtime import _terminate_tree

    descendants = _linux_descendants(process.pid)
    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        _terminate_tree(process)
    finally:
        for pid in reversed(descendants):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_strands_native(case: Any, model: ModelSettings, settings: RuntimeSettings, trace_dir: Path) -> RuntimeResult:
    """Materialize public source files and let Strands finish in its own loop."""
    if settings.command or settings.extra_args:
        raise ValueError("command and extra_args are only supported by CLI runtimes")
    if not math.isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0:
        raise ValueError("Strands timeout must be finite and positive")
    if not math.isfinite(model.timeout_seconds) or model.timeout_seconds <= 0:
        raise ValueError("model timeout must be finite and positive")
    model.origin
    tools = tuple(settings.strands.tools)
    if not tools or len(set(tools)) != len(tools) or any(name not in NATIVE_TOOLS for name in tools):
        raise ValueError("Strands native tools must be unique supported official tool names")
    trace_dir = Path(trace_dir).resolve()
    trace_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        trace_dir.chmod(0o700)
    metadata = {"runtime": "strands", "strands_mode": "native", "tools": list(tools),
                "events_file": "agent.events.jsonl", "messages_file": "agent.messages.json",
                "native_final_file": "native-final.json", "worker_file": "native-worker.json"}
    started = time.monotonic()
    status, error, exit_code = "failed", None, None
    try:
        from .cli_runtime import _environment

        with tempfile.TemporaryDirectory(prefix="mint-strands-") as temporary:
            home = Path(temporary)
            workspace = home / "workspace"
            workspace.mkdir(mode=0o700)
            prompt = prepare_workspace(case, workspace, trace_dir, runtime="strands")
            environment = _environment(home, model)
            # These settings belong only to the isolated worker process. The
            # selected tools are explicitly enabled for unattended synthesis.
            environment.update({"STRANDS_NON_INTERACTIVE": "true", "BYPASS_TOOL_CONSENT": "true",
                                "SHELL_DEFAULT_TIMEOUT": str(max(1, int(min(model.timeout_seconds, settings.timeout_seconds))))})
            package_root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, (package_root, environment.get("PYTHONPATH"))))
            payload = {"trace_dir": str(trace_dir), "prompt": prompt, "tools": list(tools),
                       "model": {"model": model.model, "base_url": model.origin + "/v1",
                                 "temperature": model.temperature, "max_tokens": model.max_tokens,
                                 "timeout_seconds": model.timeout_seconds}}
            try:
                with _private_output(trace_dir / "worker.stdout.log") as stdout, _private_output(trace_dir / "worker.stderr.log") as stderr:
                    options = {"start_new_session": True} if os.name == "posix" else {
                        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
                    process = subprocess.Popen([sys.executable, "-m", "mint_synth.agents.strands_native_worker"],
                                               stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                               cwd=workspace, env=environment, **options)
                    try:
                        process.communicate(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                            timeout=settings.timeout_seconds)
                        exit_code = process.returncode
                    except subprocess.TimeoutExpired:
                        status, error = "timeout", "Strands exceeded the rollout deadline"
                        _terminate_worker(process)
                        exit_code = process.returncode
                    except BaseException:
                        _terminate_worker(process)
                        raise
            finally:
                finish_workspace(workspace, trace_dir)
            if status != "timeout":
                worker_path = trace_dir / "native-worker.json"
                if not worker_path.is_file():
                    error = "Strands native worker did not finish successfully"
                else:
                    worker = json.loads(worker_path.read_text(encoding="utf-8"))
                    status, error = worker["status"], worker.get("error")
                    if status not in {"completed", "failed", "incomplete"}:
                        raise ValueError("Strands worker returned an unsupported status")
                    if exit_code != 0 and status == "completed":
                        raise ValueError("Strands worker exited with an error after its final reply")
                    if status == "completed" and not (trace_dir / "native-final.json").is_file():
                        raise ValueError("Strands worker omitted its final reply")
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        status, error = "failed", f"Strands native runtime failed ({type(exc).__name__})"
    metadata.update(exit_code=exit_code, elapsed_seconds=round(time.monotonic() - started, 3))
    atomic_json(trace_dir / "strands.json", {**metadata, "status": status, "error": error})
    return RuntimeResult(status, error, metadata)
