"""Collect raw runtime traces and publish only verified domain trajectories."""

from __future__ import annotations

import hashlib
import os
import threading
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import atomic_json
from mint_synth.harness.registry import ToolRegistry
from mint_synth.harness.session import HarnessSession, case_from_task
from mint_synth.trace_proxy import TraceProxyConfig, TraceProxyServer

from .base import ModelSettings, RuntimeResult, RuntimeSettings
from .bridge import FinanceToolBridge
from .trajectory import normalized_sample
from .model_trace import load_model_turns
from .cc_native import export_cc_native
from .cc_workspace import audit_native
from .native_capture import audit_native_capture, export_native_capture


def collect_runtime(
    task: Mapping[str, Any], index_path: Path, config: Mapping[str, Any], registry: ToolRegistry | None,
    *, model: ModelSettings, runtime: RuntimeSettings, trace_dir: Path, sample_key: str | None,
) -> tuple[dict[str, Any] | None, str]:
    trace_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.name == "posix":
        trace_dir.chmod(0o700)
    run_id = hashlib.sha256((sample_key or str(task["task_id"])).encode()).hexdigest()
    result = RuntimeResult("failed", "runtime did not start")
    snapshot: dict[str, Any] = {}
    reason = "runtime did not start"
    accepted = False
    sample = None
    try:
        case = case_from_task(task, index_path, cutoff=str(config["run"]["cutoff"]),
                              available_at_field=str(config["input"]["available_at_field"]))
        training = config["training"]
        claude_native = runtime.name == "claude_code" and runtime.native_tools
        context = nullcontext(case) if runtime.native_tools else FinanceToolBridge(lambda: HarnessSession(
            case, registry=registry, max_steps=int(training["max_tool_steps"]) + 1,
            max_tool_calls=int(training["max_tool_steps"]),
            max_context_chars=int(training["max_context_chars"]),
            answer_tolerance=str(config["quality"]["answer_tolerance"]),
        ), call_timeout=min(model.timeout_seconds, runtime.timeout_seconds))
        with context as bridge:
            proxy = TraceProxyServer(TraceProxyConfig(upstream=model.origin, output_dir=trace_dir / "model",
                                                     run_id=run_id, port=0))
            thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            thread.start()
            try:
                traced_model = replace(model, base_url=f"http://127.0.0.1:{proxy.server_port}/v1")
                if runtime.name == "strands":
                    from .strands_runtime import run_strands
                    result = run_strands(bridge, traced_model, runtime, trace_dir)
                elif runtime.name in {"codex", "claude_code"}:
                    from .cli_runtime import run_cli
                    result = run_cli(bridge, traced_model, runtime, trace_dir)
                else:
                    raise ValueError(f"unsupported external runtime: {runtime.name}")
            finally:
                proxy.shutdown()
                idle = proxy.wait_idle(timeout=2)
                result.metadata["model_capture_complete"] = idle
                if not idle and result.status == "completed":
                    result = RuntimeResult("failed", "model capture was still active when the agent ended", result.metadata)
                proxy.server_close()
                thread.join(timeout=5)
            if runtime.native_tools:
                if result.status == "completed":
                    if claude_native:
                        snapshot = audit_native(case, trace_dir, tolerance=str(config["quality"]["answer_tolerance"]))
                    else:
                        snapshot = audit_native_capture(case, trace_dir, runtime.name, tolerance=str(config["quality"]["answer_tolerance"]))
            else:
                snapshot = bridge.snapshot()
        verdict = snapshot.get("audit", {}).get("verdict", {})
        if result.status == "completed" and bool(verdict.get("accepted")):
            if claude_native:
                sample = export_cc_native(snapshot, trace_dir)
            elif runtime.native_tools:
                sample = export_native_capture(snapshot, trace_dir, runtime.name)
            else:
                turns = load_model_turns(trace_dir, {item["function"]["name"] for item in snapshot["tools"]})
                sample = normalized_sample(snapshot, case.query, cutoff=case.cutoff, model_turns=turns,
                                           reasoning_format=config["training"].get("reasoning_format", "plain"),
                                           trace_dir=trace_dir)
            accepted = True
        reason = "accepted" if accepted else result.error or verdict.get("reason") or "agent ended without a verified final reply"
        return sample, reason
    except Exception as exc:
        # Keep details in the private artifact; credentials are never part of the
        # model settings saved here, nor included in a rejected dataset row.
        detail = str(exc)
        if model.api_key:
            detail = detail.replace(model.api_key, "[REDACTED]")
        reason = f"{type(exc).__name__}: {detail}"
        result = RuntimeResult("failed", reason)
        return None, reason
    finally:
        atomic_json(trace_dir / "environment.json", snapshot)
        atomic_json(trace_dir / "result.json", {
            "format_version": 2, "run_id": run_id, "task_id": task["task_id"], "runtime": runtime.name,
            "model": model.model, "status": result.status, "accepted": accepted, "reason": reason,
            "claude_code_mode": runtime.claude_code.mode if runtime.name == "claude_code" else None,
            "tool_mode": runtime.tool_mode,
            "runtime_metadata": result.metadata, "model_calls": "model/calls",
            "sample_format": sample.get("sample_format", "teacher_interactions_v2") if sample else None,
            "capture": "complete_model_exchanges_v2",
            "trajectory_id": sample["trajectory_id"] if sample else None,
            "trajectory": "trajectory.json" if sample else None,
            "sample_count": sample.get("sample_count", 1) if sample else 0,
        })
