"""Collect raw runtime traces and publish only verified domain trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import atomic_json
from mint_synth.harness.registry import ToolRegistry
from mint_synth.harness.session import HarnessSession, case_from_task
from mint_synth.trace_proxy import TraceProxyConfig, TraceProxyServer

from .base import ModelSettings, RuntimeResult, RuntimeSettings
from .bridge import FinanceToolBridge


def normalized_sample(snapshot: Mapping[str, Any], query: str) -> dict[str, Any]:
    """Canonical SFT actions; exact SDK/API messages live in the raw artifacts."""
    verdict = snapshot["audit"]["verdict"]
    flags = iter(verdict["step_relevance"])
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": snapshot["system_prompt"], "loss": False},
        {"role": "user", "content": query, "loss": False},
    ]
    for entry in snapshot["trace"]:
        action = entry["action"]
        if "final" in action:
            messages.append({"role": "assistant", "content": action["final"].strip(), "loss": True})
        else:
            messages.extend([
                {"role": "tool_call", "content": json.dumps({"name": entry["tool"], "arguments": entry["arguments"]}, ensure_ascii=False, separators=(",", ":")),
                 "loss": next(flags),},
                {"role": "tool_response", "content": json.dumps(entry["observation"], ensure_ascii=False, separators=(",", ":")), "loss": False},
            ])
    return {"messages": messages, "tools": json.dumps(snapshot["tools"], ensure_ascii=False, separators=(",", ":"))}


def collect_runtime(
    task: Mapping[str, Any], index_path: Path, config: Mapping[str, Any], registry: ToolRegistry,
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
    try:
        case = case_from_task(task, index_path, cutoff=str(config["run"]["cutoff"]),
                              available_at_field=str(config["input"]["available_at_field"]))
        training = config["training"]
        with FinanceToolBridge(lambda: HarnessSession(
            case, registry=registry, max_steps=int(training["max_tool_steps"]) + 1,
            max_tool_calls=int(training["max_tool_steps"]),
            max_context_chars=int(training["max_context_chars"]),
            answer_tolerance=str(config["quality"]["answer_tolerance"]),
        ), call_timeout=min(model.timeout_seconds, runtime.timeout_seconds)) as bridge:
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
            snapshot = bridge.snapshot()
        verdict = snapshot.get("audit", {}).get("verdict", {})
        accepted = result.status == "completed" and bool(verdict.get("accepted"))
        reason = "accepted" if accepted else result.error or verdict.get("reason") or "agent ended without submit_final"
        return (normalized_sample(snapshot, case.query) if accepted else None), reason
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
            "format_version": 1, "run_id": run_id, "task_id": task["task_id"], "runtime": runtime.name,
            "model": model.model, "status": result.status, "accepted": accepted, "reason": reason,
            "runtime_metadata": result.metadata, "model_calls": "model/calls",
            "sample_format": "normalized_financial_actions",
        })
