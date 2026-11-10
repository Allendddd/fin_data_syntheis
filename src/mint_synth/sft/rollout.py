from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from mint_synth.harness.registry import ToolRegistry, default_tool_registry
from mint_synth.harness.session import HarnessSession, case_from_task
from mint_synth.harness.state import _json
from mint_synth.agents.base import ModelSettings, RuntimeSettings


def long_rollout(
    task: Mapping[str, Any], client: Any, index_path: Path,
    config: Mapping[str, Any], registry: ToolRegistry | None = None,
    *, sample_key: str | None = None,
    runtime: RuntimeSettings | None = None, model: ModelSettings | None = None,
    trace_dir: Path | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if runtime is not None and runtime.name != "native":
        if model is None or trace_dir is None:
            raise ValueError("external teacher runtimes require model settings and a trace directory")
        from mint_synth.agents.collector import collect_runtime

        return collect_runtime(task, index_path, config, registry or default_tool_registry(),
                               model=model, runtime=runtime, trace_dir=trace_dir, sample_key=sample_key)
    try:
        sample, reason = _native_rollout(task, client, index_path, config, registry,
                                         sample_key=sample_key, trace_dir=trace_dir)
    except Exception as exc:
        _record_early_failure(trace_dir, task, f"native teacher failed ({type(exc).__name__})")
        raise
    _record_early_failure(trace_dir, task, reason)
    return sample, reason


def _record_early_failure(trace_dir: Path | None, task: Mapping[str, Any], reason: str) -> None:
    # Case/snapshot initialization can fail before a session exists. Every
    # candidate reference must still resolve to an explicit attempt result.
    if trace_dir is not None and not (trace_dir / "result.json").exists():
        from mint_synth.core.run_state import atomic_json

        trace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_json(trace_dir / "result.json", {"runtime": "native", "task_id": task.get("task_id"),
                                              "accepted": False, "reason": reason})


def _native_rollout(
    task: Mapping[str, Any], client: Any, index_path: Path, config: Mapping[str, Any],
    registry: ToolRegistry | None, *, sample_key: str | None, trace_dir: Path | None,
) -> tuple[dict[str, Any] | None, str]:
    from mint_synth.agents.collector import normalized_sample
    from mint_synth.core.run_state import atomic_json

    training = config["training"]
    cutoff = str(config["run"]["cutoff"])
    available_at_field = str(config["input"]["available_at_field"])
    try:
        case = case_from_task(task, index_path, cutoff=cutoff, available_at_field=available_at_field)
    except ValueError as exc:
        return None, str(exc)
    with HarnessSession(
        case, registry=registry or default_tool_registry(),
        max_steps=int(training["max_tool_steps"]) + 1,
        max_tool_calls=int(training["max_tool_steps"]),
        max_context_chars=int(training["max_context_chars"]),
        answer_tolerance=str(config["quality"]["answer_tolerance"]),
    ) as session:
        events: list[dict[str, Any]] = []
        reason = "tool step limit exceeded"
        accepted = False
        try:
            for _ in range(session.max_steps):
                user = _json(session.observe())
                action = client.complete_json(
                    system=session.system_prompt, user=user, temperature=0.2, max_tokens=2400,
                    **({"sample_key": sample_key} if sample_key is not None else {}),
                )
                events.append({"system": session.system_prompt, "user": user, "parsed_response": action})
                if not isinstance(action, dict):
                    reason = "teacher returned non-object"
                    return None, reason
                result = session.step(action)
                observation = result["observation"]
                if isinstance(observation, dict) and "error" in observation:
                    reason = str(observation["error"])
                    return None, reason
                if "final" in action:
                    snapshot = {"trace": session.trace, "system_prompt": session.system_prompt,
                                "tools": session.tool_definitions, "audit": session.audit_record()}
                    accepted = snapshot["audit"]["verdict"]["accepted"]
                    reason = "accepted" if accepted else str(snapshot["audit"]["verdict"]["reason"])
                    return (normalized_sample(snapshot, case.query) if accepted else None), reason
            return None, reason
        except Exception as exc:
            reason = f"native teacher failed ({type(exc).__name__})"
            raise
        finally:
            if trace_dir is not None:
                trace_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
                atomic_json(trace_dir / "model.parsed.json", events)
                atomic_json(trace_dir / "environment.json", {
                    "trace": session.trace, "done": session.done,
                    "audit": session.audit_record() if session.done else None,
                })
                atomic_json(trace_dir / "result.json", {
                    "runtime": "native", "task_id": task["task_id"], "accepted": accepted,
                    "reason": reason, "capture": "parsed_json_responses",
                })
