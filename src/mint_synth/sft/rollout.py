from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from mint_synth.harness.registry import ToolRegistry, default_tool_registry
from mint_synth.harness.session import HarnessSession, case_from_task
from mint_synth.harness.state import _json
from mint_synth.agents.base import ModelSettings, RuntimeSettings
from mint_synth.inference.exchange import EXCHANGE_VERSION
from mint_synth.inference.llm import LLMExchangeError
from mint_synth.agents.trajectory import TrajectoryError


def long_rollout(
    task: Mapping[str, Any], client: Any, index_path: Path,
    config: Mapping[str, Any], registry: ToolRegistry | None = None,
    *, sample_key: str | None = None,
    runtime: RuntimeSettings | None = None, model: ModelSettings | None = None,
    trace_dir: Path | None = None, verifier: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    if runtime is not None and runtime.name != "native":
        if model is None or trace_dir is None:
            raise ValueError("external teacher runtimes require model settings and a trace directory")
        from mint_synth.agents.collector import collect_runtime

        from mint_synth.harness.registry import load_tool_registry
        tools = registry if runtime.native_tools else registry or load_tool_registry(retrieval=config.get("harness", {}).get("retrieval"))
        return collect_runtime(task, index_path, config, tools,
                               model=model, runtime=runtime, trace_dir=trace_dir, sample_key=sample_key, verifier=verifier)
    try:
        sample, reason = _native_rollout(task, client, index_path, config, registry,
                                         sample_key=sample_key, trace_dir=trace_dir, verifier=verifier)
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
    registry: ToolRegistry | None, *, sample_key: str | None, trace_dir: Path | None, verifier: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    from mint_synth.agents.collector import normalized_sample
    from mint_synth.core.run_state import atomic_json

    training = config["training"]
    cutoff = str(config["run"]["cutoff"])
    available_at_field = str(config["input"]["available_at_field"])
    try:
        case = case_from_task(task, index_path, cutoff=cutoff, available_at_field=available_at_field,
                              retrieval=config.get("harness", {}).get("retrieval"),
                              archive_path=trace_dir / "sources" if trace_dir is not None else None)
    except ValueError as exc:
        return None, str(exc)
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    with HarnessSession(
        case, registry=registry,
        max_steps=int(training["max_tool_steps"]) + 1,
        max_tool_calls=int(training["max_tool_steps"]),
        max_context_chars=int(training["max_context_chars"]),
        answer_tolerance=str(config["quality"]["answer_tolerance"]),
        solution_reviewer=verifier, sample_key=sample_key,
    ) as session:
        events: list[dict[str, Any]] = []
        model_turns: list[dict[str, Any]] = []
        reason = "tool step limit exceeded"
        accepted = False
        sample = None
        try:
            complete = getattr(client, "complete_json_exchange", None)
            if not callable(complete):
                reason = "teacher client only returns parsed actions; complete exchange capture is required"
                return None, reason
            for step in range(session.max_steps):
                user = _json(session.observe())
                source = f"model/calls/{step + 1:06d}.json"
                try:
                    exchange = complete(
                        system=session.system_prompt, user=user, temperature=0.2, max_tokens=2400,
                        native_action=True, **({"sample_key": sample_key} if sample_key is not None else {}),
                    )
                except LLMExchangeError as exc:
                    if trace_dir is not None:
                        atomic_json(trace_dir / source, exc.exchange)
                    raise
                if trace_dir is not None:
                    atomic_json(trace_dir / source, exchange)
                if (exchange.get("format_version") != EXCHANGE_VERSION
                        or not {"response", "request", "assistant_parts", "call_id"} <= exchange.keys()):
                    reason = "incomplete or legacy teacher exchange; tool-only export is disabled"
                    return None, reason
                action = exchange["parsed_response"]
                model_turns.append({"call_id": exchange["call_id"], "source": source,
                                    "assistant_parts": exchange["assistant_parts"],
                                    "cache_hit": exchange.get("cache_hit", False)})
                events.append({"system": session.system_prompt, "user": user, "parsed_response": action,
                               "source": source, "call_id": exchange["call_id"]})
                if not isinstance(action, dict):
                    reason = "teacher returned non-object"
                    return None, reason
                result = session.step(action)
                session.record_model_text("\n\n".join(
                    part["text"] for turn in model_turns for part in turn["assistant_parts"]
                    if part.get("kind") in {"text", "reasoning", "reasoning_summary"} and isinstance(part.get("text"), str)))
                observation = result["observation"]
                if isinstance(observation, dict) and "error" in observation:
                    reason = str(observation["error"])
                    if session.done:
                        return None, reason
                    continue
                if "final" in action:
                    snapshot = {"trace": session.trace, "system_prompt": session.system_prompt,
                                "tools": session.tool_definitions, "audit": session.audit_record()}
                    if snapshot["audit"]["verdict"]["accepted"]:
                        sample = normalized_sample(
                            snapshot, case.query, cutoff=case.cutoff, model_turns=model_turns,
                            reasoning_format=training.get("reasoning_format", "plain"), trace_dir=trace_dir,
                        )
                        accepted, reason = True, "accepted"
                    else:
                        reason = str(snapshot["audit"]["verdict"]["reason"])
                    return sample, reason
            return None, reason
        except Exception as exc:
            reason = str(exc) if isinstance(exc, TrajectoryError) else f"native teacher failed ({type(exc).__name__})"
            raise
        finally:
            if trace_dir is not None:
                atomic_json(trace_dir / "model.parsed.json", events)
                atomic_json(trace_dir / "environment.json", {
                    "trace": session.trace, "done": session.done,
                    "audit": session.audit_record() if session.done else None,
                })
                atomic_json(trace_dir / "result.json", {
                    "format_version": 2,
                    "runtime": "native", "task_id": task["task_id"], "accepted": accepted,
                    "status": "completed" if accepted else "failed",
                    "reason": reason, "capture": "complete_model_exchanges_v2",
                    "trajectory_id": sample["trajectory_id"] if sample else None,
                    "model_call_count": len(model_turns), "trajectory": "trajectory.json" if sample else None,
                })
