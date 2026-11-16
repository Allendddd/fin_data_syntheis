"""Convert captured model content to SFT and bind every action to execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import atomic_json
from mint_synth.harness.state import tool_call_system


class TrajectoryError(ValueError):
    pass


def normalized_sample(
    snapshot: Mapping[str, Any], query: str, *, cutoff: str,
    model_turns: list[dict[str, Any]], reasoning_format: str = "plain", trace_dir: Path | None = None,
) -> dict[str, Any]:
    """Preserve returned text; environment records only authenticate actions."""
    if reasoning_format not in {"plain", "think"}:
        raise TrajectoryError("reasoning_format must be plain or think")
    if not model_turns:
        raise TrajectoryError("complete model turns are missing; tool-only export is disabled")
    verdict = snapshot["audit"]["verdict"]
    flags = iter(verdict["step_relevance"])
    relevance = {entry["step"]: next(flags) for entry in snapshot["trace"] if entry["tool"]}
    executed = iter(enumerate(snapshot["trace"], 1))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": tool_call_system(cutoff), "loss": False},
        {"role": "user", "content": query, "loss": False},
    ]
    origins = []
    final_seen = False

    def append(message: dict[str, Any], turn: Mapping[str, Any], part: Mapping[str, Any], step=None,
               *, observation: bool = False):
        sdk_error = observation and step is None
        origin = {"message_index": len(messages), "call_id": turn["call_id"],
                  "source": part["observation_source"] if sdk_error else "environment.json" if observation else turn["source"],
                  "field": part["observation_field"] if sdk_error else f"trace[{step - 1}].observation" if observation else part["field"],
                  "kind": "observation" if observation else part["kind"], "environment_step": step}
        if not observation:
            origin["response_file"] = turn.get("response_file", turn["source"])
            if "segment" in part:
                origin["segment"] = part["segment"]
        origins.append(origin)
        messages.append(message)

    action_offset = 0
    for turn in model_turns:
        pending_observations = []
        entries, sdk_errors = [], set()
        for index, part in enumerate(turn["assistant_parts"]):
            if part["kind"] not in {"tool_call", "final"}:
                continue
            expected = snapshot["trace"][action_offset] if action_offset < len(snapshot["trace"]) else None
            if ("failed_observation" in part and (expected is None or expected["tool"] != part["name"]
                    or expected["arguments"] != part["arguments"])):
                sdk_errors.add(index)
            elif expected is not None:
                entries.append(expected)
                action_offset += 1
        # Mixed responses retain good calls, but their shared reasoning is not
        # attributed to a particular call and is conservatively masked.
        text_loss = not sdk_errors and all(relevance.get(entry["step"], True) for entry in entries)
        for index, part in enumerate(turn["assistant_parts"]):
            kind = part["kind"]
            if kind in {"text", "reasoning", "reasoning_summary", "memory"}:
                if final_seen:
                    raise TrajectoryError("model text after final submission cannot be bound to an active episode")
                if kind == "memory":
                    content = "<memory>" + json.dumps(part["value"], ensure_ascii=False, separators=(",", ":")) + "</memory>"
                else:
                    content = part["text"]
                    if not isinstance(content, str):
                        raise TrajectoryError("assistant text must be a string")
                    if not content:
                        continue
                    if kind == "reasoning" and reasoning_format == "think" and "<think>" not in content:
                        content = "<think>\n" + content + "\n</think>"
                append({"role": "assistant", "content": content, "loss": text_loss}, turn, part)
                continue
            if kind not in {"tool_call", "final"}:
                raise TrajectoryError(f"unsupported model content kind: {kind}")
            if index in sdk_errors:
                append({"role": "tool_call", "content": json.dumps({"name": part["name"],
                        "arguments": part["arguments"]}, ensure_ascii=False), "loss": False}, turn, part)
                pending_observations.append((part, None, part["failed_observation"]))
                continue
            try:
                step, entry = next(executed)
            except StopIteration as exc:
                raise TrajectoryError("model action has no matching environment execution") from exc
            action = entry["action"]
            if kind == "final":
                if pending_observations:
                    raise TrajectoryError("final proof batched with unobserved tool calls cannot use the final-proof protocol")
                proof = part.get("text", part.get("proof"))
                proof = proof if isinstance(proof, str) else json.dumps(proof, ensure_ascii=False)
                if "final" not in action or proof != action["final"]:
                    try:
                        same_legacy = json.loads(proof) == json.loads(action.get("final", ""))
                    except (TypeError, ValueError):
                        same_legacy = False
                    if not same_legacy:
                        raise TrajectoryError("model final response differs from the verified environment submission")
                append({"role": "assistant", "content": proof, "loss": True}, turn, part, step)
                final_seen = True
            else:
                if (final_seen or "final" in action or part["name"] != entry["tool"]
                        or part["arguments"] != entry["arguments"]):
                    raise TrajectoryError("captured model tool call differs from the environment execution")
                append({"role": "tool_call", "content": json.dumps({"name": part["name"],
                        "arguments": part["arguments"]}, ensure_ascii=False, separators=(",", ":")),
                        "loss": relevance[entry["step"]]}, turn, part, step)
                pending_observations.append((part, step, entry["observation"]))
        # All content in a model response precedes execution observations,
        # including text following a tool call and parallel calls in one turn.
        for part, step, observation in pending_observations:
            content = observation if step is None and isinstance(observation, str) else json.dumps(
                observation, ensure_ascii=False, separators=(",", ":"))
            append({"role": "tool_response", "content": content, "loss": False}, turn, part, step, observation=True)
    if next(executed, None) is not None or not final_seen:
        raise TrajectoryError("captured model trajectory does not cover the complete verified episode")
    # ms-swift combines adjacent calls using the first call's loss flag.
    # Refuse mixed masks instead of accidentally supervising irrelevant calls.
    group = []
    for message in [*messages, {"role": "end"}]:
        if message["role"] == "tool_call":
            group.append(message["loss"])
        else:
            if len(set(group)) > 1:
                raise TrajectoryError("batched tool calls have mixed loss masks; use sequential calls for ms-swift")
            group = []
    tools = json.dumps(snapshot["tools"], ensure_ascii=False, separators=(",", ":"))
    trajectory_id = hashlib.sha256(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False,
                                              sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if trace_dir is not None:
        atomic_json(trace_dir / "trajectory.json", {
            "format_version": 2, "trajectory_id": trajectory_id, "reasoning_format": reasoning_format,
            "model_turns": model_turns, "message_origins": origins,
        })
    return {"messages": messages, "tools": tools, "trajectory_id": trajectory_id}
