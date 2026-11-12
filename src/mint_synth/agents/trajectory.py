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
    executed = iter(enumerate(snapshot["trace"], 1))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": tool_call_system(cutoff), "loss": False},
        {"role": "user", "content": query, "loss": False},
    ]
    origins = []
    final_seen = False

    def append(message: dict[str, Any], turn: Mapping[str, Any], part: Mapping[str, Any], step=None,
               *, observation: bool = False):
        origin = {"message_index": len(messages), "call_id": turn["call_id"],
                  "source": "environment.json" if observation else turn["source"],
                  "field": f"trace[{step - 1}].observation" if observation else part["field"],
                  "kind": "observation" if observation else part["kind"], "environment_step": step}
        if not observation:
            origin["response_file"] = turn.get("response_file", turn["source"])
            if "segment" in part:
                origin["segment"] = part["segment"]
        origins.append(origin)
        messages.append(message)

    for turn in model_turns:
        pending_observations = []
        for part in turn["assistant_parts"]:
            kind = part["kind"]
            if kind in {"text", "reasoning", "reasoning_summary", "memory"}:
                if final_seen:
                    raise TrajectoryError("model text after final submission cannot satisfy the final-proof protocol")
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
                append({"role": "assistant", "content": content, "loss": True}, turn, part)
                continue
            if kind not in {"tool_call", "final"}:
                raise TrajectoryError(f"unsupported model content kind: {kind}")
            try:
                step, entry = next(executed)
            except StopIteration as exc:
                raise TrajectoryError("model action has no matching environment execution") from exc
            action = entry["action"]
            if kind == "final":
                if pending_observations:
                    raise TrajectoryError("final proof batched with unobserved tool calls cannot use the final-proof protocol")
                proof = part["proof"]
                proof = proof if isinstance(proof, str) else json.dumps(proof, ensure_ascii=False)
                if "final" not in action or json.loads(proof) != json.loads(action["final"]):
                    raise TrajectoryError("model final proof differs from the verified environment submission")
                append({"role": "assistant", "content": proof, "loss": True}, turn, part, step)
                final_seen = True
            else:
                if (final_seen or "final" in action or part["name"] != entry["tool"]
                        or part["arguments"] != entry["arguments"]):
                    raise TrajectoryError("captured model tool call differs from the environment execution")
                append({"role": "tool_call", "content": json.dumps({"name": part["name"],
                        "arguments": part["arguments"]}, ensure_ascii=False, separators=(",", ":")),
                        "loss": next(flags)}, turn, part, step)
                pending_observations.append((part, step, entry["observation"]))
        # All content in a model response precedes execution observations,
        # including text following a tool call and parallel calls in one turn.
        for part, step, observation in pending_observations:
            append({"role": "tool_response", "content": json.dumps(observation,
                    ensure_ascii=False, separators=(",", ":")), "loss": False}, turn, part, step, observation=True)
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
