from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from mint_synth.harness.registry import ToolRegistry, default_tool_registry
from mint_synth.harness.session import HarnessSession, case_from_task
from mint_synth.harness.state import _json


def long_rollout(
    task: Mapping[str, Any], client: Any, index_path: Path,
    config: Mapping[str, Any], registry: ToolRegistry | None = None,
) -> tuple[dict[str, Any] | None, str]:
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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": session.system_prompt},
            {"role": "user", "content": case.query},
        ]
        for _ in range(session.max_steps):
            action = client.complete_json(
                system=session.system_prompt, user=_json(session.observe()),
                temperature=0.2, max_tokens=700,
            )
            if not isinstance(action, dict):
                return None, "teacher returned non-object"
            result = session.step(action)
            observation = result["observation"]
            if isinstance(observation, dict) and "error" in observation:
                return None, str(observation["error"])
            if "final" in action:
                verdict = session.grade()
                if not verdict["accepted"]:
                    return None, str(verdict["reason"])
                for index, relevant in enumerate(verdict["step_relevance"]):
                    messages[2 + 2 * index]["loss"] = relevant
                messages.append({"role": "assistant", "content": action["final"].strip(), "loss": True})
                return {"messages": messages, "tools": _json(session.tool_definitions)}, "accepted"
            entry = session.trace[-1]
            messages.append({"role": "tool_call", "content": _json({"name": entry["tool"], "arguments": entry["arguments"]}), "loss": True})
            messages.append({"role": "tool_response", "content": _json(observation), "loss": False})
        return None, "tool step limit exceeded"
