from __future__ import annotations

from typing import Any, Mapping


def atomic_messages(task: Mapping[str, Any], *, response: str | None = None) -> dict[str, Any]:
    teacher_response = response is not None
    trace = task.get("reasoning_trace") or []
    lines = [str(step.get("statement", "")).strip() for step in trace if isinstance(step, dict)]
    lines = [line for line in lines if line]
    answer = task.get("answer") or {}
    final_value = answer.get("text")
    if final_value is None or final_value == "":
        final_value = answer.get("value")
    final = str(final_value).strip() if final_value is not None else ""
    if not final:
        raise ValueError("atomic answer is empty")
    if response is None:
        response = "\n".join(lines + [f"Final answer: {final}"])
    return {"messages": [
        {"role": "system", "content": (
            "Answer the financial question from the supplied context with a complete JSON evidence and calculation proof."
            if teacher_response else
            "Answer the financial question from the supplied context. Give an auditable derivation and final answer."
        )},
        {"role": "user", "content": f"Context:\n{task['input']['context']}\n\nQuestion:\n{task['question']}"},
        {"role": "assistant", "content": response, "loss": True},
    ]}
