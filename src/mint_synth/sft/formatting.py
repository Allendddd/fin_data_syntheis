from __future__ import annotations

from typing import Any, Mapping


def atomic_messages(task: Mapping[str, Any]) -> dict[str, Any]:
    trace = task.get("reasoning_trace") or []
    lines = [str(step.get("statement", "")).strip() for step in trace if isinstance(step, dict)]
    lines = [line for line in lines if line]
    answer = task.get("answer") or {}
    final = str(answer.get("text") or answer.get("value") or "").strip()
    if not final:
        raise ValueError("atomic answer is empty")
    response = "\n".join(lines + [f"Final answer: {final}"])
    return {"messages": [
        {"role": "system", "content": "Answer the financial question from the supplied context. Give an auditable derivation and final answer."},
        {"role": "user", "content": f"Context:\n{task['input']['context']}\n\nQuestion:\n{task['question']}"},
        {"role": "assistant", "content": response, "loss": True},
    ]}
