from __future__ import annotations

from typing import Any

from mint_synth.synthesis.audit import audit_atomic, replay_summary
from mint_synth.inference.llm import JsonLLM
from mint_synth.inference.prompts import ATOMIC_SYSTEM, AUDIT_SYSTEM, atomic_audit_user, atomic_user
from mint_synth.synthesis.replay import ReplayError, replay_graph
from mint_synth.core.util import stable_id


class AtomicSynthesizer:
    def __init__(
        self,
        constructor: JsonLLM,
        auditor: JsonLLM,
        *,
        language: str = "auto",
        answer_tolerance: str = "1e-6",
    ):
        self.constructor = constructor
        self.auditor = auditor
        self.language = language
        self.answer_tolerance = answer_tolerance

    def synthesize(
        self,
        chunk: dict[str, Any],
        facts: list[dict[str, Any]],
        capability: str,
        *,
        candidate_index: int = 0,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        response = self.constructor.complete_json(
            system=ATOMIC_SYSTEM,
            user=atomic_user(chunk, facts, capability, self.language),
            temperature=0.4,
            max_tokens=5000,
        )
        candidate = response.get("task") if isinstance(response, dict) else None
        if not isinstance(candidate, dict):
            return None, self._rejection(chunk, capability, response.get("reason", "constructor returned no task") if isinstance(response, dict) else "invalid constructor response", response)

        task = dict(candidate)
        task["task_id"] = stable_id(
            "atom",
            chunk["chunk_id"],
            capability,
            candidate_index,
            task.get("question", ""),
        )
        task["task_type"] = "atomic"
        task["capability"] = capability
        task["input"] = {"context": chunk["text"], "question": task.get("question", "")}
        task["source_id"] = chunk["source_id"]
        task["chunk_id"] = chunk["chunk_id"]
        fact_index = {fact["fact_id"]: fact for fact in facts}

        try:
            replay = replay_graph(task.get("evidence_graph", {}), fact_index)
        except ReplayError as exc:
            return None, self._rejection(chunk, capability, f"pre-audit replay failed: {exc}", task)
        evidence = [fact_index[fact_id] for fact_id in task.get("evidence_ids", []) if fact_id in fact_index]
        audit_response = self.auditor.complete_json(
            system=AUDIT_SYSTEM,
            user=atomic_audit_user(task, evidence, replay_summary(replay)),
            temperature=0.0,
            max_tokens=1200,
        )
        task["semantic_audit"] = audit_response if isinstance(audit_response, dict) else {}
        result = audit_atomic(task, fact_index, answer_tolerance=self.answer_tolerance)
        task["verification"] = {
            "accepted": result.accepted,
            "reasons": list(result.reasons),
            "replay": replay_summary(result.replay) if result.replay else None,
        }
        if not result.accepted:
            return None, self._rejection(chunk, capability, "; ".join(result.reasons), task)
        return task, None

    @staticmethod
    def _rejection(
        chunk: dict[str, Any], capability: str, reason: str, candidate: Any
    ) -> dict[str, Any]:
        return {
            "stage": "atomic_task",
            "source_id": chunk.get("source_id"),
            "chunk_id": chunk.get("chunk_id"),
            "capability": capability,
            "reason": reason,
            "candidate": candidate,
        }


def atomic_to_sft(task: dict[str, Any]) -> dict[str, Any]:
    reasoning = task.get("reasoning_trace", [])
    answer = task.get("answer", {})
    trace_lines = [str(step.get("statement", "")).strip() for step in reasoning if isinstance(step, dict)]
    response = "\n".join(f"{index + 1}. {line}" for index, line in enumerate(trace_lines) if line)
    final_text = str(answer.get("text") or answer.get("value") or "").strip()
    if response:
        response += "\n\n"
    response += f"Final answer: {final_text}"
    return {
        "messages": [
            {
                "role": "system",
                "content": "Answer the financial question using only the supplied context. Show a complete, auditable derivation and then give the final answer.",
            },
            {
                "role": "user",
                "content": f"Context:\n{task['input']['context']}\n\nQuestion:\n{task['question']}",
            },
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "task_id": task["task_id"],
            "task_type": "atomic",
            "capability": task["capability"],
            "evidence_ids": task["evidence_ids"],
        },
    }
