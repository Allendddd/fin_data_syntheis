"""Reusable, stateful financial harness for teacher and external rollouts.

The model-facing interface is observe/step. Hidden grading stays trainer-side.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mint_synth.harness.registry import ToolRegistry, default_tool_registry
from mint_synth.harness.snapshot import SnapshotTools, published_date
from mint_synth.harness.state import harness_system, visible_state
from mint_synth.harness.verifier import _answer_ok
from mint_synth.core.provenance import sha256_file


@dataclass(frozen=True)
class HarnessCase:
    query_id: str
    query: str
    snapshot_path: Path
    cutoff: str
    entity_id: str
    dated_only: bool
    task: Mapping[str, Any] = field(repr=False)
    toolset_fingerprint: str | None = None
    available_at_field: str = "published_at"


def case_from_task(
    task: Mapping[str, Any], snapshot_path: str | Path, *, cutoff: str,
    available_at_field: str = "published_at", query_id: str | None = None,
    query: str | None = None, toolset_fingerprint: str | None = None,
) -> HarnessCase:
    """Check a hidden task before exposing its query to a rollout."""
    evidence = task.get("evidence") or []
    if not isinstance(evidence, list) or not all(isinstance(fact, Mapping) for fact in evidence):
        raise ValueError("task evidence must be a list of records")
    required_chunks = {str(fact.get("chunk_id", "")) for fact in evidence}
    if not required_chunks or "" in required_chunks:
        raise ValueError("missing evidence chunk IDs")
    expected_pack = {(str(fact.get("source_id") or ""), str(fact.get("chunk_id") or "")) for fact in evidence}
    source_pack = task.get("source_pack")
    if source_pack is not None and (
        not isinstance(source_pack, list) or not all(isinstance(item, Mapping) for item in source_pack)
    ):
        raise ValueError("source pack must be a list of records")
    declared_pack = {
        (str(item.get("source_id") or ""), str(item.get("chunk_id") or ""))
        for item in (source_pack or [])
    }
    if not expected_pack or (source_pack is not None and expected_pack != declared_pack):
        raise ValueError("source pack differs from the task evidence")
    dated_only = bool(task.get("task_spec", {}).get("time_sensitive"))
    for fact in evidence:
        available_at = published_date(fact.get("source_metadata") or {}, available_at_field)
        if dated_only and not available_at:
            raise ValueError("time-sensitive task has undated evidence")
        if available_at and available_at > cutoff:
            raise ValueError("task evidence is after cutoff")
    entity_id = str(task.get("canonical_entity_id") or "")
    if not entity_id:
        raise ValueError("task has no canonical entity ID")
    if task.get("cutoff") and str(task["cutoff"]) != cutoff:
        raise ValueError("task cutoff differs from the frozen snapshot")
    if query_id and str(task.get("task_id") or "") != query_id:
        raise ValueError("query ID differs from the hidden task")
    text = str(query if query is not None else task.get("question") or "").strip()
    if not text or text != str(task.get("question") or "").strip():
        raise ValueError("query differs from the hidden validated task")
    case_id = str(query_id or task.get("task_id") or "")
    if not case_id:
        raise ValueError("task has no query ID")
    path = Path(snapshot_path).resolve()
    if not path.is_file():
        raise ValueError(f"snapshot index is missing: {path}")
    return HarnessCase(
        query_id=case_id, query=text,
        snapshot_path=path, cutoff=cutoff, entity_id=entity_id,
        dated_only=dated_only, task=task, toolset_fingerprint=toolset_fingerprint,
        available_at_field=available_at_field,
    )


class HarnessSession:
    """Public actions and observations; grading is a separate trainer-side call."""

    def __init__(
        self, case: HarnessCase, *, registry: ToolRegistry | None = None,
        max_steps: int = 17, max_tool_calls: int | None = None,
        max_context_chars: int = 24000,
        answer_tolerance: str = "1e-6",
        allow_toolset_override: bool = False,
    ) -> None:
        if max_steps < 1 or max_context_chars < 1:
            raise ValueError("harness limits must be positive")
        if max_tool_calls is not None and max_tool_calls < 1:
            raise ValueError("tool call limit must be positive")
        self.case = case
        self.registry = (registry or default_tool_registry()).copy()
        if case.toolset_fingerprint and self.registry.fingerprint != case.toolset_fingerprint and not allow_toolset_override:
            raise ValueError("session toolset differs from the query export")
        self.allow_toolset_override = allow_toolset_override
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls if max_tool_calls is not None else max_steps - 1
        self.max_context_chars = max_context_chars
        self.answer_tolerance = answer_tolerance
        self.tools: SnapshotTools | None = None
        self._trace: list[dict[str, Any]] = []
        self._steps = 0
        self._tool_calls = 0
        self._done = False
        self._final: str | None = None
        self._ledger: list[dict[str, Any]] = []
        self._memory: dict[str, Any] = {}
        self._invalid_actions = 0
        self._snapshot_hash: str | None = None
        self.reset()

    def __enter__(self) -> HarnessSession:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.tools is not None:
            self.tools.close()
            self.tools = None

    def reset(self) -> dict[str, Any]:
        self.close()
        self.tools = SnapshotTools(
            self.case.snapshot_path, self.case.cutoff,
            entity_id=self.case.entity_id, dated_only=self.case.dated_only,
        )
        self._trace = []
        self._steps = 0
        self._tool_calls = 0
        self._done = False
        self._final = None
        self._ledger = []
        self._memory = {}
        self._invalid_actions = 0
        try:
            return self.observe()
        except BaseException:
            self.close()
            raise

    @property
    def done(self) -> bool:
        return self._done

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        return self.registry.definitions

    @property
    def system_prompt(self) -> str:
        return harness_system(self.case.cutoff, self.registry.names)

    @property
    def trace(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._trace)

    @property
    def evidence_ledger(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._ledger)

    @property
    def working_memory(self) -> dict[str, Any]:
        return copy.deepcopy(self._memory)

    @property
    def failed_paths(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(entry) for entry in self._trace
            if isinstance(entry.get("observation"), dict)
            and ("error" in entry["observation"]
                 or (entry.get("tool") == "search_corpus" and not entry["observation"].get("hits")))
        ]

    def observe(self) -> dict[str, Any]:
        if self.tools is None:
            raise RuntimeError("harness session is closed")
        visible = visible_state(
            {"question": self.case.query}, self.case.cutoff, self.tools,
            self._trace, self.max_context_chars, self.registry.definitions,
            evidence_ledger=self._ledger, working_memory=self._memory,
            budgets={"steps_remaining": self.max_steps - self._steps,
                     "tool_calls_remaining": self.max_tool_calls - self._tool_calls},
        )
        if visible is None:
            raise RuntimeError("context limit exceeded")
        return json.loads(visible)

    def step(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a model action; never return hidden answer or verifier feedback."""
        if self.tools is None:
            raise RuntimeError("harness session is closed")
        if self._done:
            raise RuntimeError("harness episode is finished; call reset for a new rollout")
        self._steps += 1
        prior_evidence_count = len(self._ledger)
        prior_memory = copy.deepcopy(self._memory)
        prior_seen = self.tools.seen_search.copy()
        prior_opened = self.tools.opened.copy()
        raw_action = copy.deepcopy({
            key: value for key, value in action.items()
            if key in {"tool", "arguments", "final", "memory"}
        }) if isinstance(action, Mapping) else {"invalid_action": "non-object action"}
        if isinstance(action, Mapping) and action.get("invalid_action") == "non-JSON action":
            raw_action = {"invalid_action": "non-JSON action"}
        if isinstance(action, Mapping) and set(action) - {"tool", "arguments", "final", "memory"}:
            raw_action["__invalid_fields__"] = True
        try:
            json.dumps(raw_action, ensure_ascii=False, allow_nan=False)
            non_json_action = False
        except (TypeError, ValueError):
            raw_action = {"invalid_action": "non-JSON action"}
            non_json_action = True
        name = ""
        values: dict[str, Any] = {}
        valid = True
        observation: dict[str, Any] | None = None
        if non_json_action:
            observation = {"error": "action must be JSON serializable"}
        elif isinstance(action, Mapping) and action.get("invalid_action") == "non-JSON action":
            observation = {"error": "action must be JSON serializable"}
        elif not isinstance(action, Mapping) or ("final" in action) == ("tool" in action):
            observation = {"error": "action must contain exactly one of tool or final"}
        elif set(action) - ({"final"} if "final" in action else {"tool", "arguments", "memory"}):
            observation = {"error": "action contains unsupported fields"}
        elif "final" in action:
            if not isinstance(action["final"], str) or not action["final"].strip():
                observation = {"error": "final answer must be a non-empty string"}
            else:
                self._final = action["final"].strip()
                self._done = True
        else:
            name = str(action.get("tool") or "")
            arguments = action.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = None
            if not isinstance(arguments, Mapping):
                observation = {"error": "tool arguments must be a JSON object"}
            elif self._tool_calls >= self.max_tool_calls:
                observation = {"error": "tool call limit exceeded"}
            else:
                values = dict(arguments)
                self._tool_calls += 1
                observation = self.registry.call(
                    self.tools, name, values, max_observation_chars=self.max_context_chars,
                )
                if "error" not in observation:
                    try:
                        records = self.registry.distill(self.tools, name, values, observation)
                        acquired = [self._evidence_record(name, record, observation) for record in records]
                        self._ledger.extend(acquired)
                    except Exception:
                        observation = {"error": "tool produced invalid evidence provenance"}
            if observation is not None and "error" not in observation and "memory" in action:
                try:
                    self._update_memory(action["memory"])
                except ValueError as exc:
                    observation = {"error": str(exc)}
        if observation is not None and "error" in observation:
            del self._ledger[prior_evidence_count:]
            self._memory = prior_memory
            self.tools.seen_search = prior_seen
            self.tools.opened = prior_opened
            valid = False
            self._invalid_actions += 1
        self._trace.append({
            "step": self._steps, "action": raw_action, "tool": name,
            "arguments": values, "observation": observation,
            "valid": valid, "evidence_count": len(self._ledger),
        })
        if self._steps >= self.max_steps:
            self._done = True
        return {"done": self._done, "observation": observation}

    def _update_memory(self, update: Any) -> None:
        if not isinstance(update, Mapping) or set(update) - {"plan", "confirmed", "unresolved"}:
            raise ValueError("memory must contain only plan, confirmed and unresolved")
        if "plan" in update and not isinstance(update["plan"], str):
            raise ValueError("memory.plan must be a string")
        for key in ("confirmed", "unresolved"):
            if key in update and (not isinstance(update[key], list) or not all(isinstance(item, str) for item in update[key])):
                raise ValueError(f"memory.{key} must be a list of strings")
        memory = {**self._memory, **dict(update)}
        if len(json.dumps(memory, ensure_ascii=False)) > min(4000, self.max_context_chars // 4):
            raise ValueError("working memory exceeds its limit")
        self._memory = memory

    def _evidence_record(
        self, tool_name: str, record: Mapping[str, Any], observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.tools is None:
            raise RuntimeError("harness session is closed")
        value = copy.deepcopy(dict(record))
        kind = value.get("kind")
        if kind == "source_chunk":
            chunk_id = str(value.get("chunk_id") or "")
            row = self.tools.db.execute(
                "SELECT c.source_id,c.available_at,c.text FROM chunks c "
                "JOIN source_entities e ON e.source_id=c.source_id "
                "WHERE c.chunk_id=? AND e.canonical_entity_id=? "
                "AND (?=0 OR c.available_at IS NOT NULL) "
                "AND (c.available_at IS NULL OR c.available_at<=?)",
                (chunk_id, self.case.entity_id, int(self.case.dated_only), self.case.cutoff),
            ).fetchone()
            if row is None or value.get("source_id") != row[0] or value.get("available_at") != row[1]:
                raise ValueError("source chunk is absent from the permitted snapshot")
            excerpt = value.get("excerpt", "")
            observed_text = observation.get("text")
            if (not isinstance(observed_text, str) or not observed_text
                    or observed_text not in row[2] or not isinstance(excerpt, str)
                    or excerpt not in observed_text):
                raise ValueError("evidence excerpt is not in its source chunk")
            value["artifact_ref"] = f"snapshot:{chunk_id}"
        elif kind == "calculation":
            if not isinstance(value.get("value"), str):
                raise ValueError("calculation record needs a value")
        else:
            raise ValueError("unknown evidence record kind")
        value.update({"tool": tool_name, "step": self._steps, "status": "ok"})
        if len(json.dumps(value, ensure_ascii=False)) > self.max_context_chars:
            raise ValueError("evidence record exceeds context limit")
        return value

    def checkpoint(self) -> dict[str, Any]:
        """Return a model-visible pre-action state for critical-turn resampling."""
        if self._done or self.tools is None:
            raise RuntimeError("only an active episode can be checkpointed")
        if self._snapshot_hash is None:
            self._snapshot_hash = sha256_file(self.case.snapshot_path)
        return {
            "version": 1, "query_id": self.case.query_id, "query": self.case.query,
            "query_sha256": hashlib.sha256(self.case.query.encode("utf-8")).hexdigest(),
            "snapshot_sha256": self._snapshot_hash,
            "toolset_fingerprint": self.registry.fingerprint,
            "cutoff": self.case.cutoff,
            "limits": {"max_steps": self.max_steps, "max_tool_calls": self.max_tool_calls,
                       "max_context_chars": self.max_context_chars},
            "history": [{"action": entry["action"], "observation": entry["observation"]}
                        for entry in self._trace],
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        """Replay a checkpoint against the same frozen snapshot and tool versions."""
        if checkpoint.get("version") != 1 or checkpoint.get("query_id") != self.case.query_id:
            raise ValueError("checkpoint case identity differs")
        if checkpoint.get("query_sha256") != hashlib.sha256(self.case.query.encode("utf-8")).hexdigest():
            raise ValueError("checkpoint query differs")
        if checkpoint.get("query") != self.case.query:
            raise ValueError("checkpoint query text differs")
        self._snapshot_hash = sha256_file(self.case.snapshot_path)
        if checkpoint.get("snapshot_sha256") != self._snapshot_hash:
            raise ValueError("checkpoint snapshot differs")
        if checkpoint.get("toolset_fingerprint") != self.registry.fingerprint or checkpoint.get("cutoff") != self.case.cutoff:
            raise ValueError("checkpoint toolset or cutoff differs")
        if checkpoint.get("limits") != {"max_steps": self.max_steps, "max_tool_calls": self.max_tool_calls,
                                        "max_context_chars": self.max_context_chars}:
            raise ValueError("checkpoint budgets differ")
        history = checkpoint.get("history")
        if not isinstance(history, list) or len(history) >= self.max_steps:
            raise ValueError("checkpoint history is invalid")
        self.reset()
        for previous in history:
            if not isinstance(previous, Mapping) or not isinstance(previous.get("action"), Mapping):
                raise ValueError("checkpoint action is invalid")
            result = self.step(previous["action"])
            if result["done"] or result["observation"] != previous.get("observation"):
                raise ValueError("checkpoint replay differs from the frozen environment")
        return self.observe()

    def fork(self, checkpoint: Mapping[str, Any] | None = None) -> HarnessSession:
        """Create an isolated session at this pre-action state."""
        branch = HarnessSession(
            self.case, registry=self.registry, max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls, max_context_chars=self.max_context_chars,
            answer_tolerance=self.answer_tolerance,
            allow_toolset_override=self.allow_toolset_override,
        )
        try:
            branch.restore(checkpoint if checkpoint is not None else self.checkpoint())
            return branch
        except BaseException:
            branch.close()
            raise

    def checkpoint_at(self, step: int) -> dict[str, Any]:
        """Reconstruct the pre-action state at a turn in a completed trajectory."""
        if step < 1 or step > len(self._trace):
            raise ValueError("step is outside the recorded trajectory")
        branch = HarnessSession(
            self.case, registry=self.registry, max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls, max_context_chars=self.max_context_chars,
            answer_tolerance=self.answer_tolerance,
            allow_toolset_override=self.allow_toolset_override,
        )
        try:
            for previous in self._trace[:step - 1]:
                result = branch.step(previous["action"])
                if result["done"] or result["observation"] != previous["observation"]:
                    raise ValueError("recorded trajectory cannot be replayed to the chosen turn")
            return branch.checkpoint()
        finally:
            branch.close()

    def grade(self) -> dict[str, Any]:
        """Return private verifier feedback to the trainer after the episode ends."""
        if not self._done:
            raise RuntimeError("finish the harness episode before grading")
        if self.tools is None:
            raise RuntimeError("harness session is closed")
        acquired = {
            str(record["chunk_id"]): record for record in self._ledger
            if record.get("kind") == "source_chunk" and record.get("status") == "ok"
        }
        missing: list[str] = []
        mismatched: list[str] = []
        for fact in self.case.task["evidence"]:
            chunk_id = str(fact["chunk_id"])
            record = acquired.get(chunk_id)
            if record is None:
                missing.append(chunk_id)
                continue
            row = self.tools.db.execute(
                "SELECT source_id,available_at,text FROM chunks WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            quote = str(fact.get("quote") or "")
            expected_date = published_date(fact.get("source_metadata") or {}, self.case.available_at_field)
            observed = self._trace[int(record["step"]) - 1]["observation"]
            if (row is None or str(fact.get("source_id")) != row[0]
                    or record["source_id"] != row[0] or record["available_at"] != row[1]
                    or expected_date != row[1]
                    or not quote or quote not in row[2]
                    or not isinstance(observed, Mapping) or quote not in str(observed.get("text") or "")
                    or (row[1] is not None and row[1] > self.case.cutoff)
                    or (self.case.dated_only and row[1] is None)):
                mismatched.append(chunk_id)
        answer_correct = False
        if self._final:
            try:
                answer_correct = _answer_ok(self.case.task, self._final, self.answer_tolerance)
            except Exception as exc:
                raise RuntimeError("hidden verifier task cannot be replayed") from exc
        evidence_replayable = not missing and not mismatched
        action_valid = self._invalid_actions == 0
        cutoff_valid = all(
            record.get("available_at") is None or str(record["available_at"]) <= self.case.cutoff
            for record in self._ledger if record.get("kind") == "source_chunk"
        )
        accepted = bool(self._final) and evidence_replayable and answer_correct and action_valid and cutoff_valid
        reason = (
            "accepted" if accepted else
            "no final answer" if not self._final else
            "required evidence was not acquired" if missing else
            "evidence provenance did not replay" if mismatched else
            "invalid action in trajectory" if not action_valid else
            "evidence crossed the cutoff" if not cutoff_valid else
            "final answer disagrees with replayed gold"
        )
        return {
            "accepted": accepted,
            "reward": int(accepted),
            "answer_correct": answer_correct,
            "evidence_complete": evidence_replayable,
            "evidence_replayable": evidence_replayable,
            "action_valid": action_valid,
            "cutoff_valid": cutoff_valid,
            "missing_chunk_ids": sorted(set(missing)),
            "mismatched_chunk_ids": sorted(set(mismatched)),
            "reason": reason,
            "step_relevance": self._step_relevance(),
            "invalid_actions": self._invalid_actions,
            "toolset_fingerprint": self.registry.fingerprint,
        }

    def _step_relevance(self) -> list[bool]:
        seen_actions: set[str] = set()
        seen_hits: set[str] = set()
        seen_evidence: set[str] = set()
        flags: list[bool] = []
        for entry in self._trace:
            name = entry["tool"]
            if not name:
                continue
            key = json.dumps([name, entry["arguments"]], ensure_ascii=False, sort_keys=True, default=str)
            repeated = key in seen_actions
            seen_actions.add(key)
            observation = entry["observation"]
            contributes = not repeated
            if name == "search_corpus" and isinstance(observation, Mapping):
                hits = {str(hit.get("chunk_id")) for hit in observation.get("hits", [])}
                contributes = contributes and bool(hits - seen_hits)
                seen_hits.update(hits)
            records = [record for record in self._ledger if record["step"] == entry["step"]]
            if records:
                record_keys = {
                    json.dumps(
                        {key: value for key, value in record.items()
                         if key not in {"tool", "step", "status", "artifact_ref"}},
                        ensure_ascii=False, sort_keys=True, default=str,
                    )
                    for record in records
                }
                contributes = contributes and bool(record_keys - seen_evidence)
                seen_evidence.update(record_keys)
            flags.append(bool(entry["valid"] and contributes and self.registry.relevant(
                name, self.case.task, entry["arguments"], observation,
            )))
        return flags

    def audit_record(self) -> dict[str, Any]:
        """Return a trainer-side replay record after terminal grading."""
        verdict = self.grade()
        if self._snapshot_hash is None:
            self._snapshot_hash = sha256_file(self.case.snapshot_path)
        return {
            "query_id": self.case.query_id,
            "query": self.case.query,
            "cutoff": self.case.cutoff,
            "snapshot_sha256": self._snapshot_hash,
            "toolset_fingerprint": self.registry.fingerprint,
            "trace": self.trace,
            "evidence_ledger": self.evidence_ledger,
            "working_memory": self.working_memory,
            "failed_paths": self.failed_paths,
            "final_answer": self._final,
            "verdict": verdict,
        }
