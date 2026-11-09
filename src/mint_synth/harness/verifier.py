"""Private verification of model-supplied financial derivations."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from mint_synth.core.util import normalize_text
from mint_synth.synthesis.replay import answer_matches, replay_graph


PROOF_SCHEMA = {
    "answer": {"value": "decimal string, boolean, or text", "unit": None, "scale": None, "currency": None},
    "facts": [{"id": "local ID", "quote": "exact source quote", "chunk_id": "opened chunk ID for long tasks",
               "value": "source value", "unit": None, "scale": None, "currency": None}],
    "constants": [{"id": "local ID", "value": "literal", "unit": None, "scale": None, "currency": None}],
    "steps": [{"id": "local ID", "op": "operation", "inputs": ["local ID"], "params": {},
               "result": "computed value", "unit": None, "scale": None, "currency": None}],
    "terminal": "local ID of answer-producing fact or step",
}


def _same_value(left: Any, right: Any, tolerance: str) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    plain = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
    a = str(left).strip() if isinstance(left, (int, float, Decimal, str)) else ""
    b = str(right).strip() if isinstance(right, (int, float, Decimal, str)) else ""
    numeric_left, numeric_right = bool(plain.fullmatch(a)), bool(plain.fullmatch(b))
    if numeric_left or numeric_right:
        if not numeric_left or not numeric_right:
            return False
        try:
            x, y, limit = Decimal(a), Decimal(b), Decimal(tolerance)
            return abs(x - y) <= limit * max(Decimal(1), abs(y))
        except InvalidOperation:
            return False
    return normalize_text(left) == normalize_text(right)


def typed_answer_ok(task: Mapping[str, Any], answer: Any, tolerance: str, replay: Any = None) -> bool:
    if not isinstance(answer, Mapping) or "value" not in answer or not _dims(answer, task["answer"]):
        return False
    if replay is None:
        facts = {str(f["fact_id"]): f for f in task.get("evidence", [])}
        replay = replay_graph(task["evidence_graph"], facts)
    return (_same_value(answer["value"], replay.terminal.json_value(), tolerance)
            and answer_matches(dict(answer), replay, tolerance=tolerance))


def _dims(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    for key in ("unit", "scale", "currency"):
        if key not in candidate:
            return False
        wanted, offered = reference.get(key), candidate[key]
        if wanted is None:
            if offered is not None:
                return False
        elif not isinstance(wanted, str) or not isinstance(offered, str) or wanted.strip().casefold() != offered.strip().casefold():
            return False
    return True


def _fact_value(fact: Mapping[str, Any]) -> Any:
    return fact.get("value_num") if fact.get("value_num") is not None else fact.get("value_text")


def verify_proof(
    task: Mapping[str, Any], proof: Any, *, context: str | None = None,
    acquired: Mapping[str, str] | None = None, tolerance: str = "1e-6",
) -> tuple[bool, str]:
    """Bind model-visible quotes and local operations to every hidden ancestor."""
    if not isinstance(proof, Mapping) or set(proof) != {"answer", "facts", "constants", "steps", "terminal"}:
        return False, "final requires answer, facts, constants, steps, and terminal"
    facts = {str(f["fact_id"]): f for f in task.get("evidence", [])}
    graph = task.get("evidence_graph")
    try:
        gold = replay_graph(graph, facts)
    except Exception as exc:
        raise RuntimeError("hidden verifier task cannot be replayed") from exc
    if not answer_matches(task["answer"], gold, tolerance=tolerance):
        raise RuntimeError("hidden answer disagrees with its graph")
    if not isinstance(proof["answer"], Mapping) or not _dims(proof["answer"], task["answer"]):
        return False, "answer dimensions are missing or incorrect"
    if not typed_answer_ok(task, proof["answer"], tolerance, gold):
        return False, "final answer disagrees with replayed gold"
    if not all(isinstance(proof[key], list) for key in ("facts", "constants", "steps")):
        return False, "proof facts, constants, and steps must be arrays"
    answer_fields = {"value", "unit", "scale", "currency"}
    fact_fields = {"id", "quote", "value", "unit", "scale", "currency"}
    constant_fields = {"id", "value", "unit", "scale", "currency"}
    step_fields = {"id", "op", "inputs", "params", "result", "unit", "scale", "currency"}
    if (set(proof["answer"]) != answer_fields
            or any(not isinstance(item, Mapping) or set(item) not in (fact_fields, fact_fields | {"chunk_id"})
                   for item in proof["facts"])
            or any(not isinstance(item, Mapping) or set(item) != constant_fields for item in proof["constants"])
            or any(not isinstance(item, Mapping) or set(item) != step_fields for item in proof["steps"])):
        return False, "proof contains missing or unsupported fields"
    nodes = {node["id"]: node for node in graph["nodes"]}
    gold_facts = {key: node for key, node in nodes.items() if node["kind"] == "fact"}
    gold_constants = {key: node for key, node in nodes.items() if node["kind"] == "constant"}
    gold_steps = {key: node for key, node in nodes.items() if node["kind"] == "operation"}
    if (len(proof["facts"]) != len(gold_facts) or len(proof["constants"]) != len(gold_constants)
            or len(proof["steps"]) != len(gold_steps)):
        return False, "proof omits or adds graph ancestors"
    mapped: dict[str, str] = {}
    used: set[str] = set()
    proof_nodes: list[dict[str, Any]] = []

    def local_id(item: Any) -> str | None:
        name = item.get("id") if isinstance(item, Mapping) else None
        return name if isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) and name not in mapped else None

    for item in proof["facts"]:
        name = local_id(item)
        quote = item.get("quote") if isinstance(item, Mapping) else None
        if name is None or not isinstance(quote, str) or not quote:
            return False, "invalid or repeated proof fact"
        chunk_id = item.get("chunk_id")
        if context is not None and quote not in context:
            return False, "fact quote is absent from supplied context"
        if acquired is not None and (not isinstance(chunk_id, str) or quote not in acquired.get(chunk_id, "")):
            return False, "fact quote was not acquired from its cited chunk"
        choices = []
        for node_id, node in gold_facts.items():
            fact = facts[str(node.get("fact_id", node_id))]
            if (node_id not in used and fact.get("quote") == quote
                    and (acquired is None or str(fact.get("chunk_id")) == chunk_id)
                    and _dims(item, fact) and "value" in item
                    and _same_value(item["value"], _fact_value(fact), tolerance)):
                choices.append(node_id)
        if len(choices) != 1:
            return False, "fact cannot be uniquely bound to hidden evidence"
        node_id = choices[0]
        mapped[name] = node_id
        used.add(node_id)
        proof_nodes.append({"id": name, "kind": "fact", "fact_id": gold_facts[node_id].get("fact_id", node_id)})

    for item in proof["constants"]:
        name = local_id(item)
        if name is None or "value" not in item:
            return False, "invalid or repeated proof constant"
        choices = [node_id for node_id, node in gold_constants.items() if node_id not in used
                   and _same_value(item["value"], node.get("value"), tolerance) and _dims(item, node)]
        if len(choices) != 1:
            return False, "constant cannot be uniquely bound to hidden graph"
        node_id = choices[0]
        mapped[name] = node_id
        used.add(node_id)
        proof_nodes.append({"id": name, **{key: value for key, value in gold_constants[node_id].items() if key != "id"}})

    for item in proof["steps"]:
        name = local_id(item)
        inputs = item.get("inputs") if isinstance(item, Mapping) else None
        if (name is None or not isinstance(inputs, list) or not inputs
                or not all(isinstance(value, str) and value in mapped for value in inputs)
                or not isinstance(item.get("op"), str) or "result" not in item):
            return False, "operation step is incomplete or out of order"
        params = item.get("params")
        if not isinstance(params, Mapping):
            return False, "operation params must be an object"
        choices = [node_id for node_id, node in gold_steps.items() if node_id not in used
                   and node.get("op") == item["op"] and node.get("inputs") == [mapped[value] for value in inputs]
                   and (node.get("params") or {}) == dict(params)]
        if len(choices) != 1:
            return False, "operation does not follow hidden graph dependencies"
        node_id = choices[0]
        mapped[name] = node_id
        used.add(node_id)
        proof_nodes.append({"id": name, **{key: value for key, value in gold_steps[node_id].items()
                                             if key not in {"id", "inputs"}}, "inputs": inputs})
    if (used != set(nodes) or not isinstance(proof["terminal"], str)
            or mapped.get(proof["terminal"]) != graph["terminal_id"]):
        return False, "proof does not reach the complete answer graph"
    derived = replay_graph({"nodes": proof_nodes, "terminal_id": proof["terminal"]}, facts)
    outputs = {entry["node_id"]: entry for entry in derived.trace}
    for item in proof["steps"]:
        actual = outputs[item["id"]]
        if not _same_value(item["result"], actual["result"], tolerance) or not _dims(item, actual):
            return False, "reported operation result or dimensions are incorrect"
    return True, "accepted"


def _answer_ok(task: Mapping[str, Any], proposed: str, tolerance: str) -> bool:
    """Compatibility wrapper for callers needing a boolean final check."""
    try:
        proof = json.loads(proposed)
    except (TypeError, ValueError):
        return False
    return verify_proof(task, proof, tolerance=tolerance)[0]
