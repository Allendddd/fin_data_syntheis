from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mint_synth.synthesis.replay import ReplayError, ReplayResult, answer_matches, replay_graph
from mint_synth.synthesis.decontamination import QuestionIndex
from mint_synth.core.util import canonical_decimal, jaccard, normalize_text


CAPABILITIES = {"knowledge", "extraction", "calculation", "analysis", "verification"}


@dataclass(frozen=True)
class AuditResult:
    accepted: bool
    reasons: tuple[str, ...]
    replay: ReplayResult | None = None


def replay_summary(result: ReplayResult) -> dict[str, Any]:
    return {
        "value": result.terminal.json_value(),
        "unit": result.terminal.unit,
        "scale": result.terminal.scale,
        "currency": result.terminal.currency,
        "depth": result.depth,
        "fact_ids": sorted(result.fact_ids),
        "source_ids": sorted(result.source_ids),
        "trace": list(result.trace),
    }


def _pattern_reasons(task: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    trace = task.get("reasoning_trace")
    if not isinstance(trace, list) or not trace:
        return ["reasoning_trace must be a non-empty list"]
    statements: list[str] = []
    for step in trace:
        if not isinstance(step, dict) or not normalize_text(step.get("statement")):
            reasons.append("reasoning trace contains an empty or invalid step")
            continue
        statements.append(normalize_text(step["statement"]))
    if len(statements) != len(set(statements)):
        reasons.append("reasoning trace repeats an identical step")
    question = str(task.get("question", ""))
    joined = " ".join(statements)
    if question and len(joined) > 30 and jaccard(question, joined) > 0.9:
        reasons.append("reasoning trace mostly restates the question")
    return reasons


def _trace_support_reasons(task: dict[str, Any], fact_index: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    raw_declared = task.get("evidence_ids", [])
    declared = {value for value in raw_declared if isinstance(value, str)} if isinstance(raw_declared, list) else set()
    graph = task.get("evidence_graph")
    raw_nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    graph_nodes = {
        node.get("id")
        for node in raw_nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    trace = task.get("reasoning_trace")
    for index, step in enumerate(trace if isinstance(trace, list) else [], 1):
        if not isinstance(step, dict):
            continue
        cited = step.get("evidence_ids")
        operation = step.get("operation_node")
        if not isinstance(cited, list):
            reasons.append(f"reasoning step {index} has invalid evidence_ids")
            continue
        if not all(isinstance(fact_id, str) for fact_id in cited):
            reasons.append(f"reasoning step {index} has non-string evidence_ids")
            cited = [fact_id for fact_id in cited if isinstance(fact_id, str)]
        if not cited and not operation:
            reasons.append(f"reasoning step {index} cites neither evidence nor an operation")
        unknown = [fact_id for fact_id in cited if fact_id not in fact_index or fact_id not in declared]
        if unknown:
            reasons.append(f"reasoning step {index} cites undeclared evidence: {unknown}")
        if operation and operation not in graph_nodes:
            reasons.append(f"reasoning step {index} cites unknown operation node {operation!r}")
    return reasons


def _semantic_flags(task: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    audit = task.get("semantic_audit")
    if not isinstance(audit, dict):
        return ["independent semantic audit is missing"]
    return [f"semantic audit failed: {flag}" for flag in required if audit.get(flag) is not True]


def _constant_reasons(task: dict[str, Any]) -> list[str]:
    graph = task.get("evidence_graph", {})
    nodes = graph.get("nodes", []) if isinstance(graph, dict) and isinstance(graph.get("nodes", []), list) else []
    consumers: dict[str, list[str]] = {}
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") != "operation":
            continue
        inputs = node.get("inputs", [])
        for input_id in inputs if isinstance(inputs, list) else []:
            consumers.setdefault(str(input_id), []).append(str(node.get("op", "")))
    reasons: list[str] = []
    comparisons = {"equals", "greater_than", "less_than", "greater_or_equal", "less_or_equal"}
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") != "constant":
            continue
        node_id = str(node.get("id", ""))
        role = node.get("role")
        used_by = set(consumers.get(node_id, []))
        if role == "formula_parameter":
            if canonical_decimal(node.get("value")) not in {"-1", "1", "100"}:
                reasons.append(f"constant {node_id} is not an allowed formula parameter")
            if not used_by or not used_by.issubset({"multiply", "divide"}):
                reasons.append(f"constant {node_id} has an invalid formula consumer")
        elif role == "claim_value":
            if not used_by or not used_by.issubset(comparisons):
                reasons.append(f"constant {node_id} has an invalid claim consumer")
            claim = node.get("value")
            numeric_claim = canonical_decimal(claim)
            question = str(task.get("question", ""))
            if numeric_claim is not None:
                question_numbers = {
                    value
                    for token in re.findall(r"\(?[-+]?\d[\d,]*(?:\.\d+)?%?\)?", question)
                    if (value := canonical_decimal(token)) is not None
                }
                if numeric_claim not in question_numbers:
                    reasons.append(f"claim constant {node_id} is not stated in the question")
            else:
                normalized_claim = normalize_text(claim)
                if not normalized_claim or normalized_claim not in normalize_text(question):
                    reasons.append(f"claim constant {node_id} is not stated in the question")
        else:
            reasons.append(f"constant {node_id} lacks a valid role")
    return reasons


def audit_atomic(
    task: dict[str, Any],
    fact_index: dict[str, dict[str, Any]],
    *,
    answer_tolerance: str = "1e-6",
) -> AuditResult:
    reasons: list[str] = []
    if task.get("capability") not in CAPABILITIES:
        reasons.append("invalid capability")
    if not normalize_text(task.get("question")):
        reasons.append("question is empty")
    evidence_ids = task.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        reasons.append("evidence_ids must be a non-empty list")
        evidence_ids = []
    elif not all(isinstance(value, str) and value for value in evidence_ids):
        reasons.append("evidence_ids must contain only non-empty strings")
        evidence_ids = [value for value in evidence_ids if isinstance(value, str) and value]
    if len(evidence_ids) != len(set(evidence_ids)):
        reasons.append("evidence_ids contains duplicates")
    missing = [fact_id for fact_id in evidence_ids if fact_id not in fact_index]
    if missing:
        reasons.append(f"unknown evidence ids: {missing}")
    reasons.extend(_pattern_reasons(task))
    reasons.extend(_trace_support_reasons(task, fact_index))
    reasons.extend(_constant_reasons(task))

    replay: ReplayResult | None = None
    try:
        replay = replay_graph(task.get("evidence_graph", {}), fact_index)
        if not replay.fact_ids:
            reasons.append("terminal answer uses no source-backed fact")
        if replay.fact_ids != set(evidence_ids):
            reasons.append("declared evidence does not exactly match graph support")
        answer = task.get("answer")
        if not isinstance(answer, dict) or not answer_matches(answer, replay, answer_tolerance):
            reasons.append("answer does not match deterministic graph replay")
        if task.get("capability") in {"calculation", "analysis"} and replay.depth < 1:
            reasons.append("calculation/analysis task has no operation")
        if task.get("capability") in {"calculation", "analysis"} and replay.trace and all(
            step.get("op") == "identity" for step in replay.trace
        ):
            reasons.append("calculation/analysis task uses only identity operations")
    except ReplayError as exc:
        reasons.append(f"graph replay failed: {exc}")

    reasons.extend(
        _semantic_flags(
            task,
            ("fully_supported", "answer_unique", "financially_valid", "no_unsupported_jumps", "clear"),
        )
    )
    return AuditResult(not reasons, tuple(reasons), replay)


def _looks_exposed(question: str, task: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    folded = question.casefold()
    graph = task.get("evidence_graph")
    nodes = graph.get("nodes", []) if isinstance(graph, dict) and isinstance(graph.get("nodes", []), list) else []
    for internal in [task.get("task_id"), *nodes]:
        value = internal.get("id") if isinstance(internal, dict) else internal
        if isinstance(value, str) and len(value) >= 8 and value.casefold() in folded:
            reasons.append("query exposes an internal identifier")
            break
    for fact in evidence:
        locator = fact.get("locator", {})
        for value in (fact.get("fact_id"), fact.get("chunk_id"), fact.get("source_id"), locator.get("char_start")):
            if isinstance(value, str) and len(value) >= 8 and value.casefold() in folded:
                reasons.append("query exposes a provenance locator")
                return reasons
        quote = normalize_text(fact.get("quote"))
        if len(quote) >= 80 and normalize_text(question).find(quote) >= 0:
            reasons.append("query copies a long verbatim evidence span")
            return reasons
    return reasons


def _numeric_leak_reasons(
    question: str, task: dict[str, Any], evidence: list[dict[str, Any]], replay: ReplayResult
) -> list[str]:
    query_numbers = {
        value
        for token in re.findall(r"\(?[-+]?\d[\d,]*(?:\.\d+)?%?\)?", question)
        if (value := canonical_decimal(token)) is not None
    }
    allowed_periods = {
        value
        for fact in evidence
        for token in re.findall(r"\d{4}", str(fact.get("period", "")))
        if (value := canonical_decimal(token)) is not None
    }
    hidden: set[str] = set()
    for fact in evidence:
        value = canonical_decimal(fact.get("value_num"))
        if value is not None:
            hidden.add(value)
    terminal = canonical_decimal(replay.terminal.value)
    if terminal is not None:
        hidden.add(terminal)
    for step in replay.trace[:-1]:
        value = canonical_decimal(step.get("result"))
        if value is not None:
            hidden.add(value)
    leaked = sorted((query_numbers & hidden) - allowed_periods)
    return [f"query exposes hidden numeric value(s): {', '.join(leaked)}"] if leaked else []


def audit_long(
    task: dict[str, Any],
    fact_index: dict[str, dict[str, Any]],
    *,
    min_depth: int,
    min_sources: int,
    seed_text: str = "",
    evaluation_questions: list[str] | None = None,
    evaluation_index: QuestionIndex | None = None,
    overlap_threshold: float = 0.82,
    answer_tolerance: str = "1e-6",
) -> AuditResult:
    reasons: list[str] = []
    question = str(task.get("question", ""))
    if not normalize_text(question):
        reasons.append("question is empty")
    evidence_ids = task.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        reasons.append("evidence_ids must be a non-empty list")
        evidence_ids = []
    elif not all(isinstance(value, str) and value for value in evidence_ids):
        reasons.append("evidence_ids must contain only non-empty strings")
        evidence_ids = [value for value in evidence_ids if isinstance(value, str) and value]
    if len(evidence_ids) != len(set(evidence_ids)):
        reasons.append("evidence_ids contains duplicates")
    evidence = [fact_index[fact_id] for fact_id in evidence_ids if fact_id in fact_index]
    if len(evidence) != len(evidence_ids):
        reasons.append("one or more evidence ids are unknown")
    reasons.extend(_looks_exposed(question, task, evidence))
    reasons.extend(_constant_reasons(task))
    if seed_text and jaccard(question, seed_text) >= overlap_threshold:
        reasons.append("query too closely reproduces its seed")
    if evaluation_index is not None:
        if evaluation_index.overlaps(question, overlap_threshold):
            reasons.append("query overlaps an evaluation task")
    else:
        for benchmark_question in evaluation_questions or []:
            if jaccard(question, benchmark_question) >= overlap_threshold:
                reasons.append("query overlaps an evaluation task")
                break
    task_spec = task.get("task_spec", {})
    if isinstance(task_spec, dict) and task_spec.get("time_sensitive") is True:
        cutoff = str(task.get("cutoff", ""))
        if cutoff and cutoff not in question:
            reasons.append("time-sensitive query does not state its cutoff")

    replay: ReplayResult | None = None
    try:
        replay = replay_graph(task.get("evidence_graph", {}), fact_index)
        if replay.depth < min_depth:
            reasons.append(f"graph depth {replay.depth} is below {min_depth}")
        if any(step.get("op") == "identity" for step in replay.trace):
            reasons.append("long-horizon graph contains an identity padding operation")
        if len(replay.source_ids) < min_sources:
            reasons.append(f"source breadth {len(replay.source_ids)} is below {min_sources}")
        if replay.fact_ids != set(evidence_ids):
            reasons.append("declared evidence does not exactly match graph support")
        answer = task.get("answer")
        if not isinstance(answer, dict) or not answer_matches(answer, replay, answer_tolerance):
            reasons.append("answer does not match deterministic graph replay")
        for fact_id in replay.fact_ids:
            locator = fact_index[fact_id].get("locator", {})
            if not isinstance(locator.get("char_start"), int) or not isinstance(locator.get("char_end"), int):
                reasons.append(f"fact {fact_id} lacks a localized provenance span")
        reasons.extend(_numeric_leak_reasons(question, task, evidence, replay))
    except ReplayError as exc:
        reasons.append(f"graph replay failed: {exc}")

    reasons.extend(
        _semantic_flags(
            task,
            ("answer_unique", "fully_supported", "financially_valid", "non_exposing", "seed_not_copied"),
        )
    )
    return AuditResult(not reasons, tuple(reasons), replay)
