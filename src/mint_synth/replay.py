from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, DecimalException, InvalidOperation, localcontext
from typing import Any

from .util import canonical_decimal, normalize_text


class ReplayError(ValueError):
    pass


@dataclass(frozen=True)
class ReplayValue:
    value: Decimal | str | bool
    unit: str | None = None
    scale: str | None = None
    currency: str | None = None

    def json_value(self) -> str | bool:
        if isinstance(self.value, Decimal):
            rendered = canonical_decimal(self.value)
            if rendered is None:
                raise ReplayError("non-finite decimal result")
            return rendered
        return self.value


@dataclass(frozen=True)
class ReplayResult:
    terminal: ReplayValue
    depth: int
    fact_ids: frozenset[str]
    source_ids: frozenset[str]
    trace: tuple[dict[str, Any], ...]


def _decimal(value: ReplayValue, node_id: str) -> Decimal:
    if isinstance(value.value, bool):
        raise ReplayError(f"node {node_id}: boolean is not numeric")
    try:
        return Decimal(str(value.value))
    except InvalidOperation as exc:
        raise ReplayError(f"node {node_id}: value is not numeric") from exc


def _same_dimension(values: list[ReplayValue], node_id: str) -> None:
    dimensions = {(v.unit or "", v.scale or "", v.currency or "") for v in values}
    if len(dimensions) > 1:
        raise ReplayError(f"node {node_id}: incompatible unit/scale/currency inputs")


def _round(number: Decimal, params: dict[str, Any], node_id: str) -> Decimal:
    decimals = params.get("round")
    if decimals is None:
        return number
    if not isinstance(decimals, int) or decimals < 0 or decimals > 12:
        raise ReplayError(f"node {node_id}: round must be an integer in [0, 12]")
    quantum = Decimal(1).scaleb(-decimals)
    try:
        return number.quantize(quantum, rounding=ROUND_HALF_UP)
    except DecimalException as exc:
        raise ReplayError(f"node {node_id}: decimal rounding failed") from exc


def _operation(node: dict[str, Any], inputs: list[ReplayValue]) -> ReplayValue:
    node_id = str(node.get("id", "?"))
    op = str(node.get("op", "")).lower()
    params = node.get("params") if isinstance(node.get("params"), dict) else {}
    if not inputs:
        raise ReplayError(f"node {node_id}: operation has no inputs")

    if op == "identity":
        if len(inputs) != 1:
            raise ReplayError(f"node {node_id}: identity expects one input")
        return inputs[0]
    if op in {"equals", "greater_than", "less_than", "greater_or_equal", "less_or_equal"}:
        if len(inputs) != 2:
            raise ReplayError(f"node {node_id}: {op} expects two inputs")
        _same_dimension(inputs, node_id)
        left, right = inputs
        left_num, right_num = canonical_decimal(left.value), canonical_decimal(right.value)
        if left_num is not None and right_num is not None:
            a, b = Decimal(left_num), Decimal(right_num)
        else:
            a, b = normalize_text(left.value), normalize_text(right.value)
        outcome = {
            "equals": a == b,
            "greater_than": a > b,
            "less_than": a < b,
            "greater_or_equal": a >= b,
            "less_or_equal": a <= b,
        }[op]
        return ReplayValue(outcome)
    if op in {"and", "or"}:
        if len(inputs) < 2:
            raise ReplayError(f"node {node_id}: {op} expects at least two inputs")
        if not all(isinstance(v.value, bool) for v in inputs):
            raise ReplayError(f"node {node_id}: {op} expects boolean inputs")
        outcome = all(v.value for v in inputs) if op == "and" else any(v.value for v in inputs)
        return ReplayValue(outcome)

    numbers = [_decimal(value, node_id) for value in inputs]
    with localcontext() as context:
        context.prec = 34
        try:
            if op in {"add", "sum"}:
                if len(numbers) < 2:
                    raise ReplayError(f"node {node_id}: {op} expects at least two inputs")
                _same_dimension(inputs, node_id)
                result = sum(numbers, Decimal(0))
            elif op in {"subtract", "difference"}:
                if len(numbers) != 2:
                    raise ReplayError(f"node {node_id}: {op} expects two inputs")
                _same_dimension(inputs, node_id)
                result = numbers[0] - numbers[1]
            elif op == "multiply":
                if len(numbers) != 2:
                    raise ReplayError(f"node {node_id}: multiply expects two inputs")
                result = numbers[0] * numbers[1]
            elif op in {"divide", "ratio"}:
                if len(numbers) != 2:
                    raise ReplayError(f"node {node_id}: {op} expects two inputs")
                result = numbers[0] / numbers[1]
            elif op == "percent_change":
                if len(numbers) != 2:
                    raise ReplayError(f"node {node_id}: percent_change expects [new, old]")
                _same_dimension(inputs, node_id)
                result = (numbers[0] - numbers[1]) / abs(numbers[1]) * Decimal(100)
            elif op == "average":
                if len(numbers) < 2:
                    raise ReplayError(f"node {node_id}: average expects at least two inputs")
                _same_dimension(inputs, node_id)
                result = sum(numbers, Decimal(0)) / Decimal(len(numbers))
            elif op == "min":
                if len(numbers) < 2:
                    raise ReplayError(f"node {node_id}: min expects at least two inputs")
                _same_dimension(inputs, node_id)
                result = min(numbers)
            elif op == "max":
                if len(numbers) < 2:
                    raise ReplayError(f"node {node_id}: max expects at least two inputs")
                _same_dimension(inputs, node_id)
                result = max(numbers)
            elif op == "abs":
                if len(numbers) != 1:
                    raise ReplayError(f"node {node_id}: abs expects one input")
                result = abs(numbers[0])
            elif op == "negate":
                if len(numbers) != 1:
                    raise ReplayError(f"node {node_id}: negate expects one input")
                result = -numbers[0]
            else:
                raise ReplayError(f"node {node_id}: unsupported operation {op!r}")
        except (DecimalException, ZeroDivisionError) as exc:
            raise ReplayError(f"node {node_id}: invalid decimal arithmetic") from exc

    result = _round(result, params, node_id)
    if op == "percent_change":
        unit, scale, currency = "percent", None, None
    elif op == "ratio":
        unit, scale, currency = node.get("unit", "ratio"), node.get("scale"), node.get("currency")
    elif op in {"multiply", "divide"}:
        unit, scale, currency = node.get("unit"), node.get("scale"), node.get("currency")
    else:
        first = inputs[0]
        unit = node.get("unit", first.unit)
        scale = node.get("scale", first.scale)
        currency = node.get("currency", first.currency)
    return ReplayValue(result, unit=unit, scale=scale, currency=currency)


def replay_graph(graph: dict[str, Any], facts: dict[str, dict[str, Any]]) -> ReplayResult:
    nodes = graph.get("nodes")
    terminal_id = graph.get("terminal_id")
    if not isinstance(nodes, list) or not nodes:
        raise ReplayError("graph.nodes must be a non-empty list")
    if not isinstance(terminal_id, str) or not terminal_id:
        raise ReplayError("graph.terminal_id must be a non-empty string")

    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise ReplayError("every graph node must have a string id")
        if node["id"] in by_id:
            raise ReplayError(f"duplicate graph node id: {node['id']}")
        by_id[node["id"]] = node
    if terminal_id not in by_id:
        raise ReplayError("terminal_id does not reference a node")

    values: dict[str, ReplayValue] = {}
    depths: dict[str, int] = {}
    used_facts: dict[str, frozenset[str]] = {}
    sources: dict[str, frozenset[str]] = {}
    trace: list[dict[str, Any]] = []
    pending = dict(by_id)

    while pending:
        progress = False
        for node_id, node in list(pending.items()):
            kind = node.get("kind")
            if kind == "fact":
                fact_id = node.get("fact_id", node_id)
                fact = facts.get(str(fact_id))
                if fact is None:
                    raise ReplayError(f"node {node_id}: unknown fact_id {fact_id!r}")
                raw_value = fact.get("value_num")
                if raw_value is not None:
                    canonical = canonical_decimal(raw_value)
                    if canonical is None:
                        raise ReplayError(f"fact {fact_id}: invalid value_num")
                    value: Decimal | str | bool = Decimal(canonical)
                else:
                    raw_text = fact.get("value_text")
                    if isinstance(raw_text, bool):
                        value = raw_text
                    elif raw_text is None:
                        raise ReplayError(f"fact {fact_id}: missing value")
                    else:
                        value = str(raw_text)
                values[node_id] = ReplayValue(
                    value,
                    unit=fact.get("unit"),
                    scale=fact.get("scale"),
                    currency=fact.get("currency"),
                )
                source_id = fact.get("source_id") or fact.get("record_id")
                if source_id is None or not str(source_id).strip():
                    raise ReplayError(f"fact {fact_id}: missing source identifier")
                depths[node_id] = 0
                used_facts[node_id] = frozenset({str(fact_id)})
                sources[node_id] = frozenset({str(source_id)})
            elif kind == "constant":
                raw = node.get("value")
                numeric = canonical_decimal(raw)
                value = Decimal(numeric) if numeric is not None else raw
                if not isinstance(value, (Decimal, str, bool)):
                    raise ReplayError(f"node {node_id}: unsupported constant value")
                values[node_id] = ReplayValue(value, node.get("unit"), node.get("scale"), node.get("currency"))
                depths[node_id] = 0
                used_facts[node_id] = frozenset()
                sources[node_id] = frozenset()
            elif kind == "operation":
                input_ids = node.get("inputs")
                if not isinstance(input_ids, list) or not all(isinstance(v, str) for v in input_ids):
                    raise ReplayError(f"node {node_id}: inputs must be a list of node ids")
                if len(input_ids) != len(set(input_ids)):
                    raise ReplayError(f"node {node_id}: operation inputs must be distinct")
                missing = [value for value in input_ids if value not in values]
                if missing:
                    continue
                values[node_id] = _operation(node, [values[value] for value in input_ids])
                depths[node_id] = 1 + max(depths[value] for value in input_ids)
                used_facts[node_id] = frozenset().union(*(used_facts[value] for value in input_ids))
                sources[node_id] = frozenset().union(*(sources[value] for value in input_ids))
                trace.append(
                    {
                        "node_id": node_id,
                        "op": node.get("op"),
                        "inputs": input_ids,
                        "result": values[node_id].json_value(),
                        "unit": values[node_id].unit,
                    }
                )
            else:
                raise ReplayError(f"node {node_id}: unknown kind {kind!r}")
            del pending[node_id]
            progress = True
        if not progress:
            unresolved = ", ".join(sorted(pending))
            raise ReplayError(f"graph contains a cycle or unresolved references: {unresolved}")

    reachable = {terminal_id}
    stack = [terminal_id]
    while stack:
        node = by_id[stack.pop()]
        if node.get("kind") != "operation":
            continue
        for input_id in node.get("inputs", []):
            if input_id not in reachable:
                reachable.add(input_id)
                stack.append(input_id)
    disconnected = sorted(set(by_id) - reachable)
    if disconnected:
        raise ReplayError(f"graph contains nodes disconnected from terminal: {', '.join(disconnected)}")

    return ReplayResult(
        terminal=values[terminal_id],
        depth=depths[terminal_id],
        fact_ids=used_facts[terminal_id],
        source_ids=sources[terminal_id],
        trace=tuple(trace),
    )


def answer_matches(answer: dict[str, Any], result: ReplayResult, tolerance: str = "1e-6") -> bool:
    expected = answer.get("value")
    actual = result.terminal.value
    expected_num = canonical_decimal(expected)
    actual_num = canonical_decimal(actual)
    if expected_num is not None and actual_num is not None:
        a, b = Decimal(expected_num), Decimal(actual_num)
        limit = max(Decimal(tolerance), abs(b) * Decimal(tolerance))
        if abs(a - b) > limit:
            return False
    elif isinstance(actual, bool):
        if isinstance(expected, bool):
            if expected is not actual:
                return False
        elif normalize_text(expected) not in ({"true", "yes", "supported"} if actual else {"false", "no", "unsupported"}):
            return False
    elif normalize_text(expected) != normalize_text(actual):
        return False

    for field in ("unit", "scale", "currency"):
        wanted = answer.get(field)
        got = getattr(result.terminal, field)
        if normalize_text(wanted) != normalize_text(got):
            return False
    return True
