from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .replay import ReplayError, replay_graph
from .util import normalize_text, stable_id


@dataclass(frozen=True)
class FormulaSpec:
    formula_id: str
    version: int
    generator: str
    operations: tuple[str, ...]
    description: str


_BUILTIN_FORMULAS = (
    FormulaSpec("temporal_absolute_change", 1, "temporal_pair", ("subtract", "abs"), "Absolute magnitude of change between periods."),
    FormulaSpec("temporal_percent_change", 1, "temporal_pair", ("percent_change", "abs"), "Absolute percentage change between periods."),
    FormulaSpec("cross_source_ratio", 1, "compatible_ratio", ("ratio",), "Ratio from two disclosed facts."),
    FormulaSpec("cross_source_average", 1, "compatible_average", ("average",), "Average of compatible cross-source facts."),
    FormulaSpec("cross_source_verification", 1, "compatible_verification", ("equals",), "Verify agreement across disclosures."),
    FormulaSpec("three_period_trend", 1, "temporal_trend", ("subtract", "average", "abs"), "Magnitude of the average period-over-period change across three periods."),
    FormulaSpec("temporal_ratio_change", 1, "ratio_change", ("ratio", "subtract", "abs"), "Absolute change in a same-scope ratio across periods."),
    FormulaSpec("multi_source_verification", 1, "multi_verification", ("equals", "and"), "Verify agreement across three sources."),
)

_KNOWN_OPERATIONS = {
    "identity", "add", "sum", "subtract", "difference", "multiply", "divide", "ratio",
    "percent_change", "average", "min", "max", "abs", "negate", "equals", "greater_than",
    "less_than", "greater_or_equal", "less_or_equal", "and", "or",
}

_GENERATOR_OPERATIONS: dict[str, frozenset[tuple[str, ...]]] = {
    "temporal_pair": frozenset({("subtract", "abs"), ("percent_change", "abs")}),
    "compatible_ratio": frozenset({("ratio",)}),
    "compatible_average": frozenset({("average",)}),
    "compatible_verification": frozenset({("equals",)}),
    "temporal_trend": frozenset({("subtract", "average", "abs")}),
    "ratio_change": frozenset({("ratio", "subtract", "abs")}),
    "multi_verification": frozenset({("equals", "and")}),
}


def load_formula_registry(path: str) -> tuple[FormulaSpec, ...]:
    try:
        import yaml
    except ImportError as exc:  # dependency-free unit tests may use the compiled fallback
        if path == "builtin":
            return _BUILTIN_FORMULAS
        raise RuntimeError("PyYAML is required for a custom formula registry") from exc
    if path == "builtin":
        text = resources.files("mint_synth").joinpath("resources/formulas.default.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or not isinstance(raw.get("version"), int):
        raise ValueError("formula registry requires an integer version")
    entries = raw.get("formulas") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("formula registry must contain a non-empty formulas list")
    formulas: list[FormulaSpec] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each formula must be a mapping")
        formula_id = str(entry.get("id", "")).strip()
        generator = str(entry.get("generator", "")).strip()
        operations = entry.get("operations")
        if not formula_id or formula_id in seen or generator not in _GENERATOR_OPERATIONS:
            raise ValueError(f"invalid or duplicate formula: {formula_id!r}")
        if (
            not isinstance(operations, list)
            or not operations
            or not all(isinstance(op, str) and op in _KNOWN_OPERATIONS for op in operations)
        ):
            raise ValueError(f"formula {formula_id} has invalid operations")
        if tuple(operations) not in _GENERATOR_OPERATIONS[generator]:
            supported = [list(value) for value in sorted(_GENERATOR_OPERATIONS[generator])]
            raise ValueError(
                f"formula {formula_id} operations must be one of {supported} "
                f"for generator {generator!r}; custom executable topology is not supported"
            )
        version = int(entry.get("version", 1))
        if version < 1:
            raise ValueError(f"formula {formula_id} version must be positive")
        description = str(entry.get("description", "")).strip()
        if not description:
            raise ValueError(f"formula {formula_id} requires a non-empty description")
        seen.add(formula_id)
        formulas.append(
            FormulaSpec(
                formula_id,
                version,
                generator,
                tuple(operations),
                description,
            )
        )
    return tuple(formulas)


def _dimension(fact: Mapping[str, Any]) -> tuple[str, str, str]:
    return tuple(normalize_text(fact.get(key)) for key in ("unit", "scale", "currency"))  # type: ignore[return-value]


def _metric(fact: Mapping[str, Any]) -> str:
    return normalize_text(fact.get("metric"))


def _period(fact: Mapping[str, Any]) -> str:
    return normalize_text(fact.get("period"))


def _period_key(fact: Mapping[str, Any]) -> tuple[str, int, int] | None:
    """Return a conservative sortable key without mixing annual, quarterly, or point-in-time scopes."""
    value = _period(fact)
    if not value:
        return None
    match = re.fullmatch(r"(?:fy\s*)?(\d{4})(?:\s*fy)?|((?:19|20)\d{2})年", value)
    if match:
        year = int(match.group(1) or match.group(2))
        return "year", year, 1
    match = re.fullmatch(
        r"q([1-4])\s*[-/]?\s*((?:19|20)\d{2})|((?:19|20)\d{2})\s*[-/]?\s*q([1-4])|"
        r"((?:19|20)\d{2})年(?:第)?([1-4])季度",
        value,
    )
    if match:
        quarter = int(match.group(1) or match.group(4) or match.group(6))
        year = int(match.group(2) or match.group(3) or match.group(5))
        return "quarter", year, quarter
    match = re.fullmatch(
        r"h([12])\s*[-/]?\s*((?:19|20)\d{2})|((?:19|20)\d{2})\s*[-/]?\s*h([12])|"
        r"((?:19|20)\d{2})年(?:第)?([12])半年度?",
        value,
    )
    if match:
        half = int(match.group(1) or match.group(4) or match.group(6))
        year = int(match.group(2) or match.group(3) or match.group(5))
        return "half", year, half
    match = re.fullmatch(r"((?:19|20)\d{2})-(0[1-9]|1[0-2])", value)
    if match:
        return "month", int(match.group(1)), int(match.group(2))
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return "date", parsed.year, parsed.toordinal()


def _source(fact: Mapping[str, Any]) -> str:
    return str(fact.get("source_id") or fact.get("record_id") or "")


def _numeric(fact: Mapping[str, Any]) -> bool:
    return fact.get("value_num") is not None


def _first_per_source(values: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for fact in values:
        source = _source(fact)
        current = result.get(source)
        if current is None or str(fact.get("fact_id")) < str(current.get("fact_id")):
            result[source] = fact
    return result


def _source_pairs(
    left: Mapping[str, dict[str, Any]],
    right: Mapping[str, dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for left_source in sorted(left):
        for right_source in sorted(right):
            if left_source != right_source:
                yield left[left_source], right[right_source]


def _candidate_groups(
    facts: Sequence[dict[str, Any]],
    formula: FormulaSpec,
    *,
    limit_hint: int,
    min_sources: int,
) -> Iterable[tuple[dict[str, Any], ...]]:
    emitted = 0
    scanned = 0
    scan_limit = max(16, limit_hint)

    def examine() -> bool:
        nonlocal scanned
        if scanned >= scan_limit:
            return False
        scanned += 1
        return True

    def emit(group: tuple[dict[str, Any], ...]) -> bool:
        nonlocal emitted
        if len({_source(fact) for fact in group}) < min_sources:
            return False
        emitted += 1
        return True

    if formula.generator == "temporal_pair":
        grouped: dict[tuple[Any, ...], dict[tuple[str, int, int], list[dict[str, Any]]]] = {}
        for fact in facts:
            period_key = _period_key(fact)
            if period_key is None:
                continue
            grouped.setdefault((_metric(fact), _dimension(fact), period_key[0]), {}).setdefault(period_key, []).append(fact)
        for key in sorted(grouped, key=repr):
            by_period = grouped[key]
            for old_key, new_key in itertools.combinations(sorted(by_period), 2):
                if not examine():
                    return
                old_values = _first_per_source(by_period[old_key])
                new_values = _first_per_source(by_period[new_key])
                for old, new in _source_pairs(old_values, new_values):
                    if not examine():
                        return
                    group = (new, old)
                    if emit(group):
                        yield group
                        if emitted >= limit_hint:
                            return
        return

    if formula.generator in {"compatible_average", "compatible_verification"}:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for fact in facts:
            period_key = _period_key(fact)
            if period_key is None:
                continue
            grouped.setdefault((_metric(fact), period_key, _dimension(fact)), []).append(fact)
        for key in sorted(grouped, key=repr):
            by_source = _first_per_source(grouped[key])
            for left_source, right_source in itertools.combinations(sorted(by_source), 2):
                if not examine():
                    return
                group = (by_source[left_source], by_source[right_source])
                if emit(group):
                    yield group
                    if emitted >= limit_hint:
                        return
        return

    if formula.generator == "compatible_ratio":
        grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
        for fact in facts:
            period_key = _period_key(fact)
            if period_key is None:
                continue
            grouped.setdefault((period_key, _dimension(fact)), {}).setdefault(_metric(fact), []).append(fact)
        for key in sorted(grouped, key=repr):
            by_metric = grouped[key]
            for numerator_metric, denominator_metric in itertools.combinations(sorted(by_metric), 2):
                if not examine():
                    return
                numerator = _first_per_source(by_metric[numerator_metric])
                denominator = _first_per_source(by_metric[denominator_metric])
                for left, right in _source_pairs(numerator, denominator):
                    if not examine():
                        return
                    group = (left, right)
                    if emit(group):
                        yield group
                        if emitted >= limit_hint:
                            return
        return

    if formula.generator == "temporal_trend":
        grouped: dict[tuple[Any, ...], dict[tuple[str, int, int], list[dict[str, Any]]]] = {}
        for fact in facts:
            period_key = _period_key(fact)
            if period_key is not None:
                grouped.setdefault((_metric(fact), _dimension(fact), period_key[0]), {}).setdefault(period_key, []).append(fact)
        for key in sorted(grouped, key=repr):
            by_period = grouped[key]
            for periods in itertools.combinations(sorted(by_period), 3):
                if not examine():
                    return
                candidate_lists = [
                    [by_source[source] for source in sorted(by_source)]
                    for by_source in (_first_per_source(by_period[period]) for period in periods)
                ]
                for group in itertools.product(*candidate_lists):
                    if not examine():
                        return
                    value = tuple(group)
                    if emit(value):
                        yield value
                        if emitted >= limit_hint:
                            return
        return

    if formula.generator == "ratio_change":
        scopes: dict[
            tuple[Any, ...],
            dict[tuple[str, int, int], dict[str, dict[str, dict[str, Any]]]],
        ] = {}
        for fact in facts:
            period_key = _period_key(fact)
            if period_key is not None:
                scope = (period_key[0], _dimension(fact))
                by_source = (
                    scopes.setdefault(scope, {})
                    .setdefault(period_key, {})
                    .setdefault(_metric(fact), {})
                )
                source = _source(fact)
                current = by_source.get(source)
                if current is None or str(fact.get("fact_id")) < str(current.get("fact_id")):
                    by_source[source] = fact
        for scope in sorted(scopes, key=repr):
            by_period = scopes[scope]
            metric_periods: dict[str, set[tuple[str, int, int]]] = {}
            for period_key, by_metric in by_period.items():
                for metric in by_metric:
                    metric_periods.setdefault(metric, set()).add(period_key)
            for numerator_metric, denominator_metric in itertools.combinations(sorted(metric_periods), 2):
                if not examine():
                    return
                common_periods = sorted(metric_periods[numerator_metric] & metric_periods[denominator_metric])
                if len(common_periods) < 2:
                    continue
                for old_period, new_period in itertools.combinations(common_periods, 2):
                    if not examine():
                        return
                    old_numerators = by_period[old_period][numerator_metric]
                    old_denominators = by_period[old_period][denominator_metric]
                    new_numerators = by_period[new_period][numerator_metric]
                    new_denominators = by_period[new_period][denominator_metric]
                    source_products = itertools.product(
                        sorted(new_numerators),
                        sorted(new_denominators),
                        sorted(old_numerators),
                        sorted(old_denominators),
                    )
                    for new_n_source, new_d_source, old_n_source, old_d_source in source_products:
                        if not examine():
                            return
                        group = (
                            new_numerators[new_n_source],
                            new_denominators[new_d_source],
                            old_numerators[old_n_source],
                            old_denominators[old_d_source],
                        )
                        if len({str(fact.get("fact_id")) for fact in group}) < 4:
                            continue
                        if emit(group):
                            yield group
                            if emitted >= limit_hint:
                                return
        return

    if formula.generator == "multi_verification":
        grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
        for fact in facts:
            period_key = _period_key(fact)
            if period_key is None:
                continue
            key = (_metric(fact), period_key, _dimension(fact))
            by_source = grouped.setdefault(key, {})
            source = _source(fact)
            current = by_source.get(source)
            if current is None or str(fact.get("fact_id")) < str(current.get("fact_id")):
                by_source[source] = fact
        for key in sorted(grouped, key=repr):
            by_source = grouped[key]
            values = [by_source[source] for source in sorted(by_source)]
            for group in itertools.combinations(values, 3):
                if not examine():
                    return
                if emit(group):
                    yield group
                    if emitted >= limit_hint:
                        return


def _graph(formula: FormulaSpec, facts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    inputs: list[str] = []
    for index, fact in enumerate(facts):
        node_id = f"fact_{index}"
        nodes.append({"id": node_id, "kind": "fact", "fact_id": fact["fact_id"], "role": f"input_{index}"})
        inputs.append(node_id)
    def operation_node(node_id: str, operation: str, operation_inputs: list[str]) -> dict[str, Any]:
        node: dict[str, Any] = {
            "id": node_id,
            "kind": "operation",
            "op": operation,
            "inputs": operation_inputs,
            "formula_id": formula.formula_id,
            "formula_version": formula.version,
        }
        if operation == "ratio":
            node.update({"unit": "ratio", "scale": None, "currency": None})
        return node

    if formula.generator == "temporal_trend":
        nodes.extend(
            [
                operation_node("delta_0", "subtract", ["fact_1", "fact_0"]),
                operation_node("delta_1", "subtract", ["fact_2", "fact_1"]),
                operation_node("mean_delta", "average", ["delta_0", "delta_1"]),
                operation_node("magnitude", "abs", ["mean_delta"]),
            ]
        )
        return {"nodes": nodes, "terminal_id": "magnitude"}
    if formula.generator == "ratio_change":
        nodes.extend(
            [
                operation_node("new_ratio", "ratio", ["fact_0", "fact_1"]),
                operation_node("old_ratio", "ratio", ["fact_2", "fact_3"]),
                operation_node("ratio_delta", "subtract", ["new_ratio", "old_ratio"]),
                operation_node("magnitude", "abs", ["ratio_delta"]),
            ]
        )
        return {"nodes": nodes, "terminal_id": "magnitude"}
    if formula.generator == "multi_verification":
        nodes.extend(
            [
                operation_node("check_0", "equals", ["fact_0", "fact_1"]),
                operation_node("check_1", "equals", ["fact_1", "fact_2"]),
                operation_node("all_checks", "and", ["check_0", "check_1"]),
            ]
        )
        return {"nodes": nodes, "terminal_id": "all_checks"}

    for index, operation in enumerate(formula.operations):
        node_id = f"op_{index}"
        operation_inputs = inputs if index == 0 else [f"op_{index - 1}"]
        nodes.append(operation_node(node_id, operation, operation_inputs))
    return {"nodes": nodes, "terminal_id": f"op_{len(formula.operations) - 1}"}


def _answer(result: Any) -> dict[str, Any]:
    value = result.terminal.json_value()
    return {
        "value": value,
        "unit": result.terminal.unit,
        "scale": result.terminal.scale,
        "currency": result.terminal.currency,
        "text": str(value),
    }


def _task_spec(formula: FormulaSpec, facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    objective = {
        "temporal_pair": "change",
        "compatible_ratio": "ratio",
        "compatible_average": "aggregation",
        "compatible_verification": "verification",
        "temporal_trend": "change",
        "ratio_change": "change",
        "multi_verification": "verification",
    }[formula.generator]
    role_names = {
        "temporal_pair": ("new_period_value", "old_period_value"),
        "compatible_ratio": ("numerator", "denominator"),
        "compatible_average": ("observation_1", "observation_2"),
        "compatible_verification": ("claim_observation", "comparison_observation"),
        "temporal_trend": ("old_period_value", "middle_period_value", "new_period_value"),
        "ratio_change": ("new_period_numerator", "new_period_denominator", "old_period_numerator", "old_period_denominator"),
        "multi_verification": ("observation_1", "observation_2", "observation_3"),
    }[formula.generator]
    fact_roles = [
        {
            "role": role_names[index],
            "entity": fact.get("canonical_entity_name") or fact.get("entity"),
            "period": fact.get("period"),
            "metric": fact.get("metric"),
            "unit": fact.get("unit"),
            "scale": fact.get("scale"),
            "currency": fact.get("currency"),
        }
        for index, fact in enumerate(facts)
    ]
    return {
        "objective_type": objective,
        "entities": sorted({str(fact.get("canonical_entity_name") or fact.get("entity")) for fact in facts}),
        "periods": sorted({str(fact.get("period")) for fact in facts if fact.get("period") is not None}),
        "metrics": [str(fact.get("metric")) for fact in facts],
        "fact_roles": fact_roles,
        "requested_output": formula.description,
        "time_sensitive": False,
        "formula_id": formula.formula_id,
        "formula_version": formula.version,
    }


class RuleDagPlanner:
    def __init__(
        self,
        formulas: Sequence[FormulaSpec],
        *,
        allowed_operations: Iterable[str],
        min_depth: int,
        max_depth: int,
        min_sources: int,
        beam_width: int,
        max_graphs_per_entity: int,
        max_candidate_groups_per_formula: int | None = None,
    ):
        self.formulas = tuple(formulas)
        self.allowed_operations = set(allowed_operations)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_sources = min_sources
        self.beam_width = beam_width
        self.max_graphs_per_entity = max_graphs_per_entity
        self.max_candidate_groups_per_formula = (
            max_candidate_groups_per_formula
            if max_candidate_groups_per_formula is not None
            else max(beam_width * 32, beam_width)
        )

    def plan_entity(self, facts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        numeric = sorted(
            (fact for fact in facts if _numeric(fact)),
            key=lambda fact: (str(fact.get("metric")), str(fact.get("period")), str(fact.get("source_id")), str(fact.get("fact_id"))),
        )
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        seen: set[str] = set()
        for formula in self.formulas:
            if not set(formula.operations).issubset(self.allowed_operations):
                continue
            valid_for_formula = 0
            for group in _candidate_groups(
                numeric,
                formula,
                limit_hint=self.max_candidate_groups_per_formula,
                min_sources=self.min_sources,
            ):
                graph = _graph(formula, group)
                fact_index = {fact["fact_id"]: fact for fact in group}
                try:
                    replay = replay_graph(graph, fact_index)
                except ReplayError:
                    continue
                if not self.min_depth <= replay.depth <= self.max_depth:
                    continue
                if len(replay.source_ids) < self.min_sources:
                    continue
                signature = stable_id(
                    "graph",
                    formula.formula_id,
                    formula.version,
                    sorted(replay.fact_ids),
                    graph,
                    length=32,
                )
                if signature in seen:
                    continue
                seen.add(signature)
                sources = len(replay.source_ids)
                candidate = {
                    "graph_id": signature,
                    "canonical_entity_id": group[0].get("canonical_entity_id"),
                    "canonical_entity_name": group[0].get("canonical_entity_name") or group[0].get("entity"),
                    "formula_id": formula.formula_id,
                    "formula_version": formula.version,
                    "planner": "deterministic_rule",
                    "task_spec": _task_spec(formula, group),
                    "evidence_graph": graph,
                    "evidence_ids": sorted(replay.fact_ids),
                    "evidence": list(group),
                    "answer": _answer(replay),
                    "replay": {
                        "depth": replay.depth,
                        "source_ids": sorted(replay.source_ids),
                        "fact_ids": sorted(replay.fact_ids),
                        "trace": list(replay.trace),
                    },
                }
                score = (-sources, -replay.depth, formula.formula_id, signature)
                candidates.append((score, candidate))
                valid_for_formula += 1
                if valid_for_formula >= self.beam_width:
                    break
        candidates.sort(key=lambda value: value[0])
        return [candidate for _, candidate in candidates[: self.max_graphs_per_entity]]


def graph_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("evidence", [])
    return {
        "formula_id": candidate.get("formula_id"),
        "objective_type": candidate.get("task_spec", {}).get("objective_type"),
        "entities": candidate.get("task_spec", {}).get("entities"),
        "periods": candidate.get("task_spec", {}).get("periods"),
        "metrics": [fact.get("metric") for fact in evidence if isinstance(fact, dict)],
        "fact_roles": candidate.get("task_spec", {}).get("fact_roles"),
        "operation_count": len(candidate.get("replay", {}).get("trace", [])),
    }
