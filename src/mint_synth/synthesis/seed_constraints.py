"""Compile analyst seed intent into supported, pre-planning formula constraints."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from mint_synth.synthesis.dag import FormulaSpec


OBJECTIVE_BY_GENERATOR = {
    "temporal_pair": "change", "temporal_trend": "change", "ratio_change": "change",
    "compatible_ratio": "ratio", "compatible_average": "aggregation",
    "compatible_verification": "verification", "multi_verification": "verification",
}

_ROLE_SETS = {
    "temporal_pair": ({"new period value", "new value"}, {"old period value", "old value"}),
    "temporal_trend": ({"old period value"}, {"middle period value"}, {"new period value"}),
    "ratio_change": ({"new period numerator"}, {"new period denominator"},
                     {"old period numerator"}, {"old period denominator"}),
    "compatible_ratio": ({"numerator"}, {"denominator"}),
    "compatible_average": ({"two or more compatible observations", "multiple compatible observations"},),
    "compatible_verification": ({"claim observation", "first claim component"},
                                {"comparison observation", "another component"}),
    "multi_verification": ({"evidence for first claim component", "first claim component"},
                           {"evidence for another component", "another component"}),
}


def _words(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).casefold())).strip()


def _form_generator(template: Mapping[str, Any]) -> str | None:
    form = _words(template.get("reasoning_form"))
    objective = template.get("objective_type")
    if objective == "comparison":
        return None  # No comparison generator currently exists in the registered topology.
    if objective == "change":
        if "ratio" in form and "change" in form:
            return "ratio_change"
        if "trend" in form or "three period" in form:
            return "temporal_trend"
        return "temporal_pair" if "period" in form and "change" in form else None
    if objective == "ratio":
        return "compatible_ratio" if "ratio" in form else None
    if objective == "aggregation":
        return "compatible_average" if "aggregate" in form or "average" in form else None
    if objective == "verification":
        if any(term in form for term in ("combine", "boolean", "claim component", "multi part")):
            return "multi_verification"
        return "compatible_verification" if "test" in form or "verify" in form else None
    return None


def formulas_for_seed(seed: Mapping[str, Any], formulas: Sequence[FormulaSpec]) -> tuple[FormulaSpec, ...]:
    template = seed["task_spec_template"]
    generator = _form_generator(template)
    if generator is None:
        if template.get("objective_type") == "comparison" and seed.get("origin") == "built_in_approximation":
            return ()
        raise ValueError(f"seed {seed['seed_id']!r} has unsupported reasoning_form")
    required = [_words(role) for role in template["required_fact_roles"]]
    allowed = _ROLE_SETS[generator]
    if len(required) != len(allowed) or any(role not in aliases for role, aliases in zip(required, allowed)):
        raise ValueError(f"seed {seed['seed_id']!r} has unsupported required_fact_roles for {generator}")
    matches = tuple(formula for formula in formulas if formula.generator == generator)
    if not matches and seed.get("origin") != "built_in_approximation":
        raise ValueError(f"seed {seed['seed_id']!r} has no enabled formula for {generator}")
    return matches
