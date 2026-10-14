from __future__ import annotations

import hashlib
from typing import Any, Mapping


def financial_topic(task: Mapping[str, Any]) -> str:
    metrics = " ".join(str(value) for value in task.get("task_spec", {}).get("metrics", []))
    metrics += " " + " ".join(str(fact.get("metric", "")) for fact in task.get("evidence", []))
    text = metrics.casefold()
    for topic, terms in (
        ("revenue", ("revenue", "sales", "收入", "营收")),
        ("profitability", ("profit", "margin", "earnings", "利润", "毛利")),
        ("cash_flow", ("cash flow", "cashflow", "现金流")),
        ("balance_sheet", ("assets", "liabilities", "debt", "资产", "负债")),
        ("valuation", ("valuation", "p/e", "市盈率", "估值")),
    ):
        if any(term in text for term in terms):
            return topic
    return "other"


def matches_filters(task: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    training = config["training"]
    family = task.get("task_type")
    if family not in {"atomic", "long_horizon"}:
        return False
    filters = [("financial_topics", financial_topic(task))]
    if family == "atomic":
        filters.append(("capabilities", task.get("capability")))
    else:
        filters.extend((
            ("objective_types", task.get("task_spec", {}).get("objective_type")),
            ("formula_ids", task.get("formula_id")),
            ("seed_ids", task.get("seed", {}).get("seed_id")),
        ))
    for field, value in filters:
        allowed = training.get(field) or []
        if allowed and (value is None or value not in allowed):
            return False
    return True


def selected(task: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    training = config["training"]
    family = task.get("task_type")
    if not training["enabled"]:
        return False
    if family == "atomic" and not training["atomic_enabled"]:
        return False
    if family == "long_horizon" and not training["long_enabled"]:
        return False
    return matches_filters(task, config)


def split_for_task(task_id: str, fraction: float) -> str:
    value = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12], 16) / 0xFFFFFFFFFFFF
    return "val" if value < fraction else "train"
