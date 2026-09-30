from __future__ import annotations

import random
import threading
from typing import Any

from .audit import audit_long, replay_summary
from .jsonl import read_jsonl
from .llm import JsonLLM
from .prompts import (
    AUDIT_SYSTEM,
    LONG_GRAPH_SYSTEM,
    QUERY_SYSTEM,
    SEED_SYSTEM,
    long_audit_user,
    long_graph_user,
    query_user,
    seed_user,
)
from .replay import ReplayError, answer_matches, replay_graph
from .util import normalize_text, stable_id, token_set


DEFAULT_SEEDS: tuple[dict[str, Any], ...] = (
    {
        "seed_id": "seed_compare",
        "seed_query": "Compare a financially meaningful metric across two disclosed scopes and quantify the difference.",
        "task_spec_template": {
            "objective_type": "comparison",
            "reasoning_form": "retrieve comparable scoped values, normalize only when justified, and compare them",
            "required_fact_roles": ["first comparable value", "second comparable value"],
            "time_sensitive": False,
            "requested_output": "the comparison and its magnitude",
        },
        "origin": "built_in_approximation",
    },
    {
        "seed_id": "seed_change",
        "seed_query": "Determine the change in a reported financial measure between two periods and state its magnitude.",
        "task_spec_template": {
            "objective_type": "change",
            "reasoning_form": "retrieve the same compatible metric for two periods and compute absolute or percentage change",
            "required_fact_roles": ["new-period value", "old-period value"],
            "time_sensitive": False,
            "requested_output": "the direction and magnitude of change",
        },
        "origin": "built_in_approximation",
    },
    {
        "seed_id": "seed_ratio",
        "seed_query": "Use separately disclosed values to calculate a decision-relevant financial ratio.",
        "task_spec_template": {
            "objective_type": "ratio",
            "reasoning_form": "retrieve a numerator and denominator with compatible financial scope and calculate their ratio",
            "required_fact_roles": ["numerator", "denominator"],
            "time_sensitive": False,
            "requested_output": "a reproducible ratio",
        },
        "origin": "built_in_approximation",
    },
    {
        "seed_id": "seed_aggregate",
        "seed_query": "Combine multiple independently reported observations into a requested financial aggregate.",
        "task_spec_template": {
            "objective_type": "aggregation",
            "reasoning_form": "retrieve multiple compatible observations and aggregate them with an explicit operation",
            "required_fact_roles": ["two or more compatible observations"],
            "time_sensitive": False,
            "requested_output": "the aggregate and its scope",
        },
        "origin": "built_in_approximation",
    },
    {
        "seed_id": "seed_verify",
        "seed_query": "Verify a multi-part financial claim by reconciling evidence from more than one source.",
        "task_spec_template": {
            "objective_type": "verification",
            "reasoning_form": "retrieve each claim component, test it, and combine the boolean results",
            "required_fact_roles": ["evidence for first claim component", "evidence for another component"],
            "time_sensitive": False,
            "requested_output": "whether the full claim is supported",
        },
        "origin": "built_in_approximation",
    },
)


def load_seed_bank(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(seed) for seed in DEFAULT_SEEDS]
    seeds: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path), 1):
        query = row.get("seed_query", row.get("text", row.get("question")))
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}:{index}: seed requires seed_query, text, or question")
        seeds.append(
            {
                "seed_id": str(row.get("seed_id") or stable_id("seed", index, query)),
                "seed_query": query.strip(),
                "task_spec_template": row.get("task_spec_template"),
                "origin": row.get("origin", "user_seed_bank"),
            }
        )
    if not seeds:
        raise ValueError("seed bank is empty")
    return seeds


def load_evaluation_questions(path: str | None, *, max_questions: int | None = None) -> list[str]:
    if path is None:
        return []
    questions: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(read_jsonl(path), 1):
        value = row.get("question", row.get("prompt", row.get("text")))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}:{index}: evaluation row requires question, prompt, or text")
        normalized = value.strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        questions.append(normalized)
        if max_questions is not None and len(questions) > max_questions:
            raise ValueError(f"evaluation file exceeds configured maximum of {max_questions} unique questions")
    return questions


def _similarity(anchor: dict[str, Any], candidate: dict[str, Any]) -> float:
    score = 0.0
    if normalize_text(anchor.get("entity")) == normalize_text(candidate.get("entity")):
        score += 6.0
    metric_a = token_set(str(anchor.get("metric", "")))
    metric_b = token_set(str(candidate.get("metric", "")))
    if metric_a and metric_b:
        score += 5.0 * len(metric_a & metric_b) / len(metric_a | metric_b)
    if (
        normalize_text(anchor.get("unit")) == normalize_text(candidate.get("unit"))
        and normalize_text(anchor.get("scale")) == normalize_text(candidate.get("scale"))
        and normalize_text(anchor.get("currency")) == normalize_text(candidate.get("currency"))
    ):
        score += 3.0
    if normalize_text(anchor.get("period")) == normalize_text(candidate.get("period")):
        score += 1.0
    if anchor.get("source_id") != candidate.get("source_id"):
        score += 2.0
    return score


def sample_fact_bundles(
    facts: list[dict[str, Any]],
    *,
    count: int,
    bundle_size: int,
    min_sources: int,
    random_seed: int,
) -> list[list[dict[str, Any]]]:
    eligible = [fact for fact in facts if fact.get("value_num") is not None]
    sources = {str(fact.get("source_id")) for fact in eligible}
    if count <= 0 or len(sources) < min_sources:
        return []
    rng = random.Random(random_seed)
    anchors = eligible[:]
    rng.shuffle(anchors)
    bundles: list[list[dict[str, Any]]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = max(count * 5, len(anchors))
    for attempt in range(attempts):
        anchor = anchors[attempt % len(anchors)]
        ranked = [fact for fact in eligible if fact["fact_id"] != anchor["fact_id"]]
        rng.shuffle(ranked)
        ranked.sort(key=lambda fact: _similarity(anchor, fact), reverse=True)
        selected = [anchor]
        selected_ids = {anchor["fact_id"]}
        selected_sources = {str(anchor.get("source_id"))}

        # First guarantee source breadth, selecting the most semantically related fact from each new source.
        for fact in ranked:
            source_id = str(fact.get("source_id"))
            if source_id in selected_sources:
                continue
            selected.append(fact)
            selected_ids.add(fact["fact_id"])
            selected_sources.add(source_id)
            if len(selected_sources) >= min_sources:
                break
        if len(selected_sources) < min_sources:
            continue
        for fact in ranked:
            if len(selected) >= bundle_size:
                break
            if fact["fact_id"] not in selected_ids:
                selected.append(fact)
                selected_ids.add(fact["fact_id"])
        signature = tuple(sorted(selected_ids))
        if signature in seen:
            continue
        seen.add(signature)
        bundles.append(selected)
        if len(bundles) >= count:
            break
    return bundles


class LongHorizonSynthesizer:
    def __init__(self, constructor: JsonLLM, auditor: JsonLLM, *, language: str = "auto"):
        self.constructor = constructor
        self.auditor = auditor
        self.language = language
        self._parsed_seeds: dict[str, dict[str, Any]] = {}
        self._seed_lock = threading.Lock()

    def _seed_template(self, seed: dict[str, Any]) -> dict[str, Any]:
        existing = seed.get("task_spec_template")
        if isinstance(existing, dict):
            return existing
        seed_id = str(seed["seed_id"])
        if seed_id not in self._parsed_seeds:
            with self._seed_lock:
                if seed_id not in self._parsed_seeds:
                    response = self.constructor.complete_json(
                        system=SEED_SYSTEM,
                        user=seed_user(seed["seed_query"]),
                        temperature=0.0,
                        max_tokens=1200,
                    )
                    template = response.get("task_spec_template") if isinstance(response, dict) else None
                    if not isinstance(template, dict):
                        raise ValueError(f"could not parse seed {seed_id}")
                    self._parsed_seeds[seed_id] = template
        return self._parsed_seeds[seed_id]

    def synthesize(
        self,
        seed: dict[str, Any],
        facts: list[dict[str, Any]],
        *,
        cutoff: str,
        min_depth: int,
        min_sources: int,
        evaluation_questions: list[str] | None = None,
        candidate_index: int = 0,
        overlap_threshold: float = 0.82,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        try:
            template = self._seed_template(seed)
        except (ValueError, RuntimeError) as exc:
            return None, self._rejection(seed, "seed_parse", str(exc), None)
        graph_response = self.constructor.complete_json(
            system=LONG_GRAPH_SYSTEM,
            user=long_graph_user(
                {**seed, "task_spec_template": template}, facts, min_depth, min_sources
            ),
            temperature=0.5,
            max_tokens=6000,
        )
        candidate = graph_response.get("candidate") if isinstance(graph_response, dict) else None
        if not isinstance(candidate, dict):
            reason = graph_response.get("reason", "graph constructor returned no candidate") if isinstance(graph_response, dict) else "invalid graph response"
            return None, self._rejection(seed, "graph_construction", reason, graph_response)

        graph = candidate.get("evidence_graph")
        answer = candidate.get("answer")
        task_spec = candidate.get("task_spec")
        fact_index = {fact["fact_id"]: fact for fact in facts}
        try:
            replay = replay_graph(graph if isinstance(graph, dict) else {}, fact_index)
        except ReplayError as exc:
            return None, self._rejection(seed, "graph_replay", str(exc), candidate)
        if replay.depth < min_depth:
            return None, self._rejection(seed, "graph_gate", f"depth {replay.depth} < {min_depth}", candidate)
        if len(replay.source_ids) < min_sources:
            return None, self._rejection(seed, "graph_gate", f"source breadth {len(replay.source_ids)} < {min_sources}", candidate)
        if not isinstance(answer, dict) or not answer_matches(answer, replay):
            return None, self._rejection(seed, "graph_gate", "answer does not match replay", candidate)
        if not isinstance(task_spec, dict):
            return None, self._rejection(seed, "graph_gate", "task_spec is missing", candidate)

        graph_summary = self._graph_summary(graph, fact_index)
        query_response = self.constructor.complete_json(
            system=QUERY_SYSTEM,
            user=query_user(seed, task_spec, graph_summary, cutoff, self.language),
            temperature=0.4,
            max_tokens=1200,
        )
        question = query_response.get("question") if isinstance(query_response, dict) else None
        if not isinstance(question, str) or not question.strip():
            return None, self._rejection(seed, "query_realization", "query constructor returned no question", query_response)

        evidence_ids = sorted(replay.fact_ids)
        evidence = [fact_index[fact_id] for fact_id in evidence_ids]
        task: dict[str, Any] = {
            "task_id": stable_id("long", seed["seed_id"], candidate_index, question),
            "task_type": "long_horizon",
            "question": question.strip(),
            "answer": answer,
            "cutoff": cutoff,
            "task_spec": task_spec,
            "seed": {
                "seed_id": seed["seed_id"],
                "seed_query": seed["seed_query"],
                "origin": seed.get("origin"),
                "task_spec_template": template,
            },
            "source_pack": sorted(
                {
                    (str(fact["source_id"]), str(fact["chunk_id"]))
                    for fact in evidence
                }
            ),
            "evidence_ids": evidence_ids,
            "evidence_graph": graph,
        }
        # JSON has no tuple type; name the source-pack fields explicitly.
        task["source_pack"] = [
            {"source_id": source_id, "chunk_id": chunk_id}
            for source_id, chunk_id in task["source_pack"]
        ]
        audit_response = self.auditor.complete_json(
            system=AUDIT_SYSTEM,
            user=long_audit_user(task, evidence, replay_summary(replay)),
            temperature=0.0,
            max_tokens=1500,
        )
        task["semantic_audit"] = audit_response if isinstance(audit_response, dict) else {}
        result = audit_long(
            task,
            fact_index,
            min_depth=min_depth,
            min_sources=min_sources,
            seed_text=seed["seed_query"],
            evaluation_questions=evaluation_questions,
            overlap_threshold=overlap_threshold,
        )
        task["verification"] = {
            "accepted": result.accepted,
            "reasons": list(result.reasons),
            "replay": replay_summary(result.replay) if result.replay else None,
        }
        if not result.accepted:
            return None, self._rejection(seed, "long_task_audit", "; ".join(result.reasons), task)
        return task, None

    @staticmethod
    def _graph_summary(graph: dict[str, Any], fact_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
        leaves: list[dict[str, Any]] = []
        operations: list[dict[str, Any]] = []
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            if node.get("kind") == "fact":
                fact = fact_index.get(str(node.get("fact_id", node.get("id"))))
                if fact:
                    leaves.append(
                        {
                            "entity": fact.get("entity"),
                            "period": fact.get("period"),
                            "metric": fact.get("metric"),
                            "unit": fact.get("unit"),
                            "scale": fact.get("scale"),
                            "currency": fact.get("currency"),
                        }
                    )
            elif node.get("kind") == "operation":
                operations.append({"op": node.get("op"), "input_count": len(node.get("inputs", []))})
        return {"required_fact_semantics": leaves, "operation_shape": operations}

    @staticmethod
    def _rejection(seed: dict[str, Any], stage: str, reason: str, candidate: Any) -> dict[str, Any]:
        return {
            "stage": stage,
            "seed_id": seed.get("seed_id"),
            "reason": reason,
            "candidate": candidate,
        }
