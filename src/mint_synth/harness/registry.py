"""Versioned tool registry shared by teacher rollouts and external harness clients."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator

from mint_synth.harness.snapshot import TOOL_DEFINITIONS, SnapshotTools


ToolHandler = Callable[[SnapshotTools, Mapping[str, Any]], Mapping[str, Any]]
RelevanceHandler = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], bool]
DistillHandler = Callable[[SnapshotTools, Mapping[str, Any], Mapping[str, Any]], list[Mapping[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    definition: Mapping[str, Any]
    handler: ToolHandler
    relevance: RelevanceHandler | None = None
    distill: DistillHandler | None = None


class ToolRegistry:
    """Keep tool schemas, implementations and SFT relevance rules together."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", spec.name):
            raise ValueError(f"invalid harness tool name: {spec.name!r}")
        if spec.name in self._specs:
            raise ValueError(f"duplicate harness tool: {spec.name}")
        if not spec.version.strip():
            raise ValueError(f"harness tool {spec.name} needs a version")
        definition = copy.deepcopy(dict(spec.definition))
        function = definition.get("function")
        if definition.get("type") != "function" or not isinstance(function, dict):
            raise ValueError(f"harness tool {spec.name} needs a function definition")
        if function.get("name") != spec.name:
            raise ValueError(f"harness tool definition name differs from {spec.name}")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(f"harness tool {spec.name} needs an object parameters schema")
        Draft202012Validator.check_schema(parameters)
        json.dumps(definition, ensure_ascii=False, allow_nan=False)
        self._specs[spec.name] = ToolSpec(
            name=spec.name, version=spec.version, definition=definition,
            handler=spec.handler, relevance=spec.relevance, distill=spec.distill,
        )

    def copy(self) -> ToolRegistry:
        result = ToolRegistry()
        for spec in self._specs.values():
            result.register(spec)
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(dict(spec.definition)) for spec in self._specs.values()]

    @property
    def fingerprint(self) -> str:
        material = [
            {"name": spec.name, "version": spec.version, "definition": spec.definition}
            for spec in self._specs.values()
        ]
        payload = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def call(
        self, tools: SnapshotTools, name: str, arguments: Mapping[str, Any],
        *, max_observation_chars: int = 65536,
    ) -> dict[str, Any]:
        problem = self.validate(name, arguments)
        if problem:
            return {"error": problem}
        spec = self._specs[name]
        values = dict(arguments)
        try:
            response = spec.handler(tools, values)
        except Exception:
            return {"error": "tool execution failed"}
        if not isinstance(response, Mapping):
            return {"error": "tool returned a non-object observation"}
        observation = dict(response)
        try:
            encoded = json.dumps(observation, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return {"error": "tool returned an invalid JSON observation"}
        if len(encoded) > max_observation_chars:
            return {"error": "tool observation exceeds the context limit"}
        return observation

    def validate(self, name: str, arguments: Mapping[str, Any]) -> str | None:
        spec = self._specs.get(name)
        if spec is None:
            return "unknown tool"
        if not isinstance(arguments, Mapping):
            return "tool arguments must be an object"
        parameters = spec.definition["function"]["parameters"]
        errors = sorted(
            Draft202012Validator(parameters).iter_errors(dict(arguments)),
            key=lambda item: str(item.path),
        )
        return f"invalid tool arguments: {errors[0].message}" if errors else None

    def distill(
        self, tools: SnapshotTools, name: str, arguments: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        spec = self._specs.get(name)
        if spec is None or spec.distill is None or "error" in observation:
            return []
        records = spec.distill(tools, arguments, observation)
        if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
            raise ValueError(f"tool {name} returned invalid evidence records")
        return [dict(record) for record in records]

    def relevant(
        self, name: str, task: Mapping[str, Any], arguments: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> bool:
        spec = self._specs.get(name)
        if spec is None or spec.relevance is None:
            return False
        try:
            return bool(spec.relevance(task, arguments, observation))
        except Exception:
            return False


def _required_chunks(task: Mapping[str, Any]) -> set[str]:
    return {str(fact.get("chunk_id", "")) for fact in task.get("evidence", [])}


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    relevance: dict[str, RelevanceHandler] = {
        "search_corpus": lambda task, _args, observation: any(
            str(hit.get("chunk_id", "")) in _required_chunks(task)
            for hit in observation.get("hits", [])
        ),
        "open_chunk": lambda task, args, _observation: str(args.get("chunk_id", "")) in _required_chunks(task),
        "calculate": lambda _task, _args, _observation: True,
    }
    distillers: dict[str, DistillHandler] = {
        "search_corpus": lambda _tools, _args, _observation: [],
        "open_chunk": lambda _tools, _args, observation: [{
            "kind": "source_chunk", "chunk_id": observation["chunk_id"],
            "source_id": observation["source_id"],
            "available_at": observation["available_at"],
            "excerpt": str(observation["text"])[:300],
        }],
        "calculate": lambda _tools, arguments, observation: [{
            "kind": "calculation", "operation": arguments["operation"],
            "values": arguments["values"], "value": observation["value"],
        }],
    }
    for definition in TOOL_DEFINITIONS:
        name = str(definition["function"]["name"])
        registry.register(ToolSpec(
            name=name, version="1", definition=definition,
            handler=lambda tools, arguments, tool_name=name: tools.call(tool_name, arguments),
            relevance=relevance[name],
            distill=distillers[name],
        ))
    return registry


def load_tool_registry(factory: str | None = None) -> ToolRegistry:
    """Load a user factory named ``package.module:register_tools``."""
    registry = default_tool_registry()
    if not factory:
        return registry
    module_name, separator, function_name = factory.partition(":")
    if not separator or not module_name or not function_name or not function_name.isidentifier():
        raise ValueError("harness.tool_registry must be package.module:register_tools")
    callback = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(callback):
        raise ValueError(f"harness tool factory is not callable: {factory}")
    callback(registry)
    return registry
