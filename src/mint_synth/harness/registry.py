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


def resource_tool_registry(mode: str, *, enable_web_adapter: bool = True) -> ToolRegistry:
    from mint_synth.harness.resources import source_observations
    registry = ToolRegistry()
    if mode == "web" and not enable_web_adapter:
        return registry  # Native harness tools remain available; this is only an optional adapter.
    def definition(name, description, properties, required):
        return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}
    text = {"type": "string", "minLength": 1}
    limit = {"type": "integer", "minimum": 1, "maximum": 10}
    span = {"unit": {"type": "integer", "minimum": 1}, "char_start": {"type": "integer", "minimum": 0},
            "max_chars": {"type": "integer", "minimum": 100, "maximum": 16000}}
    if mode == "static":
        definitions = [definition("search_documents", "Search original documents; returns original-file locators and excerpts.",
                                 {"query": text, "limit": limit}, ["query"]),
                       definition("read_document", "Read an original PDF page, DOCX block or text document. Units are one-based.",
                                  {"document_id": text, **span}, ["document_id"])]
    elif mode == "web":
        definitions = [definition("search_web", "Search the live web. Source dates must respect the task cutoff.",
                                 {"query": text, "limit": limit}, ["query"]),
                       definition("read_web", "Fetch and read a web page or PDF; only returned text is acquired evidence.",
                                  {"url": text, **span}, ["url"])]
    else:
        raise ValueError("unknown retrieval mode")
    def distill(_tools, _args, observation):
        return [{"kind": "source_chunk", "chunk_id": item["chunk_id"], "source_id": item["source_id"],
                 "available_at": item["available_at"], "excerpt": item["text"][:300]}
                for item in source_observations(observation)]
    for item in definitions:
        name = item["function"]["name"]
        registry.register(ToolSpec(name, "1", item, lambda tools, args, n=name: tools.call(n, args),
                                  relevance=lambda _task, _args, obs: bool(source_observations(obs)), distill=distill))
    calculate = next(item for item in TOOL_DEFINITIONS if item["function"]["name"] == "calculate")
    registry.register(ToolSpec("calculate", "1", calculate, lambda tools, args: tools.call("calculate", args),
                              relevance=lambda *_args: True))
    return registry


def load_tool_registry(factory: str | None = None, *, retrieval: Mapping[str, Any] | None = None,
                       native_resources: bool = False) -> ToolRegistry:
    """Load a user factory named ``package.module:register_tools``."""
    registry = resource_tool_registry(str(retrieval["mode"]), enable_web_adapter=not native_resources and bool(retrieval.get("web", {}).get("search_endpoint"))) if retrieval else default_tool_registry()
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
