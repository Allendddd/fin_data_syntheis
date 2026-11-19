"""Bind native observations to actual resource calls or verified original reads."""

from __future__ import annotations

import json
from pathlib import Path

from mint_synth.harness.resources import ResourceTools, source_observations
from .read_binding import source_paths, observed_source_text
from .trajectory import TrajectoryError


def returned_sources(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if isinstance(value, dict):
        return source_observations(value) + [source for item in value.values() for source in returned_sources(item)]
    if isinstance(value, list):
        return [source for item in value for source in returned_sources(item)]
    return []


def bind_resource_observation(case, trace_dir, use, text, records, root, *, raw_output=None):
    binding_path = trace_dir / "search-backend.json"
    if case.retrieval.get("web", {}).get("search_endpoint") and binding_path.is_file():
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding.get("configured_endpoint") != case.retrieval["web"]["search_endpoint"]:
            raise TrajectoryError("native search binding does not match the configured endpoint")
        name = use["name"]
        selected = name == binding["native_tool"] or (binding["native_tool"] == "web.run" and name == "run" and use.get("namespace") == "web")
        if selected:
            invocations = trace_dir / "search-routing.jsonl"
            matched, observed = False, {}
            returned = {source["chunk_id"]: source for source in returned_sources(text)}
            for line in invocations.read_text(encoding="utf-8").splitlines() if invocations.is_file() else []:
                call = json.loads(line)
                arguments = use["input"]
                if binding["native_tool"] == "web.run":
                    same = call.get("protocol") == "codex_web_run" and call["request"].get("commands") == arguments
                else:
                    query = arguments.get(binding.get("query_field", "query"))
                    same = any(item.get("query") == query for item in call.get("queries", []))
                if not same or call.get("status") != "completed" or call.get("endpoint") != binding["configured_endpoint"]:
                    continue
                matched = True
                for source in call["observation"].get("results", []):
                    record = records.get(source.get("chunk_id"))
                    if record and source.get("url") in text:
                        span = source["text"] if returned.get(source["chunk_id"]) == source else observed_source_text(text, source.get("text", ""))
                        if span:
                            observed[record["chunk_id"]] = {**record, "text": span}
            if not matched:
                raise TrajectoryError("native search result is not bound to the configured backend")
            return observed
    name = use["name"].removeprefix("mcp__finance__")
    invocations = trace_dir / "resource-tools.jsonl"
    actual = {item["chunk_id"]: item for item in returned_sources(text)}
    if invocations.is_file() and name in {"search_documents", "read_document", "search_web", "read_web"}:
        matching = []
        for line in invocations.read_text(encoding="utf-8").splitlines():
            invocation = json.loads(line)
            if invocation["name"] == name and invocation["arguments"] == use["input"]:
                matching.extend(source_observations(invocation["observation"]))
        return {item["chunk_id"]: records[item["chunk_id"]] for item in matching
                if item["chunk_id"] in records and actual.get(item["chunk_id"]) == item}
    if case.retrieval["mode"] == "web":
        from .web_evidence import bind_web_observation
        return bind_web_observation(case, trace_dir, use, raw_output if raw_output is not None else text)
    paths = source_paths(use, root)
    if not paths:
        return {}
    tools = ResourceTools(case.retrieval, case.cutoff, entity_id=case.entity_id, dated_only=case.dated_only, archive=case.archive_path)
    try:
        observed = {}
        for doc_id, doc in tools.documents.items():
            path = doc["path"].resolve()
            if not any(path == p or p in path.parents for p in paths):
                continue
            for unit in tools._units(doc):
                span = observed_source_text(text, unit["text"])
                if span:
                    item = tools._publish(text=span, locator={**unit["locator"], "file": doc["file"]}, sha256=doc["sha256"],
                                          metadata=doc["metadata"], document_id=doc_id)
                    observed[item["chunk_id"]] = tools.sources[item["chunk_id"]]
        return observed
    finally:
        tools.close()
