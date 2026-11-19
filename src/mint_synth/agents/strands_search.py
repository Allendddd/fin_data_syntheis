"""Bind a Strands SDK search entry point while retaining its schema and name."""

from __future__ import annotations

import asyncio
import copy
import json

import httpx


def _spec(tool):
    value = getattr(tool, "tool_spec", getattr(tool, "TOOL_SPEC", None))
    if isinstance(value, dict):
        return value
    name = getattr(tool, "__name__", "").rsplit(".", 1)[-1]
    return getattr(getattr(tool, name, None), "tool_spec", None)


def bind_search_tool(tools, binding):
    from strands.tools.tools import PythonAgentTool
    name, query_field, limit_field = binding["tool"], binding["query_field"], binding["limit_field"]
    matches = [(index, spec) for index, tool in enumerate(tools) if isinstance(spec := _spec(tool), dict) and spec.get("name") == name]
    if len(matches) > 1:
        raise ValueError("multiple SDK tools match the configured search entry point")
    if matches:
        index, spec = matches[0]
        spec = copy.deepcopy(spec)
        schema = spec.get("inputSchema", {}).get("json", {})
        if query_field not in schema.get("properties", {}) or query_field not in schema.get("required", []):
            raise ValueError("the selected Strands search tool must have a required string query parameter")
    else:
        if name != "web_search":
            raise ValueError(f"configured Strands search tool {name!r} is not registered")
        index = len(tools)
        spec = {"name": name, "description": "Search the web for relevant sources.", "inputSchema": {"json": {
            "type": "object", "properties": {query_field: {"type": "string"}, limit_field: {"type": "integer", "minimum": 1}},
            "required": [query_field]}}}

    async def call(tool_use, **_state):
        def request():
            arguments = dict(tool_use.get("input", {}))
            arguments["query"] = arguments.pop(query_field)
            if limit_field in arguments:
                arguments["limit"] = arguments.pop(limit_field)
            response = httpx.post(binding["url"], json=arguments, timeout=60, trust_env=False)
            response.raise_for_status()
            return response.json()
        try:
            result = await asyncio.to_thread(request)
            return {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": json.dumps(result, ensure_ascii=False)}]}
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return {"toolUseId": tool_use["toolUseId"], "status": "error", "content": [{"text": str(exc)}]}

    bound = PythonAgentTool(tool_name=name, tool_spec=spec, tool_func=call)
    result = list(tools)
    if index == len(result):
        result.append(bound)
    else:
        result[index] = bound
    return result
