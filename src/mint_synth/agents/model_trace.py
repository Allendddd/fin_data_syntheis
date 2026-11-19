"""Read exposed model text and actions from the existing HTTP capture.

SDK/CLI events remain audit artifacts. Using the model response as the single
content source avoids duplicating text emitted again by an SDK callback.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from .trajectory import TrajectoryError


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TrajectoryError("model tool arguments must be a JSON object")
    return value


def _tool(name: str, arguments: Any, field: str, allowed: set[str]) -> dict[str, Any]:
    for prefix in ("mcp__finance__", "mcp__finance.", "finance."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    arguments = _arguments(arguments)
    if name == "submit_final":
        if set(arguments) == {"answer"} and isinstance(arguments["answer"], str):
            return {"kind": "final", "text": arguments["answer"], "field": field}
        if set(arguments) == {"proof"} and isinstance(arguments["proof"], dict):
            return {"kind": "final", "proof": arguments["proof"], "field": field}
        raise TrajectoryError("model final submission has invalid arguments")
    return {"kind": "tool_call", "name": name, "arguments": arguments, "field": field,
            "known_tool": name in allowed}


def _text(kind: str, value: Any, field: str) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, str):
        raise TrajectoryError(f"unsupported model text at {field}")
    return [{"kind": kind, "text": value, "field": field}]


def response_parts(body: dict[str, Any], protocol: str, allowed: set[str]) -> list[dict[str, Any]]:
    parts = []
    if protocol == "chat_completions":
        choices = body.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") in {"length", "content_filter"}:
            raise TrajectoryError("model response is incomplete or has an ambiguous choice")
        message = choices[0]["message"]
        for field in ("reasoning_content", "reasoning", "analysis"):
            parts.extend(_text("reasoning", message.get(field), "choices[0].message." + field))
        if message.get("reasoning_details"):
            raise TrajectoryError("reasoning_details needs an explicit content adapter")
        content = message.get("content")
        if isinstance(content, list):
            for index, block in enumerate(content):
                if block.get("type") not in {"text", "output_text"}:
                    raise TrajectoryError("unsupported Chat Completions content block")
                parts.extend(_text("text", block.get("text"), f"choices[0].message.content[{index}].text"))
        else:
            parts.extend(_text("text", content, "choices[0].message.content"))
        parts.extend(_text("text", message.get("refusal"), "choices[0].message.refusal"))
        for index, call in enumerate(message.get("tool_calls") or []):
            if call.get("type", "function") != "function":
                raise TrajectoryError("unsupported Chat Completions tool call")
            function = call["function"]
            part = _tool(function["name"], function["arguments"], f"choices[0].message.tool_calls[{index}]", allowed)
            part["tool_call_id"] = call.get("id")
            parts.append(part)
    elif protocol == "responses":
        if body.get("status", "completed") != "completed":
            raise TrajectoryError("Responses result is incomplete")
        for index, item in enumerate(body.get("output", [])):
            field = f"output[{index}]"
            kind = item["type"]
            if kind == "message":
                for offset, block in enumerate(item.get("content", [])):
                    if block["type"] not in {"output_text", "refusal"}:
                        raise TrajectoryError("unsupported Responses message content")
                    key = "text" if block["type"] == "output_text" else "refusal"
                    parts.extend(_text("text", block.get(key), f"{field}.content[{offset}].{key}"))
            elif kind == "reasoning":
                # Summaries are exposed summaries, never reconstructed private reasoning.
                for key, output_kind in (("summary", "reasoning_summary"), ("content", "reasoning")):
                    for offset, block in enumerate(item.get(key) or []):
                        if block.get("type") not in {"summary_text", "reasoning_text", "text"}:
                            raise TrajectoryError("unsupported exposed reasoning block")
                        parts.extend(_text(output_kind, block.get("text"), f"{field}.{key}[{offset}].text"))
            elif kind in {"function_call", "mcp_call"}:
                if kind == "mcp_call" and item.get("server_label") != "finance":
                    raise TrajectoryError("Responses MCP call is outside the finance tool set")
                part = _tool(item["name"], item["arguments"], field, allowed)
                part["tool_call_id"] = item.get("call_id", item.get("id"))
                parts.append(part)
            elif kind not in {"compaction", "mcp_list_tools"}:
                raise TrajectoryError(f"unsupported Responses output item: {kind}")
    elif protocol == "anthropic":
        if body.get("stop_reason") in {"max_tokens", "model_context_window_exceeded"}:
            raise TrajectoryError("Anthropic result is incomplete")
        for index, block in enumerate(body.get("content", [])):
            field = f"content[{index}]"
            kind = block["type"]
            if kind == "text":
                parts.extend(_text("text", block.get("text"), field + ".text"))
            elif kind == "thinking":
                parts.extend(_text("reasoning", block.get("thinking"), field + ".thinking"))
            elif kind == "tool_use":
                part = _tool(block["name"], block["input"], field, allowed)
                part["tool_call_id"] = block.get("id")
                parts.append(part)
            elif kind != "redacted_thinking":
                raise TrajectoryError(f"unsupported Anthropic content block: {kind}")
    else:
        raise TrajectoryError(f"unsupported model protocol: {protocol}")
    return parts


def _sse_events(text: str) -> tuple[list[dict[str, Any]], bool]:
    events, done = [], False
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
        if not data:
            continue
        if data == "[DONE]":
            done = True
        else:
            event = json.loads(data)
            if not isinstance(event, dict) or event.get("type") == "error" or "error" in event:
                raise TrajectoryError("model stream contains an error")
            events.append(event)
    return events, done


def streamed_parts(text: str, protocol: str, allowed: set[str]) -> list[dict[str, Any]]:
    events, done = _sse_events(text)
    if protocol == "chat_completions":
        parts: list[dict[str, Any]] = []
        calls: dict[int, dict[str, Any]] = {}
        finish = None

        def fragment(kind: str, value: Any):
            if value in (None, ""):
                return
            if not isinstance(value, str):
                raise TrajectoryError("unsupported streamed assistant content")
            if parts and parts[-1]["kind"] == kind:
                parts[-1]["text"] += value
            else:
                parts.append({"kind": kind, "text": value, "field": f"SSE.delta.{kind}"})

        for event in events:
            choices = event.get("choices", [])
            if not choices:
                continue  # Usage-only chunks.
            if len(choices) != 1 or choices[0].get("index", 0) != 0:
                raise TrajectoryError("ambiguous streamed model choices")
            choice = choices[0]
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            fragment("reasoning", delta.get("reasoning_content"))
            fragment("text", delta.get("content"))
            fragment("text", delta.get("refusal"))
            for call in delta.get("tool_calls") or []:
                index = call["index"]
                if index not in calls:
                    calls[index] = {"kind": "pending_tool", "name": "", "arguments": "",
                                    "field": f"SSE.delta.tool_calls[{index}]", "tool_call_id": None}
                    parts.append(calls[index])
                function = call.get("function", {})
                calls[index]["name"] += function.get("name") or ""
                calls[index]["tool_call_id"] = call.get("id") or calls[index]["tool_call_id"]
                calls[index]["arguments"] += function.get("arguments") or ""
        if not done or finish not in {"stop", "tool_calls", "function_call"}:
            raise TrajectoryError("Chat Completions stream has no successful terminal event")
        return [{**_tool(part["name"], part["arguments"], part["field"], allowed), "tool_call_id": part["tool_call_id"]}
                if part["kind"] == "pending_tool" else part for part in parts]
    if protocol == "responses":
        response = None
        items = {}
        fragments: dict[tuple, str] = {}
        for event in events:
            kind = event.get("type", "")
            if kind in {"response.failed", "response.incomplete"}:
                raise TrajectoryError("Responses stream failed or is incomplete")
            if kind == "response.completed":
                response = event["response"]
            elif kind == "response.output_item.done":
                items[event["output_index"]] = event["item"]
            elif kind.endswith(".delta"):
                if kind not in {"response.output_text.delta", "response.reasoning_summary_text.delta",
                                "response.reasoning_text.delta", "response.function_call_arguments.delta"}:
                    raise TrajectoryError(f"unsupported Responses stream delta: {kind}")
                key = (kind, event.get("output_index"), event.get("content_index"), event.get("summary_index"))
                fragments[key] = fragments.get(key, "") + event["delta"]
        if response is None:
            raise TrajectoryError("Responses stream has no completed response")
        if not response.get("output") and items:
            response = {**response, "output": [items[key] for key in sorted(items)]}
        parts = response_parts(response, protocol, allowed)
        # Never silently drop streamed text absent from the completed payload.
        for (kind, *_), value in fragments.items():
            if kind == "response.function_call_arguments.delta":
                continue
            if value and not any(value in part.get("text", "") for part in parts):
                raise TrajectoryError("streamed model text is missing from the completed response")
        return parts
    if protocol == "anthropic":
        blocks, arguments = {}, {}
        stopped, stop_reason = False, None
        closed = set()
        for event in events:
            kind = event.get("type")
            if kind == "content_block_start":
                blocks[event["index"]] = dict(event["content_block"])
            elif kind == "content_block_delta":
                index, delta = event["index"], event["delta"]
                block = blocks[index]
                delta_kind = delta["type"]
                if delta_kind in {"text_delta", "thinking_delta"}:
                    field = "text" if delta_kind == "text_delta" else "thinking"
                    block[field] = block.get(field, "") + delta[field]
                elif delta_kind == "input_json_delta":
                    arguments[index] = arguments.get(index, "") + delta["partial_json"]
                elif delta_kind != "signature_delta":
                    raise TrajectoryError(f"unsupported Anthropic stream delta: {delta_kind}")
            elif kind == "content_block_stop":
                closed.add(event["index"])
            elif kind == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason") or stop_reason
            elif kind == "message_stop":
                stopped = True
        if not stopped or not stop_reason or set(blocks) != closed:
            raise TrajectoryError("Anthropic stream has incomplete content blocks")
        for index, value in arguments.items():
            blocks[index]["input"] = _arguments(value)
        return response_parts({"content": [blocks[key] for key in sorted(blocks)],
                               "stop_reason": stop_reason}, protocol, allowed)
    raise TrajectoryError("unsupported streaming model protocol")


def iter_captured_calls(trace_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield verified raw payloads without changing tool names or schemas."""
    calls = trace_dir / "model" / "calls"
    records = []
    for path in calls.glob("*.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("capture_kind") == "search_backend":
            continue  # Native search-service RPCs are observations, not teacher predictions.
        target = metadata.get("request", {}).get("target", "").split("?", 1)[0].rstrip("/")
        protocol = next((value for suffix, value in (("/chat/completions", "chat_completions"),
                        ("/responses", "responses"), ("/messages", "anthropic")) if target.endswith(suffix)), None)
        if protocol is None:
            continue  # e.g. a CLI's GET /models probe; the raw record is retained.
        records.append((metadata.get("started_at", ""), path, metadata, protocol))
    for _, path, metadata, protocol in sorted(records, key=lambda row: row[0]):
        response = metadata.get("response", {})
        status = response.get("status", 0)
        if status >= 400:
            continue  # Failed HTTP attempts have no executed model action.
        if metadata.get("state") != "complete" or not response.get("complete") or status != 200:
            raise TrajectoryError("model HTTP capture is incomplete")

        def body(record):
            name = record["body_file"]
            if Path(name).name != name:
                raise TrajectoryError("model capture contains an invalid body path")
            value = (path.parent / name).read_bytes()
            if hashlib.sha256(value).hexdigest() != record.get("body_sha256"):
                raise TrajectoryError("model capture body fingerprint mismatch")
            return value

        request = json.loads(body(metadata["request"]))
        raw = body(response)
        headers = {name.lower(): value for name, value in response.get("headers", [])}
        encoding = headers.get("content-encoding", "identity").lower()
        if encoding == "gzip":
            raw = gzip.decompress(raw)
        elif encoding == "deflate":
            raw = zlib.decompress(raw)
        elif encoding not in {"", "identity"}:
            raise TrajectoryError(f"unsupported model capture encoding: {encoding}")
        text = raw.decode("utf-8")
        streaming = "text/event-stream" in headers.get("content-type", "") or text.lstrip().startswith(("data:", "event:"))
        yield {"call_id": metadata["call_id"], "source": path.relative_to(trace_dir).as_posix(),
               "response_file": (path.parent / response["body_file"]).relative_to(trace_dir).as_posix(),
               "protocol": protocol, "request": request, "response_text": text, "streaming": streaming}


def load_model_turns(trace_dir: Path, allowed: set[str]) -> list[dict[str, Any]]:
    turns = []
    failures = {}
    for position, call in enumerate(iter_captured_calls(trace_dir)):
        request = call["request"]
        for index, message in enumerate(request.get("messages", [])):
            if message.get("role") == "tool":
                failures.setdefault(message.get("tool_call_id"), (message.get("content"), call["source"], f"request.messages[{index}].content", position))
            elif isinstance(message.get("content"), list):
                for block_index, block in enumerate(message["content"]):
                    if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                        failures.setdefault(block.get("tool_use_id"), (block, call["source"], f"request.messages[{index}].content[{block_index}]", position))
        for index, item in enumerate(request.get("input", []) if isinstance(request.get("input"), list) else []):
            if item.get("type") == "function_call_output":
                failures.setdefault(item.get("call_id"), (item.get("output"), call["source"], f"request.input[{index}].output", position))
        text, protocol = call["response_text"], call["protocol"]
        parts = streamed_parts(text, protocol, allowed) if call["streaming"] else response_parts(json.loads(text), protocol, allowed)
        turns.append({key: call[key] for key in ("call_id", "source", "response_file", "protocol")})
        turns[-1].update(assistant_parts=parts, cache_hit=False, capture_position=position,
                         request_messages_field=next((key for key in ("messages", "input") if key in call["request"]), None))
    if not turns:
        raise TrajectoryError("no complete model exchanges were captured; tool-only export is disabled")
    from .native_audit import _failed_result

    final_position = next((turn["capture_position"] for turn in turns
                           if any(part["kind"] == "final" for part in turn["assistant_parts"])), len(turns))
    for turn in turns:
        for part in turn["assistant_parts"]:
            if part.get("kind") == "tool_call" and part.get("known_tool") is False:
                failure = failures.get(part.get("tool_call_id"))
                if (failure is not None and turn["capture_position"] < failure[3] <= final_position
                        and _failed_result(failure[0])):
                    part.update(failed_observation=failure[0], observation_source=failure[1], observation_field=failure[2])
    return turns
