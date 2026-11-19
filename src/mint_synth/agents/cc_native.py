"""Per-request Claude Code samples with native Anthropic content and tools."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import atomic_json

from .model_trace import _sse_events, iter_captured_calls
from .trajectory import TrajectoryError


SAMPLE_FORMAT = "claude_code_native_v2"


def anthropic_response(text: str, *, streaming: bool) -> dict[str, Any]:
    """Reassemble the visible response, including signatures and tool IDs."""
    if not streaming:
        body = json.loads(text)
    else:
        events, _ = _sse_events(text)
        body, blocks, arguments, closed = None, {}, {}, set()
        stopped = False
        for event in events:
            kind = event.get("type")
            if kind == "message_start":
                if body is not None:
                    raise TrajectoryError("multiple Anthropic messages in one model response")
                body = copy.deepcopy(event["message"])
            elif kind == "content_block_start":
                index = event["index"]
                if index in blocks:
                    raise TrajectoryError("duplicate Anthropic content block")
                blocks[index] = copy.deepcopy(event["content_block"])
            elif kind == "content_block_delta":
                index, delta = event["index"], event["delta"]
                block, delta_kind = blocks[index], delta["type"]
                fields = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}
                if delta_kind in fields:
                    field = fields[delta_kind]
                    block[field] = block.get(field, "") + delta[field]
                elif delta_kind == "input_json_delta":
                    arguments[index] = arguments.get(index, "") + delta["partial_json"]
                elif delta_kind == "citations_delta":
                    block.setdefault("citations", []).append(copy.deepcopy(delta["citation"]))
                else:
                    raise TrajectoryError(f"unsupported native Anthropic delta: {delta_kind}")
            elif kind == "content_block_stop":
                closed.add(event["index"])
            elif kind == "message_delta":
                if body is None:
                    raise TrajectoryError("Anthropic stream is missing message_start")
                body.update(copy.deepcopy(event.get("delta", {})))
                body.setdefault("usage", {}).update(copy.deepcopy(event.get("usage", {})))
            elif kind == "message_stop":
                stopped = True
            elif kind != "ping":
                raise TrajectoryError(f"unsupported native Anthropic event: {kind}")
        if body is None or not stopped or set(blocks) != closed:
            raise TrajectoryError("native Anthropic stream is incomplete")
        for index, value in arguments.items():
            blocks[index]["input"] = json.loads(value)
        body["content"] = [blocks[key] for key in sorted(blocks)]
    if (not isinstance(body, dict) or not isinstance(body.get("content"), list)
            or body.get("role", "assistant") != "assistant"
            or body.get("stop_reason") not in {"end_turn", "tool_use", "stop_sequence", "pause_turn"}):
        raise TrajectoryError("native Anthropic response is incomplete or unsupported")
    return body


def export_cc_native(snapshot: Mapping[str, Any], trace_dir: Path) -> dict[str, Any]:
    """Keep request-specific tools/context and supervise only this response.

    The file is a native Anthropic dataset, not an ms-swift tool template.
    Its final reply and source reads must already have passed offline grading.
    """
    fingerprint = hashlib.sha256()
    for path in sorted((trace_dir / "model/calls").glob("*.json")):
        fingerprint.update(path.read_bytes())
    trajectory_id = fingerprint.hexdigest()
    if not snapshot.get("audit", {}).get("verdict", {}).get("accepted") or not snapshot.get("final_call_id"):
        raise TrajectoryError("CC native export requires an accepted offline audit")
    source_map = []
    sample_file = trace_dir / "cc-native.jsonl"
    descriptor = os.open(sample_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    final_seen = False
    web = bool(snapshot.get("unrestricted_web"))
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for call in iter_captured_calls(trace_dir):
            if call["protocol"] != "anthropic":
                raise TrajectoryError("CC native export requires captured Anthropic Messages requests")
            request = call["request"]
            context = request.get("messages")
            tools = request.get("tools", [])
            if not isinstance(context, list) or not context or not isinstance(tools, list):
                raise TrajectoryError("CC native request has invalid messages or tools")
            names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
            if web:
                if any(not isinstance(tool, dict) or not isinstance(tool.get("name", tool.get("type")), str) for tool in tools):
                    raise TrajectoryError("CC native request has invalid tool definitions")
                # Provider-managed MCP toolsets identify the server rather than
                # enumerate every tool name in the client request.
                names = [tool["name"] for tool in tools if "name" in tool]
            if (not web and len(names) != len(tools)) or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
                raise TrajectoryError("CC native request has invalid or duplicate tool definitions")
            from .native_capture import resource_mcp
            if any(name.startswith("mcp__") and not (web or snapshot.get("resource_mode") and resource_mcp(name)) for name in names):
                raise TrajectoryError("CC native request contains MCP tool definitions")
            response = anthropic_response(call["response_text"], streaming=call["streaming"])
            content = response["content"]
            block_loss = []
            flags = snapshot.get("action_relevance", {})
            uses = [block for block in content if block.get("type") in ({"tool_use", "server_tool_use", "mcp_tool_use"} if web else {"tool_use"})]
            if any(block.get("id") not in flags for block in uses):
                raise TrajectoryError("CC native export has no action relevance decision")
            text_loss = all(flags.get(block.get("id"), {}).get("relevant", True) for block in uses)
            for block in content:
                if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                    raise TrajectoryError("CC native response contains an invalid block")
                loss = block["type"] not in {"redacted_thinking", "tool_result"} and not block["type"].endswith("_tool_result")
                loss = loss and text_loss
                if block["type"] in {"tool_use", "server_tool_use", "mcp_tool_use"}:
                    name, arguments = block.get("name"), block.get("input")
                    if ((name not in names and block.get("id") not in snapshot.get("failed_tool_ids", []) and not web)
                            or not isinstance(arguments, dict) or not isinstance(block.get("id"), str)):
                        raise TrajectoryError("CC native tool call is absent from its request tools or has invalid input/ID")
                    if name.startswith("mcp__") and not (web or snapshot.get("resource_mode") and resource_mcp(name)):
                        raise TrajectoryError("CC native capture contains an MCP tool call")
                    loss = flags.get(block["id"], {}).get("relevant", True)
                block_loss.append(loss)
            if call["call_id"] == snapshot["final_call_id"]:
                if response.get("id") != snapshot["final_message_id"] or response.get("stop_reason") != "end_turn":
                    raise TrajectoryError("CC native final response differs from the offline audit")
                final_seen = True
            sample = copy.deepcopy(request)
            sample["messages"] = [*sample["messages"], {"role": "assistant", "content": copy.deepcopy(content)}]
            sample.update(sample_format=SAMPLE_FORMAT, trajectory_id=trajectory_id, call_id=call["call_id"],
                          target_message_index=len(context), loss_mask=[False] * len(context) + [any(block_loss)],
                          target_block_loss_mask=block_loss,
                          response_metadata={key: value for key, value in response.items() if key not in {"content", "role"}})
            if web:
                sample["target_block_field_loss_mask"] = {
                    str(index): {key: key in {"text", "thinking", "name", "input"} for key in block}
                    for index, block in enumerate(content)}
            stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            source_map.append({"sample_index": len(source_map), "call_id": call["call_id"],
                               "source": call["source"], "response_file": call["response_file"],
                               "target_message_index": len(context),
                               "read_bindings": [item for item in snapshot.get("read_bindings", [])
                                                 if item["call_id"] == call["call_id"]]})
            if final_seen:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if not source_map or not final_seen:
        raise TrajectoryError("CC native model capture does not cover the verified final reply")
    atomic_json(trace_dir / "trajectory.json", {"format_version": 1, "sample_format": SAMPLE_FORMAT,
                "trajectory_id": trajectory_id, "sample_file": sample_file.name, "sample_origins": source_map})
    return {"sample_format": SAMPLE_FORMAT, "trajectory_id": trajectory_id,
            "sample_file": str(sample_file), "sample_count": len(source_map)}
