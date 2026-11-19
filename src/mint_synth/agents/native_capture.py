"""Offline audit and protocol-preserving exports for built-in agent tools.

The model capture is the content source. SDK/CLI final events bind the final
reply; captured call IDs bind source reads to their returned tool outputs.
No functions in this module are exposed as agent tools.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import atomic_json
from mint_synth.harness.session import HarnessCase
from .cc_workspace import _read_path
from .model_trace import _sse_events, iter_captured_calls
from .trajectory import TrajectoryError


NATIVE_FORMATS = {
    "claude_code_native_v2": "cc-native",
    "codex_native_v2": "codex-native",
    "strands_native_v2": "strands-native",
}


def native_response(call: Mapping[str, Any]) -> dict[str, Any]:
    """Reassemble native outputs without mapping their tools to finance tools."""
    protocol = call["protocol"]
    if protocol not in {"responses", "chat_completions"}:
        raise TrajectoryError("native capture requires Responses or Chat Completions")
    if not call["streaming"]:
        response = json.loads(call["response_text"])
    else:
        events, done = _sse_events(call["response_text"])
        if protocol == "responses":
            completed = [event["response"] for event in events if event.get("type") == "response.completed"]
            if len(completed) != 1 or any(event.get("type") in {"response.failed", "response.incomplete"} for event in events):
                raise TrajectoryError("native Responses stream is incomplete")
            response = copy.deepcopy(completed[0])
            if not response.get("output"):
                items = {event["output_index"]: event["item"] for event in events if event.get("type") == "response.output_item.done"}
                if sorted(items) != list(range(len(items))):
                    raise TrajectoryError("native Responses output indices are incomplete")
                response["output"] = [items[index] for index in sorted(items)]
        else:
            response = {"choices": [{"index": 0, "message": {"role": "assistant"}, "finish_reason": None}]}
            choice = response["choices"][0]
            tools: dict[int, dict[str, Any]] = {}
            for event in events:
                for key in ("id", "model", "created", "system_fingerprint", "usage"):
                    if key in event:
                        response[key] = copy.deepcopy(event[key])
                choices = event.get("choices", [])
                if not choices:
                    continue
                if len(choices) != 1 or choices[0].get("index", 0) != 0:
                    raise TrajectoryError("native chat stream has ambiguous choices")
                part = choices[0]
                if part.get("finish_reason"):
                    choice["finish_reason"] = part["finish_reason"]
                for key, value in part.get("delta", {}).items():
                    if key == "tool_calls":
                        for fragment in value or []:
                            index = fragment["index"]
                            target = tools.setdefault(index, {"type": "function", "function": {"name": "", "arguments": ""}})
                            if fragment.get("id"):
                                target["id"] = fragment["id"]
                            if fragment.get("type"):
                                target["type"] = fragment["type"]
                            for field, text in fragment.get("function", {}).items():
                                if not isinstance(text, str):
                                    raise TrajectoryError("invalid native chat tool fragment")
                                target["function"][field] = target["function"].get(field, "") + text
                    elif value is not None:
                        if key in {"content", "reasoning_content", "reasoning", "analysis", "refusal"}:
                            if not isinstance(value, str):
                                raise TrajectoryError("unsupported structured native chat delta")
                            choice["message"][key] = choice["message"].get(key, "") + value
                        elif key == "role":
                            choice["message"][key] = value
                        else:
                            raise TrajectoryError(f"unsupported native chat delta: {key}")
            if tools:
                if sorted(tools) != list(range(len(tools))):
                    raise TrajectoryError("native chat tool indices are incomplete")
                choice["message"]["tool_calls"] = [tools[key] for key in sorted(tools)]
            if not done:
                raise TrajectoryError("native chat stream has no DONE marker")
    if not isinstance(response, dict):
        raise TrajectoryError("native response is not an object")
    if protocol == "responses":
        if response.get("status") != "completed" or not isinstance(response.get("output"), list):
            raise TrajectoryError("native Responses result is incomplete")
    else:
        choices = response.get("choices", [])
        if (len(choices) != 1 or choices[0].get("index", 0) != 0
                or choices[0].get("finish_reason") not in {"stop", "tool_calls"}
                or not isinstance(choices[0].get("message"), dict)
                or choices[0]["message"].get("role") != "assistant"):
            raise TrajectoryError("native chat result is incomplete or ambiguous")
    return response


def resource_mcp(name: str) -> bool:
    return str(name) in {"mcp__finance__" + n for n in ("search_documents", "read_document", "search_web", "read_web", "calculate")}


def _declared_tools(request: Mapping[str, Any], *, allow_resources: bool = False, unrestricted_web: bool = False) -> set[str]:
    def names(tools: Any) -> set[str]:
        if not isinstance(tools, list):
            raise TrajectoryError("native request tools must be an array")
        result = set()
        for tool in tools:
            if not isinstance(tool, dict):
                raise TrajectoryError("invalid native tool definition")
            if tool.get("type") == "mcp" and not unrestricted_web:
                raise TrajectoryError("native capture contains MCP definitions")
            name = tool.get("name") or tool.get("function", {}).get("name")
            if isinstance(name, str):
                if name.startswith("mcp__") and not (unrestricted_web or allow_resources and resource_mcp(name)):
                    raise TrajectoryError("native capture contains MCP definitions")
                result.add(name)
            if tool.get("type") == "namespace":
                result.update(names(tool.get("tools", [])))
        return result
    return names(request.get("tools", []))


def _uses(call: Mapping[str, Any], response: Mapping[str, Any], *, allow_unknown: bool = False, allow_resources: bool = False,
          unrestricted_web: bool = False) -> list[dict[str, Any]]:
    declared = _declared_tools(call["request"], allow_resources=allow_resources, unrestricted_web=unrestricted_web)
    result = []
    if call["protocol"] == "responses":
        for item in response["output"]:
            kind = item.get("type", "")
            if kind in {"function_call", "custom_tool_call"}:
                name = item.get("name")
                if (name not in declared and not allow_unknown) or (str(name).startswith("mcp__") and not (unrestricted_web or allow_resources and resource_mcp(name) and name in declared)):
                    raise TrajectoryError("native response used an undeclared or MCP tool")
                arguments = item.get("arguments") if kind == "function_call" else item.get("input")
                if kind == "function_call":
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    if not isinstance(arguments, dict):
                        raise TrajectoryError("native function arguments must be an object")
                else:
                    # Free-form input (for example apply_patch) stays free-form.
                    arguments = {"_raw_input": arguments}
                result.append({"id": item.get("call_id"), "name": name, "namespace": item.get("namespace"), "input": arguments, "known_tool": name in declared})
            elif unrestricted_web and (kind.endswith("_call") or kind == "mcp_call") and kind != "shell_call":
                # Hosted tools execute inside the provider response. Preserve
                # their own names, arguments and embedded observations.
                arguments = item.get("action", item.get("arguments", {}))
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = {"_raw_input": arguments}
                client_call = kind in {"computer_call", "local_shell_call", "apply_patch_call"}
                hosted = not client_call and (kind in {"web_search_call", "file_search_call", "code_interpreter_call", "image_generation_call", "mcp_call"}
                                               or any(key in item for key in ("output", "results")))
                result.append({"id": item.get("call_id") or item.get("id"),
                               "name": item.get("name") or kind, "input": arguments,
                               "known_tool": True, "hosted": hosted, "item_type": kind})
            elif kind.startswith("mcp_") and not unrestricted_web:
                raise TrajectoryError("native capture contains an MCP operation")
            elif kind == "shell_call":
                if not any(tool.get("type") == "shell" for tool in call["request"].get("tools", [])):
                    raise TrajectoryError("native response used an undeclared shell tool")
                result.append({"id": item.get("call_id"), "name": "shell_call", "input": item.get("action", {})})
            elif kind.endswith("_call"):
                raise TrajectoryError(f"unsupported native Responses tool item: {kind}")
    else:
        for item in response["choices"][0]["message"].get("tool_calls", []) or []:
            function = item.get("function", {})
            name = function.get("name")
            if (name not in declared and not allow_unknown) or (str(name).startswith("mcp__") and not (unrestricted_web or allow_resources and resource_mcp(name) and name in declared)):
                raise TrajectoryError("native response used an undeclared or MCP tool")
            arguments = function.get("arguments", {})
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(arguments, dict):
                raise TrajectoryError("native function arguments must be an object")
            result.append({"id": item.get("id"), "name": name, "input": arguments, "known_tool": name in declared})
    if any(not isinstance(item["id"], str) or not item["id"] for item in result):
        raise TrajectoryError("native tool call has no call ID")
    return result


def _reply(call: Mapping[str, Any], response: Mapping[str, Any], *, allow_unknown: bool = False, allow_resources: bool = False,
           unrestricted_web: bool = False) -> tuple[str, str | None]:
    uses = _uses(call, response, allow_unknown=allow_unknown, allow_resources=allow_resources, unrestricted_web=unrestricted_web)
    if any(not use.get("hosted") for use in uses):
        return "", None
    if call["protocol"] == "responses":
        messages = [item for item in response["output"] if item.get("type") == "message" and item.get("phase") != "commentary"]
        text = "\n".join(block["text"] for message in messages for block in message.get("content", [])
                         if block.get("type") == "output_text" and isinstance(block.get("text"), str)).strip()
        return text, messages[-1].get("id") if messages else None
    message = response["choices"][0]["message"]
    content = message.get("content", "")
    text = content if isinstance(content, str) else "\n".join(block.get("text", "") for block in content or [] if isinstance(block, dict))
    return text.strip(), response.get("id")


def _results(call: Mapping[str, Any]):
    request = call["request"]
    if call["protocol"] == "responses":
        for item in request.get("input", []) if isinstance(request.get("input"), list) else []:
            if item.get("type", "").endswith("_output") and item.get("call_id"):
                yield item.get("call_id"), item.get("output")
    else:
        for item in request.get("messages", []):
            if item.get("role") == "tool":
                yield item.get("tool_call_id"), item.get("content")


def hosted_results(response: Mapping[str, Any], uses: list[dict[str, Any]]):
    """Return provider-executed results without inventing client tool turns."""
    items = {item.get("call_id") or item.get("id"): item for item in response.get("output", [])}
    annotations = [annotation for item in response.get("output", []) for block in item.get("content", [])
                   for annotation in block.get("annotations", []) if isinstance(annotation, dict)]
    web_ids = [use["id"] for use in uses if use.get("item_type") == "web_search_call"]
    for use in uses:
        if not use.get("hosted"):
            continue
        item = items.get(use["id"], {})
        if item.get("status") not in {None, "completed", "failed"}:
            continue
        result = copy.deepcopy(item)
        if use.get("item_type") == "web_search_call":
            if item.get("status") not in {"completed", "failed"}:
                continue
            action = item.get("action", {})
            from .web_evidence import _url, _urls
            source_urls = set(_urls(action.get("sources", [])))
            if action.get("url"):
                source_urls.add(_url(action["url"]))
            citations = [annotation for annotation in annotations if _url(annotation.get("url")) in source_urls] if source_urls else (
                annotations if len(web_ids) == 1 else [{**annotation, "evidence_attribution": "provider_response",
                                                      "tool_use_ids": web_ids} for annotation in annotations]
                if use["id"] == web_ids[-1] else [])
            result = {"status": item.get("status"), "sources": action.get("sources", []),
                      "provider_citations": citations}
            if action.get("type") in {"open_page", "find_in_page"} and action.get("url"):
                result["opened_page"] = {"url": action["url"]}
        elif use.get("item_type") == "mcp_call":
            if item.get("output") is None and not item.get("error"):
                continue
            result = {key: item[key] for key in ("output", "error", "status") if key in item}
        yield use["id"], result


def _returned_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = None
        if isinstance(decoded, list) or isinstance(decoded, dict) and any(
            key in decoded for key in ("content", "stdout", "output", "text", "is_error", "status", "outcome")):
            return _returned_text(decoded)
        # The official reader wraps source text with this heading. Status-like
        # lines inside that source are evidence, not shell execution metadata.
        if value.startswith("Content of "):
            return value
        exits = re.findall(r"Process exited with code (-?\d+)", value)
        shell_report = re.search(r"(?m)^Command:\s", value) and re.search(r"(?m)^Exit Code:\s", value)
        failed = False
        if shell_report:
            exits.extend(re.findall(r"(?m)^\s*Exit Code:\s*(-?\d+)\s*$", value))
            failed = re.search(r"(?mi)^\s*Status:\s*(?:error|failed|timeout)\s*$", value)
        return "" if failed or any(int(code) != 0 for code in exits) else value
    if isinstance(value, dict):
        if value.get("is_error") or value.get("status") in {"error", "failed"} or value.get("metadata", {}).get("exit_code", 0) != 0:
            return ""
        outcome = value.get("outcome", {})
        if isinstance(outcome, dict) and (outcome.get("type") == "timeout" or outcome.get("exit_code", 0) != 0):
            return ""
        if "stdout" in value:
            return _returned_text(value["stdout"])
        return _returned_text(value.get("content", value.get("output", value.get("text", ""))))
    if isinstance(value, list):
        return "\n".join(_returned_text(item) for item in value)
    return ""


def _source_path(use: Mapping[str, Any], *, runtime: str | None = None) -> str | None:
    name, arguments = use["name"], use["input"]
    if name == "file_read":
        return arguments.get("path") if arguments.get("mode", "view") == "view" else None
    if name in {"exec_command", "shell_command", "shell", "shell_call"}:
        command = arguments.get("cmd", arguments.get("command", arguments.get("commands")))
        if isinstance(command, list):
            if len(command) == 1:
                command = command[0]
            elif runtime == "strands":
                return None  # strands_tools.shell treats a list as multiple commands, not argv.
            elif len(command) == 3 and Path(command[0]).name in {"bash", "sh", "zsh"} and command[1] in {"-c", "-lc"}:
                command = command[2]
            elif len(command) in {2, 3} and command[0] == "cat":
                command = shlex.join(command)
            else:
                return None
        return _read_path({"name": "Bash", "input": {"command": command}})
    return None


def audit_native_capture(case: HarnessCase, trace_dir: Path, runtime: str, *, tolerance: str = "1e-6",
                         reviewer: Any = None) -> dict[str, Any]:
    from .native_audit import audit_trace

    if runtime not in {"codex", "strands"}:
        raise ValueError("native capture runtime must be codex or strands")
    return audit_trace(case, trace_dir, runtime, tolerance=tolerance, reviewer=reviewer)


def export_native_capture(snapshot: Mapping[str, Any], trace_dir: Path, runtime: str) -> dict[str, Any]:
    sample_format = runtime + "_native_v2"
    if runtime not in {"codex", "strands"} or not snapshot.get("audit", {}).get("verdict", {}).get("accepted"):
        raise TrajectoryError("native export requires an accepted offline audit")
    fingerprint = hashlib.sha256()
    for path in sorted((trace_dir / "model/calls").glob("*.json")):
        fingerprint.update(path.read_bytes())
    trajectory_id = fingerprint.hexdigest()
    sample_file = trace_dir / (runtime + "-native.jsonl")
    origins, final_seen, response_ids = [], False, set()
    descriptor = os.open(sample_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for call in iter_captured_calls(trace_dir):
            expected = "responses" if runtime == "codex" else "chat_completions"
            if call["protocol"] != expected:
                raise TrajectoryError("native export protocol mismatch")
            response = native_response(call)
            web = bool(snapshot.get("unrestricted_web"))
            uses = _uses(call, response, allow_unknown=True, allow_resources=bool(snapshot.get("resource_mode")), unrestricted_web=web)
            flags = snapshot.get("action_relevance", {})
            if any(use["id"] not in flags for use in uses):
                raise TrajectoryError("native export has no action relevance decision")
            for use in uses:
                if use.get("known_tool") is False and use["id"] not in snapshot.get("failed_tool_ids", []) and not web:
                    raise TrajectoryError("native response used an undeclared tool without a recorded failure")
            call_loss = all(flags.get(use["id"], {}).get("relevant", True) for use in uses)
            sample = copy.deepcopy(call["request"])
            if runtime == "codex":
                context = sample.get("input")
                if not isinstance(context, (str, list)) or not context:
                    raise TrajectoryError("native Responses request has no input")
                previous = sample.get("previous_response_id")
                if previous and previous not in response_ids:
                    raise TrajectoryError("native Responses context references an uncaptured response")
                sample["output"] = copy.deepcopy(response["output"])
                sample["input_loss_mask"] = [False] * (len(context) if isinstance(context, list) else 1)
                sample["target_output_loss_mask"] = [
                    (flags.get(item.get("call_id") or item.get("id"), {}).get("relevant", call_loss)
                     if item.get("type", "").endswith("_call") else call_loss)
                    and not item.get("type", "").endswith("_output")
                    and item.get("type") not in {"mcp_list_tools", "mcp_approval_request", "compaction"}
                    and not (item.get("type") == "reasoning" and not (item.get("content") or item.get("summary")))
                    for item in response["output"]
                ]
                sample["target_output_field_loss_mask"] = {
                    str(index): {key: key in {"name", "arguments", "input", "action", "queries", "code", "prompt"} for key in item}
                    | ({"action.sources": False} if isinstance(item.get("action"), dict) and "sources" in item["action"] else {})
                    for index, item in enumerate(response["output"])
                    if any(use.get("hosted") and use["id"] == (item.get("call_id") or item.get("id")) for use in uses)}
                if web:
                    sample["target_output_field_loss_mask"].update({
                        str(index): {"content": True, "content.annotations": False}
                        for index, item in enumerate(response["output"]) if item.get("type") == "message"})
                sample["target_reasoning_text_fields"] = {
                    str(index): {"content": bool(item.get("content")), "summary": bool(item.get("summary")),
                                 "encrypted_content": False}
                    for index, item in enumerate(response["output"]) if item.get("type") == "reasoning"}
                sample["response_metadata"] = {key: value for key, value in response.items() if key != "output"}
                if response.get("id"):
                    response_ids.add(response["id"])
            else:
                context = sample.get("messages")
                if not isinstance(context, list) or not context:
                    raise TrajectoryError("native chat request has no messages")
                sample["messages"].append(copy.deepcopy(response["choices"][0]["message"]))
                sample["target_message_index"] = len(context) - 1
                tool_mask = {use["id"]: flags.get(use["id"], {}).get("relevant", True) for use in uses}
                sample["loss_mask"] = [False] * (len(context) - 1) + [bool(call_loss or any(tool_mask.values()))]
                sample["target_text_loss"] = call_loss
                sample["target_tool_call_loss_mask"] = tool_mask
                sample["response_metadata"] = {key: value for key, value in response.items() if key != "choices"}
                sample["response_metadata"]["choice"] = {key: value for key, value in response["choices"][0].items() if key != "message"}
            if call["call_id"] == snapshot["final_call_id"]:
                _, message_id = _reply(call, response, allow_unknown=web, allow_resources=bool(snapshot.get("resource_mode")), unrestricted_web=web)
                if message_id != snapshot.get("final_message_id"):
                    raise TrajectoryError("native final response differs from its audit")
                final_seen = True
            sample.update(sample_format=sample_format, trajectory_id=trajectory_id, call_id=call["call_id"])
            stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            origins.append({"sample_index": len(origins), "call_id": call["call_id"], "source": call["source"],
                            "response_file": call["response_file"],
                            "read_bindings": [item for item in snapshot.get("read_bindings", []) if item["call_id"] == call["call_id"]]})
            if final_seen:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if not origins or not final_seen:
        raise TrajectoryError("native capture does not cover the verified final reply")
    atomic_json(trace_dir / "trajectory.json", {"format_version": 1, "sample_format": sample_format,
                "trajectory_id": trajectory_id, "sample_file": sample_file.name, "sample_origins": origins})
    return {"sample_format": sample_format, "trajectory_id": trajectory_id,
            "sample_file": str(sample_file), "sample_count": len(origins)}
