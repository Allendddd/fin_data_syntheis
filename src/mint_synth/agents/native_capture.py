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
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import atomic_json
from mint_synth.harness.session import HarnessCase
from mint_synth.harness.snapshot import published_date
from mint_synth.harness.verifier import typed_answer_ok, verify_proof

from .cc_workspace import _digest, _proof_from_text, _read_path
from .model_trace import _sse_events, iter_captured_calls
from .trajectory import TrajectoryError


NATIVE_FORMATS = {
    "claude_code_native_v1": "cc-native",
    "codex_native_v1": "codex-native",
    "strands_native_v1": "strands-native",
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


def _declared_tools(request: Mapping[str, Any]) -> set[str]:
    def names(tools: Any) -> set[str]:
        if not isinstance(tools, list):
            raise TrajectoryError("native request tools must be an array")
        result = set()
        for tool in tools:
            if not isinstance(tool, dict):
                raise TrajectoryError("invalid native tool definition")
            if tool.get("type") == "mcp":
                raise TrajectoryError("native capture contains MCP definitions")
            name = tool.get("name") or tool.get("function", {}).get("name")
            if isinstance(name, str):
                if name.startswith("mcp__"):
                    raise TrajectoryError("native capture contains MCP definitions")
                result.add(name)
            if tool.get("type") == "namespace":
                result.update(names(tool.get("tools", [])))
        return result
    return names(request.get("tools", []))


def _uses(call: Mapping[str, Any], response: Mapping[str, Any]) -> list[dict[str, Any]]:
    declared = _declared_tools(call["request"])
    result = []
    if call["protocol"] == "responses":
        for item in response["output"]:
            kind = item.get("type", "")
            if kind in {"function_call", "custom_tool_call"}:
                name = item.get("name")
                if name not in declared or str(name).startswith("mcp__"):
                    raise TrajectoryError("native response used an undeclared or MCP tool")
                arguments = item.get("arguments") if kind == "function_call" else item.get("input")
                if kind == "function_call":
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    if not isinstance(arguments, dict):
                        raise TrajectoryError("native function arguments must be an object")
                else:
                    # Free-form input (for example apply_patch) stays free-form.
                    arguments = {"_raw_input": arguments}
                result.append({"id": item.get("call_id"), "name": name, "input": arguments})
            elif kind.startswith("mcp_"):
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
            if name not in declared or str(name).startswith("mcp__"):
                raise TrajectoryError("native response used an undeclared or MCP tool")
            arguments = function.get("arguments", {})
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(arguments, dict):
                raise TrajectoryError("native function arguments must be an object")
            result.append({"id": item.get("id"), "name": name, "input": arguments})
    if any(not isinstance(item["id"], str) or not item["id"] for item in result):
        raise TrajectoryError("native tool call has no call ID")
    return result


def _reply(call: Mapping[str, Any], response: Mapping[str, Any]) -> tuple[str, str | None]:
    if _uses(call, response):
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
            if item.get("type") in {"function_call_output", "custom_tool_call_output", "shell_call_output"}:
                yield item.get("call_id"), item.get("output")
    else:
        for item in request.get("messages", []):
            if item.get("role") == "tool":
                yield item.get("tool_call_id"), item.get("content")


def _returned_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)):
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


def audit_native_capture(case: HarnessCase, trace_dir: Path, runtime: str, *, tolerance: str = "1e-6") -> dict[str, Any]:
    manifest = json.loads((trace_dir / "native-workspace.json").read_text(encoding="utf-8"))
    if not manifest.get("integrity", {}).get("valid") or manifest.get("cutoff") != case.cutoff:
        raise TrajectoryError("native corpus workspace failed its integrity check")
    root = Path(manifest["workspace_root"])
    sources = {(root / item["file"]).resolve(): item for item in manifest["sources"]}
    if runtime == "codex":
        from .codex_events import native_final_reply
        final_text, final_id = native_final_reply(trace_dir / "stdout.jsonl")
        protocol = "responses"
    elif runtime == "strands":
        final = json.loads((trace_dir / "native-final.json").read_text(encoding="utf-8"))
        final_text, final_id = final["text"].strip(), None
        protocol = "chat_completions"
    else:
        raise ValueError("native capture runtime must be codex or strands")
    positions, uses, final_matches = {}, {}, []
    for position, call in enumerate(iter_captured_calls(trace_dir)):
        if call["protocol"] != protocol or call["call_id"] in positions:
            raise TrajectoryError("native capture has an unexpected protocol or duplicate call ID")
        positions[call["call_id"]] = position
        response = native_response(call)
        text, message_id = _reply(call, response)
        if final_text and text == final_text and (final_id is None or final_id == message_id):
            final_matches.append((call["call_id"], message_id))
        for use in _uses(call, response):
            if use["id"] in uses:
                raise TrajectoryError("native capture has an ambiguous tool call ID")
            uses[use["id"]] = (call["call_id"], use)
    if len(final_matches) != 1:
        raise TrajectoryError("native final reply is not uniquely bound to a captured model response")
    final_call_id, message_id = final_matches[0]
    failed_tool_ids = set()
    if runtime == "strands" and (trace_dir / "agent.messages.json").is_file():
        # The OpenAI adapter may flatten toolResult.content and omit status.
        # Keep a known SDK execution failure from becoming successful evidence.
        for message in json.loads((trace_dir / "agent.messages.json").read_text(encoding="utf-8")):
            for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
                result = block.get("toolResult", {}) if isinstance(block, dict) else {}
                if result.get("status") == "error":
                    failed_tool_ids.add(result.get("toolUseId"))
    acquired, bindings, seen = {}, [], set()
    for call in iter_captured_calls(trace_dir):
        if positions[call["call_id"]] > positions[final_call_id]:
            continue
        for use_id, output in _results(call):
            if use_id not in uses or use_id in failed_tool_ids:
                continue
            use_call_id, use = uses[use_id]
            if positions[use_call_id] >= positions[call["call_id"]]:
                continue
            path = _source_path(use, runtime=runtime)
            if not isinstance(path, str):
                continue
            source = sources.get((root / Path(path)).resolve())
            text = _returned_text(output)
            if source is None or not text or (use_id, text) in seen:
                continue
            if use["name"] == "file_read":
                text = re.sub(r"(?m)^\s*\d+(?:→|\t|\s*\|\s)", "", text)
            seen.add((use_id, text))
            chunk_id = source["chunk_id"]
            acquired[chunk_id] = acquired.get(chunk_id, "") + "\n" + text
            bindings.append({"tool_use_id": use_id, "tool": use["name"], "chunk_id": chunk_id,
                             "call_id": use_call_id, "observed_in_call": call["call_id"], "text_sha256": _digest(text.encode())})
    missing, mismatched = [], []
    by_chunk = {source["chunk_id"]: source for source in manifest["sources"]}
    with closing(sqlite3.connect(case.snapshot_path.as_uri() + "?mode=ro", uri=True)) as db:
        for fact in case.task["evidence"]:
            chunk_id = str(fact["chunk_id"])
            source = by_chunk.get(chunk_id)
            if chunk_id not in acquired:
                missing.append(chunk_id)
                continue
            row = db.execute("SELECT source_id,available_at,text FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
            quote = str(fact.get("quote") or "")
            if (source is None or row is None or source["source_id"] != row[0] or fact.get("source_id") != row[0]
                    or source["available_at"] != row[1]
                    or published_date(fact.get("source_metadata") or {}, case.available_at_field) != row[1]
                    or source["sha256"] != _digest(row[2].encode())
                    or not quote or quote not in row[2] or quote not in acquired[chunk_id]):
                mismatched.append(chunk_id)
    proof = _proof_from_text(final_text)
    answer_correct = typed_answer_ok(case.task, proof.get("answer"), tolerance)
    derivation_valid, derivation_reason = verify_proof(case.task, proof, acquired=acquired, tolerance=tolerance)
    cutoff_valid = all((source["available_at"] is None or source["available_at"] <= case.cutoff)
                       and (not case.dated_only or source["available_at"] is not None)
                       for source in manifest["sources"] if source["chunk_id"] in acquired)
    evidence_valid = not missing and not mismatched
    accepted = evidence_valid and derivation_valid and cutoff_valid
    verdict = {"accepted": accepted, "answer_correct": answer_correct, "derivation_valid": derivation_valid,
               "evidence_replayable": evidence_valid, "cutoff_valid": cutoff_valid,
               "missing_chunk_ids": missing, "mismatched_chunk_ids": mismatched,
               "reason": ("accepted" if accepted else "required evidence was not acquired" if missing else
                          "evidence provenance did not replay" if mismatched else
                          "evidence crossed the cutoff" if not cutoff_valid else derivation_reason)}
    record = {"format_version": 1, "mode": runtime + "_native", "audit": {"verdict": verdict},
              "read_bindings": bindings, "final_call_id": final_call_id, "final_message_id": message_id}
    atomic_json(trace_dir / "native-audit.json", record)
    return record


def export_native_capture(snapshot: Mapping[str, Any], trace_dir: Path, runtime: str) -> dict[str, Any]:
    sample_format = runtime + "_native_v1"
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
            _uses(call, response)  # Validate actual calls against this request's native definitions.
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
                    not item.get("type", "").endswith("_output")
                    and not (item.get("type") == "reasoning" and (item.get("encrypted_content")
                              or not (item.get("content") or item.get("summary"))))
                    for item in response["output"]
                ]
                sample["response_metadata"] = {key: value for key, value in response.items() if key != "output"}
                if response.get("id"):
                    response_ids.add(response["id"])
            else:
                context = sample.get("messages")
                if not isinstance(context, list) or not context:
                    raise TrajectoryError("native chat request has no messages")
                sample["messages"].append(copy.deepcopy(response["choices"][0]["message"]))
                sample["target_message_index"] = len(context) - 1
                sample["loss_mask"] = [False] * (len(context) - 1) + [True]
                sample["response_metadata"] = {key: value for key, value in response.items() if key != "choices"}
                sample["response_metadata"]["choice"] = {key: value for key, value in response["choices"][0].items() if key != "message"}
            if call["call_id"] == snapshot["final_call_id"]:
                _, message_id = _reply(call, response)
                if message_id != snapshot.get("final_message_id"):
                    raise TrajectoryError("native final response differs from its audit")
                final_seen = True
            sample.update(sample_format=sample_format, trajectory_id=trajectory_id, call_id=call["call_id"])
            stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            origins.append({"sample_index": len(origins), "call_id": call["call_id"], "source": call["source"],
                            "response_file": call["response_file"],
                            "read_bindings": [item for item in snapshot.get("read_bindings", []) if item["call_id"] == call["call_id"]]})
        stream.flush()
        os.fsync(stream.fileno())
    if not origins or not final_seen:
        raise TrajectoryError("native capture does not cover the verified final reply")
    atomic_json(trace_dir / "trajectory.json", {"format_version": 1, "sample_format": sample_format,
                "trajectory_id": trajectory_id, "sample_file": sample_file.name, "sample_origins": origins})
    return {"sample_format": sample_format, "trajectory_id": trajectory_id,
            "sample_file": str(sample_file), "sample_count": len(origins)}
