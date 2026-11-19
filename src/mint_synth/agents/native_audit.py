"""Shared offline grading of actual native tool reads and visible model output."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing, nullcontext
from pathlib import Path
from typing import Any

from mint_synth.core.run_state import atomic_json
from mint_synth.harness.solution import assess_solution

from .read_binding import observed_source_text, source_paths
from .trajectory import TrajectoryError


def _failed_result(value: Any) -> bool:
    if isinstance(value, str):
        if value.startswith("Content of "):
            return False
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            return _failed_result(decoded)
        return bool(re.search(r"Process exited with code (?!0\b)-?\d+", value)
                    or re.search(r"(?m)^\s*Exit Code:\s*(?!0\b)-?\d+", value))
    if isinstance(value, list):
        return any(_failed_result(item) for item in value if not isinstance(item, dict) or item.get("type") != "text")
    if isinstance(value, dict):
        outcome = value.get("outcome", {})
        return bool(value.get("is_error") or value.get("error") or value.get("status") in {"error", "failed"}
                    or value.get("error_code") or value.get("type", "").endswith("tool_result_error")
                    or isinstance(value.get("content"), (dict, list)) and _failed_result(value["content"])
                    or isinstance(value.get("observation"), dict) and "error" in value["observation"]
                    or value.get("metadata", {}).get("exit_code", 0) != 0
                    or isinstance(outcome, dict) and (outcome.get("type") == "timeout" or outcome.get("exit_code", 0) != 0))
    return False


def audit_trace(case, trace_dir: Path, runtime: str, *, tolerance: str, reviewer: Any = None):
    # Imports stay local because runtime capture modules also expose these audits.
    from .cc_native import anthropic_response
    from .cc_workspace import _digest, _proof_from_text, native_final_reply
    from .model_trace import iter_captured_calls
    from .native_capture import _reply, _results, _returned_text, _uses, hosted_results, native_response

    manifest = json.loads((trace_dir / "native-workspace.json").read_text(encoding="utf-8"))
    if not manifest.get("integrity", {}).get("valid") or manifest.get("cutoff") != case.cutoff:
        raise TrajectoryError("native corpus workspace failed its integrity check")
    root = Path(manifest["workspace_root"])
    web = case.retrieval.get("mode") == "web"
    source_records = {}
    if case.retrieval:
        from .resource_capture import bind_resource_observation
        if case.archive_path:
            for path in case.archive_path.glob("evidence_*.json"):
                item = json.loads(path.read_text(encoding="utf-8"))
                source_records[item["chunk_id"]] = item
        sources = {}
    else:
        sources = {(root / source["file"]).resolve(): source for source in manifest["sources"]}
    if runtime == "claude_code":
        final_text, final_id = native_final_reply(trace_dir / "stdout.jsonl")
        protocol = "anthropic"
    elif runtime == "codex":
        from .codex_events import native_final_reply as codex_final
        final_text, final_id = codex_final(trace_dir / "stdout.jsonl", allow_mcp=bool(case.retrieval))
        protocol = "responses"
    else:
        final = json.loads((trace_dir / "native-final.json").read_text(encoding="utf-8"))
        final_text, final_id, protocol = final["text"].strip(), None, "chat_completions"

    def response_of(call):
        return (anthropic_response(call["response_text"], streaming=call["streaming"])
                if protocol == "anthropic" else native_response(call))

    def uses_of(call, response):
        if protocol != "anthropic":
            return _uses(call, response, allow_unknown=True, allow_resources=bool(case.retrieval), unrestricted_web=web)
        declared = {tool.get("name") for tool in call["request"].get("tools", [])}
        uses = []
        for block in response["content"]:
            if block.get("type") in ({"tool_use", "server_tool_use", "mcp_tool_use"} if web else {"tool_use"}):
                from .native_capture import resource_mcp
                if str(block.get("name", "")).startswith("mcp__") and not (web or case.retrieval and resource_mcp(block["name"]) and block["name"] in declared):
                    raise TrajectoryError("native capture contains an MCP tool call")
                uses.append({"id": block.get("id"), "name": block.get("name"),
                             "input": block.get("input", {}), "known_tool": block.get("name") in declared,
                             "hosted": block.get("type") != "tool_use"})
        return uses

    positions, uses, final_matches = {}, {}, []
    for position, call in enumerate(iter_captured_calls(trace_dir)):
        if call["protocol"] != protocol or call["call_id"] in positions:
            raise TrajectoryError("native capture has an unexpected protocol or duplicate call ID")
        positions[call["call_id"]] = position
        response = response_of(call)
        calls = uses_of(call, response)
        if protocol == "anthropic":
            text = "\n".join(block["text"] for block in response["content"] if block.get("type") == "text").strip()
            message_id = response.get("id")
            normal = response.get("stop_reason") == "end_turn" and not any(not use.get("hosted") for use in calls)
        else:
            text, message_id = _reply(call, response, allow_unknown=True, allow_resources=bool(case.retrieval), unrestricted_web=web)
            normal = bool(text)
        if normal and text == final_text and (final_id is None or message_id == final_id):
            final_matches.append((call["call_id"], message_id))
        for use in calls:
            if not isinstance(use["id"], str) or not use["id"] or use["id"] in uses:
                raise TrajectoryError("native capture has an ambiguous tool call ID")
            uses[use["id"]] = (call["call_id"], use)
    if len(final_matches) != 1:
        raise TrajectoryError("native final reply is not bound to a unique captured model response")
    final_call_id, message_id = final_matches[0]
    final_position = positions[final_call_id]
    acquired, bindings, seen, outputs, model_text = {}, [], set(), {}, []
    failed_ids = set()
    if runtime == "strands" and (trace_dir / "agent.messages.json").is_file():
        for message in json.loads((trace_dir / "agent.messages.json").read_text(encoding="utf-8")):
            for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
                result = block.get("toolResult", {}) if isinstance(block, dict) else {}
                if result.get("status") == "error":
                    failed_ids.add(result.get("toolUseId"))
    with (nullcontext(None) if case.retrieval else closing(sqlite3.connect(case.snapshot_path.as_uri() + "?mode=ro", uri=True))) as db:
        rows = {}
        for path, source in sources.items():
            row = db.execute("SELECT source_id,available_at,text FROM chunks WHERE chunk_id=?", (source["chunk_id"],)).fetchone()
            if (row is None or row[0] != source["source_id"] or row[1] != source["available_at"]
                    or _digest(row[2].encode()) != source["sha256"]):
                raise TrajectoryError("native source provenance did not replay")
            rows[path] = row
        for call in iter_captured_calls(trace_dir):
            if positions[call["call_id"]] > final_position:
                continue
            response = response_of(call)
            # Preserve actual visible CoT in the private material, including
            # derivations preceding the final reply. Exports retain raw blocks.
            if protocol == "anthropic":
                for block in response["content"]:
                    if block.get("type") in {"text", "thinking"}:
                        model_text.append(block.get("text", block.get("thinking", "")))
                results = [(item.get("tool_use_id"), item)
                           for message in call["request"].get("messages", []) if message.get("role") == "user"
                           for item in message.get("content", []) if isinstance(message.get("content"), list)
                           and isinstance(item, dict) and item.get("type") == "tool_result"]
                if web:
                    from .web_evidence import _url, _urls
                    citations = [citation for block in response["content"] for citation in block.get("citations", [])]
                    for block in response["content"]:
                        if block.get("type", "").endswith("tool_result"):
                            urls = set(_urls(block))
                            results.append((block.get("tool_use_id"), {**block, "provider_citations": [
                                citation for citation in citations if _url(citation.get("url")) in urls]}))
            else:
                if protocol == "chat_completions":
                    message = response["choices"][0]["message"]
                    model_text.extend(message.get(key, "") for key in ("reasoning_content", "reasoning", "analysis", "content")
                                      if isinstance(message.get(key, ""), str))
                else:
                    for item in response["output"]:
                        for block in [*item.get("content", []), *item.get("summary", [])]:
                            if isinstance(block, dict) and isinstance(block.get("text"), str):
                                model_text.append(block["text"])
                results = list(_results(call))
                if web and protocol == "responses":
                    results.extend(hosted_results(response, uses_of(call, response)))
            for use_id, output in results:
                if use_id not in uses or use_id in seen:
                    continue
                use_call_id, use = uses[use_id]
                if positions[use_call_id] > positions[call["call_id"]] or (
                        positions[use_call_id] == positions[call["call_id"]] and not use.get("hosted")):
                    continue
                seen.add(use_id)
                text = _returned_text(output)
                failed = use_id in failed_ids or _failed_result(output)
                if failed:
                    failed_ids.add(use_id)
                if web and not text:
                    text = json.dumps(output, ensure_ascii=False)
                outputs[use_id] = {"id": use_id, "name": use["name"], "arguments": use["input"],
                                   "observation": text, "failed": failed,
                                   "known_tool": True if web and not failed else use.get("known_tool", True)}
                if failed:
                    continue
                if case.retrieval:
                    observed_records = bind_resource_observation(case, trace_dir, use, text, source_records, root, raw_output=output)
                    for chunk_id, record in observed_records.items():
                        source_records[chunk_id] = record
                        acquired[chunk_id] = acquired.get(chunk_id, "") + "\n" + record["text"]
                        bindings.append({"tool_use_id": use_id, "tool": use["name"], "chunk_id": chunk_id,
                                         "call_id": use_call_id, "observed_in_call": call["call_id"],
                                         "evidence_visibility": record.get("metadata", {}).get("evidence_visibility", "tool_observation"),
                                         "text_sha256": _digest(record["text"].encode())})
                    continue
                paths = source_paths(use, root)
                resolved = []
                for path in paths:
                    if path in sources:
                        resolved.append(path)
                    else:
                        resolved.extend(file for file in sources if path in file.parents)
                for path in dict.fromkeys(resolved):
                    source = sources.get(path)
                    if source is None:
                        continue
                    observed = observed_source_text(text, rows[path][2])
                    if not observed:
                        continue
                    chunk_id = source["chunk_id"]
                    acquired[chunk_id] = acquired.get(chunk_id, "") + "\n" + observed
                    bindings.append({"tool_use_id": use_id, "tool": use["name"], "chunk_id": chunk_id,
                                     "call_id": use_call_id, "observed_in_call": call["call_id"],
                                     "text_sha256": _digest(observed.encode())})
    actions = []
    for use_id, (call_id, use) in uses.items():
        if positions[call_id] <= final_position:
            action = outputs.get(use_id, {"id": use_id, "name": use["name"], "arguments": use["input"],
                                          "observation": None, "failed": True, "known_tool": use.get("known_tool", True)})
            if action["known_tool"] is False and use_id not in failed_ids and not web:
                raise TrajectoryError("undeclared native tool has no confirmed failure observation")
            actions.append(action)
    if reviewer is None:
        # Explicit compatibility for old captures, never imposed on new prompts.
        proof = _proof_from_text(final_text)
        evaluation_text = json.dumps(proof, ensure_ascii=False)
    else:
        evaluation_text = "\n\n".join([*model_text, *[
            json.dumps([action["arguments"], action["observation"]], ensure_ascii=False) for action in actions], final_text])
    verdict = assess_solution(case.task, evaluation_text, reviewer, acquired=acquired, actions=actions,
                              tolerance=tolerance, sample_key=final_call_id, final_response=final_text,
                              source_records=source_records if case.retrieval else None)
    required = {str(fact["chunk_id"]) for fact in case.task["evidence"]}
    private_review = verdict.get("review")
    expected_facts = case.task["evidence"] if reviewer is None else (
        private_review.get("facts", []) if isinstance(private_review, dict) else [])
    if not isinstance(expected_facts, list):
        expected_facts = []
    missing = sorted({str(fact["chunk_id"]) for fact in expected_facts
                      if isinstance(fact, dict) and isinstance(fact.get("chunk_id"), str)
                      and (not isinstance(fact.get("quote"), str)
                           or fact["quote"] not in acquired.get(fact["chunk_id"], ""))})
    used_chunks = {str(fact["chunk_id"]) for fact in expected_facts if isinstance(fact, dict) and fact.get("chunk_id")}
    cutoff_valid = all((source["available_at"] is None or source["available_at"] <= case.cutoff)
                       and (not case.dated_only or source["available_at"] is not None)
                       for source in (source_records.values() if case.retrieval else manifest["sources"])
                       if source["chunk_id"] in (used_chunks if web else acquired))
    flags = verdict.get("action_relevance", {})
    if reviewer is None:
        relevant_reads = {binding["tool_use_id"] for binding in bindings if binding["chunk_id"] in required}
        flags = {action["id"]: {"relevant": action["id"] in relevant_reads and not action["failed"],
                                "reason": "legacy source read relevance"} for action in actions}
    accepted = verdict["accepted"] and not missing and cutoff_valid
    verdict.update(accepted=accepted, evidence_replayable=not missing, cutoff_valid=cutoff_valid,
                   missing_chunk_ids=missing, mismatched_chunk_ids=[])
    if missing:
        verdict["reason"] = "required evidence was not acquired"
    elif not cutoff_valid:
        verdict["reason"] = "evidence crossed the cutoff"
    record = {"format_version": 2, "mode": runtime + "_native", "resource_mode": bool(case.retrieval), "unrestricted_web": web,
              "resource_fingerprint": manifest.get("resource_fingerprint"), "audit": {"verdict": verdict},
              "read_bindings": bindings, "final_call_id": final_call_id, "final_message_id": message_id,
              "action_relevance": flags, "failed_tool_ids": sorted(str(value) for value in failed_ids if value),
              "actions": actions}
    atomic_json(trace_dir / "native-audit.json", record)
    return record
