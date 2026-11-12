from __future__ import annotations

import copy
import hashlib
import json

import pytest

from mint_synth.agents.model_trace import load_model_turns, response_parts, streamed_parts
from mint_synth.agents.trajectory import TrajectoryError, normalized_sample
from mint_synth.harness import HarnessSession, case_from_task

from test_teacher_proof import proof, snapshot, task


ALLOWED = {"search_corpus", "open_chunk", "calculate"}


def sse(*events, done=False):
    result = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return result + ("data: [DONE]\n\n" if done else "")


def test_chat_stream_preserves_reasoning_text_and_argument_fragments():
    text = sse(
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Find "}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "the filing."}}]},
        {"choices": [{"index": 0, "delta": {"content": "I will search."}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "type": "function",
            "function": {"name": "search_corpus", "arguments": '{"query":'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
            "function": {"arguments": '"Revenue"}'}}]}}]},
        {"choices": [{"index": 0, "delta": {"content": "I will inspect the hits."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}, done=True)
    parts = streamed_parts(text, "chat_completions", ALLOWED)
    assert [part["kind"] for part in parts] == ["reasoning", "text", "tool_call", "text"]
    assert parts[0]["text"] == "Find the filing."
    assert parts[2]["arguments"] == {"query": "Revenue"}
    with pytest.raises(TrajectoryError, match="terminal"):
        streamed_parts(text.replace("data: [DONE]\n\n", ""), "chat_completions", ALLOWED)


def test_responses_stream_preserves_exposed_summary_and_function_call():
    response = {"status": "completed", "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Locate the report."}],
         "encrypted_content": "opaque-provider-data"},
        {"type": "message", "content": [{"type": "output_text", "text": "I will search."}]},
        {"type": "function_call", "name": "mcp__finance__search_corpus", "arguments": '{"query":"Revenue"}'}]}
    text = sse(
        {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0,
         "delta": "Locate the report."},
        {"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "I will search."},
        {"type": "response.completed", "response": response})
    parts = streamed_parts(text, "responses", ALLOWED)
    assert [part["kind"] for part in parts] == ["reasoning_summary", "text", "tool_call"]
    assert parts[0]["text"] == "Locate the report."
    assert "opaque-provider-data" not in json.dumps(parts)
    missing = copy.deepcopy(response)
    missing["output"][0]["summary"] = []
    with pytest.raises(TrajectoryError, match="missing"):
        streamed_parts(text.replace(json.dumps(response), json.dumps(missing)), "responses", ALLOWED)


def test_anthropic_stream_preserves_thinking_and_tool_input():
    text = sse(
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Locate both periods."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "opaque"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": "I will search."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {
            "type": "tool_use", "id": "call", "name": "mcp__finance__search_corpus", "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"query":'}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '"Revenue"}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"})
    parts = streamed_parts(text, "anthropic", ALLOWED)
    assert [part["kind"] for part in parts] == ["reasoning", "text", "tool_call"]
    assert parts[2]["arguments"] == {"query": "Revenue"}
    with pytest.raises(TrajectoryError, match="incomplete"):
        streamed_parts(text.replace('data: {"type": "message_stop"}\n\n', ""), "anthropic", ALLOWED)


def test_unsupported_content_is_explicitly_rejected():
    with pytest.raises(TrajectoryError, match="unsupported Responses"):
        response_parts({"output": [{"type": "web_search_call"}]}, "responses", ALLOWED)
    parts = response_parts({"content": [{"type": "redacted_thinking", "data": "unavailable"}]}, "anthropic", ALLOWED)
    assert parts == []


def test_capture_fingerprints_are_checked(tmp_path):
    root = tmp_path / "model/calls"
    root.mkdir(parents=True)
    request = b'{"messages":[]}'
    response = json.dumps({"choices": [{"message": {"content": "Actual model text."}}]}).encode()
    (root / "call.request.bin").write_bytes(request)
    (root / "call.response.bin").write_bytes(response)
    metadata = {"call_id": "call", "state": "complete", "started_at": "2025-01-01T00:00:00Z",
                "request": {"target": "/v1/chat/completions", "body_file": "call.request.bin",
                            "body_sha256": hashlib.sha256(request).hexdigest()},
                "response": {"status": 200, "complete": True, "headers": [], "body_file": "call.response.bin",
                             "body_sha256": hashlib.sha256(response).hexdigest()}}
    (root / "call.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert load_model_turns(tmp_path, ALLOWED)[0]["assistant_parts"][0]["text"] == "Actual model text."
    (root / "call.response.bin").write_bytes(response + b" ")
    with pytest.raises(TrajectoryError, match="fingerprint"):
        load_model_turns(tmp_path, ALLOWED)


def test_batched_calls_precede_observations_and_repeated_calls_keep_loss_mask(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    case = case_from_task(task(), index, cutoff="2025-12-31")
    calls = [("search_corpus", {"query": "Revenue"}), ("search_corpus", {"query": "Revenue"}),
             ("open_chunk", {"chunk_id": "c1"}), ("open_chunk", {"chunk_id": "c2"})]
    with HarnessSession(case, max_steps=6, max_tool_calls=5) as session:
        for name, arguments in calls:
            session.step({"tool": name, "arguments": arguments})
        session.step({"final": json.dumps(proof())})
        record = {"audit": session.audit_record(), "trace": session.trace, "tools": session.tool_definitions}
    turns = [{"call_id": "search", "source": "search.json", "assistant_parts": [
        {"kind": "reasoning", "text": "<think>Find the evidence.</think>", "field": "reasoning"},
        *[{"kind": "tool_call", "name": name, "arguments": args, "field": f"call[{i}]"}
          for i, (name, args) in enumerate(calls[:2])],
        {"kind": "text", "text": "Now wait for the searches.", "field": "content"}]},
        {"call_id": "open", "source": "open.json", "assistant_parts": [
            {"kind": "tool_call", "name": name, "arguments": args, "field": f"call[{i}]"}
            for i, (name, args) in enumerate(calls[2:])]},
        {"call_id": "final", "source": "final.json", "assistant_parts": [
            {"kind": "final", "proof": proof(), "field": "proof"}]}]
    with pytest.raises(TrajectoryError, match="mixed loss masks"):
        normalized_sample(record, case.query, cutoff=case.cutoff, model_turns=turns, reasoning_format="think")
    # Split the repeated search into its actual sequential model turn so the
    # target template can retain its separate loss flag. The opens stay batched.
    first = turns[0]
    repeated = {"call_id": "repeated", "source": "repeated.json", "assistant_parts": [first["assistant_parts"][2]]}
    first["assistant_parts"] = [first["assistant_parts"][0], first["assistant_parts"][1], first["assistant_parts"][3]]
    turns.insert(1, repeated)
    sample = normalized_sample(record, case.query, cutoff=case.cutoff, model_turns=turns, reasoning_format="think")
    messages = sample["messages"]
    assert [message["role"] for message in messages[2:8]] == [
        "assistant", "tool_call", "assistant", "tool_response", "tool_call", "tool_response"]
    assert [message["role"] for message in messages[8:12]] == [
        "tool_call", "tool_call", "tool_response", "tool_response"]
    assert messages[2]["content"].count("<think>") == 1
    tool_messages = [message for message in messages if message["role"] == "tool_call"]
    assert tool_messages[0]["loss"] is True
    assert tool_messages[1]["loss"] is False
    incorrect = copy.deepcopy(turns)
    incorrect[0]["assistant_parts"][1]["arguments"]["query"] = "Something else"
    with pytest.raises(TrajectoryError, match="differs"):
        normalized_sample(record, case.query, cutoff=case.cutoff, model_turns=incorrect)
