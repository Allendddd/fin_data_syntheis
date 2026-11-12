"""Mock model responses exercise the real HTTP client, cache and rollout."""

from __future__ import annotations

import json
import httpx
import pytest

from mint_synth.inference.exchange import native_content
from mint_synth.inference.llm import ClientConfig, LLMExchangeError, OpenAICompatibleClient
from mint_synth.inference.runtime import ConfiguredRoleClient, SqliteResponseCache
from mint_synth.sft.rollout import long_rollout

from test_teacher_proof import proof, snapshot, task


CONFIG = {"training": {"max_tool_steps": 6, "max_context_chars": 24000},
          "run": {"cutoff": "2025-12-31"}, "input": {"available_at_field": "published_at"},
          "quality": {"answer_tolerance": "1e-6"}}


def model_client(handler):
    client = OpenAICompatibleClient(ClientConfig("mock-teacher", retries=0))
    client.close()
    client._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def teacher_api(requests, *, analysis=True, memory=True):
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        state = json.loads(payload["messages"][1]["content"])
        step = len(state["history"])
        actions = [
            {"tool": "search_corpus", "arguments": {"query": "Revenue"}},
            {"tool": "open_chunk", "arguments": {"chunk_id": "c1"}},
            {"tool": "open_chunk", "arguments": {"chunk_id": "c2"}},
            {"tool": "calculate", "arguments": {"operation": "add", "values": [100, 60]}},
            {"final": json.dumps(proof())},
        ]
        action = actions[step]
        if memory and step == 0:
            action["memory"] = {"plan": "Find both annual disclosures.", "confirmed": [],
                                "unresolved": ["The two revenue values."]}
        prefix = "I need evidence for both periods.\n" if analysis and step == 0 else ""
        suffix = "\nI will inspect the returned reports next." if analysis and step == 0 else ""
        message = {"role": "assistant", "content": prefix + json.dumps(action) + suffix}
        if analysis:
            message["reasoning_content"] = f"Mock exposed analysis at decision {step}."
        return httpx.Response(200, json={"id": f"mock-{step}", "choices": [
            {"message": message, "finish_reason": "stop"}], "usage": {"total_tokens": 12},
            "provider_metadata": {"preserve": "all original fields"}})
    return respond


def test_complete_native_text_memory_and_order_are_exported(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    requests = []
    client = model_client(teacher_api(requests))
    trace = tmp_path / "trace"
    try:
        sample, reason = long_rollout(task(), client, index, CONFIG, trace_dir=trace)
    finally:
        client.close()
    assert reason == "accepted"
    messages = sample["messages"]
    contents = [message["content"] for message in messages]
    assert "Mock exposed analysis at decision 0." in contents
    assert "I need evidence for both periods.\n" in contents
    memory = next(message for message in messages if message["content"].startswith("<memory>"))
    assert memory["role"] == "assistant" and memory["loss"] is True
    assert json.loads(memory["content"][8:-9])["plan"] == "Find both annual disclosures."
    call = next(i for i, message in enumerate(messages) if message["role"] == "tool_call")
    observation = next(i for i, message in enumerate(messages) if message["role"] == "tool_response")
    tail = contents.index("\nI will inspect the returned reports next.")
    assert call < tail < observation  # Suffix was emitted before the tool ran.
    assert "memory" not in json.loads(messages[call]["content"])["arguments"]
    second_input = json.loads(requests[1]["messages"][1]["content"])
    assert second_input["working_memory"]["plan"] == "Find both annual disclosures."
    assert json.loads(messages[-1]["content"]) == proof()
    assert all(message["loss"] is False for message in messages if message["role"] in {"system", "user", "tool_response"})
    assert "evidence_graph" not in json.dumps(sample)
    record = json.loads((trace / "model/calls/000001.json").read_text(encoding="utf-8"))
    assert record["request"] == requests[0]
    assert record["response"]["provider_metadata"]["preserve"] == "all original fields"
    assert record["response"]["choices"][0]["message"]["content"].endswith("reports next.")
    provenance = json.loads((trace / "trajectory.json").read_text(encoding="utf-8"))
    assert provenance["trajectory_id"] == sample["trajectory_id"]
    for origin in provenance["message_origins"]:
        assert (trace / origin["source"]).is_file()
    assert len(list((trace / "model/calls").glob("*.json"))) == 5


def test_no_reasoning_or_memory_is_invented(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    client = model_client(teacher_api([], analysis=False, memory=False))
    try:
        sample, reason = long_rollout(task(), client, index, CONFIG)
    finally:
        client.close()
    assert reason == "accepted"
    assistants = [message for message in sample["messages"] if message["role"] == "assistant"]
    assert len(assistants) == 1
    assert json.loads(assistants[0]["content"]) == proof()
    assert "<memory>" not in json.dumps(assistants)


def test_full_exchange_cache_is_separate_from_legacy_cache_and_survives_restart(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "reasoning_content": "Keep this exposed plan.",
            "content": '{"tool":"search_corpus","arguments":{"query":"Revenue"}}'}}]})

    cache_path = tmp_path / "responses.sqlite3"
    def configured():
        return ConfiguredRoleClient(model_client(handler), temperature=0.2, max_tokens=None,
            cache=SqliteResponseCache(cache_path, policy="immutable", busy_timeout_seconds=1))

    first = configured()
    arguments = {"system": "system", "user": "state", "sample_key": "same-slot"}
    try:
        legacy = first.complete_json(**arguments)
        complete = first.complete_json_exchange(**arguments)
        assert len(calls) == 2
        assert legacy == complete["parsed_response"]
        assert complete["cache_hit"] is False
    finally:
        first.close()
    second = configured()
    try:
        replay = second.complete_json_exchange(**arguments)
        assert replay["cache_hit"] is True
        assert replay["response"] == complete["response"]
        assert replay["assistant_parts"] == complete["assistant_parts"]
        assert replay["call_id"] == complete["call_id"]
        assert len(calls) == 2
    finally:
        second.close()


def test_failed_parse_still_records_full_response(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    client = model_client(lambda _: httpx.Response(200, json={"choices": [{"message": {
        "reasoning_content": "Exposed planning survived the invalid action.", "content": "not an action"}}]}))
    try:
        with pytest.raises(LLMExchangeError):
            long_rollout(task(), client, index, CONFIG, trace_dir=tmp_path / "trace")
    finally:
        client.close()
    record = json.loads((tmp_path / "trace/model/calls/000001.json").read_text(encoding="utf-8"))
    assert record["response"]["choices"][0]["message"]["reasoning_content"].startswith("Exposed planning")
    result = json.loads((tmp_path / "trace/result.json").read_text(encoding="utf-8"))
    assert result["accepted"] is False


def test_parsed_only_client_cannot_silently_publish(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    sample, reason = long_rollout(task(), object(), index, CONFIG)
    assert sample is None
    assert "complete exchange capture is required" in reason


def test_action_parser_preserves_surrounding_json_in_analysis():
    action, before, after = native_content('Plan metadata: {"period":"2024"}\n'
        '{"tool":"search_corpus","arguments":{"query":"Revenue"}}\nInspect the results.')
    assert action["tool"] == "search_corpus"
    assert '{"period":"2024"}' in before
    assert after == "\nInspect the results."
