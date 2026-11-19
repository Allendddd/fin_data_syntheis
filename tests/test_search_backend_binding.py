"""Native search tools reach the configured HTTP backend, not an extra tool."""

import asyncio
import copy
import json
import os
import shutil
import sys
import threading
import types
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from mint_synth.agents.base import ClaudeCodeSettings, ModelSettings, RuntimeSettings, StrandsSettings
from mint_synth.agents.collector import collect_runtime
from mint_synth.agents.model_trace import iter_captured_calls
from mint_synth.agents.search_router import NativeSearchRouter, SearchRoutingError
from mint_synth.harness import case_from_task
from mint_synth.trace_proxy import TraceProxyConfig, TraceProxyServer

from test_natural_sft import SOLUTION
from test_research_resources import SourceReviewer
from test_teacher_proof import task


URL = "https://example.org/revenue"
SOURCE = "Revenue was $100 million in 2023. Revenue was $60 million in 2024."


@contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class Services(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.server.seen.append((self.path, None, None))
        body = ('<html><meta property="article:published_time" content="2025-01-01">'
                '<p>Revenue report</p><p>' + SOURCE + '</p><a href="/detail.html">Source details</a></html>')
        self.reply(body.encode(), media="text/html")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        self.server.seen.append((self.path, body, self.headers.get("Authorization")))
        if self.path == "/configured-search":
            if self.server.failure:
                self.reply({"error": "backend unavailable"}, status=503)
            else:
                self.reply(getattr(self.server, "backend_payload", {"results": [{"url": URL, "title": "Revenue report", "content": SOURCE, "published_at": "2025-01-01"}]}))
            return
        self.server.model_requests += 1
        first = self.server.model_requests == 1
        if self.path.endswith("/responses"):
            assert any(tool.get("name") == "web" for tool in body["tools"])
            item = {"type": "function_call", "id": "fc_search", "call_id": "search_1", "name": "run", "namespace": "web",
                    "arguments": json.dumps({"search_query": [{"q": "Revenue report"}]}), "status": "completed"} if first else {
                    "type": "message", "id": "final_1", "role": "assistant", "phase": "final_answer",
                    "content": [{"type": "output_text", "text": SOLUTION, "annotations": []}]}
            result = {"id": f"resp_{self.server.model_requests}", "object": "response", "created_at": 0,
                      "model": body["model"], "status": "completed", "output": [item],
                      "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
            frames = [{"type": "response.created", "response": {**result, "status": "in_progress", "output": []}},
                      {"type": "response.output_item.added", "output_index": 0, "item": item},
                      {"type": "response.output_item.done", "output_index": 0, "item": item},
                      {"type": "response.completed", "response": result}]
        else:
            # The native WebSearch helper request must be handled by the backend
            # router; it must never be forwarded to the teacher model service.
            assert not any(str(tool.get("type", "")).startswith("web_search_") for tool in body.get("tools", []))
            block = {"type": "tool_use", "id": "search_1", "name": "WebSearch", "input": {"query": "Revenue report"}} if first else {
                "type": "text", "text": SOLUTION}
            stop = "tool_use" if first else "end_turn"
            result = {"id": f"msg_{self.server.model_requests}", "type": "message", "role": "assistant", "model": body["model"],
                      "content": [block], "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
            start = {**block, "input": {}} if first else {**block, "text": ""}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])} if first else {"type": "text_delta", "text": SOLUTION}
            frames = [{"type": "message_start", "message": {**result, "content": [], "stop_reason": None}},
                      {"type": "content_block_start", "index": 0, "content_block": start},
                      {"type": "content_block_delta", "index": 0, "delta": delta},
                      {"type": "content_block_stop", "index": 0},
                      {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 1}},
                      {"type": "message_stop"}]
        if body.get("stream"):
            payload = "".join("event: " + frame["type"] + "\ndata: " + json.dumps(frame) + "\n\n" for frame in frames).encode()
            self.reply(payload, media="text/event-stream")
        else:
            self.reply(result)

    def reply(self, body, *, status=200, media="application/json"):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def services(*, failure=False):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Services)
    server.seen, server.model_requests, server.failure = [], 0, failure
    with running(server):
        yield server


def configuration(server, *, protocol="json"):
    return {"run": {"cutoff": "2025-12-31"}, "input": {"available_at_field": "published_at"},
            "harness": {"retrieval": {"mode": "web", "web": {"search_endpoint": f"http://127.0.0.1:{server.server_port}/configured-search",
                "search_protocol": protocol, "api_key_env": "TEST_SEARCH_BACKEND_KEY"}}},
            "training": {"max_tool_steps": 8, "max_context_chars": 24000}, "quality": {"answer_tolerance": "1e-6"}}


def original_task():
    original = task()
    for fact in original["evidence"]:
        fact["source_metadata"]["url"] = URL
    return original


def actual_cli(name):
    command = shutil.which("codex" if name == "codex" else "claude")
    if not command:
        pytest.skip(name + " CLI is not installed")
    if name == "codex" and os.name == "nt" and Path(command).suffix.lower() in {".cmd", ".ps1"}:
        launcher = Path(command).parent / "node_modules/@openai/codex/bin/codex.js"
        if not launcher.is_file() or not shutil.which("node"):
            pytest.skip("Codex Node launcher is unavailable")
        return (shutil.which("node"), str(launcher))
    return (command,)


@pytest.mark.parametrize("name,protocol", [("codex", "json"), ("codex", "tavily"), ("claude_code", "json"), ("claude_code", "tavily")])
def test_actual_cli_uses_native_search_tool_with_configured_backend(tmp_path, monkeypatch, name, protocol):
    prefix = actual_cli(name)
    monkeypatch.setenv("TEST_SEARCH_BACKEND_KEY", "backend-only-key")
    with services() as server:
        config = configuration(server, protocol=protocol)
        runtime = RuntimeSettings(name=name, command=prefix, timeout_seconds=30,
                                  claude_code=ClaudeCodeSettings(allowed_tools=("WebSearch",)),
                                  strands=StrandsSettings(search_query_field="irrelevant_field"))
        model = ModelSettings("mock-finance" if name == "codex" else "claude-sonnet-4-6",
                              f"http://127.0.0.1:{server.server_port}/v1", "model-only-key", timeout_seconds=20)
        trace = tmp_path / "trace"
        sample, reason = collect_runtime(original_task(), None, config, None, model=model, runtime=runtime,
                                         trace_dir=trace, sample_key="backend-binding", verifier=SourceReviewer())
        assert sample is not None, (reason, (trace / "result.json").read_text())
        assert sample["sample_count"] == 2
        backend = [item for item in server.seen if item[0] == "/configured-search"]
        assert len(backend) == 1
        expected = {"query": "Revenue report", "limit": 5} if protocol == "json" else {
            "query": "Revenue report", "max_results": 5, "include_answer": False,
            "include_raw_content": False, "include_published_date": True}
        assert backend[0][1] == expected
        assert backend[0][2] == "Bearer backend-only-key"
        assert server.model_requests == 2
    state = json.loads((trace / "search-backend.json").read_text())
    assert state["binding"] == "bound" and state["rpc_requests"] == 1
    calls = list(iter_captured_calls(trace))
    assert len(calls) == 2
    assert "mcp__finance__search_web" not in json.dumps([call["request"].get("tools") for call in calls])
    assert list((trace / "sources").glob("backend_*.json"))
    raw = [json.loads(path.read_text()) for path in (trace / "model/calls").glob("*.json")]
    assert sum(call.get("capture_kind") == "search_backend" for call in raw) == 1


def router(tmp_path, server, name="codex", **runtime_options):
    trace = tmp_path / "trace"
    trace.mkdir()
    config = configuration(server)
    case = case_from_task(original_task(), None, cutoff="2025-12-31", retrieval=config["harness"]["retrieval"], archive_path=trace / "sources")
    return NativeSearchRouter(case, RuntimeSettings(name, **runtime_options), trace)


def test_unsupported_codex_provider_fails_before_model_or_backend_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_BACKEND_KEY", "backend-only-key")
    with services() as server:
        native = router(tmp_path, server)
        request = {"model": "mock", "tools": [{"type": "web_search"}, {"type": "function", "name": "shell"}]}
        original = copy.deepcopy(request)
        try:
            with pytest.raises(SearchRoutingError, match="standalone web.run"):
                native.handle("POST", "/v1/responses", json.dumps(request).encode(), call_id="unsupported")
            assert request == original and server.seen == []
            assert native.record["binding"] == "unsupported"
        finally:
            native.close()


def test_backend_failure_is_recorded_without_provider_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_BACKEND_KEY", "backend-only-key")
    with services(failure=True) as server:
        native = router(tmp_path, server)
        try:
            reply = native.handle("POST", "/v1/alpha/search", json.dumps({"commands": {"search_query": [{"q": "Revenue report"}]}}).encode(), call_id="failed")
            assert reply.status == 502 and native.record["failed_rpc_requests"] == 1
            assert len(server.seen) == 1 and server.model_requests == 0
            assert json.loads(next((native.trace_dir / "sources").glob("backend_*.json")).read_text())["status"] == "failed"
        finally:
            native.close()


def test_backend_response_protocol_mismatch_is_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_BACKEND_KEY", "backend-only-key")
    with services() as server:
        server.backend_payload = {"data": []}
        native = router(tmp_path, server)
        try:
            reply = native.handle("POST", "/v1/alpha/search", json.dumps({"commands": {"search_query": [{"q": "Revenue report"}]}}).encode(), call_id="bad-protocol")
            assert reply.status == 502 and "results array" in reply.body.decode()
            assert len(server.seen) == 1 and server.model_requests == 0
        finally:
            native.close()


def test_native_search_cannot_credit_observations_from_an_unbound_backend(tmp_path, monkeypatch):
    from mint_synth.agents.resource_capture import bind_resource_observation
    from mint_synth.agents.trajectory import TrajectoryError
    monkeypatch.setenv("TEST_SEARCH_BACKEND_KEY", "backend-only-key")
    with services() as server:
        native = router(tmp_path, server, "claude_code")
        try:
            use = {"id": "fake-read", "name": "WebSearch", "input": {"query": "Revenue report"}}
            with pytest.raises(TrajectoryError, match="not bound"):
                bind_resource_observation(native.case, native.trace_dir, use, URL + "\n" + SOURCE, {}, tmp_path)
            assert server.seen == []
        finally:
            native.close()


def test_search_backend_preflight_catches_missing_native_entry_point():
    from mint_synth.agents.base import search_backend_contract
    config = {"harness": {"retrieval": {"mode": "web", "web": {"search_endpoint": "https://search.example"}}},
              "training": {"agent_runtime": {"default": "claude_code", "claude_code": {"tools": ["Read", "Bash"]}}}}
    with pytest.raises(ValueError, match="omits WebSearch"):
        search_backend_contract(config)
    config["training"]["agent_runtime"]["claude_code"]["tools"] = ["default"]
    status = search_backend_contract(config)
    assert status["runtime_verification"] == "pending" and status["bindings"][0]["search_tool"] == "WebSearch"


def test_codex_open_find_and_click_use_real_pages_and_reference_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_BACKEND_KEY", "backend-only-key")
    with services() as server:
        native = router(tmp_path, server)
        def command(value, key):
            reply = native.handle("POST", "/v1/alpha/search", json.dumps({"commands": value}).encode(), call_id=key)
            assert reply.status == 200, reply.body
            return json.loads(reply.body)
        try:
            opened = command({"open": [{"ref_id": f"http://127.0.0.1:{server.server_port}/original.html", "lineno": 1}]}, "open")
            assert SOURCE in opened["output"] and "Link 0:" in opened["output"]
            ref = opened["results"][0]["ref_id"]
            clicked = command({"click": [{"ref_id": ref, "id": 0}]}, "click")
            assert SOURCE in clicked["output"] and clicked["results"][0]["url"].endswith("/detail.html")
            found = command({"find": [{"ref_id": ref, "pattern": "$60 million"}]}, "find")
            assert "$60 million" in found["output"]
            assert [path for path, _, _ in server.seen] == ["/original.html", "/detail.html", "/original.html"]
            assert list((native.trace_dir / "sources").glob("*.bin"))
        finally:
            native.close()


def test_general_json_backend_receives_other_native_commands_without_a_tool_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SEARCH_BACKEND_KEY", "backend-only-key")
    with services() as server:
        native = router(tmp_path, server)
        commands = {"finance": [{"ticker": "EXAMPLE", "type": "equity", "market": "USA"}],
                    "search_query": [{"q": "Revenue report"}], "response_length": "long"}
        try:
            reply = native.handle("POST", "/v1/alpha/search", json.dumps({"commands": commands}).encode(), call_id="general")
            assert reply.status == 200
            assert server.seen[0][1] == {"commands": commands, "limit": 5}
            assert server.model_requests == 0
        finally:
            native.close()


def test_strands_binding_preserves_selected_tool_schema_and_executes_backend(tmp_path, monkeypatch):
    from mint_synth.agents.strands_search import bind_search_tool
    monkeypatch.setenv("TEST_SEARCH_BACKEND_KEY", "backend-only-key")
    module = types.ModuleType("strands.tools.tools")
    class SDKTool:
        def __init__(self, *, tool_name, tool_spec, tool_func):
            self.tool_name, self.tool_spec, self.tool_func = tool_name, tool_spec, tool_func
    module.PythonAgentTool = SDKTool
    monkeypatch.setitem(sys.modules, "strands.tools.tools", module)
    spec = {"name": "analyst-search", "description": "Search financial sources", "inputSchema": {"json": {
        "type": "object", "properties": {"search_term": {"type": "string"}}, "required": ["search_term"]}}}
    original = types.SimpleNamespace(TOOL_SPEC=copy.deepcopy(spec))
    other = object()
    with services() as server:
        native = router(tmp_path, server, "strands", strands=StrandsSettings(search_tool="analyst-search", search_query_field="search_term"))
        proxy = TraceProxyServer(TraceProxyConfig(upstream=f"http://127.0.0.1:{server.server_port}", output_dir=tmp_path / "proxy",
                                                run_id="strands-binding", port=0), request_router=native)
        try:
            with running(proxy):
                bound = bind_search_tool([other, original], {"url": f"http://127.0.0.1:{proxy.server_port}/v1/mint/search",
                    "tool": "analyst-search", "query_field": "search_term", "limit_field": "limit"})
                assert bound[0] is other and bound[1].tool_spec == original.TOOL_SPEC == spec
                result = asyncio.run(bound[1].tool_func({"toolUseId": "search-1", "input": {"search_term": "Revenue report"}}))
                assert result["status"] == "success" and SOURCE in result["content"][0]["text"]
            assert server.seen[0][1] == {"query": "Revenue report", "limit": 5}
            assert native.record["rpc_requests"] == 1
        finally:
            native.close()
