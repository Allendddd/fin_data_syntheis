"""Web mode keeps native action spaces and audits actual provider observations."""

import copy
import json
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from mint_synth.agents.base import (ClaudeCodeSettings, CodexSettings, ModelSettings, RuntimeSettings,
                                    StrandsSettings, runtime_for_retrieval)
from mint_synth.agents.cc_native import export_cc_native
from mint_synth.agents.cc_workspace import prepare_workspace, finish_workspace
from mint_synth.agents.cli_runtime import _audit_events, _claude_command, _codex_command, _environment
from mint_synth.agents.native_audit import audit_trace
from mint_synth.agents.native_capture import export_native_capture
from mint_synth.core.config_v2 import load_config
from mint_synth.harness import case_from_task
from mint_synth.harness.registry import load_tool_registry
from mint_synth.harness.resources import validate_retrieval

from test_cc_native import write_calls, write_final_event
from test_native_capture import write_capture, write_final
from test_natural_sft import SOLUTION
from test_research_resources import SourceReviewer
from test_teacher_proof import task


URL = "https://sources.example/revenue"
SOURCE = "Revenue was $100 million in 2023. Revenue was $60 million in 2024."


def workspace(tmp_path, runtime, url=URL):
    original = task()
    for fact in original["evidence"]:
        fact["source_metadata"]["url"] = url
    settings = {"mode": "web", "corpus_path": str(tmp_path / "does-not-exist")}
    validate_retrieval(settings)
    case = case_from_task(original, None, cutoff="2025-12-31", retrieval=settings, archive_path=tmp_path / "sources")
    trace = tmp_path / "trace"
    work = trace / "workspace"
    work.mkdir(parents=True)
    prompt = prepare_workspace(case, work, trace, runtime=runtime)
    assert "only" not in prompt.lower() and "submit_final" not in prompt
    finish_workspace(work, trace)
    return case, trace, prompt


@pytest.mark.parametrize("runtime,name,date_case", [
    ("codex", "mcp__custom_research__fetch_report", "valid"),
    ("strands", "financial_news_search", "valid"),
    ("claude_code", "WebSearch", "valid"),
    ("claude_code", "mcp__custom_research__fetch_report", "valid"),
    ("codex", "mcp__custom_research__fetch_report", "unused_future_result"),
    ("codex", "mcp__custom_research__fetch_report", "used_future_result"),
])
def test_native_web_tools_are_not_filtered_by_name(tmp_path, runtime, name, date_case):
    case, trace, prompt = workspace(tmp_path, runtime)
    results = [{"url": URL, "snippet": SOURCE, "published_at": "2026-01-01" if date_case == "used_future_result" else "2025-01-01"}]
    if date_case == "unused_future_result":
        results.append({"url": "https://unrelated.example", "snippet": "Unrelated current news.", "published_at": "2026-01-01"})
    result = json.dumps({"results": results})
    args = {"query": "Revenue report"}
    if runtime == "codex":
        tool = {"type": "function", "name": name, "parameters": {"type": "object"}}
        action = {"type": "function_call", "name": name, "call_id": "source-1", "arguments": json.dumps(args)}
        context = [{"role": "user", "content": prompt}]
        first = {"id": "resp-1", "status": "completed", "output": [action]}
        calls = [({"model": "mock", "tools": [tool], "input": copy.deepcopy(context)}, first)]
        context += [action, {"type": "function_call_output", "call_id": "source-1", "output": result}]
        second = {"id": "resp-2", "status": "completed", "output": [{"type": "message", "id": "final",
                  "role": "assistant", "content": [{"type": "output_text", "text": SOLUTION}]}]}
        calls.append(({"model": "mock", "tools": [tool], "input": context}, second))
        write_capture(trace, calls, runtime)
        write_final(trace, SOLUTION, runtime)
    elif runtime == "claude_code":
        tool = {"name": name, "input_schema": {"type": "object"}}
        action = {"type": "tool_use", "name": name, "id": "source-1", "input": args}
        context = [{"role": "user", "content": prompt}]
        first = {"id": "resp-1", "role": "assistant", "content": [action], "stop_reason": "tool_use"}
        calls = [({"model": "mock", "tools": [tool], "messages": copy.deepcopy(context)}, first)]
        context += [{"role": "assistant", "content": [action]}, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "source-1", "content": result}]}]
        second = {"id": "resp-2", "role": "assistant", "content": [{"type": "text", "text": SOLUTION}], "stop_reason": "end_turn"}
        calls.append(({"model": "mock", "tools": [tool], "messages": context}, second))
        write_calls(trace, calls)
        write_final_event(trace, second)
    else:
        tool = {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        action = {"type": "function", "id": "source-1", "function": {"name": name, "arguments": json.dumps(args)}}
        context = [{"role": "user", "content": prompt}]
        first = {"id": "resp-1", "choices": [{"message": {"role": "assistant", "tool_calls": [action]}, "finish_reason": "tool_calls"}]}
        calls = [({"model": "mock", "tools": [tool], "messages": copy.deepcopy(context)}, first)]
        context += [first["choices"][0]["message"], {"role": "tool", "tool_call_id": "source-1", "content": result}]
        second = {"id": "resp-2", "choices": [{"message": {"role": "assistant", "content": SOLUTION}, "finish_reason": "stop"}]}
        calls.append(({"model": "mock", "tools": [tool], "messages": context}, second))
        write_capture(trace, calls, runtime)
        write_final(trace, SOLUTION, runtime)
    audit = audit_trace(case, trace, runtime, tolerance="1e-6", reviewer=SourceReviewer())
    assert audit["audit"]["verdict"]["accepted"] is (date_case != "used_future_result"), audit
    if date_case == "used_future_result":
        assert not audit["audit"]["verdict"]["cutoff_valid"]
        return
    assert audit["unrestricted_web"] and audit["read_bindings"][0]["tool"] == name
    bundle = export_cc_native(audit, trace) if runtime == "claude_code" else export_native_capture(audit, trace, runtime)
    assert bundle["sample_count"] == 2
    rows = [json.loads(line) for line in Path(bundle["sample_file"]).read_text(encoding="utf-8").splitlines()]
    assert SOURCE in json.dumps(rows[1])
    if runtime == "codex":
        assert not any(rows[1]["input_loss_mask"])
    else:
        assert not any(rows[1]["loss_mask"][:-1])


def test_codex_hosted_search_and_final_in_same_response(tmp_path):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(("<html><p>" + SOURCE + "</p></html>").encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/report"
        case, trace, prompt = workspace(tmp_path, "codex", url)
        response = {"id": "resp-1", "status": "completed", "output": [
            {"type": "web_search_call", "id": "hosted-1", "status": "completed", "action": {
                "type": "search", "query": "Revenue", "sources": [{"type": "url", "url": url}]}},
            {"type": "message", "id": "final", "role": "assistant", "content": [{"type": "output_text", "text": SOLUTION,
                "annotations": [{"type": "url_citation", "url": url, "title": "Report", "start_index": 0, "end_index": 50}]}]}]}
        write_capture(trace, [({"model": "mock", "tools": [{"type": "web_search"}], "input": prompt}, response)], "codex")
        write_final(trace, SOLUTION, "codex")
        audit = audit_trace(case, trace, "codex", tolerance="1e-6", reviewer=SourceReviewer())
        assert audit["audit"]["verdict"]["accepted"], audit
        assert requests == ["/report"]
        assert {binding["evidence_visibility"] for binding in audit["read_bindings"]} == {"verifier_snapshot"}
        replay = audit_trace(case, trace, "codex", tolerance="1e-6", reviewer=SourceReviewer())
        assert replay["audit"]["verdict"]["accepted"] and requests == ["/report"]
        bundle = export_native_capture(audit, trace, "codex")
        row = json.loads(Path(bundle["sample_file"]).read_text())
        assert row["output"] == response["output"]
        assert SOURCE not in json.dumps(row)
        assert row["target_output_loss_mask"] == [True, True]
        assert row["target_output_field_loss_mask"]["0"]["action.sources"] is False
        assert (case.archive_path / (next(case.archive_path.glob("*.bin")).name)).is_file()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_claude_hosted_search_preserves_encrypted_result_and_masks_observation(tmp_path, monkeypatch):
    case, trace, prompt = workspace(tmp_path, "claude_code")
    monkeypatch.setattr("httpx.stream", lambda *_a, **_kw: pytest.fail("visible cited text needs no background fetch"))
    response = {"id": "resp-1", "role": "assistant", "stop_reason": "end_turn", "content": [
        {"type": "server_tool_use", "id": "hosted-1", "name": "web_search", "input": {"query": "Revenue"}},
        {"type": "web_search_tool_result", "tool_use_id": "hosted-1", "content": [
            {"type": "web_search_result", "url": URL, "title": "Report", "encrypted_content": "opaque-original"}]},
        {"type": "text", "text": SOLUTION, "citations": [{"type": "web_search_result_location", "url": URL,
            "cited_text": SOURCE, "encrypted_index": "opaque-index"}]}]}
    write_calls(trace, [({"model": "mock", "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                         "messages": [{"role": "user", "content": prompt}]}, response)])
    write_final_event(trace, response)
    audit = audit_trace(case, trace, "claude_code", tolerance="1e-6", reviewer=SourceReviewer())
    assert audit["audit"]["verdict"]["accepted"], audit
    assert audit["read_bindings"][0]["evidence_visibility"] == "tool_observation"
    bundle = export_cc_native(audit, trace)
    row = json.loads(Path(bundle["sample_file"]).read_text())
    assert row["messages"][-1]["content"] == response["content"]
    assert row["target_block_loss_mask"] == [True, False, True]
    assert row["target_block_field_loss_mask"]["2"]["citations"] is False


def test_hosted_mcp_tools_keep_embedded_output_as_observation(tmp_path):
    case, trace, prompt = workspace(tmp_path, "codex")
    result = json.dumps({"url": URL, "text": SOURCE, "published_at": "2025-01-01"})
    response = {"id": "resp-1", "status": "completed", "output": [
        {"type": "mcp_list_tools", "id": "list-1", "server_label": "research", "tools": [{"name": "fetch"}]},
        {"type": "mcp_call", "id": "mcp-1", "server_label": "research", "name": "fetch",
         "arguments": json.dumps({"url": URL}), "output": result},
        {"type": "message", "id": "final", "role": "assistant", "content": [{"type": "output_text", "text": SOLUTION}]}]}
    write_capture(trace, [({"model": "mock", "tools": [{"type": "mcp", "server_label": "research", "server_url": "https://mcp.example"}],
                           "input": prompt}, response)], "codex")
    write_final(trace, SOLUTION, "codex")
    audit = audit_trace(case, trace, "codex", tolerance="1e-6", reviewer=SourceReviewer())
    assert audit["audit"]["verdict"]["accepted"], audit
    row = json.loads(Path(export_native_capture(audit, trace, "codex")["sample_file"]).read_text())
    assert row["output"] == response["output"]
    assert row["target_output_loss_mask"] == [False, True, True]
    assert row["target_output_field_loss_mask"]["1"]["output"] is False


def test_claude_hosted_mcp_toolset_needs_no_client_tool_name_whitelist(tmp_path):
    case, trace, prompt = workspace(tmp_path, "claude_code")
    response = {"id": "resp-1", "role": "assistant", "stop_reason": "end_turn", "content": [
        {"type": "mcp_tool_use", "id": "mcp-1", "server_name": "research", "name": "fetch", "input": {"url": URL}},
        {"type": "mcp_tool_result", "tool_use_id": "mcp-1", "content": json.dumps({"url": URL, "text": SOURCE})},
        {"type": "text", "text": SOLUTION}]}
    write_calls(trace, [({"model": "mock", "tools": [{"type": "mcp_toolset", "mcp_server_name": "research"}],
                         "messages": [{"role": "user", "content": prompt}]}, response)])
    write_final_event(trace, response)
    audit = audit_trace(case, trace, "claude_code", tolerance="1e-6", reviewer=SourceReviewer())
    assert audit["audit"]["verdict"]["accepted"], audit
    row = json.loads(Path(export_cc_native(audit, trace)["sample_file"]).read_text())
    assert row["target_block_loss_mask"] == [True, False, True]


def test_plaintext_search_results_keep_actual_url_and_text_pairs(tmp_path):
    from mint_synth.agents.web_evidence import bind_web_observation
    case, trace, _ = workspace(tmp_path, "codex")
    result = f"Report A ({URL})\nRevenue was $100 million in 2023.\n\nReport B (https://other.example/report)\nRevenue was $60 million in 2024."
    records = bind_web_observation(case, trace, {"id": "plain-1", "name": "AnySearch", "input": {}}, result)
    assert len(records) == 2
    first = next(record for record in records.values() if record["url"] == URL)
    assert "$100 million" in first["text"] and "$60 million" not in first["text"]


def test_unexecuted_client_calls_and_empty_hosted_search_do_not_fabricate_results():
    from mint_synth.agents.native_capture import _uses, hosted_results
    call = {"protocol": "responses", "request": {"tools": [{"type": "computer_use_preview"}, {"type": "web_search"}]}}
    response = {"output": [
        {"type": "computer_call", "id": "comp-1", "call_id": "client-1", "action": {"type": "screenshot"}},
        {"type": "web_search_call", "id": "web-1", "status": "completed", "action": {"type": "search", "query": URL}}]}
    uses = _uses(call, response, unrestricted_web=True)
    assert uses[0]["hosted"] is False
    results = list(hosted_results(response, uses))
    assert [key for key, _ in results] == ["web-1"]
    assert URL not in json.dumps(results)  # A URL in the query is not a returned source.


def test_web_configuration_needs_no_provider_and_does_not_register_forced_tools(tmp_path):
    config = yaml.safe_load(Path("config/mint-synth.example.yaml").read_text(encoding="utf-8"))
    config["harness"]["retrieval"] = {"mode": "web"}
    path = tmp_path / "web.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    retrieval = load_config(path)["harness"]["retrieval"]
    assert retrieval["web"]["search_endpoint"] is None
    validate_retrieval(retrieval)
    assert not load_tool_registry(retrieval=retrieval).definitions
    for name in ("codex", "claude_code", "strands"):
        runtime = runtime_for_retrieval(RuntimeSettings(name), retrieval)
        assert runtime.native_tools
    StrandsSettings(mode="native", tools=("tavily", "my_research.browser"))
    ClaudeCodeSettings(mode="native", allowed_tools=("mcp__custom__*",))


@pytest.mark.parametrize("runtime", ["codex", "claude_code"])
def test_web_cli_keeps_native_tools_and_accepts_other_mcp_events(tmp_path, runtime, monkeypatch):
    from types import SimpleNamespace
    home = tmp_path / "home"
    home.mkdir()
    bridge = SimpleNamespace(case=SimpleNamespace(retrieval={"mode": "web"}), tool_definitions=[])
    model = ModelSettings("mock", "http://localhost:9876/v1", None)
    monkeypatch.setenv("CUSTOM_SEARCH_API_KEY", "test-key")
    environment = _environment(home, model, inherit_tools=True)
    assert environment["CUSTOM_SEARCH_API_KEY"] == "test-key"
    if runtime == "codex":
        args = _codex_command(("codex",), home, tmp_path, model, bridge, None, CodexSettings())
        config = tomllib.loads((home / ".codex/config.toml").read_text())
        assert config["web_search"] == "live" and config["sandbox_workspace_write"]["network_access"] is True
        assert "shell_tool" not in config["features"] and "mcp_servers" not in config
        assert "--ignore-rules" not in args and "model_instructions_file" not in config
        event = {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "custom", "tool": "browser"}}
    else:
        args = _claude_command(("claude",), home, model, bridge, None, None, environment, ClaudeCodeSettings())
        assert not {"--disallowedTools", "--tools", "--allowedTools", "--strict-mcp-config", "--system-prompt"}.intersection(args)
        event = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__custom__browser"}]}}
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert _audit_events(events, runtime, set(), unrestricted_web=True) == (1, None)
