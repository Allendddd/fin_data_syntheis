"""Native CC configuration and lossless export; all model responses are mock."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mint_synth.agents.base import ClaudeCodeSettings, RuntimeResult, settings_for_role
from mint_synth.agents.cc_native import SAMPLE_FORMAT, anthropic_response, export_cc_native
from mint_synth.agents.cc_workspace import audit_native, finish_workspace, prepare_workspace
from mint_synth.agents.trajectory import TrajectoryError
from mint_synth.core.config_v2 import ConfigError, load_config
from mint_synth.core.run_state import RunLayout, atomic_json
from mint_synth.harness import case_from_task
from mint_synth.workflow.stages.publication import stage_export
from mint_synth.workflow.stages.sft_data import stage_sft_prepare

from test_teacher_proof import proof, snapshot, task


def native_record(index, root, mutate=None):
    case = case_from_task(task(), index, cutoff="2025-12-31")
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    prompt = prepare_workspace(case, workspace, root)
    calls = native_calls(workspace, prompt)
    if mutate:
        mutate(calls)
    write_calls(root, calls)
    write_final_event(root, calls[-1][1])
    finish_workspace(workspace, root)
    return audit_native(case, root), calls


def native_calls(workspace, prompt):
    tools = [{"name": name, "description": "Mock native tool", "input_schema": {"type": "object"}}
             for name in ("Read", "Bash", "EndConversation")]
    actions = [("Read", {"file_path": str(workspace / "corpus-manifest.json")}),
               ("Bash", {"command": "rg Revenue corpus"}),
               ("Read", {"file_path": str(workspace / "corpus/c1.txt")}),
               ("Read", {"file_path": str(workspace / "corpus/c2.txt")}),
               ("Bash", {"command": "python3 -c 'from decimal import Decimal; print(Decimal(100)+Decimal(60))'"}),
               (None, {})]
    observations = [(workspace / "corpus-manifest.json").read_text(encoding="utf-8"),
                    "corpus/c1.txt:Revenue\ncorpus/c2.txt:Revenue",
                    "     1→Revenue was $100 million in 2023.", "     1→Revenue was $60 million in 2024.", "160"]
    context = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    calls = []
    for step, (name, arguments) in enumerate(actions):
        request = {"model": "mock-cc-model", "max_tokens": 4096, "stream": False,
                   "system": [{"type": "text", "text": "Mock original CC system.", "cache_control": {"type": "ephemeral"}}],
                   "messages": copy.deepcopy(context), "tools": copy.deepcopy(tools),
                   "tool_choice": {"type": "auto"}}
        if step:
            request["tools"].append({"name": "Grep", "input_schema": {"type": "object"}})
        content = [{"type": "thinking", "thinking": f"Mock planning at {step}.", "signature": "signed-original"}]
        content.append({"type": "tool_use", "name": name, "id": f"call-{step}", "input": copy.deepcopy(arguments)} if name else
                       {"type": "text", "text": "```json\n" + json.dumps(proof()) + "\n```"})
        if step == 0:
            content.insert(1, {"type": "redacted_thinking", "data": "opaque-original"})
        response = {"id": f"msg-{step}", "type": "message", "role": "assistant", "model": "mock-cc-model",
                    "content": content, "stop_reason": "tool_use" if name else "end_turn", "usage": {"output_tokens": 10}}
        calls.append((request, response))
        context.append({"role": "assistant", "content": copy.deepcopy(content)})
        if name:
            context.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"call-{step}",
                "content": [{"type": "text", "text": observations[step]}]}]})
    return calls


def write_final_event(root, response):
    text = "\n".join(block["text"] for block in response["content"] if block["type"] == "text")
    events = [{"type": "assistant", "parent_tool_use_id": None, "message": {**response, "content": [block]}}
              for block in response["content"]]
    events.append({"type": "result", "subtype": "success", "is_error": False, "result": text})
    (root / "stdout.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def write_calls(root, calls):
    directory = root / "model/calls"
    directory.mkdir(parents=True, exist_ok=True)
    for index, (request, response) in enumerate(calls):
        call_id = f"mock-{index:06d}"
        request_bytes, response_bytes = json.dumps(request).encode(), json.dumps(response).encode()
        (directory / f"{call_id}.request.bin").write_bytes(request_bytes)
        (directory / f"{call_id}.response.bin").write_bytes(response_bytes)
        atomic_json(directory / f"{call_id}.json", {
            "call_id": call_id, "state": "complete", "started_at": f"2026-01-01T00:00:{index:02d}Z",
            "request": {"target": "/v1/messages", "body_file": f"{call_id}.request.bin",
                        "body_sha256": hashlib.sha256(request_bytes).hexdigest()},
            "response": {"status": 200, "complete": True, "headers": [], "body_file": f"{call_id}.response.bin",
                         "body_sha256": hashlib.sha256(response_bytes).hexdigest()},
        })


def test_native_configuration_round_trip_and_invalid_mode(tmp_path):
    example = Path(__file__).parents[1] / "config/mint-synth.example.yaml"
    value = yaml.safe_load(example.read_text(encoding="utf-8"))
    value["llm"]["endpoints"]["inference"]["require_api_key"] = False
    runtime = value["training"]["agent_runtime"]
    runtime.update(default="claude_code", claude_code={"mode": "native", "tools": ["Read", "Bash"],
                                                     "allowed_tools": ["Read", "Bash(echo *)"]})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    _, settings = settings_for_role(load_config(path), "teacher")
    assert settings.claude_code == ClaudeCodeSettings("native", ("Read", "Bash"), ("Read", "Bash(echo *)"))
    runtime["claude_code"]["mode"] = "typo"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
    runtime["claude_code"]["mode"] = "native"
    runtime["claude_code"]["tools"] = ["default", "Read"]
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_native_export_preserves_each_request_and_its_target(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    root = tmp_path / "trace"
    record, calls = native_record(index, root)
    bundle = export_cc_native(record, root)
    rows = [json.loads(line) for line in Path(bundle["sample_file"]).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == bundle["sample_count"] == 6
    for row, (request, response) in zip(rows, calls):
        assert row["sample_format"] == SAMPLE_FORMAT
        assert row["system"] == request["system"]
        assert row["tools"] == request["tools"]
        assert row["messages"][:-1] == request["messages"]
        assert row["messages"][-1] == {"role": "assistant", "content": response["content"]}
        assert row["loss_mask"] == [False] * len(request["messages"]) + [any(row["target_block_loss_mask"])]
        assert row["trajectory_id"] == bundle["trajectory_id"]
    assert rows[0]["tools"] != rows[1]["tools"]
    assert rows[0]["target_block_loss_mask"] == [False, False, False]
    assert rows[2]["target_block_loss_mask"] == [True, True]
    assert rows[4]["messages"][-1]["content"][-1]["name"] == "Bash"
    assert rows[5]["messages"][-1]["content"][-1]["type"] == "text"
    assert all(not tool["name"].startswith("mcp__") for row in rows for tool in row["tools"])
    assert record["audit"]["verdict"]["accepted"]
    assert rows[1]["messages"][-2]["content"][0]["tool_use_id"] == "call-0"


def test_native_export_rejects_undeclared_tools(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    root = tmp_path / "trace"
    record, calls = native_record(index, root)
    calls[0][1]["content"][-1]["name"] = "UndeclaredTool"
    write_calls(root, calls)
    with pytest.raises(TrajectoryError, match="absent from its request tools"):
        export_cc_native(record, root)


def test_native_request_start_order_is_not_assumed_to_be_execution_order(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    root = tmp_path / "trace"
    record, calls = native_record(index, root)
    # Simulate two helper requests: c2 started first but c1 executed first.
    # Each helper has its own actual request context, not the other's history.
    calls[2][0]["messages"] = [{"role": "user", "content": "Mock helper: inspect c1."}]
    calls[3][0]["messages"] = [{"role": "user", "content": "Mock helper: inspect c2."}]
    calls[2], calls[3] = calls[3], calls[2]
    write_calls(root, calls)
    record = audit_native(case_from_task(task(), index, cutoff="2025-12-31"), root)
    bundle = export_cc_native(record, root)
    provenance = json.loads((root / "trajectory.json").read_text(encoding="utf-8"))
    assert bundle["sample_count"] == 6
    assert provenance["sample_origins"][2]["read_bindings"][0]["chunk_id"] == "c2"
    assert provenance["sample_origins"][3]["read_bindings"][0]["chunk_id"] == "c1"


def test_native_stream_preserves_signatures_and_ids():
    events = [
        {"type": "message_start", "message": {"id": "msg", "role": "assistant", "content": [],
            "usage": {"input_tokens": 20, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Mock plan."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "signed-"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "original"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "read-1", "name": "Read", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"file_path":"notes.txt"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 10}},
        {"type": "message_stop"},
    ]
    text = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    response = anthropic_response(text, streaming=True)
    assert response["content"][0]["signature"] == "signed-original"
    assert response["content"][1] == {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": "notes.txt"}}
    assert response["usage"] == {"input_tokens": 20, "output_tokens": 10}


def test_native_rollout_prepare_and_publication_are_separate_from_ms_swift(tmp_path):
    layout = RunLayout(tmp_path / "run")
    index = layout.shard("retrieval_index", 0) / "snapshot.sqlite3"
    index.parent.mkdir(parents=True)
    snapshot(index)
    prepared = layout.shard("sft_prepare", 0)
    prepared.mkdir(parents=True)
    config = {"version": 3, "run": {"cutoff": "2025-12-31"}, "input": {"available_at_field": "published_at"},
        "quality": {"answer_tolerance": "1e-6"}, "reader": {"batch_rows": 16}, "dag": {"entity_buckets": 1},
        "query_export": {"enabled": False}, "output": {"export_formats": ["jsonl"], "artifact_checksums": True},
        "llm": {"concurrency_per_worker": 1, "roles": {"teacher": {"model": "mock", "endpoint": "local"}},
                "endpoints": {"local": {"base_url": "http://127.0.0.1:9876/v1", "api_key_env": "UNUSED", "require_api_key": False}}},
        "training": {"enabled": True, "atomic_enabled": False, "long_enabled": True,
            "long_teacher_roles": ["teacher"], "teacher_role": "teacher", "rollouts_per_task": 1,
            "agent_runtime": {"default": "claude_code", "claude_code": {"mode": "native"}},
            "split_validation_fraction": 0, "max_tool_steps": 4, "max_context_chars": 24000,
            "min_atomic_samples": 1, "min_long_samples": 1}}

    def mock_cli(case, _model, settings, root):
        assert settings.claude_code.mode == "native"
        assert not hasattr(case, "call_tool")
        workspace = root / "workspace"
        workspace.mkdir()
        prompt = prepare_workspace(case, workspace, root)
        calls = native_calls(workspace, prompt)
        write_calls(root, calls)
        write_final_event(root, calls[-1][1])
        finish_workspace(workspace, root)
        return RuntimeResult("completed")

    with (patch("mint_synth.workflow.stages.sft_data.iter_payloads", return_value=iter([task()])),
          patch("mint_synth.workflow.stages.sft_data._role_client", return_value=__import__("test_natural_sft").Reviewer()) as json_client,
          patch("mint_synth.agents.cli_runtime.run_cli", side_effect=mock_cli),
          patch("mint_synth.agents.collector.FinanceToolBridge", side_effect=AssertionError("no finance bridge"))):
        metrics = stage_sft_prepare(layout, config, {}, 0, prepared)
    assert json_client.call_count == 1 and json_client.call_args.args[-1] == "auditor"
    assert metrics["long_accepted"] == 1 and metrics["cc_native_requests"] == 6
    assert (prepared / "long.jsonl").read_text() == ""
    atomic_json(prepared / "_SUCCESS.json", {"metrics": metrics})
    plan = {"stages": [{"name": "validation", "shard_count": 1}, {"name": "sft_prepare", "shard_count": 1}]}
    output = tmp_path / "export"
    output.mkdir()
    with patch("mint_synth.workflow.stages.publication._enforce_global_quality", return_value={}):
        exported = stage_export(layout, config, plan, 0, output)
    assert exported["formats"] == ["jsonl"]
    root = output / "sft"
    rows = [json.loads(line) for line in (root / "cc-native-train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and all("_split" not in row for row in rows)
    assert (root / "long-train.jsonl").read_text() == ""
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["formats"]["cc-native"] == SAMPLE_FORMAT
    assert manifest["counts"]["cc_native_train_trajectories"] == 1
    entry = json.loads((root / "trajectory-index.jsonl").read_text())
    assert entry["claude_code_mode"] == "native" and entry["trajectory_id"] == rows[0]["trajectory_id"]
    assert (root / entry["trace"] / "trajectory.json").is_file()
    # Six requests from one task must not satisfy a two-trajectory minimum.
    config["training"]["min_long_samples"] = 2
    second_output = tmp_path / "export-too-few"
    second_output.mkdir()
    with (patch("mint_synth.workflow.stages.publication._enforce_global_quality", return_value={}),
          pytest.raises(RuntimeError, match="requires 2")):
        stage_export(layout, config, plan, 0, second_output)


def test_native_workspace_contains_all_visible_sources_without_gold(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    with closing(sqlite3.connect(index)) as db:
        for chunk, source, date, entity in [("distractor", "s3", "2025-02-01", "entity"),
                                            ("future", "s4", "2026-02-01", "entity"),
                                            ("other", "s5", "2025-02-01", "another"),
                                            ("undated", "s6", None, "entity")]:
            db.execute("INSERT INTO chunks VALUES (?,?,?,?)", (chunk, source, date, "Unrelated source text."))
            db.execute("INSERT INTO source_entities VALUES (?,?)", (source, entity))
        db.commit()
    for dated_only in (False, True):
        hidden = task()
        hidden["task_spec"]["time_sensitive"] = dated_only
        root = tmp_path / str(dated_only)
        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        case = case_from_task(hidden, index, cutoff="2025-12-31")
        prompt = prepare_workspace(case, workspace, root)
        public = json.loads((workspace / "corpus-manifest.json").read_text())
        chunks = {source["chunk_id"] for source in public["sources"]}
        assert chunks == {"c1", "c2", "distractor"} | (set() if dated_only else {"undated"})
        assert "160" not in prompt and "evidence_graph" not in prompt and "source_pack" not in prompt
        assert all(set(source) == {"chunk_id", "source_id", "available_at", "file", "sha256"} for source in public["sources"])
        finish_workspace(workspace, root)


@pytest.mark.parametrize("problem", ["partial", "error", "printed", "unread", "swapped"])
def test_offline_audit_requires_actual_source_reads(tmp_path, problem):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)

    def mutate(calls):
        if problem in {"printed", "swapped"}:
            block = calls[2][1]["content"][-1]
            if problem == "printed":
                block.update(name="Bash", input={"command": "echo 'Revenue was $100 million in 2023.'"})
            else:
                block["input"]["file_path"] = str(tmp_path / "trace/workspace/corpus/c2.txt")
        for request, _response in calls:
            for message in request["messages"]:
                for block in message["content"] if isinstance(message["content"], list) else []:
                    if block.get("type") == "tool_result" and block["tool_use_id"] == "call-2":
                        if problem == "partial":
                            block["content"] = "1→A partial read without the required quote."
                        elif problem == "error":
                            block["is_error"] = True
                        elif problem == "unread":
                            block["tool_use_id"] = "not-a-captured-call"

    record, _calls = native_record(index, tmp_path / "trace", mutate)
    assert not record["audit"]["verdict"]["accepted"]
    assert not record["audit"]["verdict"]["evidence_replayable"]


def test_simple_native_bash_cat_can_supply_evidence(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)

    def mutate(calls):
        block = calls[2][1]["content"][-1]
        path = Path(block["input"]["file_path"]).as_posix()
        block.update(name="Bash", input={"command": f"cat -- '{path}'"})
        for request, _ in calls:
            for message in request["messages"]:
                for result in message["content"] if isinstance(message["content"], list) else []:
                    if result.get("type") == "tool_result" and result["tool_use_id"] == "call-2":
                        result["content"] = "Revenue was $100 million in 2023."

    record, _ = native_record(index, tmp_path / "trace", mutate)
    assert record["audit"]["verdict"]["accepted"]
    assert any(binding["tool"] == "Bash" for binding in record["read_bindings"])


def test_native_final_is_not_selected_from_correct_earlier_drafts(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)

    def mutate(calls):
        calls[4][1]["content"].insert(1, {"type": "text", "text": json.dumps(proof())})
        incorrect = proof()
        incorrect["answer"]["value"] = "999"
        calls[-1][1]["content"][-1]["text"] = json.dumps(incorrect)

    record, _ = native_record(index, tmp_path / "trace", mutate)
    assert not record["audit"]["verdict"]["accepted"]
    assert not record["audit"]["verdict"]["answer_correct"]


def test_native_rejects_multiple_proof_candidates(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    with pytest.raises(TrajectoryError, match="exactly one JSON proof"):
        native_record(index, tmp_path / "trace", lambda calls: calls[-1][1]["content"][-1].update(
            text=json.dumps(proof()) + "\n" + json.dumps(proof())))


def test_native_reassembles_final_content_blocks_and_ignores_subagent_reply(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    root = tmp_path / "trace"
    record, calls = native_record(index, root, lambda calls: calls[-1][1]["content"].insert(
        1, {"type": "text", "text": "The two reported values sum to 160 million USD."}))
    assert record["audit"]["verdict"]["accepted"]
    events = [json.loads(line) for line in (root / "stdout.jsonl").read_text().splitlines()]
    # Even if result.result contains just the last block, check the full reply.
    events[-1]["result"] = calls[-1][1]["content"][-1]["text"]
    events.append({"type": "assistant", "parent_tool_use_id": "helper-call", "message": {
        "id": "subagent-final", "content": [{"type": "text", "text": "A different subagent response."}]}})
    (root / "stdout.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    record = audit_native(case_from_task(task(), index, cutoff="2025-12-31"), root)
    assert record["audit"]["verdict"]["accepted"]


def test_native_does_not_backfill_reads_from_requests_after_final(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    root = tmp_path / "trace"
    _, calls = native_record(index, root)
    late_request, late_response = copy.deepcopy(calls[-1])
    late_response.update(id="late-helper", content=[{"type": "text", "text": "A later helper reply."}])
    for request, _ in calls:
        for message in request["messages"]:
            if message["role"] == "user" and isinstance(message["content"], list):
                message["content"] = [block for block in message["content"]
                                      if block.get("tool_use_id") != "call-2"]
    calls.append((late_request, late_response))
    write_calls(root, calls)
    record = audit_native(case_from_task(task(), index, cutoff="2025-12-31"), root)
    assert not record["audit"]["verdict"]["accepted"]
    assert record["audit"]["verdict"]["missing_chunk_ids"] == ["c1"]


def test_native_rejects_changed_corpus_and_unbound_final(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    root = tmp_path / "trace"
    _, calls = native_record(index, root)
    final = copy.deepcopy(calls[-1][1])
    final["content"][-1]["text"] = "Forged final reply"
    write_final_event(root, final)
    case = case_from_task(task(), index, cutoff="2025-12-31")
    with pytest.raises(TrajectoryError, match="not bound"):
        audit_native(case, root)
    path = root / "workspace/corpus/c1.txt"
    path.write_text("Modified evidence", encoding="utf-8")
    finish_workspace(root / "workspace", root)
    with pytest.raises(TrajectoryError, match="integrity"):
        audit_native(case, root)


@pytest.mark.parametrize("field", ["tools", "allowed_tools"])
def test_native_configuration_accepts_explicit_mcp_tools(field):
    configured = ClaudeCodeSettings(mode="native", **{field: ("mcp__custom__search",)})
    assert getattr(configured, field) == ("mcp__custom__search",)


def test_native_export_rejects_even_unused_mcp_definitions(tmp_path):
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    root = tmp_path / "trace"
    record, calls = native_record(index, root)
    calls[0][0]["tools"].append({"name": "mcp__finance__search_corpus", "input_schema": {"type": "object"}})
    write_calls(root, calls)
    with pytest.raises(TrajectoryError, match="MCP tool definitions"):
        export_cc_native(record, root)
