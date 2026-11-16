"""Mock native protocols exercise real capture checksums and offline grading."""

from __future__ import annotations

import copy
import hashlib
import json
import shlex
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mint_synth.agents.base import CodexSettings, RuntimeResult, StrandsSettings, settings_for_role
from mint_synth.agents.cc_workspace import finish_workspace, prepare_workspace
from mint_synth.agents.native_capture import audit_native_capture, export_native_capture, native_response
from mint_synth.agents.trajectory import TrajectoryError
from mint_synth.core.run_state import RunLayout, atomic_json
from mint_synth.core.config_v2 import ConfigError, load_config
from mint_synth.harness import case_from_task
from mint_synth.workflow.stages.publication import stage_export
from mint_synth.workflow.stages.sft_data import stage_sft_prepare

from test_teacher_proof import proof, snapshot, task


def write_capture(root: Path, calls: list[tuple[dict, dict | str]], runtime: str) -> None:
    directory = root / "model/calls"
    directory.mkdir(parents=True, exist_ok=True)
    for index, (request, response) in enumerate(calls):
        call_id = f"mock-{index:06d}"
        request_bytes = json.dumps(request).encode()
        response_bytes = response.encode() if isinstance(response, str) else json.dumps(response).encode()
        (directory / f"{call_id}.request.bin").write_bytes(request_bytes)
        (directory / f"{call_id}.response.bin").write_bytes(response_bytes)
        atomic_json(directory / f"{call_id}.json", {
            "call_id": call_id, "state": "complete", "started_at": f"2026-01-01T00:00:{index:02d}Z",
            "request": {"target": "/v1/responses" if runtime == "codex" else "/v1/chat/completions",
                        "body_file": f"{call_id}.request.bin", "body_sha256": hashlib.sha256(request_bytes).hexdigest()},
            "response": {"status": 200, "complete": True,
                "headers": [["content-type", "text/event-stream"]] if isinstance(response, str) else [],
                "body_file": f"{call_id}.response.bin", "body_sha256": hashlib.sha256(response_bytes).hexdigest()},
        })


def responses_calls(workspace: Path, prompt: str) -> list[tuple[dict, dict]]:
    context = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    tools = [{"type": "function", "name": "exec_command", "description": "Mock native shell",
              "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
             {"type": "custom", "name": "apply_patch", "format": {"type": "text"}}]
    calls = []
    for step in range(4):
        request = {"model": "mock-codex", "stream": False, "instructions": "Mock original Codex instructions",
                   "input": copy.deepcopy(context), "tools": copy.deepcopy(tools),
                   "reasoning": {"effort": "medium"}, "temperature": 0.3}
        if step:
            request["tools"].append({"type": "function", "name": "shell_command", "parameters": {"type": "object"}})
        output = [{"type": "reasoning", "id": f"reason-{step}", "summary": [{"type": "summary_text", "text": "Mock plan"}],
                   "encrypted_content": "opaque-original"}]
        if step < 2:
            path = workspace / f"corpus/c{step + 1}.txt"
            output.append({"type": "function_call", "id": f"fc-{step}", "call_id": f"read-{step}",
                           "name": "exec_command", "arguments": json.dumps({"cmd": "cat -- " + shlex.quote(str(path.resolve()))})})
            result = {"type": "function_call_output", "call_id": f"read-{step}",
                      "output": "Process exited with code 0\nFinal output:\n" + path.read_text(encoding="utf-8")}
        elif step == 2:
            output.append({"type": "custom_tool_call", "id": "patch-item", "call_id": "patch-0", "name": "apply_patch",
                           "input": "*** Begin Patch\n*** Add File: scratch.txt\n+temporary calculation\n*** End Patch"})
            result = {"type": "custom_tool_call_output", "call_id": "patch-0", "output": "Success. Updated scratch.txt"}
        else:
            output.append({"type": "message", "id": "final-model-message", "role": "assistant", "phase": "final_answer",
                           "content": [{"type": "output_text", "text": json.dumps(proof()), "annotations": []}]})
            result = None
        response = {"id": f"resp-{step}", "status": "completed", "model": "mock-codex", "output": output,
                    "usage": {"input_tokens": 50, "output_tokens": 10}}
        calls.append((request, response))
        context.extend(copy.deepcopy(output))
        if result:
            context.append(result)
    return calls


def chat_calls(workspace: Path, prompt: str, *, shell: bool = False) -> list[tuple[dict, dict]]:
    context = [{"role": "system", "content": "Mock original Strands system"}, {"role": "user", "content": prompt}]
    tools = [{"type": "function", "function": {"name": "file_read", "description": "Mock official file reader",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
             {"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}]
    calls = []
    for step in range(3):
        request = {"model": "mock-strands", "stream": False, "messages": copy.deepcopy(context),
                   "tools": copy.deepcopy(tools), "temperature": 0.4, "max_tokens": 500}
        if step:
            request["tools"].append({"type": "function", "function": {"name": "file_write", "parameters": {"type": "object"}}})
        message = {"role": "assistant", "content": "Mock planning" if step < 2 else json.dumps(proof()),
                   "reasoning_content": "Mock exposed reasoning"}
        if step < 2:
            path = workspace / f"corpus/c{step + 1}.txt"
            name = "shell" if shell else "file_read"
            arguments = {"command": "cat -- " + shlex.quote(str(path.resolve()))} if shell else {"path": str(path.resolve()), "mode": "view"}
            message["tool_calls"] = [{"id": f"read-{step}", "type": "function",
                                      "function": {"name": name, "arguments": json.dumps(arguments)}}]
            output = path.read_text(encoding="utf-8")
            if not shell:
                output = "1 | " + output
            result = {"role": "tool", "tool_call_id": f"read-{step}",
                      "content": json.dumps({"status": "success", "content": [{"text": output}]})}
        else:
            result = None
        response = {"id": f"chat-{step}", "model": "mock-strands", "choices": [
            {"index": 0, "message": copy.deepcopy(message), "finish_reason": "tool_calls" if step < 2 else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        calls.append((request, response))
        context.append(copy.deepcopy(message))
        if result:
            context.append(result)
    return calls


def final_text(calls: list[tuple[dict, dict]], runtime: str) -> str:
    response = calls[-1][1]
    if runtime == "codex":
        return response["output"][-1]["content"][0]["text"]
    return response["choices"][0]["message"]["content"]


def write_final(root: Path, text: str, runtime: str) -> None:
    if runtime == "codex":
        rows = [{"type": "turn.started"}, {"type": "item.completed", "item": {
            "type": "agent_message", "id": "cli-item-final", "text": text}}, {"type": "turn.completed"}]
        (root / "stdout.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        atomic_json(root / "native-final.json", {"text": text})


def record(tmp_path: Path, runtime: str, mutate=None, *, shell: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    case = case_from_task(task(), index, cutoff="2025-12-31")
    root = tmp_path / "trace"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    prompt = prepare_workspace(case, workspace, root, runtime=runtime)
    calls = responses_calls(workspace, prompt) if runtime == "codex" else chat_calls(workspace, prompt, shell=shell)
    expected_final = final_text(calls, runtime)
    if mutate:
        mutate(calls)
    write_capture(root, calls, runtime)
    write_final(root, expected_final, runtime)
    finish_workspace(workspace, root)
    return case, root, calls


def outputs(request: dict, runtime: str):
    if runtime == "codex":
        return [item for item in request["input"] if item.get("type") == "function_call_output"]
    return [item for item in request["messages"] if item.get("role") == "tool"]


@pytest.mark.parametrize("runtime,shell", [("codex", False), ("strands", False), ("strands", True)])
def test_native_reads_are_audited_and_request_specific_protocol_is_preserved(tmp_path, runtime, shell):
    case, root, calls = record(tmp_path, runtime, shell=shell)
    audited = audit_native_capture(case, root, runtime)
    assert audited["audit"]["verdict"]["accepted"]
    assert {item["chunk_id"] for item in audited["read_bindings"]} == {"c1", "c2"}
    bundle = export_native_capture(audited, root, runtime)
    rows = [json.loads(line) for line in Path(bundle["sample_file"]).read_text().splitlines()]
    assert len(rows) == len(calls) == bundle["sample_count"]
    assert rows[0]["tools"] != rows[1]["tools"]
    for row, (request, response) in zip(rows, calls):
        for key, value in request.items():
            if key == "messages":
                assert row[key][:-1] == value
            else:
                assert row[key] == value
        assert row["sample_format"] == runtime + "_native_v1"
        if runtime == "codex":
            assert row["output"] == response["output"]
            assert row["input_loss_mask"] == [False] * len(request["input"])
        else:
            assert row["messages"][-1] == response["choices"][0]["message"]
            assert row["loss_mask"] == [False] * len(request["messages"]) + [True]
            assert row["target_message_index"] == len(request["messages"])
    if runtime == "codex":
        assert rows[2]["output"][-1]["input"] == calls[2][1]["output"][-1]["input"]
    provenance = json.loads((root / "trajectory.json").read_text())
    assert len(provenance["sample_origins"]) == len(calls)
    assert all(row["trajectory_id"] == provenance["trajectory_id"] for row in rows)


@pytest.mark.parametrize("runtime", ["codex", "strands"])
@pytest.mark.parametrize("problem", ["partial", "error", "echo", "missing_result_id"])
def test_native_audit_rejects_unacquired_or_failed_source_reads(tmp_path, runtime, problem):
    def mutate(calls):
        if problem == "echo":
            if runtime == "codex":
                calls[0][1]["output"][-1]["arguments"] = json.dumps({"cmd": "echo 'Revenue was $100 million in 2023.'"})
            else:
                tool = calls[0][1]["choices"][0]["message"]["tool_calls"][0]["function"]
                tool.update(name="shell", arguments=json.dumps({"command": "echo 'Revenue was $100 million in 2023.'"}))
        for request, _response in calls:
            for output in outputs(request, runtime):
                key, content = ("call_id", "output") if runtime == "codex" else ("tool_call_id", "content")
                if output[key] != "read-0":
                    continue
                if problem == "partial": output[content] = "Partial read without the required quote."
                elif problem == "error": output[content] = json.dumps({"is_error": True, "content": "Revenue was $100 million in 2023."})
                elif problem == "missing_result_id": output[key] = "uncaptured-tool"
    case, root, _calls = record(tmp_path, runtime, mutate)
    audited = audit_native_capture(case, root, runtime)
    assert not audited["audit"]["verdict"]["accepted"]
    assert not audited["audit"]["verdict"]["evidence_replayable"]


@pytest.mark.parametrize("runtime", ["codex", "strands"])
@pytest.mark.parametrize("problem", ["undeclared", "missing_use_id", "mcp", "unused_mcp"])
def test_native_audit_rejects_undeclared_ambiguous_and_mcp_calls(tmp_path, runtime, problem):
    def mutate(calls):
        if problem == "unused_mcp":
            calls[0][0]["tools"].append({"type": "mcp", "server_label": "foreign"})
            return
        if runtime == "codex":
            use = calls[0][1]["output"][-1]
            if problem == "missing_use_id": use["call_id"] = None
            else: use["name"] = "mcp__finance__search_corpus" if problem == "mcp" else "unknown-tool"
        else:
            use = calls[0][1]["choices"][0]["message"]["tool_calls"][0]
            if problem == "missing_use_id": use["id"] = None
            else: use["function"]["name"] = "mcp__finance__search_corpus" if problem == "mcp" else "unknown-tool"
    case, root, _calls = record(tmp_path, runtime, mutate)
    with pytest.raises(TrajectoryError):
        audit_native_capture(case, root, runtime)


@pytest.mark.parametrize("runtime", ["codex", "strands"])
def test_native_results_after_final_cannot_backfill_evidence(tmp_path, runtime):
    case, root, calls = record(tmp_path, runtime)
    late = copy.deepcopy(calls[-1])
    if runtime == "codex":
        late[1]["id"] = "late-response"
        late[1]["output"] = [{"type": "message", "id": "late-message", "role": "assistant", "content": [{"type": "output_text", "text": "Late helper reply"}]}]
    else:
        late[1]["id"] = "late-response"
        late[1]["choices"][0]["message"]["content"] = "Late helper reply"
    for request, _response in calls:
        field = "input" if runtime == "codex" else "messages"
        request[field] = [item for item in request[field]
                          if item.get("call_id" if runtime == "codex" else "tool_call_id") != "read-0"]
    calls.append(late)
    write_capture(root, calls, runtime)
    audited = audit_native_capture(case, root, runtime)
    assert not audited["audit"]["verdict"]["accepted"]
    assert audited["audit"]["verdict"]["missing_chunk_ids"] == ["c1"]


@pytest.mark.parametrize("runtime", ["codex", "strands"])
def test_native_final_binding_and_answer_are_independently_checked(tmp_path, runtime):
    case, root, calls = record(tmp_path, runtime)
    incorrect = proof(); incorrect["answer"]["value"] = "999"
    text = json.dumps(incorrect)
    if runtime == "codex": calls[-1][1]["output"][-1]["content"][0]["text"] = text
    else: calls[-1][1]["choices"][0]["message"]["content"] = text
    write_capture(root, calls, runtime)
    write_final(root, text, runtime)
    audited = audit_native_capture(case, root, runtime)
    assert not audited["audit"]["verdict"]["accepted"]
    assert not audited["audit"]["verdict"]["answer_correct"]
    write_final(root, "Unbound CLI or SDK final", runtime)
    with pytest.raises(TrajectoryError, match="not uniquely bound"):
        audit_native_capture(case, root, runtime)


def test_responses_stream_keeps_custom_patch_and_native_reasoning_payload(tmp_path):
    case, root, calls = record(tmp_path, "codex")
    expected = copy.deepcopy(calls[2][1])
    stream = "".join("data: " + json.dumps(event) + "\n\n" for event in [
        {"type": "response.output_item.done", "output_index": index, "item": item}
        for index, item in enumerate(expected["output"])
    ])
    stream += "data: " + json.dumps({"type": "response.completed", "response": expected}) + "\n\n"
    calls[2] = ({**calls[2][0], "stream": True}, stream)
    write_capture(root, calls, "codex")
    audited = audit_native_capture(case, root, "codex")
    bundle = export_native_capture(audited, root, "codex")
    rows = [json.loads(line) for line in Path(bundle["sample_file"]).read_text().splitlines()]
    assert rows[2]["output"] == expected["output"]
    assert rows[2]["output"][-1]["input"].startswith("*** Begin Patch\n")
    assert rows[2]["output"][0]["encrypted_content"] == "opaque-original"


def test_native_shell_call_must_be_declared_by_its_request():
    from mint_synth.agents.native_capture import _uses
    call = {"protocol": "responses", "request": {"tools": []}}
    response = {"status": "completed", "output": [{"type": "shell_call", "call_id": "shell-1", "action": {"commands": ["cat /tmp/source.txt"]}}]}
    with pytest.raises(TrajectoryError, match="undeclared|declared|shell"):
        _uses(call, response)


def test_chat_stream_preserves_tool_ids_arguments_and_exposed_reasoning(tmp_path):
    case, root, calls = record(tmp_path, "strands")
    original = copy.deepcopy(calls[0][1])
    message = original["choices"][0]["message"]
    tool = message["tool_calls"][0]
    arguments = tool["function"]["arguments"]
    fragments = [
        {"role": "assistant", "reasoning_content": message["reasoning_content"], "content": message["content"]},
        {"tool_calls": [{"index": 0, "id": tool["id"], "type": "function", "function": {
            "name": tool["function"]["name"], "arguments": arguments[:10]}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[10:]}}]},
    ]
    events = [{"id": original["id"], "model": original["model"], "choices": [
        {"index": 0, "delta": delta, "finish_reason": None}]} for delta in fragments]
    events.append({"id": original["id"], "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                   "usage": original["usage"]})
    streamed = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
    calls[0] = ({**calls[0][0], "stream": True}, streamed)
    write_capture(root, calls, "strands")
    audited = audit_native_capture(case, root, "strands")
    assert audited["audit"]["verdict"]["accepted"]
    bundle = export_native_capture(audited, root, "strands")
    rows = [json.loads(line) for line in Path(bundle["sample_file"]).read_text().splitlines()]
    assert rows[0]["messages"][-1] == message
    assert rows[0]["response_metadata"]["usage"] == original["usage"]


@pytest.mark.parametrize("runtime", ["codex", "strands"])
def test_native_configuration_round_trip_and_finance_defaults(tmp_path, runtime):
    example = Path(__file__).parents[1] / "config/mint-synth.example.yaml"
    configured = yaml.safe_load(example.read_text(encoding="utf-8"))
    configured["llm"]["endpoints"]["inference"]["require_api_key"] = False
    agent = configured["training"]["agent_runtime"]
    agent["default"] = runtime
    agent["codex"] = {"mode": "native", "sandbox": "read-only"}
    agent["strands"] = {"mode": "native", "tools": ["file_read", "file_write", "shell", "python_repl"]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(configured), encoding="utf-8")
    _model, settings = settings_for_role(load_config(path), "teacher")
    assert settings.native_tools and settings.tool_mode == "native"
    assert settings.codex == CodexSettings("native", "read-only")
    assert settings.strands == StrandsSettings("native", ("file_read", "file_write", "shell", "python_repl"))
    agent.pop("codex"); agent.pop("strands")
    path.write_text(yaml.safe_dump(configured), encoding="utf-8")
    _model, settings = settings_for_role(load_config(path), "teacher")
    assert not settings.native_tools and settings.tool_mode == "finance"


@pytest.mark.parametrize("runtime,options", [
    ("codex", {"mode": "typo"}),
    ("codex", {"mode": "native", "sandbox": "danger-full-access"}),
    ("codex", {"mode": "native", "sandbox": "--yolo"}),
    ("strands", {"mode": "typo"}),
    ("strands", {"mode": "native", "tools": []}),
    ("strands", {"mode": "native", "tools": ["file_read", "file_read"]}),
    ("strands", {"mode": "native", "tools": ["unknown_tool"]}),
    ("strands", {"mode": "native", "tools": ["mcp__finance__search_corpus"]}),
])
def test_native_configuration_rejects_unsupported_modes_tools_and_bypass(tmp_path, runtime, options):
    configured = yaml.safe_load((Path(__file__).parents[1] / "config/mint-synth.example.yaml").read_text(encoding="utf-8"))
    configured["training"]["agent_runtime"][runtime] = options
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(configured), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_codex_and_strands_native_stage_publish_separately_and_count_trajectories(tmp_path):
    layout = RunLayout(tmp_path / "run")
    index = layout.shard("retrieval_index", 0) / "snapshot.sqlite3"
    index.parent.mkdir(parents=True)
    snapshot(index)
    prepared = layout.shard("sft_prepare", 0)
    prepared.mkdir(parents=True)
    tasks = [task(), task()]
    tasks[0]["task_id"] = "long_codex"
    tasks[1]["task_id"] = "long_strands"
    config = {"version": 3, "run": {"cutoff": "2025-12-31"}, "input": {"available_at_field": "published_at"},
        "quality": {"answer_tolerance": "1e-6"}, "reader": {"batch_rows": 16}, "dag": {"entity_buckets": 1},
        "query_export": {"enabled": False}, "output": {"export_formats": ["jsonl"], "artifact_checksums": True},
        "llm": {"concurrency_per_worker": 1,
                "roles": {name: {"model": "mock-" + name, "endpoint": "local"} for name in ("codex_teacher", "strands_teacher")},
                "endpoints": {"local": {"base_url": "http://127.0.0.1:9876/v1", "api_key_env": "UNUSED", "require_api_key": False}}},
        "training": {"enabled": True, "atomic_enabled": False, "long_enabled": True,
            "long_teacher_roles": ["codex_teacher", "strands_teacher"], "teacher_role": "codex_teacher", "rollouts_per_task": 2,
            "agent_runtime": {"default": "codex", "by_role": {"strands_teacher": "strands"},
                              "codex": {"mode": "native"}, "strands": {"mode": "native"}},
            "split_validation_fraction": 0, "max_tool_steps": 4, "max_context_chars": 24000,
            "min_atomic_samples": 1, "min_long_samples": 2}}

    def run_mock(case, _model, settings, root):
        assert settings.native_tools and not hasattr(case, "call_tool")
        if case.task["task_id"] == "long_strands" and settings.name == "codex":
            return RuntimeResult("failed", "Mock Codex failed; try the next configured teacher")
        workspace = root / "workspace"
        workspace.mkdir()
        prompt = prepare_workspace(case, workspace, root, runtime=settings.name)
        calls = responses_calls(workspace, prompt) if settings.name == "codex" else chat_calls(workspace, prompt)
        write_capture(root, calls, settings.name)
        write_final(root, final_text(calls, settings.name), settings.name)
        finish_workspace(workspace, root)
        return RuntimeResult("completed")

    with (patch("mint_synth.workflow.stages.sft_data.iter_payloads", return_value=iter(tasks)),
          patch("mint_synth.workflow.stages.sft_data._role_client") as client,
          patch("mint_synth.workflow.stages.sft_data.load_tool_registry", side_effect=AssertionError("no finance registry")),
          patch("mint_synth.agents.collector.FinanceToolBridge", side_effect=AssertionError("no finance bridge")),
          patch("mint_synth.agents.cli_runtime.run_cli", side_effect=run_mock),
          patch("mint_synth.agents.strands_runtime.run_strands", side_effect=run_mock)):
        metrics = stage_sft_prepare(layout, config, {}, 0, prepared)
    client.assert_not_called()
    assert metrics["long_accepted"] == 2
    assert metrics["codex_native_requests"] == 4
    assert metrics["strands_native_requests"] == 3
    assert (prepared / "long.jsonl").read_text() == ""
    atomic_json(prepared / "_SUCCESS.json", {"metrics": metrics})
    plan = {"stages": [{"name": "validation", "shard_count": 1}, {"name": "sft_prepare", "shard_count": 1}]}
    output = tmp_path / "published"
    output.mkdir()
    with patch("mint_synth.workflow.stages.publication._enforce_global_quality", return_value={}):
        stage_export(layout, config, plan, 0, output)
    root = output / "sft"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format"] == "mixed"
    for runtime, count in (("codex", 4), ("strands", 3)):
        rows = [json.loads(line) for line in (root / f"{runtime}-native-train.jsonl").read_text().splitlines()]
        assert len(rows) == count
        assert all(row["sample_format"] == runtime + "_native_v1" and "_split" not in row for row in rows)
        assert manifest["formats"][runtime + "-native"] == runtime + "_native_v1"
        assert manifest["counts"][runtime + "_native_train_trajectories"] == 1
    assert (root / "long-train.jsonl").read_text() == ""
    entries = [json.loads(line) for line in (root / "trajectory-index.jsonl").read_text().splitlines()]
    assert len(entries) == 3
    assert sum(entry["accepted"] for entry in entries) == 2
    for entry in entries:
        assert entry["tool_mode"] == "native"
        if entry["accepted"]:
            assert (root / entry["trace"] / "trajectory.json").is_file()
    config["training"]["min_long_samples"] = 3
    rejected = tmp_path / "too-few-trajectories"
    rejected.mkdir()
    with (patch("mint_synth.workflow.stages.publication._enforce_global_quality", return_value={}),
          pytest.raises(RuntimeError, match="requires 3")):
        stage_export(layout, config, plan, 0, rejected)
