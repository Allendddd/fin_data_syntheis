"""Natural solutions, visible CoT, alternative derivations and content-based gates."""

from __future__ import annotations

import copy
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from mint_synth.core.run_state import RunLayout, atomic_json
from mint_synth.harness import HarnessSession, case_from_task
from mint_synth.harness.solution import CHECKS, pattern_rejection, verify_review
from mint_synth.inference.llm import ClientConfig, OpenAICompatibleClient
from mint_synth.inference.runtime import ConfiguredRoleClient, SqliteResponseCache
from mint_synth.sft.atomic_teacher import sample_atomic, sample_reference
from mint_synth.sft.rollout import long_rollout
from mint_synth.workflow.stages.publication import stage_export
from mint_synth.workflow.stages.sft_data import stage_sft_prepare

from test_teacher_proof import SequenceClient, proof, snapshot, task


SOLUTION = ("The sources report $100 million for 2023 and $60 million for 2024. "
            "Adding them, 100 + 60 = 160 million USD. Combined revenue is $160 million.")


def content_review(candidate=SOLUTION, actions=()):
    value = copy.deepcopy(proof())
    value.update(checks={key: True for key in CHECKS}, answer_quote="$160 million.", reason="grounded complete sum")
    value["facts"][0]["response_quote"] = "$100 million"
    value["facts"][1]["response_quote"] = "$60 million"
    value["steps"][0].update(op="sum", response_quote="100 + 60 = 160 million USD.")
    if "100 + 60 = 160" not in candidate:
        # Legacy capture fixtures remain useful for protocol preservation tests.
        value["steps"][0]["response_quote"] = '"result": "160"'
        value["answer_quote"] = '"value": "160"'
    value["actions"] = [{"id": action["id"], "relevant": not action.get("failed"),
                          "reason": "supports the financial task"} for action in actions]
    return value


class Reviewer:
    def __init__(self, change=None):
        self.calls = []
        self.change = change

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        request = json.loads(kwargs["user"])
        value = content_review(request["candidate"], request.get("actions", []))
        if self.change:
            self.change(value)
        return value

    def stats(self):
        return {"requests": len(self.calls)}


class NaturalTeacher:
    def __init__(self, texts=(SOLUTION,), *, reasoning="", finish="stop", barrier=None):
        self.texts = iter(texts)
        self.calls = []
        self.reasoning, self.finish, self.barrier = reasoning, finish, barrier

    def complete_text_exchange(self, **kwargs):
        self.calls.append(kwargs)
        if self.barrier:
            self.barrier.wait(timeout=5)
        content = next(self.texts)
        parts = ([{"field": "reasoning_content", "text": self.reasoning}] if self.reasoning else [])
        parts.append({"field": "content", "text": content})
        return {"text": "\n\n".join(part["text"] for part in parts), "text_parts": parts,
                "finish_reason": self.finish, "request": {"messages": [kwargs["system"], kwargs["user"]]},
                "response": {"choices": [{"message": {"content": content, "reasoning_content": self.reasoning}}]}}


def test_teacher_natural_text_and_visible_cot_are_preserved_without_hidden_reference():
    teacher, reviewer = NaturalTeacher(reasoning="I will add the reported revenues."), Reviewer()
    sample, audit, reason = sample_atomic(task(), {"teacher": teacher}, attempts=1, policy="judge",
        fallback=False, tolerance="1e-6", verifier=reviewer, reasoning_format="think")
    assert reason == "accepted"
    assert sample["messages"][-1]["content"] == "<think>\nI will add the reported revenues.\n</think>\n\n" + SOLUTION
    assert [message["loss"] for message in sample["messages"]] == [False, False, True]
    assert "JSON" not in teacher.calls[0]["system"]
    for hidden in ("evidence_graph", "fact_id", "reasoning_trace", '"value": "160"'):
        assert hidden not in teacher.calls[0]["user"]
    assert audit["candidates"][0]["review"]["steps"][0]["op"] == "sum"


def test_judge_prefers_longer_complete_candidate_and_concise_is_an_alias():
    longer = SOLUTION + " Both amounts use the same currency and scale."
    judge = SequenceClient([{"candidate_id": "candidate-2", "reason": "Clear period and dimensional explanation."}])
    sample, audit, _ = sample_atomic(task(), {"teacher": NaturalTeacher([SOLUTION, longer])}, attempts=2,
        policy="concise", fallback=False, tolerance="1e-6", verifier=Reviewer(), judge=judge)
    assert sample["messages"][-1]["content"] == longer
    assert audit["selection"]["policy"] == "judge" and "migration" in audit["selection"]
    keys = [row["sha256"] for row in audit["candidates"]]
    assert len(set(keys)) == 2


@pytest.mark.parametrize("decision", [{"candidate_id": "unknown", "reason": "invalid"}, {}, None])
def test_bad_judge_cannot_fall_back_to_first_or_shortest(decision):
    sample, audit, reason = sample_atomic(task(), {"teacher": NaturalTeacher([SOLUTION, SOLUTION + " Checked."])},
        attempts=2, policy="judge", fallback=True, tolerance="1e-6", verifier=Reviewer(), judge=SequenceClient([decision]))
    assert sample is None and audit["selected"] is None and "judge failed" in reason


def test_sampling_keys_are_distinct_without_changing_visible_prompts():
    teacher = NaturalTeacher([SOLUTION, SOLUTION])
    sample_atomic(task(), {"teacher": teacher}, attempts=2, policy="first_valid", fallback=False,
                  tolerance="1e-6", verifier=Reviewer())
    assert teacher.calls[0]["user"] == teacher.calls[1]["user"]
    assert teacher.calls[0]["sample_key"] != teacher.calls[1]["sample_key"]


@pytest.mark.parametrize("field", ["atomic_verifier_role", "atomic_judge_role", "long_verifier_role"])
def test_missing_private_role_is_rejected_before_sampling(tmp_path, field):
    import yaml
    from mint_synth.core.config_v2 import ConfigError, load_config
    config = yaml.safe_load(Path("config/mint-synth.example.yaml").read_text(encoding="utf-8"))
    config["training"][field] = "missing-reviewer"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(path)


def test_missing_or_unverified_reference_is_not_fabricated_from_gold():
    source = task()
    source["reasoning_trace"] = []
    sample, audit = sample_reference(source, Reviewer(), tolerance="1e-6")
    assert sample is None and not audit["accepted"] and "missing" in audit["reason"]


@pytest.mark.parametrize("change", [
    lambda r: r["answer"].update(value="999"),
    lambda r: r["answer"].update(unit="EUR"),
    lambda r: r["steps"][0].update(result="159"),
    lambda r: r["steps"][0].update(response_quote="a computation the teacher never wrote"),
    lambda r: r["facts"][0].update(quote="an invented source"),
    lambda r: r["facts"][0].update(value="200"),
    lambda r: r.update(steps=[]),
])
def test_private_review_cannot_launder_wrong_or_missing_material(change):
    review = content_review()
    change(review)
    assert not verify_review(task(), SOLUTION, review, tolerance="1e-6")[0]


def test_final_number_alone_cannot_supply_a_calculation_derivation():
    assert not verify_review(task(), "Combined revenue is $160 million.", content_review(), tolerance="1e-6")[0]
    assert not verify_review(task(), SOLUTION.replace("160", "999"), content_review(), tolerance="1e-6")[0]


def test_candidate_can_use_a_different_graph_and_equivalent_scale():
    candidate = SOLUTION.replace("100 + 60 = 160 million USD", "(100 + 60) / 2 = 80; 80 * 2 = 160 million USD")
    review = content_review()
    dims = {"unit": "USD", "scale": "million", "currency": "USD"}
    review["constants"] = [{"id": "two", "value": "2", "response_quote": "80 * 2 = 160", "unit": None, "scale": None, "currency": None}]
    review["steps"] = [
        {"id": "mean", "op": "average", "inputs": ["a", "b"], "params": {}, "result": "80",
         "response_quote": "(100 + 60) / 2 = 80", **dims},
        {"id": "total", "op": "multiply", "inputs": ["mean", "two"], "params": {}, "result": "160",
         "response_quote": "80 * 2 = 160 million USD", **dims}]
    assert verify_review(task(), candidate, review, tolerance="1e-6")[0]
    candidate = SOLUTION.replace("Combined revenue is $160 million.", "Combined revenue is $0.16 billion.")
    review = content_review()
    review["answer_quote"] = "$0.16 billion."
    assert verify_review(task(), candidate, review, tolerance="1e-6")[0]


def test_textual_knowledge_does_not_require_arithmetic_or_reference_wording():
    source = task()
    source.update(answer={"value": "earnings before interest, taxes, depreciation and amortization",
                          "unit": None, "scale": None, "currency": None}, evidence_graph={"nodes": []})
    source["input"]["context"] = "EBITDA means earnings before interest, taxes, depreciation and amortization."
    response = "It means earnings before interest, taxes, depreciation and amortization."
    review = {"checks": {name: True for name in CHECKS}, "answer": source["answer"], "answer_quote": response,
              "facts": [{"id": "definition", "quote": source["input"]["context"], "response_quote": response,
                         **source["answer"]}], "steps": [], "constants": [], "terminal": None}
    assert verify_review(source, response, review, tolerance="1e-6")[0]


def test_pattern_filters_are_content_based_and_do_not_require_an_answer_marker():
    assert pattern_rejection(SOLUTION, task()["question"], "") is None
    assert "finish" in pattern_rejection(SOLUTION, "question", "", "length")
    assert "repeats" in pattern_rejection(("The values in both annual reports use the same financial scale.\n") * 4, "q", "")
    assert "prompt" in pattern_rejection((task()["question"] + "\n") * 8, task()["question"], "")


def test_explicit_wrong_currency_cannot_be_normalized_away_by_the_reviewer():
    response = SOLUTION.replace("$160 million.", "160 million EUR.")
    review = content_review()
    review["answer_quote"] = "160 million EUR."
    assert not verify_review(task(), response, review, tolerance="1e-6")[0]


def test_text_client_and_cache_preserve_actual_reasoning_without_json_mode(tmp_path):
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        message = {"role": "assistant", "content": '{"ok":true}'} if "response_format" in payload else {
            "role": "assistant", "reasoning_content": "Actual exposed reasoning.", "content": SOLUTION}
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})
    def build():
        delegate = OpenAICompatibleClient(ClientConfig("test", strict_json=True, retries=0))
        delegate.close()
        delegate._http_client = httpx.Client(transport=httpx.MockTransport(respond))
        return ConfiguredRoleClient(delegate, temperature=0.2, max_tokens=None,
            cache=SqliteResponseCache(tmp_path / "cache.sqlite3", policy="immutable", busy_timeout_seconds=1),
            cache_namespace={"model": "test", "revision": "v1"})
    client = build()
    assert client.complete_json(system="s", user="u", sample_key="1") == {"ok": True}
    first = client.complete_text_exchange(system="s", user="u", sample_key="1", max_tokens=1000)
    assert first["text"] == "Actual exposed reasoning.\n\n" + SOLUTION
    assert "response_format" not in requests[-1]
    client.close()
    client = build()
    assert client.complete_text_exchange(system="s", user="u", sample_key="1", max_tokens=1000)["cache_hit"]
    client.complete_text_exchange(system="s", user="u", sample_key="2", max_tokens=1000)
    client.complete_text_exchange(system="s", user="u", sample_key="1", max_tokens=2000)
    client.close()
    assert len(requests) == 4


@pytest.mark.parametrize("runtime", ["strands", "codex", "claude_code"])
def test_finance_runtimes_preserve_real_visible_cot_and_submit_natural_answer(tmp_path, runtime):
    from test_agent_collection import CONFIG, Upstream
    from mint_synth.agents.base import ModelSettings, RuntimeResult, RuntimeSettings
    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    index = tmp_path / "snapshot.sqlite3"
    snapshot(index)
    def run(bridge, model, settings, root):
        endpoint = {"strands": "/chat/completions", "codex": "/responses", "claude_code": "/messages"}[runtime]
        with httpx.Client(trust_env=False) as client:
            for name, args in [("missing_tool", {}), ("search_corpus", {"query": "Revenue"}),
                               ("open_chunk", {"chunk_id": "c1"}), ("open_chunk", {"chunk_id": "c2"}),
                               ("submit_final", {"answer": SOLUTION})]:
                client.post(model.base_url + endpoint, json={"mock_tool": name, "mock_arguments": args}).raise_for_status()
                bridge.call_tool(name, args)
        return RuntimeResult("completed")
    function = "mint_synth.agents.strands_runtime.run_strands" if runtime == "strands" else "mint_synth.agents.cli_runtime.run_cli"
    try:
        with patch(function, side_effect=run):
            sample, reason = long_rollout(task(), None, index, CONFIG,
                model=ModelSettings("mock", f"http://127.0.0.1:{server.server_port}/v1", None),
                runtime=RuntimeSettings(runtime), trace_dir=tmp_path / "trace", verifier=Reviewer())
        assert reason == "accepted"
        assert sample["messages"][-1]["content"] == SOLUTION
        assert sample["messages"][2]["loss"] is False
        assert any("Mock teacher plan" in row["content"] and row["loss"] for row in sample["messages"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_parallel_preparation_remains_bounded_and_ordered(tmp_path):
    barrier = threading.Barrier(2)
    tasks = [task(), task()]
    tasks[1]["task_id"] = "second"
    for source in tasks:
        source["task_type"] = "atomic"
    config = {"reader": {"batch_rows": 16}, "llm": {"concurrency_per_worker": 2},
              "training": {"enabled": True, "atomic_enabled": True, "long_enabled": False,
                           "atomic_teacher_roles": ["teacher"], "atomic_samples_per_task": 1,
                           "atomic_include_reference": False, "split_validation_fraction": 0},
              "quality": {"answer_tolerance": "1e-6"}}
    # A new per-call iterator is unnecessary because each task samples once.
    teacher = NaturalTeacher([SOLUTION, SOLUTION], barrier=barrier)
    def role(_layout, _config, _stage, _shard, name):
        return teacher if name == "teacher" else Reviewer()
    layout = RunLayout(tmp_path / "run")
    output = tmp_path / "prepared"
    output.mkdir()
    with patch("mint_synth.workflow.stages.sft_data.iter_payloads", return_value=iter(tasks)), \
         patch("mint_synth.workflow.stages.sft_data._role_client", side_effect=role):
        metrics = stage_sft_prepare(layout, config, {}, 0, output)
    assert metrics["atomic_accepted"] == 2
    rows = [json.loads(line) for line in (output / "sidecar.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["task_id"] for row in rows] == [source["task_id"] for source in tasks]


def test_natural_long_rollout_can_recover_and_masks_related_cot(tmp_path):
    path = tmp_path / "snapshot.sqlite3"
    snapshot(path)
    actions = [{"tool": "missing_tool", "arguments": {}},
               {"tool": "search_corpus", "arguments": {"query": "Revenue"}},
               {"tool": "open_chunk", "arguments": {"chunk_id": "c1"}},
               {"tool": "open_chunk", "arguments": {"chunk_id": "c2"}}, {"final": SOLUTION}]
    client = SequenceClient(actions)
    original = client.complete_json_exchange
    def complete(**kwargs):
        result = original(**kwargs)
        result["assistant_parts"].insert(0, {"kind": "reasoning", "text": "Actual planning for this action.", "field": "reasoning_content"})
        return result
    client.complete_json_exchange = complete
    config = {"training": {"max_tool_steps": 6, "max_context_chars": 24000}, "run": {"cutoff": "2025-12-31"},
              "input": {"available_at_field": "published_at"}, "quality": {"answer_tolerance": "1e-6"}}
    sample, reason = long_rollout(task(), client, path, config, verifier=Reviewer(), trace_dir=tmp_path / "trace")
    assert reason == "accepted"
    assert sample["messages"][2]["content"] == "Actual planning for this action."
    assert sample["messages"][2]["loss"] is False and sample["messages"][3]["loss"] is False
    assert sample["messages"][-1]["content"] == SOLUTION
    assert all(not row["loss"] for row in sample["messages"] if row["role"] == "tool_response")


def test_reference_is_independent_audited_source_and_stage_exports_both(tmp_path):
    source = task()
    source.update(task_type="atomic", verification={"accepted": True}, reasoning_trace=[{"statement": SOLUTION}])
    reference, record = sample_reference(source, Reviewer(), tolerance="1e-6")
    assert reference is not None and record["source"] == "audited_reference"
    layout = RunLayout(tmp_path / "run")
    prepared = layout.shard("sft_prepare", 0)
    prepared.mkdir(parents=True)
    config = {"version": 3, "reader": {"batch_rows": 16}, "training": {"enabled": True, "atomic_enabled": True,
        "long_enabled": False, "atomic_teacher_roles": ["teacher"], "atomic_samples_per_task": 1,
        "atomic_include_reference": True, "atomic_reference_fraction": 1, "split_validation_fraction": 0,
        "min_atomic_samples": 1, "min_long_samples": 1}, "quality": {"answer_tolerance": "1e-6"},
        "query_export": {"enabled": False}, "output": {"export_formats": ["jsonl"], "artifact_checksums": False}}
    def role(_layout, _config, _stage, _shard, name):
        return NaturalTeacher() if name == "teacher" else Reviewer()
    with patch("mint_synth.workflow.stages.sft_data.iter_payloads", return_value=iter([source])), \
         patch("mint_synth.workflow.stages.sft_data._role_client", side_effect=role):
        metrics = stage_sft_prepare(layout, config, {}, 0, prepared)
    assert metrics["atomic_train"] == metrics["atomic_accepted"] == 2
    atomic_json(prepared / "_SUCCESS.json", {"metrics": metrics})
    plan = {"stages": [{"name": "validation", "shard_count": 1}, {"name": "sft_prepare", "shard_count": 1}]}
    output = tmp_path / "export"
    output.mkdir()
    with patch("mint_synth.workflow.stages.publication._enforce_global_quality", return_value={}):
        stage_export(layout, config, plan, 0, output)
    rows = [json.loads(line) for line in (output / "sft/atomic-train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({row["sample_id"] for row in rows}) == 2
    assert rows[0]["messages"][-1]["content"] == SOLUTION
    assert all("evidence_graph" not in json.dumps(row) and "JSON" not in row["messages"][0]["content"] for row in rows)
    audit = [json.loads(line) for line in (output / "sft/audit-sidecar.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(audit) == 2 and all(row["task_id"] == source["task_id"] and row["split"] == "train" for row in audit)


@pytest.mark.parametrize("runtime", ["claude_code", "codex", "strands"])
def test_native_protocols_accept_natural_final_and_preserve_cot_and_masks(tmp_path, runtime):
    if runtime == "claude_code":
        from test_cc_native import native_record, write_calls, write_final_event
        from mint_synth.agents.cc_workspace import audit_native
        from mint_synth.agents.cc_native import export_cc_native
        index = tmp_path / "snapshot.sqlite3"
        snapshot(index)
        root = tmp_path / "trace"
        _, calls = native_record(index, root)
        calls[-1][1]["content"][-1]["text"] = SOLUTION
        write_calls(root, calls)
        write_final_event(root, calls[-1][1])
        case = case_from_task(task(), index, cutoff="2025-12-31")
        audited = audit_native(case, root, reviewer=Reviewer())
        bundle = export_cc_native(audited, root)
    else:
        from test_native_capture import record, write_capture, write_final
        from mint_synth.agents.native_capture import audit_native_capture, export_native_capture
        case, root, calls = record(tmp_path, runtime)
        if runtime == "codex":
            calls[-1][1]["output"][-1]["content"][0]["text"] = SOLUTION
        else:
            calls[-1][1]["choices"][0]["message"]["content"] = SOLUTION
        write_capture(root, calls, runtime)
        write_final(root, SOLUTION, runtime)
        audited = audit_native_capture(case, root, runtime, reviewer=Reviewer())
        bundle = export_native_capture(audited, root, runtime)
    assert audited["audit"]["verdict"]["accepted"]
    rows = [json.loads(line) for line in Path(bundle["sample_file"]).read_text(encoding="utf-8").splitlines()]
    if runtime == "claude_code":
        assert rows[-1]["messages"][-1]["content"] == calls[-1][1]["content"]
        assert rows[-1]["target_block_loss_mask"] == [True, True]
    elif runtime == "codex":
        assert rows[-1]["output"] == calls[-1][1]["output"]
    else:
        assert rows[-1]["messages"][-1] == calls[-1][1]["choices"][0]["message"]
    assert all("JSON proof" not in json.dumps(request) for request, _ in calls[:1])


@pytest.mark.parametrize("command", ["cat corpus/c1.txt", "sed -n '1,20p' corpus/c1.txt",
    "head -n 20 corpus/c1.txt", "cd corpus && cat c1.txt", "rg -n Revenue corpus",
    "python3 -c \"from pathlib import Path; print(Path('corpus/c1.txt').read_text())\""])
def test_common_native_read_forms_bind_actual_returned_evidence(tmp_path, command):
    from test_native_capture import record, write_capture
    from mint_synth.agents.native_capture import audit_native_capture
    case, root, calls = record(tmp_path, "codex")
    calls[0][1]["output"][-1]["arguments"] = json.dumps({"cmd": command, "workdir": str(root / "workspace")})
    write_capture(root, calls, "codex")
    audited = audit_native_capture(case, root, "codex")
    assert audited["audit"]["verdict"]["accepted"]
    assert any(row["tool_use_id"] == "read-0" and row["chunk_id"] == "c1" for row in audited["read_bindings"])


@pytest.mark.parametrize("code", [
    "from pathlib import Path; print('Revenue was $100 million in 2023.'); Path('corpus/c1.txt')",
    "print('Revenue was $100 million in 2023.'); open('corpus/c1.txt')",
    "from other_module import Path; print(Path('corpus/c1.txt').read_text())",
])
def test_a_path_mention_and_printed_quote_do_not_prove_a_python_read(tmp_path, code):
    import shlex
    from test_native_capture import record, write_capture
    from mint_synth.agents.native_capture import audit_native_capture
    case, root, calls = record(tmp_path, "codex")
    calls[0][1]["output"][-1]["arguments"] = json.dumps({"cmd": "python3 -c " + shlex.quote(code)})
    write_capture(root, calls, "codex")
    audited = audit_native_capture(case, root, "codex")
    assert not audited["audit"]["verdict"]["accepted"]


def test_native_loss_gate_masks_unrelated_work_and_its_visible_reasoning(tmp_path):
    from test_native_capture import record
    from mint_synth.agents.native_capture import audit_native_capture, export_native_capture
    case, root, _ = record(tmp_path, "codex")
    def mark_patch(review):
        for action in review["actions"]:
            if action["id"] == "patch-0":
                action.update(relevant=False, reason="Unrelated scratch file does not support the answer")
    audited = audit_native_capture(case, root, "codex", reviewer=Reviewer(mark_patch))
    bundle = export_native_capture(audited, root, "codex")
    rows = [json.loads(line) for line in Path(bundle["sample_file"]).read_text(encoding="utf-8").splitlines()]
    assert rows[2]["target_output_loss_mask"] == [False, False]
    assert rows[0]["target_output_loss_mask"] == [True, True]
    assert rows[0]["target_reasoning_text_fields"]["0"]["encrypted_content"] is False
    assert rows[2]["output"][0]["encrypted_content"] == "opaque-original"


@pytest.mark.parametrize("late_error", [False, True])
def test_sdk_local_failure_uses_authenticated_observation_before_final(tmp_path, late_error):
    from test_native_capture import write_capture
    from mint_synth.agents.model_trace import load_model_turns
    from mint_synth.agents.trajectory import TrajectoryError, normalized_sample
    error = '{"error":"Unknown tool missing_tool"}'
    messages = [{"role": "user", "content": "Solve the task"}]
    unknown = {"role": "assistant", "reasoning_content": "Actual thought before a failed action.", "tool_calls": [
        {"id": "failed-call", "type": "function", "function": {"name": "missing_tool", "arguments": "{}"}}]}
    final = {"role": "assistant", "tool_calls": [
        {"id": "final-call", "type": "function", "function": {"name": "submit_final", "arguments": json.dumps({"answer": SOLUTION})}}]}
    error_message = {"role": "tool", "tool_call_id": "failed-call", "content": error}
    contexts = [messages, messages + [unknown] + ([] if late_error else [error_message]),
                messages + [unknown, error_message, final]]
    replies = [unknown, final, {"role": "assistant", "content": ""}]
    calls = [({"messages": context, "tools": []}, {"choices": [{"index": 0, "message": reply,
              "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop"}]})
             for context, reply in zip(contexts, replies)]
    write_capture(tmp_path, calls, "strands")
    turns = load_model_turns(tmp_path, set())
    snapshot = {"trace": [{"step": 1, "action": {"final": SOLUTION}, "tool": "", "arguments": {}, "observation": None}],
                "audit": {"verdict": {"step_relevance": []}}, "tools": []}
    if late_error:
        assert "failed_observation" not in turns[0]["assistant_parts"][-1]
        with pytest.raises(TrajectoryError):
            normalized_sample(snapshot, "question", cutoff="2025-12-31", model_turns=turns)
    else:
        sample = normalized_sample(snapshot, "question", cutoff="2025-12-31", model_turns=turns)
        assert sample["messages"][2]["loss"] is False
        assert sample["messages"][3]["loss"] is False
        assert sample["messages"][4] == {"role": "tool_response", "content": error, "loss": False}
        assert sample["messages"][-1]["content"] == SOLUTION
