"""Original documents, live HTTP sources, evidence gates and query-only seeds."""

import copy
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from mint_synth.core.run_state import RunLayout
from mint_synth.harness import HarnessSession, case_from_task
from mint_synth.harness.registry import load_tool_registry
from mint_synth.harness.resources import ResourceTools, resource_fingerprint
from mint_synth.harness.solution import assess_solution
from mint_synth.harness.solution import verify_review
from mint_synth.synthesis.long_horizon import load_seed_bank
from mint_synth.workflow.stages.seeds import stage_seed_prepare, prepared_seeds
from test_natural_sft import SOLUTION, content_review
from test_teacher_proof import task


def write_pdf(path, text):
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    content = DecodedStreamObject()
    content.set_data(("BT /F1 12 Tf 30 700 Td (" + text + ") Tj ET").encode())
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(path)


def corpus(tmp_path, mapping=True):
    root = tmp_path / "originals"
    root.mkdir()
    write_pdf(root / "annual.pdf", "ACME 2023 Revenue was $100 million in 2023.")
    with zipfile.ZipFile(root / "quarter.docx", "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                         '<w:p><w:r><w:t>ACME 2024 Revenue was $60 million in 2024.</w:t></w:r></w:p>'
                         '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Unrelated table row</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
                         '</w:body></w:document>')
    settings = {"mode": "static", "corpus_path": str(root)}
    if mapping:
        manifest = tmp_path / "originals-manifest.jsonl"
        manifest.write_text("\n".join(json.dumps({"file": name, "source_ids": [source], "published_at": "2025-01-01"})
                                      for name, source in [("annual.pdf", "s1"), ("quarter.docx", "s2")]), encoding="utf-8")
        settings["manifest_path"] = str(manifest)
    return settings


class SourceReviewer:
    def complete_json(self, **kwargs):
        request = json.loads(kwargs["user"])
        review = content_review(request["candidate"], request.get("actions", []))
        review["checks"]["reference_consistent"] = True
        for fact in review["facts"]:
            fact["chunk_id"] = next(key for key, text in request["context"].items() if fact["quote"] in text)
        return review


def test_original_pdf_docx_and_tables_remain_accessible(tmp_path):
    settings = corpus(tmp_path)
    tools = ResourceTools(settings, "2025-12-31", entity_id="entity")
    try:
        hits = tools.call("search_documents", {"query": "Revenue"})["hits"]
        assert len(hits) == 2
        assert {hit["locator"]["kind"] for hit in hits} == {"pdf_page", "docx_block"}
        docx = next(key for key, doc in tools.documents.items() if doc["file"].endswith("docx"))
        table = tools.call("read_document", {"document_id": docx, "unit": 2})
        assert table["locator"]["table"] is True
        assert table["text"] == "Unrelated table row"
        assert "error" in tools.call("search_web", {"query": "Revenue"})
    finally:
        tools.close()


def test_resource_harness_uses_originals_without_any_snapshot(tmp_path):
    settings = corpus(tmp_path)
    case = case_from_task(task(), None, cutoff="2025-12-31", retrieval=settings, archive_path=tmp_path / "sources")
    with HarnessSession(case, solution_reviewer=SourceReviewer()) as session:
        state = session.observe()
        assert "answer" not in state and "evidence_graph" not in state
        assert {x["function"]["name"] for x in state["tools"]} == {"search_documents", "read_document", "calculate"}
        result = session.step({"tool": "search_documents", "arguments": {"query": "Revenue"}})
        assert len(result["observation"]["hits"]) == 2
        session.step({"final": SOLUTION})
        assert session.grade()["accepted"], session.grade()
        assert len(session.audit_record()["snapshot_sha256"]) == 64
    assert len(list((tmp_path / "sources").glob("evidence_*.json"))) == 2


def test_different_corpus_needs_task_evidence_correspondence_not_global_manifest(tmp_path):
    settings = corpus(tmp_path, mapping=False)
    tools = ResourceTools(settings, "2025-12-31", entity_id="entity")
    try:
        hits = tools.call("search_documents", {"query": "Revenue"})["hits"]
        original = task()
        for number, fact in enumerate(original["evidence"], 1):
            fact.update(entity="ACME", period=str(2022 + number), metric="revenue")
        class CorrespondenceReviewer(SourceReviewer):
            def complete_json(self, **kwargs):
                review = super().complete_json(**kwargs)
                contexts = json.loads(kwargs["user"])["context"]
                for n, fact in enumerate(review["facts"], 1):
                    fact["reference_bindings"] = [{"fact_id": f"f{n}", "entity": "ACME", "period": str(2022 + n),
                        "metric": "revenue", "scope_quotes": [contexts[fact["chunk_id"]].strip()]}]
                return review
        result = assess_solution(original, SOLUTION, CorrespondenceReviewer(),
                                 acquired={h["chunk_id"]: h["text"] for h in hits}, source_records=tools.sources)
        assert result["accepted"], result
        result = assess_solution(original, SOLUTION, SourceReviewer(),
                                 acquired={h["chunk_id"]: h["text"] for h in hits}, source_records=tools.sources)
        assert not result["accepted"]
    finally:
        tools.close()


def test_original_mutation_is_detected_and_changes_resource_fingerprint(tmp_path):
    settings = corpus(tmp_path)
    tools = ResourceTools(settings, "2025-12-31", entity_id="entity")
    try:
        before = resource_fingerprint(settings)
        assert before == tools.fingerprint
        path = Path(settings["corpus_path"]) / "quarter.docx"
        path.write_bytes(b"changed")
        assert before != resource_fingerprint(settings)
        doc_id = next(k for k, d in tools.documents.items() if d["file"] == path.name)
        assert "changed" in tools.call("read_document", {"document_id": doc_id})["error"]
    finally:
        tools.close()


def test_live_http_search_and_page_read_never_touch_static_corpus(tmp_path):
    requests = []
    class Server(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            query = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(query)
            data = json.dumps({"results": [{"url": f"http://127.0.0.1:{self.server.server_port}/report", "title": "Annual report",
                "content": "Revenue was $100 million in 2023.", "published_date": "2024-01-01"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            data = b'<html><meta property="article:published_time" content="2024-01-01"><p>Revenue was $100 million in 2023.</p></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = {"mode": "web", "corpus_path": str(tmp_path / "must-not-exist"),
                "web": {"api_key_env": None, "search_endpoint": f"http://127.0.0.1:{server.server_port}/search"}}
    tools = ResourceTools(settings, "2025-12-31", entity_id="entity", dated_only=True, archive=tmp_path / "archive")
    try:
        hits = tools.call("search_web", {"query": "Revenue"})["hits"]
        page = tools.call("read_web", {"url": hits[0]["url"]})
        assert "Revenue was $100 million" in page["text"]
        assert page["available_at"] == "2024-01-01"
        assert not tools.documents
        assert "error" in tools.call("search_documents", {"query": "Revenue"})
        assert requests[0]["include_answer"] is False
        assert list((tmp_path / "archive").glob("*.bin"))
        assert {x["function"]["name"] for x in load_tool_registry(retrieval=settings).definitions} == {"search_web", "read_web", "calculate"}
        case = case_from_task(task(), None, cutoff="2025-12-31", retrieval=settings, archive_path=tmp_path / "session-sources")
        with HarnessSession(case) as session:
            session.step({"tool": "search_web", "arguments": {"query": "Revenue"}})
            checkpoint = session.checkpoint()
            count = len(requests)
            session.restore(checkpoint)
            assert len(requests) == count  # Replaying a web checkpoint must not search the changing web again.
    finally:
        tools.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_query_only_seed_is_prepared_once_and_preserves_chinese_spec(tmp_path):
    path = tmp_path / "seeds.jsonl"
    path.write_text('{"query":"比较两个期间的营业收入变化"}\n', encoding="utf-8")
    seeds = load_seed_bank(str(path))
    assert seeds[0]["task_spec_template"] is None
    config = {"query_realization": {"seed_file": str(path)}, "training": {},
              "dag": {"formula_registry": "builtin", "allowed_operations": ["subtract", "abs"]},
              "quality": {"evaluation_file": None, "overlap_threshold": 0.82}}
    class Parser:
        calls = 0
        def complete_json(self, **kwargs):
            self.calls += 1
            return {"task_spec_template": {"objective_type": "change", "generator": "temporal_pair", "reasoning_form": "比较两期变化",
                    "required_fact_roles": ["新期间指标", "旧期间指标"], "time_sensitive": False, "requested_output": "变化方向及幅度"}}
    parser = Parser()
    layout = RunLayout(tmp_path / "run")
    output = layout.shard("seed_prepare", 0)
    with patch("mint_synth.workflow.stages.seeds._role_client", return_value=parser):
        metrics = stage_seed_prepare(layout, config, {}, 0, output)
    assert metrics["evaluation_check"] == "not_configured"
    assert prepared_seeds(layout, config)[0]["task_spec_template"]["reasoning_form"] == "比较两期变化"
    assert prepared_seeds(layout, config)[0]["seed_id"] == seeds[0]["seed_id"]
    assert parser.calls == 1


def test_harness_and_resource_changes_do_not_invalidate_task_generation_signatures():
    from mint_synth.core.config_v2 import load_config
    from mint_synth.workflow.planner import _task_config_fingerprint, _planned_signatures
    original = load_config(Path(__file__).resolve().parents[1] / "config/mint-synth.example.yaml")
    changed = copy.deepcopy(original)
    changed["training"]["agent_runtime"]["default"] = "strands"
    changed["harness"]["retrieval"] = {"mode": "web"}
    changed["llm"]["roles"]["teacher"]["model"] = "another-teacher"
    first, second = _task_config_fingerprint(original), _task_config_fingerprint(changed)
    assert first == second
    stages = [{"name": "validation", "shard_count": 1, "dependencies": []},
              {"name": "sft_prepare", "shard_count": 1, "dependencies": ["validation"]}]
    a, _ = _planned_signatures("execution-a", stages, [], task_fingerprint=first)
    b, _ = _planned_signatures("execution-b", stages, [], task_fingerprint=second)
    assert a["validation"] == b["validation"]
    assert a["sft_prepare"] != b["sft_prepare"]
    changed["llm"]["roles"]["constructor"]["model"] = "different-task-constructor"
    assert _task_config_fingerprint(changed) != first


def test_atomic_numeric_premises_must_be_supported_by_declared_evidence():
    original = task()
    original["task_type"] = "atomic"
    assert verify_review(original, SOLUTION, content_review(), tolerance="1e-6")[0]
    text = "Decoy revenue was $200 million. Another decoy was $40 million."
    original["input"]["context"] += "\n" + text
    candidate = "The sources report $200 million and $40 million. Subtracting, 200 - 40 = 160 million USD. Combined revenue is $160 million."
    review = content_review()
    for item, value, quote in zip(review["facts"], ["200", "40"], text.split(". ")):
        item.update(value=value, quote=quote, response_quote="$" + value + " million")
    review["steps"][0].update(op="subtract", response_quote="200 - 40 = 160 million USD.")
    accepted, reason, _ = verify_review(original, candidate, review, tolerance="1e-6")
    assert not accepted and "declared evidence" in reason


def test_temporal_source_rejection_does_not_fall_back_to_another_corpus(tmp_path):
    settings = corpus(tmp_path)
    manifest = Path(settings["manifest_path"])
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("2025-01-01", "2026-01-01"), encoding="utf-8")
    tools = ResourceTools(settings, "2025-12-31", entity_id="entity", dated_only=True)
    try:
        assert tools.call("search_documents", {"query": "Revenue"})["hits"] == []
        doc_id = next(iter(tools.documents))
        assert "cutoff" in tools.call("read_document", {"document_id": doc_id})["error"]
        assert not tools.sources
    finally:
        tools.close()


@pytest.mark.parametrize("runtime", ["codex", "claude_code", "strands"])
@pytest.mark.parametrize("returned", [True, False])
def test_native_resource_protocols_require_actual_returned_evidence(tmp_path, runtime, returned):
    from mint_synth.agents.research_bridge import ResearchBridge
    from mint_synth.agents.cc_workspace import prepare_workspace, finish_workspace
    from mint_synth.agents.native_audit import audit_trace
    from mint_synth.agents.native_capture import export_native_capture
    from mint_synth.agents.cc_native import export_cc_native
    from test_native_capture import write_capture, write_final
    from test_cc_native import write_calls, write_final_event
    settings = corpus(tmp_path)
    trace = tmp_path / "trace"
    workspace = trace / "workspace"
    workspace.mkdir(parents=True)
    case = case_from_task(task(), None, cutoff="2025-12-31", retrieval=settings, archive_path=trace / "sources")
    name = "search_documents" if runtime == "strands" else "mcp__finance__search_documents"
    arguments = {"query": "Revenue"}
    with ResearchBridge(case) as bridge:
        prompt = prepare_workspace(bridge, workspace, trace, runtime=runtime)
        observation = bridge.call_tool("search_documents", arguments)
        encoded = json.dumps(observation)
    finish_workspace(workspace, trace)
    if runtime == "codex":
        tool = {"type": "function", "name": name, "parameters": {"type": "object"}}
        action = {"type": "function_call", "name": name, "call_id": "read-1", "arguments": json.dumps(arguments)}
        context = [{"role": "user", "content": prompt}]
        first = {"id": "model-1", "status": "completed", "output": [action]}
        second = {"id": "model-2", "status": "completed", "output": [{"type": "message", "id": "final-model-message",
            "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": SOLUTION}]}]}
        calls = [({"model": "mock", "tools": [tool], "input": copy.deepcopy(context)}, first)]
        context += [action]
        if returned:
            context.append({"type": "function_call_output", "call_id": "read-1", "output": encoded})
        calls.append(({"model": "mock", "tools": [tool], "input": context}, second))
        write_capture(trace, calls, runtime)
        write_final(trace, SOLUTION, runtime)
    elif runtime == "claude_code":
        tool = {"name": name, "description": "Research source", "input_schema": {"type": "object"}}
        action = {"type": "tool_use", "name": name, "id": "read-1", "input": arguments}
        context = [{"role": "user", "content": prompt}]
        first = {"id": "model-1", "type": "message", "role": "assistant", "content": [action], "stop_reason": "tool_use"}
        second = {"id": "model-2", "type": "message", "role": "assistant", "content": [{"type": "text", "text": SOLUTION}], "stop_reason": "end_turn"}
        calls = [({"model": "mock", "tools": [tool], "messages": copy.deepcopy(context)}, first)]
        context += [{"role": "assistant", "content": [action]}]
        if returned:
            context.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": encoded}]})
        calls.append(({"model": "mock", "tools": [tool], "messages": context}, second))
        write_calls(trace, calls)
        write_final_event(trace, second)
    else:
        tool = {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        action = {"id": "read-1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        context = [{"role": "user", "content": prompt}]
        first = {"id": "model-1", "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [action]}, "finish_reason": "tool_calls"}]}
        second = {"id": "model-2", "choices": [{"message": {"role": "assistant", "content": SOLUTION}, "finish_reason": "stop"}]}
        calls = [({"model": "mock", "tools": [tool], "messages": copy.deepcopy(context)}, first)]
        context.append(first["choices"][0]["message"])
        if returned:
            context.append({"role": "tool", "tool_call_id": "read-1", "content": encoded})
        calls.append(({"model": "mock", "tools": [tool], "messages": context}, second))
        write_capture(trace, calls, runtime)
        write_final(trace, SOLUTION, runtime)
    audited = audit_trace(case, trace, runtime, tolerance="1e-6", reviewer=SourceReviewer())
    assert audited["audit"]["verdict"]["accepted"] is returned, audited
    if returned:
        bundle = export_cc_native(audited, trace) if runtime == "claude_code" else export_native_capture(audited, trace, runtime)
        assert bundle["sample_count"] == 2


def test_recollection_reuses_frozen_tasks_and_keeps_sampling_artifacts_separate(tmp_path):
    from mint_synth.core.config_v2 import load_config, dump_config
    from mint_synth.core.provenance import json_fingerprint, sha256_file
    from mint_synth.core.run_state import atomic_json
    from mint_synth.sft.collection import collect_frozen_tasks
    config = load_config(Path(__file__).resolve().parents[1] / "config/mint-synth.example.yaml")
    config["harness"].pop("retrieval", None)
    source = RunLayout(tmp_path / "source-run")
    dump_config(config, source.root / "resolved_config.yaml")
    path = source.shard("validation", 0) / "tasks.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"frozen validated task fixture")
    digest = sha256_file(path)
    atomic_json(source.success("validation", 0), {"stage": "validation", "shard_id": 0, "signature": "frozen",
        "files": [{"path": "tasks.parquet", "bytes": path.stat().st_size, "sha256": digest}]})
    plan = {"stages": [{"name": "validation", "shard_count": 1}], "stage_signatures": {"validation": ["frozen"]}}
    plan["plan_fingerprint"] = json_fingerprint(plan)
    atomic_json(source.plan, plan)
    visited = []
    def sample(layout, cfg, _plan, shard, output):
        assert layout.shard("validation", shard) == path.parent
        visited.append(cfg["training"]["agent_runtime"]["default"])
        assert output.is_relative_to(layout.root)
        return {"atomic_train": 1, "long_train": 1}
    for runtime in ("codex", "strands"):
        config["training"]["agent_runtime"]["default"] = runtime
        config_path = tmp_path / (runtime + ".yaml")
        dump_config(config, config_path)
        with patch("mint_synth.workflow.stages.sft_data.stage_sft_prepare", side_effect=sample):
            result = collect_frozen_tasks(source.root, config_path, tmp_path / (runtime + "-collection"))
        assert result["sampling_fingerprint"]
        assert sha256_file(path) == digest
    assert visited == ["codex", "strands"]
    assert not (source.root / "cache").exists()
