from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx

from mint_synth.agents.base import ModelSettings, RuntimeResult, RuntimeSettings
from mint_synth.agents.bridge import FinanceToolBridge
from mint_synth.core.config_v2 import ConfigError, materialize_defaults, validate_cross_fields
from mint_synth.core.run_state import RunLayout, atomic_json
from mint_synth.harness import HarnessSession, case_from_task
from mint_synth.sft.rollout import long_rollout
from mint_synth.workflow.stages.sft_data import stage_sft_prepare
from mint_synth.workflow.stages.publication import stage_export

from test_teacher_proof import proof, snapshot, task


CONFIG = {"training": {"max_tool_steps": 4, "max_context_chars": 24000},
          "run": {"cutoff": "2025-12-31"}, "input": {"available_at_field": "published_at"},
          "quality": {"answer_tolerance": "1e-6"}}


def solve(bridge):
    bridge.call_tool("search_corpus", {"query": "Revenue"})
    bridge.call_tool("open_chunk", {"chunk_id": "c1"})
    bridge.call_tool("open_chunk", {"chunk_id": "c2"})
    return bridge.call_tool("submit_final", {"proof": proof()})


class Upstream(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"output":[]}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class AgentCollectionTests(unittest.TestCase):
    def test_bridge_serializes_sqlite_access_and_hides_verifier(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "snapshot.sqlite3"
            snapshot(index)
            case = case_from_task(task(), index, cutoff="2025-12-31")
            with FinanceToolBridge(lambda: HarnessSession(case, max_steps=5, max_tool_calls=4)) as bridge:
                public = json.dumps(bridge.public_state())
                self.assertNotIn("evidence_graph", public)
                self.assertNotIn('"task"', public)
                result = solve(bridge)
                self.assertTrue(result["done"])
                self.assertNotIn("verdict", result)
                self.assertNotIn("accepted", result)
                self.assertTrue(bridge.snapshot()["audit"]["verdict"]["accepted"])

    def test_all_external_runtimes_use_one_verifier_and_preserve_model_exchange(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                index = Path(temp) / "snapshot.sqlite3"
                snapshot(index)
                model = ModelSettings("local", f"http://127.0.0.1:{server.server_port}/v1", "secret-key")

                def runtime(bridge, traced_model, _settings, root):
                    with httpx.Client(trust_env=False) as client:
                        response = client.post(traced_model.base_url + "/responses", json={"input": "query"},
                                               headers={"Authorization": "Bearer secret-key"})
                    self.assertEqual(response.status_code, 200)
                    solve(bridge)
                    (root / "events.jsonl").write_text('{"event":"completed"}\n')
                    return RuntimeResult("completed", metadata={"events_file": "events.jsonl"})

                for name in ("strands", "codex", "claude_code"):
                    target = "mint_synth.agents.strands_runtime.run_strands" if name == "strands" else "mint_synth.agents.cli_runtime.run_cli"
                    root = Path(temp) / name
                    with patch(target, side_effect=runtime):
                        sample, reason = long_rollout(task(), None, index, CONFIG, model=model,
                                                      runtime=RuntimeSettings(name), trace_dir=root)
                    self.assertEqual(reason, "accepted")
                    self.assertEqual(json.loads(sample["messages"][-1]["content"]), proof())
                    self.assertTrue(all(not row["loss"] for row in sample["messages"] if row["role"] == "tool_response"))
                    self.assertNotIn("evidence_graph", json.dumps(sample))
                    metadata = json.loads((root / "result.json").read_text())
                    self.assertEqual(metadata["runtime"], name)
                    calls = list((root / "model" / "calls").glob("*.json"))
                    self.assertEqual(len(calls), 1)
                    self.assertNotIn("secret-key", calls[0].read_text())
                    call = json.loads(calls[0].read_text())
                    self.assertEqual(json.loads((calls[0].parent / call["request"]["body_file"]).read_bytes()), {"input": "query"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_failed_runtime_retains_partial_episode_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "snapshot.sqlite3"
            snapshot(index)
            root = Path(temp) / "trace"

            def fail(bridge, *_args):
                bridge.call_tool("search_corpus", {"query": "Revenue"})
                return RuntimeResult("timeout", "agent deadline exceeded")

            with patch("mint_synth.agents.cli_runtime.run_cli", side_effect=fail):
                sample, reason = long_rollout(task(), None, index, CONFIG,
                    model=ModelSettings("local", "http://127.0.0.1:8000/v1", None),
                    runtime=RuntimeSettings("codex"), trace_dir=root)
            self.assertIsNone(sample)
            self.assertEqual(reason, "agent deadline exceeded")
            self.assertEqual(json.loads((root / "result.json").read_text())["status"], "timeout")
            self.assertEqual(len(json.loads((root / "environment.json").read_text())["trace"]), 1)

    def test_runtime_defaults_and_unknown_role(self):
        configured = materialize_defaults({"version": 3, "training": {}})
        self.assertEqual(configured["training"]["agent_runtime"]["default"], "native")
        configured["training"]["agent_runtime"]["by_role"] = {"missing": "codex"}
        with self.assertRaisesRegex(ConfigError, "unknown LLM role"):
            validate_cross_fields(configured)

    def test_external_role_skips_json_client_and_publication_keeps_trace_references(self):
        with tempfile.TemporaryDirectory() as temp:
            layout = RunLayout(Path(temp) / "run")
            prepared = layout.shard("sft_prepare", 0)
            prepared.mkdir(parents=True)
            configured = {
                **CONFIG, "version": 3, "reader": {"batch_rows": 16}, "dag": {"entity_buckets": 1},
                "query_export": {"enabled": False},
                "output": {"export_formats": ["jsonl"], "artifact_checksums": True},
                "llm": {"concurrency_per_worker": 1, "roles": {"teacher": {"model": "local", "endpoint": "local"}},
                        "endpoints": {"local": {"base_url": "http://127.0.0.1:8000/v1", "api_key_env": "UNUSED", "require_api_key": False}}},
                "training": {**CONFIG["training"], "enabled": True, "atomic_enabled": False, "long_enabled": True,
                             "long_teacher_roles": ["teacher"], "teacher_role": "teacher", "rollouts_per_task": 1,
                             "agent_runtime": {"default": "codex"}, "split_validation_fraction": 0,
                             "min_atomic_samples": 1, "min_long_samples": 1},
            }

            def collect(_task, client, _index, _config, _registry, **kwargs):
                self.assertIsNone(client)
                self.assertEqual(kwargs["runtime"].name, "codex")
                root = kwargs["trace_dir"]
                root.mkdir(parents=True)
                atomic_json(root / "result.json", {"runtime": "codex", "accepted": True})
                return {"messages": [{"role": "system", "content": "system"},
                                     {"role": "user", "content": task()["question"]},
                                     {"role": "assistant", "content": json.dumps(proof()), "loss": True}],
                        "tools": "[]"}, "accepted"

            with (patch("mint_synth.workflow.stages.sft_data.iter_payloads", return_value=iter([task()])),
                  patch("mint_synth.workflow.stages.sft_data._role_client") as json_client,
                  patch("mint_synth.workflow.stages.sft_data.long_rollout", side_effect=collect)):
                metrics = stage_sft_prepare(layout, configured, {}, 0, prepared)
            json_client.assert_not_called()
            atomic_json(prepared / "_SUCCESS.json", {"metrics": metrics})
            output = Path(temp) / "export"
            output.mkdir()
            plan = {"stages": [{"name": "validation", "shard_count": 1}, {"name": "sft_prepare", "shard_count": 1}]}
            with patch("mint_synth.workflow.stages.publication._enforce_global_quality", return_value={}):
                stage_export(layout, configured, plan, 0, output)
            root = output / "sft"
            row = json.loads((root / "trajectory-index.jsonl").read_text())
            sidecar = json.loads((root / "audit-sidecar.jsonl").read_text())
            self.assertEqual(row["trace"], sidecar["teacher_audit"]["candidates"][0]["trace"])
            self.assertTrue((root / row["trace"] / "result.json").is_file())
            self.assertEqual(row["runtime"], "codex")
            sample = json.loads((root / "long-train.jsonl").read_text())
            self.assertNotIn("teacher_audit", sample)
            self.assertNotIn("evidence_graph", sample)
            artifacts = json.loads((output / "artifact_manifest.json").read_text())["artifacts"]
            self.assertTrue(any(item["path"].endswith("/result.json") for item in artifacts))


if __name__ == "__main__":
    unittest.main()
