from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


RUNTIME_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("yaml", "jsonschema", "pyarrow", "duckdb")
)


class InferenceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system = request["messages"][0]["content"]
        user = request["messages"][1]["content"]
        if "financial evidence extractor" in system:
            chunk = user.split("CHUNK BEGIN\n", 1)[1].split("\nCHUNK END", 1)[0]
            match = re.search(r"(Alpha Corp) reported revenue of \$(\d+) million in (20\d\d)\.", chunk)
            assert match
            entity, value, period = match.groups()
            payload = {
                "facts": [
                    {
                        "entity": entity,
                        "period": period,
                        "metric": "revenue",
                        "value_text": f"${value} million",
                        "value_num": value,
                        "unit": "USD",
                        "scale": "million",
                        "currency": "USD",
                        "fact_type": "reported",
                        "quote": match.group(0),
                    }
                ]
            }
        elif "realize a hidden evidence graph" in system:
            payload = {
                "question": "What was the absolute change in Alpha Corp's reported revenue from 2023 to 2024?"
            }
        elif "independent, strict financial-data auditor" in system:
            payload = {
                "answer_unique": True,
                "fully_supported": True,
                "financially_valid": True,
                "non_exposing": True,
                "seed_not_copied": True,
                "reason": "passes",
            }
        else:
            raise AssertionError(system)
        body = json.dumps(
            {
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@unittest.skipUnless(RUNTIME_DEPENDENCIES, "v2 integration requires PyYAML/jsonschema/pyarrow/duckdb")
class V2IntegrationTests(unittest.TestCase):
    def test_complete_parquet_worker_dag(self) -> None:
        import yaml

        from mint_synth.planner import create_plan, run_status
        from mint_synth.run_state import read_json
        from mint_synth.stages import run_worker

        server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data = root / "data"
                data.mkdir()
                rows = [
                    {"id": "a", "text": "Alpha Corp reported revenue of $100 million in 2023."},
                    {"id": "b", "text": "Alpha Corp reported revenue of $125 million in 2024."},
                ]
                (data / "input.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                image = root / "mint-synth.sif"
                image.write_bytes(b"test image fingerprint")
                stage_names = (
                    "normalize", "id_validation", "dedup", "facts", "atomic", "entity_resolution",
                    "repartition", "graph_planning", "query_realization", "validation", "export",
                )
                endpoint = f"http://127.0.0.1:{server.server_port}/v1"
                config = {
                    "version": 2,
                    "run": {"name": "integration", "work_dir": str(root / "work"), "cutoff": "2026-09-30"},
                    "input": {"paths": [str(data)], "text_field": "text", "id_field": "id"},
                    "output": {
                        "root": str(root / "outputs"), "parquet_row_group_rows": 2048,
                        "export_formats": ["parquet", "jsonl"],
                    },
                    "slurm": {
                        "partition": "test", "apptainer_image": str(image),
                        "stage_resources": {name: {} for name in stage_names},
                    },
                    "llm": {
                        "endpoints": {
                            "local": {
                                "base_url": endpoint, "api_key_env": "UNUSED_TEST_KEY",
                                "require_api_key": False, "strict_json": True,
                            }
                        },
                        "roles": {
                            "extractor": {"endpoint": "local", "model": "extractor"},
                            "constructor": {"endpoint": "local", "model": "constructor"},
                            "auditor": {"endpoint": "local", "model": "auditor"},
                            "entity_resolver": {"endpoint": "local", "model": "resolver"},
                        },
                        "concurrency_per_worker": 2,
                    },
                    "reader": {
                        "jsonl_byte_shard_size": 1048576,
                        "normalize_target_bytes": 1048576,
                        "dedup_buckets": 1,
                        "shuffle_partitions": 1,
                        "external_id_partitions": 1,
                        "duckdb_memory_limit": "256MB",
                        "duckdb_threads": 1,
                    },
                    "chunking": {"max_chars": 1000, "overlap_chars": 0, "min_chars": 1},
                    "atomic": {"enabled": False, "tasks_per_chunk": 0},
                    "entity_resolution": {"llm_fallback": False, "buckets": 1},
                    "dag": {
                        "allowed_operations": ["subtract", "abs"],
                        "min_depth": 2,
                        "max_depth": 2,
                        "min_sources": 2,
                        "entity_buckets": 1,
                        "llm_supplement": False,
                        "max_graphs_per_entity": 1,
                    },
                    "stages": {"atomic": False},
                }
                config_path = root / "config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                layout = create_plan(config_path)
                plan = read_json(layout.plan)
                for stage in plan["stages"]:
                    for shard_id in range(stage["shard_count"]):
                        run_worker(layout.root, stage["name"], shard_id)
                status = run_status(layout.root)
                self.assertTrue(status["complete"])
                self.assertTrue(status["published"])
                export_shard = layout.shard("export", 0)
                exported = [
                    json.loads(line)
                    for line in (export_shard / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(exported), 1)
                self.assertEqual(exported[0]["formula_id"], "temporal_absolute_change")
                self.assertEqual(exported[0]["answer"]["value"], "25")
                self.assertTrue(list((export_shard / "dataset").glob("*.parquet")))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
