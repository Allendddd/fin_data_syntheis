from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from mint_synth.planner import (
    _planned_signatures,
    _stage_plan,
    _validate_auxiliary_inputs,
    verify_plan_fingerprint,
)
from mint_synth.provenance import json_fingerprint


class Segment:
    def __init__(self, value: str):
        self.value = value

    def to_dict(self):
        return {"segment_id": self.value}


class PlannerSignatureTests(unittest.TestCase):
    def test_assignment_change_invalidates_downstream_signature(self) -> None:
        stages = [
            {"name": "normalize", "shard_count": 1, "dependencies": []},
            {"name": "dedup", "shard_count": 1, "dependencies": ["normalize"]},
        ]
        first, _ = _planned_signatures("run", stages, [[Segment("a")]])
        second, _ = _planned_signatures("run", stages, [[Segment("b")]])
        self.assertNotEqual(first["normalize"], second["normalize"])
        self.assertNotEqual(first["dedup"], second["dedup"])

    def test_plan_fingerprint_detects_tampering(self) -> None:
        plan = {"version": 2, "stages": []}
        plan["plan_fingerprint"] = json_fingerprint(plan)
        self.assertTrue(verify_plan_fingerprint(plan))
        plan["version"] = 3
        self.assertFalse(verify_plan_fingerprint(plan))

    def test_global_id_validation_precedes_content_dedup(self) -> None:
        config = {
            "reader": {"shuffle_partitions": 4, "external_id_partitions": 3},
            "entity_resolution": {"buckets": 2},
            "dag": {"entity_buckets": 5},
            "atomic": {"enabled": False},
            "stages": {
                "facts": False,
                "atomic": False,
                "entity_resolution": False,
                "graph_planning": False,
                "query_realization": False,
                "validation": False,
                "export": False,
            },
        }
        stages = _stage_plan(config, normalize_count=2)
        self.assertEqual([stage["name"] for stage in stages], ["normalize", "id_validation", "dedup"])
        self.assertEqual(stages[1]["shard_count"], 3)
        self.assertEqual(stages[2]["dependencies"], ["normalize", "id_validation"])

    def test_plan_preflight_rejects_unparsed_custom_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            seed_path = Path(temp) / "seeds.jsonl"
            seed_path.write_text(json.dumps({"seed_id": "bad", "seed_query": "query"}) + "\n", encoding="utf-8")
            config = {
                "dag": {
                    "formula_registry": "builtin",
                    "allowed_operations": ["subtract", "abs"],
                },
                "stages": {"graph_planning": True},
                "entity_resolution": {"alias_file": None},
                "query_realization": {"seed_file": str(seed_path)},
                "quality": {"evaluation_file": None, "max_evaluation_questions": 10},
            }
            with self.assertRaisesRegex(ValueError, "task_spec_template"):
                _validate_auxiliary_inputs(config)


if __name__ == "__main__":
    unittest.main()
