from __future__ import annotations

import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from mint_synth.pipeline import PipelineConfig, SynthesisPipeline, validate_output


class FakeLLM:
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        del temperature, max_tokens
        if "evidence extractor" in system:
            chunk = user.split("CHUNK BEGIN\n", 1)[1].split("\nCHUNK END", 1)[0]
            match = re.search(r"([A-Z][A-Za-z ]+) reported revenue of \$(\d+) million in (20\d\d)\.", chunk)
            assert match
            entity, value, period = match.groups()
            quote = match.group(0)
            return {
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
                        "quote": quote,
                    }
                ]
            }
        if "atomic financial training tasks" in system:
            facts_text = user.split("FACTS:\n", 1)[1].split("\n\nCONTEXT BEGIN", 1)[0]
            facts = json.loads(facts_text)
            evidence = facts[0]
            return {
                "task": {
                    "question": f"What revenue did {evidence['entity']} report for {evidence['period']}?",
                    "evidence_ids": [evidence["fact_id"]],
                    "evidence_graph": {
                        "nodes": [
                            {"id": "leaf", "kind": "fact", "fact_id": evidence["fact_id"]},
                            {"id": "out", "kind": "operation", "op": "abs", "inputs": ["leaf"]},
                        ],
                        "terminal_id": "out",
                    },
                    "reasoning_trace": [
                        {"statement": "Read the explicitly reported revenue for the requested period.", "evidence_ids": [evidence["fact_id"]], "operation_node": "out"}
                    ],
                    "answer": {
                        "value": evidence["value_num"],
                        "unit": "USD",
                        "scale": "million",
                        "currency": "USD",
                        "text": f"${evidence['value_num']} million",
                    },
                }
            }
        if "hidden, executable evidence graphs" in system:
            facts = json.loads(user.split("FACT BUNDLE:\n", 1)[1])
            first, second = facts[0], next(item for item in facts[1:] if item["source_id"] != facts[0]["source_id"])
            total = int(first["value_num"]) + int(second["value_num"])
            return {
                "candidate": {
                    "task_spec": {
                        "objective_type": "aggregation",
                        "entities": [first["entity"], second["entity"]],
                        "periods": [first["period"], second["period"]],
                        "requested_output": "combined reported revenue",
                        "time_sensitive": False,
                    },
                    "evidence_graph": {
                        "nodes": [
                            {"id": "left", "kind": "fact", "fact_id": first["fact_id"]},
                            {"id": "right", "kind": "fact", "fact_id": second["fact_id"]},
                            {"id": "sum", "kind": "operation", "op": "add", "inputs": ["left", "right"]},
                            {"id": "magnitude", "kind": "operation", "op": "abs", "inputs": ["sum"]},
                        ],
                        "terminal_id": "magnitude",
                    },
                    "answer": {
                        "value": total,
                        "unit": "USD",
                        "scale": "million",
                        "currency": "USD",
                        "text": f"${total} million",
                    },
                }
            }
        if "realize a hidden evidence graph" in system:
            return {"question": "Using the two companies' disclosures, what is their combined reported revenue for the stated periods?"}
        if "independent, strict financial-data auditor" in system:
            if "non_exposing" in user:
                return {
                    "answer_unique": True,
                    "fully_supported": True,
                    "financially_valid": True,
                    "non_exposing": True,
                    "seed_not_copied": True,
                    "reason": "passes",
                }
            return {
                "fully_supported": True,
                "answer_unique": True,
                "financially_valid": True,
                "no_unsupported_jumps": True,
                "clear": True,
                "reason": "passes",
            }
        raise AssertionError(f"unexpected prompt: {system[:80]}")


class PipelineTests(unittest.TestCase):
    def test_end_to_end_with_replayable_long_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.jsonl"
            rows = [
                {"id": "a", "text": "Alpha Corp reported revenue of $100 million in 2024."},
                {"id": "b", "text": "Beta Ltd reported revenue of $60 million in 2024."},
            ]
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "out"
            config = PipelineConfig(
                input_path=str(input_path),
                output_dir=str(output),
                chunk_chars=1000,
                min_chunk_chars=1,
                overlap_chars=0,
                atomic_per_chunk=1,
                long_candidates=1,
                long_bundle_size=2,
                min_depth=2,
                min_sources=2,
                cutoff="2026-09-29",
                concurrency=2,
            )
            llm = FakeLLM()
            manifest = SynthesisPipeline(
                config,
                extractor_llm=llm,
                constructor_llm=llm,
                auditor_llm=llm,
            ).run()
            self.assertEqual(manifest["counts"]["facts"], 2)
            self.assertEqual(manifest["counts"]["atomic_tasks"], 2)
            self.assertEqual(manifest["counts"]["long_tasks"], 1)
            report = validate_output(output)
            self.assertTrue(report["valid"], report["errors"])
            resumed = SynthesisPipeline(
                replace(config, resume=True),
                extractor_llm=llm,
                constructor_llm=llm,
                auditor_llm=llm,
            ).run()
            self.assertEqual(resumed["counts"], manifest["counts"])
            with self.assertRaisesRegex(ValueError, "differs"):
                SynthesisPipeline(
                    replace(config, resume=True, atomic_per_chunk=2),
                    extractor_llm=llm,
                    constructor_llm=llm,
                    auditor_llm=llm,
                ).run()


if __name__ == "__main__":
    unittest.main()
