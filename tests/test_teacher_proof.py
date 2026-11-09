from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import yaml

from mint_synth.harness import HarnessSession, case_from_task
from mint_synth.harness.verifier import _answer_ok, typed_answer_ok
from mint_synth.sft.atomic_teacher import sample_atomic
from mint_synth.sft.rollout import long_rollout
from mint_synth.synthesis.dag import load_formula_registry
from mint_synth.synthesis.long_horizon import DEFAULT_SEEDS
from mint_synth.synthesis.seed_constraints import formulas_for_seed
from mint_synth.core.config_v2 import load_config
from mint_synth.core.run_state import RunLayout, atomic_json
from mint_synth.sft.formatting import atomic_messages
from mint_synth.workflow.stages.questions import stage_graph_planning
from mint_synth.workflow.stages.sft_data import stage_sft_prepare
from mint_synth.workflow.stages.publication import stage_export


def task() -> dict:
    quotes = ["Revenue was $100 million in 2023.", "Revenue was $60 million in 2024."]
    facts = [
        {"fact_id": f"f{i}", "chunk_id": f"c{i}", "source_id": f"s{i}",
         "source_metadata": {"published_at": "2025-01-01"}, "quote": quote,
         "value_num": value, "value_text": value, "unit": "USD", "scale": "million", "currency": "USD"}
        for i, (quote, value) in enumerate(zip(quotes, ["100", "60"]), 1)
    ]
    return {
        "task_id": "long_1", "task_type": "long_horizon", "question": "What is the combined revenue?",
        "input": {"context": "\n".join(quotes)}, "evidence": facts,
        "canonical_entity_id": "entity", "cutoff": "2025-12-31",
        "source_pack": [{"source_id": f"s{i}", "chunk_id": f"c{i}"} for i in (1, 2)],
        "task_spec": {"time_sensitive": False},
        "evidence_graph": {"nodes": [
            {"id": "g1", "kind": "fact", "fact_id": "f1"},
            {"id": "g2", "kind": "fact", "fact_id": "f2"},
            {"id": "sum", "kind": "operation", "op": "add", "inputs": ["g1", "g2"]},
        ], "terminal_id": "sum"},
        "answer": {"value": "160", "unit": "USD", "scale": "million", "currency": "USD"},
        "reasoning_trace": [{"statement": "The values sum to 160.", "evidence_ids": ["f1", "f2"]}],
    }


def proof() -> dict:
    dimensions = {"unit": "USD", "scale": "million", "currency": "USD"}
    return {
        "answer": {"value": "160", **dimensions},
        "facts": [
            {"id": "a", "chunk_id": "c1", "quote": "Revenue was $100 million in 2023.", "value": "100", **dimensions},
            {"id": "b", "chunk_id": "c2", "quote": "Revenue was $60 million in 2024.", "value": "60", **dimensions},
        ],
        "constants": [],
        "steps": [{"id": "total", "op": "add", "inputs": ["a", "b"], "params": {},
                   "result": "160", **dimensions}],
        "terminal": "total",
    }


class SequenceClient:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.values)


def snapshot(path: Path) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY,source_id TEXT,available_at TEXT,text TEXT)")
        db.execute("CREATE TABLE source_entities(source_id TEXT,canonical_entity_id TEXT)")
        db.execute("CREATE VIRTUAL TABLE lookup USING fts5(text,content='chunks',content_rowid='rowid',tokenize='trigram')")
        for i, quote in enumerate(("Revenue was $100 million in 2023.", "Revenue was $60 million in 2024."), 1):
            cur = db.execute("INSERT INTO chunks VALUES(?,?,?,?)", (f"c{i}", f"s{i}", "2025-01-01", quote))
            db.execute("INSERT INTO lookup(rowid,text) VALUES(?,?)", (cur.lastrowid, quote))
            db.execute("INSERT INTO source_entities VALUES(?,?)", (f"s{i}", "entity"))
        db.commit()


class TeacherProofTests(unittest.TestCase):
    def test_atomic_rejects_wrong_candidates_and_keeps_prompt_private(self):
        source = task()
        source["task_type"] = "atomic"
        bad = proof()
        bad["answer"]["value"] = "999"
        shorter = proof()
        clients = {"weak": SequenceClient([bad]), "strong": SequenceClient([shorter])}
        sample, audit, reason = sample_atomic(source, clients, attempts=1, policy="concise",
                                              fallback=False, tolerance="1e-6")
        self.assertEqual(reason, "accepted")
        self.assertEqual(audit["selected"]["role"], "strong")
        self.assertEqual(len(audit["candidates"]), 2)
        self.assertFalse(audit["candidates"][0]["accepted"])
        self.assertEqual(json.loads(sample["messages"][2]["content"]), shorter)
        for client in clients.values():
            call = client.calls[0]
            self.assertNotIn("fact_id", call["user"])
            self.assertNotIn("answer", call["user"])
            self.assertNotIn("evidence_graph", call["user"])

    def test_atomic_all_fail_and_explicit_audit_fallback(self):
        source = task()
        source["task_type"] = "atomic"
        bad = proof()
        bad["steps"] = []
        rejected, audit, _ = sample_atomic(source, {"teacher": SequenceClient([bad])}, attempts=1,
                                           policy="first_valid", fallback=False, tolerance="1e-6")
        self.assertIsNone(rejected)
        self.assertIsNone(audit["selected"])
        fallback, audit, _ = sample_atomic(source, {"teacher": SequenceClient([bad])}, attempts=1,
                                           policy="first_valid", fallback=True, tolerance="1e-6")
        self.assertIsNotNone(fallback)
        self.assertEqual(audit["selected"]["source"], "audit_fallback")
        restated = copy.deepcopy(proof())
        restated["question_repeated"] = source["question"] * 20
        rejected, _, _ = sample_atomic(source, {"teacher": SequenceClient([restated])}, attempts=1,
                                       policy="concise", fallback=False, tolerance="1e-6")
        self.assertIsNone(rejected)
        repeated = copy.deepcopy(proof())
        repeated["steps"].append(copy.deepcopy(repeated["steps"][0]))
        rejected, _, _ = sample_atomic(source, {"teacher": SequenceClient([repeated])}, attempts=1,
                                       policy="concise", fallback=False, tolerance="1e-6")
        self.assertIsNone(rejected)

    def test_atomic_selection_policy_records_why_a_valid_candidate_won(self):
        source = task()
        source["task_type"] = "atomic"
        verbose = proof()
        verbose["facts"][0]["id"] = "longer_local_reference"
        verbose["steps"][0]["inputs"][0] = "longer_local_reference"
        concise = proof()
        clients = {"first": SequenceClient([verbose]), "second": SequenceClient([concise])}
        _, audit, _ = sample_atomic(source, clients, attempts=1, policy="concise",
                                    fallback=False, tolerance="1e-6")
        self.assertEqual(audit["selected"]["role"], "second")
        self.assertLess(audit["selected"]["serialized_chars"], audit["candidates"][0]["serialized_chars"])
        clients = {"first": SequenceClient([verbose]), "second": SequenceClient([concise])}
        _, audit, _ = sample_atomic(source, clients, attempts=1, policy="first_valid",
                                    fallback=False, tolerance="1e-6")
        self.assertEqual(audit["selected"]["role"], "first")

    def test_atomic_samples_use_distinct_slots_without_changing_teacher_input(self):
        source = task()
        source["task_type"] = "atomic"
        client = SequenceClient([proof(), proof()])
        sample, audit, _ = sample_atomic(source, {"teacher": client}, attempts=2,
                                         policy="first_valid", fallback=False, tolerance="1e-6")
        self.assertIsNotNone(sample)
        self.assertEqual(len(audit["candidates"]), 2)
        self.assertEqual([call["sample_key"] for call in client.calls],
                         ["atomic:long_1:teacher:1", "atomic:long_1:teacher:2"])
        self.assertEqual(client.calls[0]["user"], client.calls[1]["user"])

    def test_numeric_units_and_boolean_boundaries(self):
        source = task()
        proposed = proof()
        self.assertTrue(_answer_ok(source, json.dumps(proposed), "1e-6"))
        for value in ("160%", "0", "2024", "1600"):
            wrong = copy.deepcopy(proposed)
            wrong["answer"]["value"] = value
            self.assertFalse(_answer_ok(source, json.dumps(wrong), "1e-6"))
        wrong = copy.deepcopy(proposed)
        wrong["answer"]["unit"] = "EUR"
        self.assertFalse(_answer_ok(source, json.dumps(wrong), "1e-6"))
        wrong["answer"]["unit"] = None
        self.assertFalse(_answer_ok(source, json.dumps(wrong), "1e-6"))
        self.assertFalse(_answer_ok(source, "The answer is 160 USD million", "1e-6"))
        boolean = {"answer": {"value": False, "unit": None, "scale": None, "currency": None},
                   "evidence": [{"fact_id": "b", "source_id": "s", "quote": "The claim is false.",
                                 "value_text": False}],
                   "evidence_graph": {"nodes": [{"id": "b", "kind": "fact"}], "terminal_id": "b"}}
        boolean_proof = {"answer": dict(boolean["answer"]),
                         "facts": [{"id": "c", "quote": "The claim is false.", "value": False,
                                    "unit": None, "scale": None, "currency": None}],
                         "constants": [], "steps": [], "terminal": "c"}
        self.assertTrue(_answer_ok(boolean, json.dumps(boolean_proof), "1e-6"))
        self.assertTrue(typed_answer_ok(boolean, {"value": False, "unit": None, "scale": None, "currency": None}, "1e-6"))
        for wrong_value in (True, 0, "false", "not false"):
            self.assertFalse(typed_answer_ok(boolean, {"value": wrong_value, "unit": None,
                                                       "scale": None, "currency": None}, "1e-6"))
            wrong_proof = copy.deepcopy(boolean_proof)
            wrong_proof["answer"]["value"] = wrong_value
            self.assertFalse(_answer_ok(boolean, json.dumps(wrong_proof), "1e-6"))

    def test_long_harness_requires_derivation_and_rollout_masks_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "snapshot.sqlite3"
            snapshot(index)
            source = task()
            case = case_from_task(source, index, cutoff="2025-12-31")
            actions = [
                {"tool": "search_corpus", "arguments": {"query": "Revenue"}},
                {"tool": "open_chunk", "arguments": {"chunk_id": "c1"}},
                {"tool": "open_chunk", "arguments": {"chunk_id": "c2"}},
            ]
            with HarnessSession(case, max_steps=5, max_tool_calls=4) as session:
                for action in actions:
                    self.assertNotIn("error", session.step(action)["observation"])
                incomplete = copy.deepcopy(proof())
                incomplete["steps"] = []
                session.step({"final": json.dumps(incomplete)})
                grade = session.grade()
                self.assertTrue(grade["evidence_replayable"])
                self.assertTrue(grade["answer_correct"])
                self.assertFalse(grade["derivation_valid"])
                self.assertFalse(grade["accepted"])
            client = SequenceClient([*actions, {"final": json.dumps(proof())}])
            config = {"training": {"max_tool_steps": 4, "max_context_chars": 24000},
                      "run": {"cutoff": "2025-12-31"}, "input": {"available_at_field": "published_at"},
                      "quality": {"answer_tolerance": "1e-6"}}
            sample, reason = long_rollout(source, client, index, config)
            self.assertEqual(reason, "accepted")
            self.assertEqual([message["loss"] for message in sample["messages"] if message["role"] == "tool_response"],
                             [False, False, False])
            self.assertTrue(sample["messages"][-1]["loss"])

    def test_checkpoint_fork_preserves_strict_grade(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "snapshot.sqlite3"
            snapshot(index)
            case = case_from_task(task(), index, cutoff="2025-12-31")
            with HarnessSession(case, max_steps=5, max_tool_calls=4) as session:
                session.step({"tool": "search_corpus", "arguments": {"query": "Revenue"}})
                session.step({"tool": "open_chunk", "arguments": {"chunk_id": "c1"}})
                checkpoint = session.checkpoint()
                with session.fork(checkpoint) as branch:
                    branch.step({"tool": "open_chunk", "arguments": {"chunk_id": "c2"}})
                    branch.step({"final": json.dumps(proof())})
                    self.assertTrue(branch.grade()["accepted"])

    def test_seed_constraints_compile_before_planning(self):
        formulas = load_formula_registry("builtin")
        seed = DEFAULT_SEEDS[1]
        enabled = formulas_for_seed(seed, formulas)
        self.assertEqual({formula.generator for formula in enabled}, {"temporal_pair"})
        invalid = copy.deepcopy(seed)
        invalid["task_spec_template"]["required_fact_roles"] = ["numerator", "denominator"]
        with self.assertRaisesRegex(ValueError, "required_fact_roles"):
            formulas_for_seed(invalid, formulas)
        unsupported = copy.deepcopy(DEFAULT_SEEDS[0])
        unsupported["origin"] = "user_seed_bank"
        with self.assertRaisesRegex(ValueError, "reasoning_form"):
            formulas_for_seed(unsupported, formulas)

    def test_seed_constraints_filter_facts_and_formulas_before_planner_call(self):
        seed = copy.deepcopy(DEFAULT_SEEDS[1])
        seed["task_spec_template"]["time_sensitive"] = True
        seen = []

        class Planner:
            def __init__(self, formulas, **_kwargs):
                self.formulas = formulas

            def plan_entity(self, facts):
                seen.append((tuple(formula.generator for formula in self.formulas), tuple(facts)))
                return []

        class Writer:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def write(self, _row):
                pass

        class Layout:
            def shard(self, *_args):
                return Path("unused")

        dated = {"fact_id": "dated", "source_metadata": {"published_at": "2024-01-01"}}
        undated = {"fact_id": "undated", "source_metadata": {}}
        config = {
            "version": 3, "query_realization": {"seed_file": None}, "training": {},
            "run": {"cutoff": "2025-12-31"}, "input": {"available_at_field": "published_at"},
            "reader": {"batch_rows": 16},
            "dag": {"formula_registry": "builtin", "allowed_operations": ["subtract", "abs"],
                    "min_depth": 2, "max_depth": 4, "min_sources": 2, "beam_width": 8,
                    "max_graphs_per_entity": 8, "max_candidate_groups_per_formula": 32,
                    "max_facts_per_entity": 100, "min_rule_graphs_before_supplement": 0,
                    "llm_supplement": False},
        }
        with (patch("mint_synth.workflow.stages.questions.parquet_num_rows", return_value=2),
              patch("mint_synth.workflow.stages.questions.load_seed_bank", return_value=[seed]),
              patch("mint_synth.workflow.stages.questions._grouped_payloads",
                    return_value=iter([("entity", [dated, undated])])),
              patch("mint_synth.workflow.stages.questions.RuleDagPlanner", Planner),
              patch("mint_synth.workflow.stages.questions._envelope_writer", return_value=Writer()),
              patch("mint_synth.workflow.stages.questions._quarantine_writer", return_value=Writer())):
            stage_graph_planning(Layout(), config, {}, 0, Path("unused"))
        self.assertEqual(seen, [( ("temporal_pair", "temporal_pair"), (dated,))])

    def test_equivalent_seed_profiles_plan_each_entity_once(self):
        first = copy.deepcopy(DEFAULT_SEEDS[1])
        second = copy.deepcopy(first)
        second["seed_id"] = first["seed_id"] + "-second"
        calls = []

        class Planner:
            def __init__(self, formulas, **_kwargs):
                self.formulas = formulas

            def plan_entity(self, facts):
                calls.append(tuple(fact["fact_id"] for fact in facts))
                return []

        class Writer:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def write(self, _row):
                pass

        class Layout:
            def shard(self, *_args):
                return Path("unused")

        config = {"version": 3, "query_realization": {"seed_file": None}, "training": {},
                  "run": {"cutoff": "2025-12-31"}, "input": {"available_at_field": "published_at"},
                  "reader": {"batch_rows": 16},
                  "dag": {"formula_registry": "builtin", "allowed_operations": ["subtract", "abs"],
                          "min_depth": 2, "max_depth": 4, "min_sources": 2, "beam_width": 8,
                          "max_graphs_per_entity": 8, "max_candidate_groups_per_formula": 32,
                          "max_facts_per_entity": 100, "min_rule_graphs_before_supplement": 0,
                          "llm_supplement": False}}
        fact = {"fact_id": "f", "source_metadata": {"published_at": "2024-01-01"}}
        with (patch("mint_synth.workflow.stages.questions.parquet_num_rows", return_value=1),
              patch("mint_synth.workflow.stages.questions.load_seed_bank", return_value=[first, second]),
              patch("mint_synth.workflow.stages.questions._grouped_payloads", return_value=iter([("entity", [fact])])),
              patch("mint_synth.workflow.stages.questions.RuleDagPlanner", Planner),
              patch("mint_synth.workflow.stages.questions._envelope_writer", return_value=Writer()),
              patch("mint_synth.workflow.stages.questions._quarantine_writer", return_value=Writer())):
            metrics = stage_graph_planning(Layout(), config, {}, 0, Path("unused"))
        self.assertEqual(calls, [("f",)])
        self.assertEqual(metrics["planner_cache_hits"], 1)

    def test_sft_prepare_keeps_private_fields_in_sidecar(self):
        source = task()
        source["task_type"] = "atomic"
        client = SequenceClient([proof()])

        class Layout:
            def shard(self, *_args):
                return Path("unused")

        config = {"reader": {"batch_rows": 16}, "training": {
            "enabled": True, "atomic_enabled": True, "long_enabled": False,
            "atomic_teacher_roles": ["teacher"], "atomic_samples_per_task": 1,
            "atomic_selection_policy": "concise", "atomic_audit_fallback": False,
            "split_validation_fraction": 0,
        }, "quality": {"answer_tolerance": "1e-6"}}
        with tempfile.TemporaryDirectory() as temp:
            with (patch("mint_synth.workflow.stages.sft_data.iter_payloads", return_value=iter([source])),
                  patch("mint_synth.workflow.stages.sft_data._role_client", return_value=client)):
                stage_sft_prepare(Layout(), config, {}, 0, Path(temp))
            sample = json.loads((Path(temp) / "atomic.jsonl").read_text(encoding="utf-8"))
            sidecar = json.loads((Path(temp) / "sidecar.jsonl").read_text(encoding="utf-8"))
            self.assertEqual([entry["role"] for entry in sample["messages"]], ["system", "user", "assistant"])
            self.assertNotIn("evidence_graph", sample["messages"][1]["content"])
            self.assertNotIn("f1", sample["messages"][1]["content"])
            self.assertFalse(sample["messages"][1]["loss"])
            self.assertEqual(sidecar["answer"]["value"], "160")
            self.assertIn("evidence_graph", sidecar)
            self.assertEqual(sidecar["teacher_audit"]["selected"]["source"], "teacher")

    def test_sft_prepare_bounds_parallel_teacher_calls_and_preserves_order(self):
        first = task()
        first["task_type"] = "atomic"
        second = copy.deepcopy(first)
        second["task_id"] = "long_2"
        barrier = threading.Barrier(2)

        class ParallelTeacher:
            def complete_json(self, **_kwargs):
                barrier.wait(timeout=3)
                return proof()

        class Layout:
            def shard(self, *_args):
                return Path("unused")

        config = {"reader": {"batch_rows": 16}, "llm": {"concurrency_per_worker": 2},
                  "training": {"enabled": True, "atomic_enabled": True, "long_enabled": False,
                               "atomic_teacher_roles": ["teacher"], "atomic_samples_per_task": 1,
                               "atomic_selection_policy": "concise", "atomic_audit_fallback": False,
                               "split_validation_fraction": 0},
                  "quality": {"answer_tolerance": "1e-6"}}
        with tempfile.TemporaryDirectory() as temp:
            with (patch("mint_synth.workflow.stages.sft_data.iter_payloads", return_value=iter([first, second])),
                  patch("mint_synth.workflow.stages.sft_data._role_client", return_value=ParallelTeacher())):
                counts = stage_sft_prepare(Layout(), config, {}, 0, Path(temp))
            self.assertEqual(counts["atomic_accepted"], 2)
            sidecars = [json.loads(line) for line in (Path(temp) / "sidecar.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["task_id"] for row in sidecars], ["long_1", "long_2"])

    def test_legacy_v3_config_receives_new_defaults(self):
        example = yaml.safe_load(Path("config/mint-synth.example.yaml").read_text(encoding="utf-8"))
        for key in ("atomic_teacher_roles", "atomic_samples_per_task", "atomic_selection_policy",
                    "atomic_audit_fallback", "long_teacher_roles"):
            example["training"].pop(key)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy-v3.yaml"
            path.write_text(yaml.safe_dump(example, allow_unicode=True), encoding="utf-8")
            loaded = load_config(path)
        self.assertEqual(loaded["training"]["atomic_teacher_roles"], ["constructor"])
        self.assertEqual(loaded["training"]["long_teacher_roles"], [])
        self.assertFalse(loaded["training"]["atomic_audit_fallback"])

    def test_final_sft_export_strips_split_and_keeps_private_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            layout = RunLayout(Path(temp) / "run")
            prepared = layout.shard("sft_prepare", 0)
            prepared.mkdir(parents=True)
            sample = atomic_messages(task(), response=json.dumps(proof()))
            sample["_split"] = "train"
            (prepared / "atomic.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            (prepared / "long.jsonl").write_text("", encoding="utf-8")
            (prepared / "sidecar.jsonl").write_text(
                json.dumps({"task_id": "long_1", "answer": task()["answer"],
                            "evidence_graph": task()["evidence_graph"]}) + "\n", encoding="utf-8")
            atomic_json(prepared / "_SUCCESS.json", {"metrics": {"atomic_train": 1}})
            output = Path(temp) / "export"
            output.mkdir()
            plan = {"stages": [{"name": "validation", "shard_count": 1},
                               {"name": "sft_prepare", "shard_count": 1}]}
            config = {"version": 3, "training": {"enabled": True, "atomic_enabled": True,
                       "long_enabled": False, "min_atomic_samples": 1, "min_long_samples": 1},
                      "query_export": {"enabled": False},
                      "output": {"export_formats": ["jsonl"], "artifact_checksums": False},
                      "reader": {"batch_rows": 16}}
            with patch("mint_synth.workflow.stages.publication._enforce_global_quality", return_value={}):
                stage_export(layout, config, plan, 0, output)
            exported = json.loads((output / "sft" / "atomic-train.jsonl").read_text(encoding="utf-8"))
            sidecar = json.loads((output / "sft" / "audit-sidecar.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn("_split", exported)
            self.assertNotIn("evidence_graph", exported)
            self.assertIn("evidence_graph", sidecar)
            self.assertEqual(json.loads((output / "sft" / "manifest.json").read_text(encoding="utf-8"))["format"],
                             "ms-swift messages")


if __name__ == "__main__":
    unittest.main()
