from __future__ import annotations

import unittest

from mint_synth.dag import RuleDagPlanner, load_formula_registry
from mint_synth.replay import replay_graph
from mint_synth.stages import _llm_supplement


def fact(fact_id: str, source_id: str, period: str, value: str) -> dict:
    return {
        "fact_id": fact_id,
        "source_id": source_id,
        "canonical_entity_id": "entity_1",
        "canonical_entity_name": "Example Corp",
        "entity": "Example Corp",
        "period": period,
        "metric": "revenue",
        "value_num": value,
        "unit": "USD",
        "scale": "million",
        "currency": "USD",
    }


class DagV2Tests(unittest.TestCase):
    def test_rule_planner_builds_replayable_cross_source_graph(self) -> None:
        facts = [fact("old", "report-a", "2023", "100"), fact("new", "report-b", "2024", "125")]
        planner = RuleDagPlanner(
            load_formula_registry("builtin"),
            allowed_operations=["subtract", "percent_change", "average", "abs", "equals", "identity"],
            min_depth=2,
            max_depth=4,
            min_sources=2,
            beam_width=10,
            max_graphs_per_entity=5,
        )
        candidates = planner.plan_entity(facts)
        self.assertTrue(candidates)
        candidate = candidates[0]
        result = replay_graph(candidate["evidence_graph"], {value["fact_id"]: value for value in facts})
        self.assertGreaterEqual(result.depth, 2)
        self.assertEqual(result.source_ids, {"report-a", "report-b"})
        self.assertTrue(all("formula_id" in node for node in candidate["evidence_graph"]["nodes"] if node["kind"] == "operation"))

    def test_llm_relation_supplement_never_rewrites_original_facts(self) -> None:
        facts = [
            {**fact("a", "report-a", "2023", "100"), "metric": "net sales"},
            {**fact("b", "report-b", "2024", "125"), "metric": "revenue from customers"},
        ]
        planner = RuleDagPlanner(
            load_formula_registry("builtin"),
            allowed_operations=["subtract", "percent_change", "abs"],
            min_depth=2,
            max_depth=4,
            min_sources=2,
            beam_width=10,
            max_graphs_per_entity=5,
        )

        class Resolver:
            def complete_json(self, **kwargs):
                return {"relations": [{"relation": "same_metric", "fact_ids": ["a", "b"]}]}

        candidates = _llm_supplement(Resolver(), facts, planner)
        self.assertTrue(candidates)
        self.assertEqual(
            {value["metric"] for value in candidates[0]["evidence"]},
            {"net sales", "revenue from customers"},
        )

    def test_three_period_formula_builds_a_branched_depth_three_dag(self) -> None:
        facts = [
            fact("y1", "report-a", "2022", "80"),
            fact("y2", "report-b", "2023", "100"),
            fact("y3", "report-c", "2024", "130"),
        ]
        planner = RuleDagPlanner(
            load_formula_registry("builtin"),
            allowed_operations=["subtract", "average", "abs"],
            min_depth=3,
            max_depth=3,
            min_sources=3,
            beam_width=10,
            max_graphs_per_entity=5,
        )
        candidates = planner.plan_entity(facts)
        self.assertTrue(candidates)
        candidate = next(value for value in candidates if value["formula_id"] == "three_period_trend")
        result = replay_graph(candidate["evidence_graph"], {value["fact_id"]: value for value in facts})
        self.assertEqual(result.depth, 3)
        self.assertEqual(len(result.source_ids), 3)
        self.assertEqual(
            {step["node_id"] for step in result.trace},
            {"delta_0", "delta_1", "mean_delta", "magnitude"},
        )

    def test_invalid_early_ratio_does_not_exhaust_beam(self) -> None:
        facts = [
            {**fact("numerator", "s1", "2024", "10"), "metric": "assets"},
            {**fact("zero", "s2", "2024", "0"), "metric": "liabilities"},
            {**fact("valid", "s3", "2024", "2"), "metric": "liabilities"},
        ]
        planner = RuleDagPlanner(
            load_formula_registry("builtin"),
            allowed_operations=["ratio"],
            min_depth=1,
            max_depth=1,
            min_sources=2,
            beam_width=1,
            max_graphs_per_entity=1,
            max_candidate_groups_per_formula=10,
        )
        candidates = planner.plan_entity(facts)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["formula_id"], "cross_source_ratio")
        self.assertEqual(candidates[0]["answer"]["value"], "5")

    def test_multi_source_verification_has_real_branch_depth_without_identity_padding(self) -> None:
        facts = [
            fact("a", "s1", "2024", "100"),
            fact("b", "s2", "2024", "100"),
            fact("c", "s3", "2024", "100"),
        ]
        planner = RuleDagPlanner(
            load_formula_registry("builtin"),
            allowed_operations=["equals", "and"],
            min_depth=2,
            max_depth=2,
            min_sources=3,
            beam_width=10,
            max_graphs_per_entity=5,
        )
        candidate = next(
            value for value in planner.plan_entity(facts)
            if value["formula_id"] == "multi_source_verification"
        )
        operations = [
            node["op"] for node in candidate["evidence_graph"]["nodes"]
            if node["kind"] == "operation"
        ]
        self.assertEqual(operations, ["equals", "equals", "and"])
        self.assertNotIn("identity", operations)

    def test_four_fact_ratio_change_builds_two_real_branches(self) -> None:
        facts = [
            {**fact("old_assets", "s1", "2023", "100"), "metric": "assets"},
            {**fact("old_liabilities", "s2", "2023", "50"), "metric": "liabilities"},
            {**fact("new_assets", "s3", "2024", "120"), "metric": "assets"},
            {**fact("new_liabilities", "s4", "2024", "40"), "metric": "liabilities"},
        ]
        planner = RuleDagPlanner(
            load_formula_registry("builtin"),
            allowed_operations=["ratio", "subtract", "abs"],
            min_depth=3,
            max_depth=3,
            min_sources=4,
            beam_width=10,
            max_graphs_per_entity=5,
        )
        candidate = next(
            value for value in planner.plan_entity(facts)
            if value["formula_id"] == "temporal_ratio_change"
        )
        result = replay_graph(candidate["evidence_graph"], {value["fact_id"]: value for value in facts})
        self.assertEqual(result.depth, 3)
        self.assertEqual(result.terminal.json_value(), "1")
        self.assertEqual(len(result.source_ids), 4)

    def test_temporal_formulas_do_not_mix_annual_and_quarterly_periods(self) -> None:
        facts = [
            fact("annual", "s1", "2023", "100"),
            fact("quarter", "s2", "2024-Q1", "125"),
        ]
        planner = RuleDagPlanner(
            load_formula_registry("builtin"),
            allowed_operations=["subtract", "abs"],
            min_depth=2,
            max_depth=2,
            min_sources=2,
            beam_width=10,
            max_graphs_per_entity=5,
        )
        self.assertEqual(planner.plan_entity(facts), [])

    def test_equivalent_period_spellings_are_not_treated_as_distinct_periods(self) -> None:
        facts = [
            fact("left", "s1", "Q1 2024", "100"),
            fact("right", "s2", "2024-Q1", "125"),
        ]
        planner = RuleDagPlanner(
            load_formula_registry("builtin"),
            allowed_operations=["subtract", "abs"],
            min_depth=2,
            max_depth=2,
            min_sources=2,
            beam_width=10,
            max_graphs_per_entity=5,
        )
        self.assertEqual(planner.plan_entity(facts), [])


if __name__ == "__main__":
    unittest.main()
