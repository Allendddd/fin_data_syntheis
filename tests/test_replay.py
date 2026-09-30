from __future__ import annotations

import unittest

from mint_synth.replay import ReplayError, answer_matches, replay_graph


def fact(fact_id: str, value: str, source: str = "s1", unit: str = "USD") -> dict:
    return {
        "fact_id": fact_id,
        "source_id": source,
        "value_num": value,
        "value_text": value,
        "unit": unit,
        "scale": "million",
        "currency": "USD",
    }


class ReplayTests(unittest.TestCase):
    def test_comparison_rejects_incompatible_financial_dimensions(self) -> None:
        facts = {
            "usd": {
                "fact_id": "usd", "source_id": "s1", "value_num": "100",
                "unit": "USD", "scale": "million", "currency": "USD",
            },
            "eur": {
                "fact_id": "eur", "source_id": "s2", "value_num": "100",
                "unit": "EUR", "scale": "million", "currency": "EUR",
            },
        }
        graph = {
            "nodes": [
                {"id": "left", "kind": "fact", "fact_id": "usd"},
                {"id": "right", "kind": "fact", "fact_id": "eur"},
                {"id": "same", "kind": "operation", "op": "equals", "inputs": ["left", "right"]},
            ],
            "terminal_id": "same",
        }
        with self.assertRaisesRegex(ReplayError, "incompatible"):
            replay_graph(graph, facts)

    def test_zero_over_zero_is_a_recoverable_replay_error(self) -> None:
        facts = {"a": fact("a", "0"), "b": fact("b", "0", source="s2")}
        graph = {
            "nodes": [
                {"id": "a", "kind": "fact"},
                {"id": "b", "kind": "fact"},
                {"id": "ratio", "kind": "operation", "op": "ratio", "inputs": ["a", "b"]},
            ],
            "terminal_id": "ratio",
        }
        with self.assertRaisesRegex(ReplayError, "invalid decimal arithmetic"):
            replay_graph(graph, facts)

    def test_rejects_nodes_disconnected_from_terminal(self) -> None:
        facts = {"a": fact("a", "1"), "b": fact("b", "2", source="s2")}
        graph = {
            "nodes": [
                {"id": "a", "kind": "fact"},
                {"id": "b", "kind": "fact"},
                {"id": "out", "kind": "operation", "op": "identity", "inputs": ["a"]},
            ],
            "terminal_id": "out",
        }
        with self.assertRaisesRegex(ReplayError, "disconnected"):
            replay_graph(graph, facts)

    def test_rejects_unary_aggregate_depth_padding(self) -> None:
        facts = {"a": fact("a", "1")}
        graph = {
            "nodes": [
                {"id": "a", "kind": "fact"},
                {"id": "fake", "kind": "operation", "op": "average", "inputs": ["a"]},
            ],
            "terminal_id": "fake",
        }
        with self.assertRaisesRegex(ReplayError, "at least two"):
            replay_graph(graph, facts)

    def test_replays_graph_and_tracks_ancestors(self) -> None:
        facts = {"a": fact("a", "120", "s1"), "b": fact("b", "100", "s2")}
        graph = {
            "nodes": [
                {"id": "a_node", "kind": "fact", "fact_id": "a"},
                {"id": "b_node", "kind": "fact", "fact_id": "b"},
                {"id": "change", "kind": "operation", "op": "percent_change", "inputs": ["a_node", "b_node"]},
                {"id": "magnitude", "kind": "operation", "op": "abs", "inputs": ["change"]},
            ],
            "terminal_id": "magnitude",
        }
        result = replay_graph(graph, facts)
        self.assertEqual(result.terminal.json_value(), "20")
        self.assertEqual(result.depth, 2)
        self.assertEqual(result.source_ids, frozenset({"s1", "s2"}))
        self.assertTrue(answer_matches({"value": 20, "unit": "percent"}, result))

    def test_rejects_incompatible_addition(self) -> None:
        facts = {"a": fact("a", "2", unit="USD"), "b": fact("b", "3", unit="shares")}
        graph = {
            "nodes": [
                {"id": "a", "kind": "fact"},
                {"id": "b", "kind": "fact"},
                {"id": "c", "kind": "operation", "op": "add", "inputs": ["a", "b"]},
            ],
            "terminal_id": "c",
        }
        with self.assertRaisesRegex(ReplayError, "incompatible"):
            replay_graph(graph, facts)

    def test_rejects_cycle(self) -> None:
        graph = {
            "nodes": [
                {"id": "a", "kind": "operation", "op": "abs", "inputs": ["b"]},
                {"id": "b", "kind": "operation", "op": "abs", "inputs": ["a"]},
            ],
            "terminal_id": "a",
        }
        with self.assertRaisesRegex(ReplayError, "cycle"):
            replay_graph(graph, {})


if __name__ == "__main__":
    unittest.main()
