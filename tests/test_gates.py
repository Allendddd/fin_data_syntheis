from __future__ import annotations

import unittest

from mint_synth.audit import audit_long
from mint_synth.extract import FactExtractor
from mint_synth.replay import answer_matches, replay_graph


class GateTests(unittest.TestCase):
    def test_fact_number_must_occur_in_quote(self) -> None:
        chunk = {
            "chunk_id": "c1",
            "source_id": "s1",
            "text": "Revenue was $100 million in 2024.",
            "locator": {"char_start": 0, "char_end": 34},
        }
        candidate = {
            "entity": "Issuer",
            "period": "2024",
            "metric": "revenue",
            "value_text": "$999 million",
            "value_num": "999",
            "unit": "USD",
            "scale": "million",
            "currency": "USD",
            "fact_type": "reported",
            "quote": chunk["text"],
        }
        fact, reason = FactExtractor._localize(chunk, candidate)
        self.assertIsNone(fact)
        self.assertIn("not explicitly present", reason)

    def test_answer_requires_matching_financial_dimensions(self) -> None:
        facts = {
            "f": {
                "fact_id": "f",
                "source_id": "s",
                "value_num": "100",
                "unit": "USD",
                "scale": "million",
                "currency": "USD",
            }
        }
        result = replay_graph(
            {"nodes": [{"id": "f", "kind": "fact"}], "terminal_id": "f"}, facts
        )
        self.assertFalse(answer_matches({"value": "100", "unit": "USD"}, result))

    def test_long_query_cannot_reveal_hidden_answer(self) -> None:
        facts = {
            "f1": {
                "fact_id": "f1",
                "source_id": "s1",
                "chunk_id": "c1",
                "period": "2024",
                "value_num": "100",
                "value_text": "$100 million",
                "unit": "USD",
                "scale": "million",
                "currency": "USD",
                "quote": "A reported $100 million in 2024.",
                "locator": {"char_start": 0, "char_end": 32},
            },
            "f2": {
                "fact_id": "f2",
                "source_id": "s2",
                "chunk_id": "c2",
                "period": "2024",
                "value_num": "60",
                "value_text": "$60 million",
                "unit": "USD",
                "scale": "million",
                "currency": "USD",
                "quote": "B reported $60 million in 2024.",
                "locator": {"char_start": 0, "char_end": 31},
            },
        }
        task = {
            "task_id": "long_x",
            "question": "Which disclosures support a combined value of $160 million in 2024?",
            "evidence_ids": ["f1", "f2"],
            "evidence_graph": {
                "nodes": [
                    {"id": "a", "kind": "fact", "fact_id": "f1"},
                    {"id": "b", "kind": "fact", "fact_id": "f2"},
                    {"id": "total", "kind": "operation", "op": "add", "inputs": ["a", "b"]},
                ],
                "terminal_id": "total",
            },
            "answer": {
                "value": "160",
                "unit": "USD",
                "scale": "million",
                "currency": "USD",
            },
            "task_spec": {"time_sensitive": False},
            "semantic_audit": {
                "answer_unique": True,
                "fully_supported": True,
                "financially_valid": True,
                "non_exposing": True,
                "seed_not_copied": True,
            },
        }
        result = audit_long(task, facts, min_depth=1, min_sources=2)
        self.assertFalse(result.accepted)
        self.assertTrue(any("hidden numeric" in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
