from __future__ import annotations

import unittest

from mint_synth.extract import FactExtractor


class ExtractV2Tests(unittest.TestCase):
    def test_oversized_entity_is_rejected_before_entity_resolution(self) -> None:
        chunk = {
            "chunk_id": "chunk", "source_id": "source", "text": "Revenue was 10.",
            "locator": {"char_start": 0, "char_end": 15},
        }
        candidate = {
            "entity": "A" * 17, "period": "2024", "metric": "revenue",
            "value_text": "10", "value_num": "10", "unit": "USD", "scale": "million",
            "currency": "USD", "fact_type": "reported", "quote": "Revenue was 10.",
        }
        fact, reason = FactExtractor._localize(chunk, candidate, max_entity_chars=16)
        self.assertIsNone(fact)
        self.assertIn("max_entity_chars", reason)

    def test_same_span_can_represent_distinct_entities_without_fact_id_collision(self) -> None:
        quote = "Alpha and Beta each reported revenue of $100 million in 2024."
        chunk = {
            "source_id": "source",
            "chunk_id": "chunk",
            "text": quote,
            "locator": {"char_start": 0, "char_end": len(quote)},
        }
        base = {
            "period": "2024",
            "metric": "revenue",
            "value_text": "$100 million",
            "value_num": "100",
            "unit": "USD",
            "scale": "million",
            "currency": "USD",
            "fact_type": "reported",
            "quote": quote,
        }
        alpha, alpha_error = FactExtractor._localize(chunk, {**base, "entity": "Alpha"})
        beta, beta_error = FactExtractor._localize(chunk, {**base, "entity": "Beta"})
        self.assertEqual(alpha_error, "")
        self.assertEqual(beta_error, "")
        self.assertNotEqual(alpha["fact_id"], beta["fact_id"])


if __name__ == "__main__":
    unittest.main()
