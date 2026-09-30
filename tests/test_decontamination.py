from __future__ import annotations

import unittest

from mint_synth.decontamination import QuestionIndex


class DecontaminationTests(unittest.TestCase):
    def test_inverted_index_keeps_exact_jaccard_decision(self) -> None:
        index = QuestionIndex(
            [
                "What was Alpha Corp revenue in 2024?",
                "Calculate Beta Ltd operating margin.",
            ]
        )
        self.assertTrue(index.overlaps("What was Alpha Corp revenue in 2024?", 0.82))
        self.assertFalse(index.overlaps("How much cash did Gamma report?", 0.82))
        self.assertTrue(index.overlaps("unrelated", 0.0))


if __name__ == "__main__":
    unittest.main()
