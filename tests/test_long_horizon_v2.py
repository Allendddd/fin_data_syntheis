from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mint_synth.long_horizon import load_evaluation_questions


class LongHorizonV2Tests(unittest.TestCase):
    def test_evaluation_file_is_strict_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evaluation.jsonl"
            path.write_text(
                json.dumps({"question": "Q"}) + "\n" + json.dumps({"question": "Q"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_evaluation_questions(str(path), max_questions=1), ["Q"])
            path.write_text(json.dumps({"answer": "missing question"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires"):
                load_evaluation_questions(str(path), max_questions=10)


if __name__ == "__main__":
    unittest.main()
