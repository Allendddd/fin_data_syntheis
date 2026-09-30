from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mint_synth.jsonl import load_sources


class JsonlTests(unittest.TestCase):
    def test_content_ids_are_stable_without_explicit_id_and_duplicates_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            rows = [{"text": "alpha"}, {"text": "beta"}, {"text": "alpha"}]
            first.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            second.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows[:2])), encoding="utf-8")
            a = load_sources(first)
            b = load_sources(second)
            self.assertEqual(len(a), 2)
            self.assertEqual({row["source_id"] for row in a}, {row["source_id"] for row in b})


if __name__ == "__main__":
    unittest.main()
