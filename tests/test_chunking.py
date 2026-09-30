from __future__ import annotations

import unittest

from mint_synth.chunking import semantic_chunks


class ChunkingTests(unittest.TestCase):
    def test_chunks_preserve_exact_offsets(self) -> None:
        text = ("第一段包含一组需要保留精确位置的金融文本。" * 8) + "\n\n" + (
            "Second paragraph has reported figures and semantic boundaries. " * 8
        )
        source = {
            "source_id": "src_test",
            "text": text,
            "metadata": {},
            "locator": {"jsonl_line": 1, "text_field": "text"},
        }
        chunks = semantic_chunks(source, max_chars=220, overlap_chars=30, min_chars=80)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            start = chunk["locator"]["char_start"]
            end = chunk["locator"]["char_end"]
            self.assertEqual(text[start:end], chunk["text"])
            self.assertLess(start, end)

    def test_invalid_overlap_is_rejected(self) -> None:
        source = {"source_id": "s", "text": "x" * 300, "metadata": {}, "locator": {}}
        with self.assertRaises(ValueError):
            semantic_chunks(source, max_chars=200, overlap_chars=200, min_chars=50)


if __name__ == "__main__":
    unittest.main()
