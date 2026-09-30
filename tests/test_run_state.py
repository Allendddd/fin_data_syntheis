from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mint_synth.run_state import RunLayout, ShardTransaction, quarantine_incomplete_shard, shard_complete


class RunStateTests(unittest.TestCase):
    def test_atomic_shard_is_immutable_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            layout = RunLayout(temp)
            with ShardTransaction(layout, "normalize", 0, "sig-a") as transaction:
                (transaction.path / "data.txt").write_text("complete", encoding="utf-8")
                transaction.commit({"rows": 1})
            self.assertTrue(shard_complete(layout, "normalize", 0, "sig-a"))
            self.assertFalse(shard_complete(layout, "normalize", 0, "sig-b"))
            with ShardTransaction(layout, "normalize", 0, "sig-a") as transaction:
                transaction.commit({"rows": 1})
            self.assertEqual((layout.shard("normalize", 0) / "data.txt").read_text(encoding="utf-8"), "complete")
            (layout.shard("normalize", 0) / "data.txt").write_text("damaged", encoding="utf-8")
            self.assertFalse(shard_complete(layout, "normalize", 0, "sig-a"))
            quarantined = quarantine_incomplete_shard(layout, "normalize", 0)
            self.assertIsNotNone(quarantined)
            self.assertFalse(layout.shard("normalize", 0).exists())
            self.assertTrue((quarantined / "data.txt").is_file())


if __name__ == "__main__":
    unittest.main()
