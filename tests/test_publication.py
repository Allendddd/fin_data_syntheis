from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from mint_synth.run_state import RunLayout, atomic_json, read_json
from mint_synth.stages import _publish_export_markers


class PublicationTests(unittest.TestCase):
    def test_publication_recovery_uses_committed_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            layout = RunLayout(Path(temp) / "run")
            export = layout.shard("export", 0)
            dataset = export / "dataset" / "part-000000.parquet"
            dataset.parent.mkdir(parents=True)
            dataset.write_bytes(b"trusted")
            digest = hashlib.sha256(b"trusted").hexdigest()
            quality = export / "quality_summary.json"
            quality.write_bytes(b"quality")
            quality_digest = hashlib.sha256(b"quality").hexdigest()
            atomic_json(
                export / "artifact_manifest.json",
                {
                    "created_at": "2026-09-30T00:00:00+00:00",
                    "artifacts": [
                        {
                            "path": "dataset/part-000000.parquet",
                            "bytes": len(b"trusted"),
                            "sha256": digest,
                        },
                        {
                            "path": "quality_summary.json",
                            "bytes": len(b"quality"),
                            "sha256": quality_digest,
                        },
                    ]
                },
            )
            plan = {"run_id": "run-1", "run_fingerprint": "fingerprint"}
            _publish_export_markers(layout, plan, export, {"exported_tasks": 1})
            published = read_json(layout.root / "_PUBLISHED.json")
            self.assertEqual(published["artifacts"][0]["sha256"], digest)
            self.assertEqual(published["artifacts"][1]["sha256"], quality_digest)
            self.assertEqual(published["published_at"], "2026-09-30T00:00:00+00:00")

            _publish_export_markers(layout, plan, export, {"exported_tasks": 1})
            republished = read_json(layout.root / "_PUBLISHED.json")
            self.assertEqual(republished["published_at"], published["published_at"])

            dataset.write_bytes(b"untrust")  # same byte length: size-only validation would miss this
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                _publish_export_markers(layout, plan, export, {"exported_tasks": 1})


if __name__ == "__main__":
    unittest.main()
