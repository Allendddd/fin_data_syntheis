from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mint_synth.catalog import CatalogError, InputSegment, build_catalog, discover_files, pack_segments
from mint_synth.records import iter_segment_objects, normalize_object


class CatalogTests(unittest.TestCase):
    def test_empty_segment_set_is_rejected_before_planning(self) -> None:
        with self.assertRaisesRegex(CatalogError, "no readable segments"):
            pack_segments([], 1024)

    def test_recursive_discovery_and_byte_ranges_are_lossless(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested = root / "nested"
            nested.mkdir()
            source = nested / "input.jsonl"
            rows = [{"text": f"row {index}"} for index in range(20)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (nested / "ignored.txt").write_text("x", encoding="utf-8")
            config = {
                "run": {"fingerprint_mode": "metadata"},
                "input": {
                    "paths": [str(root)], "recursive": True, "include": ["**/*.jsonl"],
                    "exclude": [], "follow_symlinks": False, "text_field": "text",
                    "id_field": None, "metadata_fields": [],
                },
                "reader": {"jsonl_byte_shard_size": 40},
            }
            self.assertEqual(discover_files(config["input"]), [source.resolve()])
            segments = build_catalog(config)
            self.assertGreater(len(segments), 1)
            recovered = []
            for segment in segments:
                recovered.extend(
                    value
                    for value, _, error in iter_segment_objects(segment, config["input"], batch_rows=8)
                    if error is None
                )
            self.assertEqual(recovered, rows)
            packed = pack_segments(segments, 100)
            self.assertEqual(sum(len(value) for value in packed), len(segments))

    def test_jsonl_bom_and_invalid_complex_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bom.jsonl"
            path.write_text('\ufeff{"id":{"nested":1},"text":"hello"}\n', encoding="utf-8")
            stat = path.stat()
            segment = InputSegment(
                "seg", "file", str(path), "jsonl", 0, 0, stat.st_size, None,
                stat.st_size, None, stat.st_size, stat.st_mtime_ns, "fingerprint",
            )
            [(value, locator, error)] = list(
                iter_segment_objects(
                    segment,
                    {"text_field": "text", "id_field": "id", "metadata_fields": []},
                    batch_rows=8,
                )
            )
            self.assertIsNone(error)
            row, rejection = normalize_object(
                value,
                error,
                segment=segment,
                locator=locator,
                input_config={"text_field": "text", "id_field": "id", "metadata_fields": []},
                dedup_buckets=16,
                shuffle_partitions=4,
            )
            self.assertIsNone(row)
            self.assertEqual(rejection["error_code"], "invalid_external_id")

            row, rejection = normalize_object(
                {"id": "   ", "text": "hello"},
                None,
                segment=segment,
                locator=locator,
                input_config={"text_field": "text", "id_field": "id", "metadata_fields": []},
                dedup_buckets=16,
                shuffle_partitions=4,
            )
            self.assertIsNone(row)
            self.assertEqual(rejection["error_code"], "missing_external_id")

    def test_external_id_partition_is_independent_of_content_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl"
            path.write_text('{"id":"shared","text":"first"}\n', encoding="utf-8")
            stat = path.stat()
            segment = InputSegment(
                "seg", "file", str(path), "jsonl", 0, 0, stat.st_size, None,
                stat.st_size, 1, stat.st_size, stat.st_mtime_ns, "fingerprint",
            )
            common = {
                "segment": segment,
                "locator": {"byte_start": 0, "byte_end": stat.st_size},
                "input_config": {"text_field": "text", "id_field": "id", "metadata_fields": []},
                "dedup_buckets": 16,
                "shuffle_partitions": 4,
                "external_id_partitions": 7,
            }
            first, _ = normalize_object({"id": "shared", "text": "first"}, None, **common)
            second, _ = normalize_object({"id": "shared", "text": "different"}, None, **common)
            self.assertEqual(first["external_id_partition"], second["external_id_partition"])
            self.assertGreaterEqual(first["external_id_partition"], 0)
            self.assertLess(first["external_id_partition"], 7)

    def test_oversized_jsonl_line_is_bounded_and_next_record_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.jsonl"
            source.write_text(
                json.dumps({"text": "x" * 500}) + "\n" + json.dumps({"text": "ok"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "run": {"fingerprint_mode": "metadata"},
                "input": {
                    "paths": [str(source)],
                    "recursive": True,
                    "include": ["**/*.jsonl"],
                    "exclude": [],
                    "follow_symlinks": False,
                    "text_field": "text",
                    "id_field": None,
                    "metadata_fields": [],
                    "max_jsonl_line_bytes": 64,
                },
                "reader": {"jsonl_byte_shard_size": 40},
            }
            values = []
            errors = []
            for segment in build_catalog(config):
                for value, _, error in iter_segment_objects(segment, config["input"], batch_rows=8):
                    if error:
                        errors.append(error)
                    else:
                        values.append(value)
            self.assertEqual(values, [{"text": "ok"}])
            self.assertEqual(len(errors), 1)
            self.assertIn("max_jsonl_line_bytes", errors[0])

    def test_seed_and_evaluation_jsonl_are_not_ingested_as_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus.jsonl"
            seed = root / "seed.jsonl"
            evaluation = root / "evaluation.jsonl"
            for path in (corpus, seed, evaluation):
                path.write_text('{"text":"row"}\n', encoding="utf-8")
            config = {
                "run": {"fingerprint_mode": "metadata", "work_dir": str(root / "work")},
                "input": {
                    "paths": [str(root)], "recursive": True, "include": ["**/*.jsonl"],
                    "exclude": [], "follow_symlinks": False, "text_field": "text",
                    "id_field": None, "metadata_fields": [],
                },
                "output": {"root": str(root / "output")},
                "query_realization": {"seed_file": str(seed)},
                "quality": {"evaluation_file": str(evaluation)},
                "entity_resolution": {"alias_file": None},
                "dag": {"formula_registry": "builtin"},
                "reader": {"jsonl_byte_shard_size": 1024},
            }
            segments = build_catalog(config)
            self.assertEqual({Path(segment.source_uri) for segment in segments}, {corpus.resolve()})


if __name__ == "__main__":
    unittest.main()
