from __future__ import annotations

import unittest

from mint_synth.config_v2 import ConfigError, materialize_defaults, validate_cross_fields


class ConfigV2Tests(unittest.TestCase):
    def test_defaults_are_materialized(self) -> None:
        value = materialize_defaults(
            {
                "version": 2,
                "run": {"name": "test", "work_dir": "work"},
                "input": {"paths": ["data"], "text_field": "text"},
                "output": {"root": "out"},
                "slurm": {"apptainer_image": "image.sif"},
                "llm": {"endpoints": {}, "roles": {}},
            }
        )
        self.assertEqual(value["reader"]["dedup_buckets"], 4096)
        self.assertEqual(value["output"]["compression"], "zstd")
        self.assertEqual(value["dag"]["min_sources"], 2)
        self.assertEqual(value["reader"]["external_id_partitions"], 256)
        self.assertEqual(value["dag"]["max_candidate_groups_per_formula"], 4096)
        self.assertEqual(value["llm"]["cache"]["backend"], "sqlite")
        self.assertEqual(value["quality"]["min_validated_tasks"], 1)
        self.assertEqual(value["entity_resolution"]["llm_max_surfaces"], 100)
        self.assertEqual(value["extraction"]["max_entity_chars"], 256)
        self.assertEqual(value["slurm"]["partition"], "compute")
        self.assertEqual(value["slurm"]["stage_resources"], {})

    def test_cross_field_checks_chunk_overlap(self) -> None:
        with self.assertRaisesRegex(ConfigError, "overlap"):
            validate_cross_fields(
                {
                    "chunking": {"max_chars": 1000, "overlap_chars": 1000, "min_chars": 10},
                    "dag": {"min_depth": 2, "max_depth": 4, "min_sources": 2},
                    "llm": {"endpoints": {}, "roles": {}},
                    "stages": {},
                    "slurm": {"stage_resources": {"normalize": {}, "dedup": {}}},
                }
            )

    def test_shuffle_partitions_must_evenly_divide_dedup_buckets(self) -> None:
        value = materialize_defaults(
            {
                "version": 2,
                "run": {"name": "test", "work_dir": "work"},
                "input": {"paths": ["data"], "text_field": "text"},
                "output": {"root": "out"},
                "slurm": {"partition": "cpu", "apptainer_image": "image.sif", "stage_resources": {}},
                "llm": {"endpoints": {}, "roles": {}},
                "reader": {"dedup_buckets": 10, "shuffle_partitions": 6},
            }
        )
        with self.assertRaisesRegex(ConfigError, "divisible"):
            validate_cross_fields(value)


if __name__ == "__main__":
    unittest.main()
