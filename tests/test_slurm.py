from __future__ import annotations

import unittest
import tempfile
from subprocess import CompletedProcess
from pathlib import Path
from unittest.mock import patch

from mint_synth.run_state import RunLayout, atomic_json, read_json
from mint_synth.slurm import (
    SlurmError,
    _array_spec,
    _assert_no_active_jobs,
    _freeze_static_submission,
    _static_parallel_limit,
    _static_submission_files,
    _submit_script,
    _submission_lock,
    _write_shard_map,
    render_script,
    submit_run,
)


class SlurmTests(unittest.TestCase):
    def test_sparse_resume_array_is_compacted(self) -> None:
        self.assertEqual(_array_spec([0, 1, 2, 5, 7, 8], 16), "0-2,5,7-8%16")

    def test_script_uses_apptainer_and_worker_array_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = RunLayout(root / "run")
            layout.root.mkdir()
            config = {
                "run": {"work_dir": str(root / "work")},
                "output": {"root": str(root / "out")},
                "input": {"paths": [str(root / "data")]},
                "entity_resolution": {"alias_file": None},
                "quality": {"evaluation_file": None},
                "dag": {"formula_registry": "builtin"},
                "slurm": {
                    "max_parallel_tasks": 4,
                    "max_array_tasks": 1000,
                    "partition": "cpu",
                    "account": None,
                    "qos": None,
                    "apptainer_image": str(root / "image.sif"),
                    "bind_paths": [],
                    "extra_sbatch_args": [],
                    "stage_resources": {
                        "normalize": {
                            "cpus": 2, "memory": "4G", "time": "01:00:00", "gpus": 0,
                            "max_parallel_tasks": 2,
                        }
                    },
                },
            }
            script = render_script(
                layout,
                config,
                {"name": "normalize", "shard_count": 8, "dependencies": []},
                [0, 2, 3],
            ).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --array=0-2%2", script)
            self.assertIn("#SBATCH --export=ALL", script)
            self.assertIn("apptainer exec", script)
            self.assertIn("MINT_SHARD_IDS=(0 2 3)", script)
            self.assertIn("--shard-id ${MINT_SHARD_ID}", script)

    def test_generated_script_uses_defaults_when_stage_resources_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = RunLayout(root / "run")
            layout.root.mkdir()
            config = {
                "run": {"work_dir": str(root / "work")},
                "output": {"root": str(root / "out")},
                "input": {"paths": [str(root / "data")]},
                "entity_resolution": {"alias_file": None},
                "quality": {"evaluation_file": None},
                "dag": {"formula_registry": "builtin"},
                "slurm": {
                    "max_parallel_tasks": 4,
                    "partition": "cpu",
                    "account": None,
                    "qos": None,
                    "apptainer_image": str(root / "image.sif"),
                    "bind_paths": [],
                    "extra_sbatch_args": [],
                    "stage_resources": {},
                },
            }
            script = render_script(
                layout,
                config,
                {"name": "normalize", "shard_count": 2, "dependencies": []},
            ).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --cpus-per-task=4", script)
            self.assertIn("#SBATCH --mem=16G", script)
            self.assertIn("#SBATCH --time=04:00:00", script)
            self.assertIn("#SBATCH --array=0-1%4", script)

    def test_static_parallel_limit_and_atomic_shard_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / "facts.sbatch"
            script.write_text(
                "#!/usr/bin/env bash\n# MINT_MAX_PARALLEL_TASKS=7\n", encoding="utf-8"
            )
            self.assertEqual(_static_parallel_limit(script, 64), 7)
            shard_map = _write_shard_map(root / "maps" / "facts.txt", [9, 17, 23])
            self.assertEqual(shard_map.read_text(encoding="utf-8"), "9\n17\n23\n")

    def test_static_parallel_limit_rejects_a_malformed_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / "facts.sbatch"
            script.write_text("# MINT_MAX_PARALLEL_TASKS=many\n", encoding="utf-8")
            with self.assertRaisesRegex(SlurmError, "invalid MINT_MAX_PARALLEL_TASKS"):
                _static_parallel_limit(script, 64)

    def test_repository_has_static_script_for_every_stage(self) -> None:
        root = Path(__file__).resolve().parents[1]
        names = [
            "normalize",
            "id_validation",
            "dedup",
            "facts",
            "atomic",
            "entity_resolution",
            "repartition",
            "graph_planning",
            "query_realization",
            "validation",
            "export",
        ]
        runner, scripts, fingerprints = _static_submission_files(
            root / "slurm", [{"name": name} for name in names]
        )
        self.assertEqual(runner, root / "scripts" / "run_stage.sh")
        self.assertEqual(set(scripts), set(names))
        self.assertEqual(len(fingerprints["runner"]["sha256"]), 64)
        for name, script in scripts.items():
            self.assertIn(f'"${{MINT_RUNNER_SCRIPT:', script.read_text(encoding="utf-8"))
            self.assertGreaterEqual(_static_parallel_limit(script, 64), 1)

    def test_static_submission_files_are_frozen_per_submission(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        runner, scripts, fingerprints = _static_submission_files(
            repository / "slurm", [{"name": "normalize"}]
        )
        with tempfile.TemporaryDirectory() as temp:
            layout = RunLayout(Path(temp) / "run")
            layout.root.mkdir()
            frozen_runner, frozen_scripts, frozen_fingerprints = _freeze_static_submission(
                layout, "submission-1", runner, scripts, fingerprints
            )
            self.assertEqual(
                frozen_runner,
                layout.root / "slurm" / "submissions" / "submission-1" / "scripts" / "run_stage.sh",
            )
            self.assertEqual(
                frozen_runner.read_bytes(), (repository / "scripts" / "run_stage.sh").read_bytes()
            )
            self.assertEqual(
                frozen_scripts["normalize"].read_bytes(),
                (repository / "slurm" / "normalize.sbatch").read_bytes(),
            )
            self.assertEqual(
                frozen_fingerprints["stages"]["normalize"]["submitted_path"],
                str(frozen_scripts["normalize"]),
            )

    def test_resume_refuses_to_duplicate_active_jobs(self) -> None:
        manifest = {
            "submissions": [
                {"dry_run": False, "jobs": {"normalize": "123", "dedup": "124"}}
            ]
        }
        with patch(
            "mint_synth.slurm.subprocess.run",
            return_value=CompletedProcess([], 0, stdout="123|RUNNING\n", stderr=""),
        ):
            with self.assertRaisesRegex(SlurmError, "active"):
                _assert_no_active_jobs(manifest, force_resubmit=False)

    def test_resume_allows_terminal_jobs_absent_from_squeue(self) -> None:
        manifest = {"submissions": [{"dry_run": False, "jobs": {"normalize": "123"}}]}
        with patch(
            "mint_synth.slurm.subprocess.run",
            return_value=CompletedProcess([], 0, stdout="", stderr=""),
        ):
            _assert_no_active_jobs(manifest, force_resubmit=False)

    def test_resume_treats_requeued_job_as_active(self) -> None:
        manifest = {"submissions": [{"dry_run": False, "jobs": {"normalize": "123"}}]}
        with patch(
            "mint_synth.slurm.subprocess.run",
            return_value=CompletedProcess([], 0, stdout="123|REQUEUED\n", stderr=""),
        ):
            with self.assertRaisesRegex(SlurmError, "active"):
                _assert_no_active_jobs(manifest, force_resubmit=False)

    def test_submit_rejects_non_numeric_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch(
            "mint_synth.slurm.subprocess.run",
            return_value=CompletedProcess([], 0, stdout="warning-only\n", stderr=""),
        ):
            with self.assertRaisesRegex(SlurmError, "invalid job id"):
                _submit_script(Path(temp) / "job.sbatch", [], dry_run=False)

    def test_submit_passes_array_arguments_and_environment_to_sbatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch(
            "mint_synth.slurm.subprocess.run",
            return_value=CompletedProcess([], 0, stdout="321;cluster\n", stderr=""),
        ) as run:
            script = Path(temp) / "job.sbatch"
            job_id = _submit_script(
                script,
                ["123", "124"],
                dry_run=False,
                sbatch_args=("--array=0-2%2",),
                environment={"MINT_RUN_DIR": "/shared/run"},
            )
            self.assertEqual(job_id, "321")
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["sbatch", "--parsable", "--array=0-2%2"])
            self.assertIn("--dependency=afterok:123:124", command)
            self.assertEqual(run.call_args.kwargs["env"]["MINT_RUN_DIR"], "/shared/run")

    def test_static_submit_freezes_scripts_maps_shards_and_audits_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = RunLayout(root / "run")
            layout.root.mkdir()
            atomic_json(
                layout.plan,
                {
                    "run_id": "test-run",
                    "config_fingerprint": "config-fingerprint",
                    "code_fingerprint": "code-fingerprint",
                    "auxiliary_fingerprints": {},
                    "deployment_image_fingerprint": {"sha256": "image-fingerprint"},
                    "stages": [
                        {"name": "normalize", "shard_count": 3, "dependencies": []},
                        {"name": "dedup", "shard_count": 1, "dependencies": ["normalize"]},
                    ],
                },
            )
            atomic_json(layout.manifest, {"state": "PLANNED", "submissions": []})
            layout.config.write_text("version: 2\n", encoding="utf-8")

            controller = root / "controller"
            sbatch_dir = controller / "slurm"
            scripts_dir = controller / "scripts"
            sbatch_dir.mkdir(parents=True)
            scripts_dir.mkdir(parents=True)
            (scripts_dir / "run_stage.sh").write_text(
                "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
            )
            (sbatch_dir / "normalize.sbatch").write_text(
                "#!/usr/bin/env bash\n"
                "#SBATCH --cpus-per-task=2\n"
                "# MINT_MAX_PARALLEL_TASKS=2\n",
                encoding="utf-8",
            )
            (sbatch_dir / "dedup.sbatch").write_text(
                "#!/usr/bin/env bash\n"
                "#SBATCH --cpus-per-task=4\n"
                "# MINT_MAX_PARALLEL_TASKS=4\n",
                encoding="utf-8",
            )
            config = {
                "run": {"work_dir": str(root / "work")},
                "output": {"root": str(root / "output")},
                "input": {"paths": [str(root / "data")]},
                "entity_resolution": {"alias_file": None},
                "quality": {"evaluation_file": None},
                "dag": {"formula_registry": "builtin"},
                "query_realization": {"seed_file": None},
                "llm": {"endpoints": {}},
                "slurm": {
                    "apptainer_image": str(root / "mint-synth.sif"),
                    "bind_paths": [],
                    "max_parallel_tasks": 64,
                    "max_array_tasks": 2,
                },
            }
            with (
                patch("mint_synth.slurm.load_config", return_value=config),
                patch("mint_synth.slurm.verify_plan_fingerprint", return_value=True),
                patch("mint_synth.slurm.config_fingerprint", return_value="config-fingerprint"),
                patch("mint_synth.slurm.package_fingerprint", return_value="code-fingerprint"),
                patch("mint_synth.slurm.auxiliary_fingerprints", return_value={}),
                patch(
                    "mint_synth.slurm.deployment_image_fingerprint",
                    return_value={"sha256": "image-fingerprint"},
                ),
                patch(
                    "mint_synth.slurm._submit_script", side_effect=["981", "982", "983"]
                ) as submit,
                patch("mint_synth.slurm.run_status", return_value={"complete": False}),
            ):
                result = submit_run(layout.root, script_dir=sbatch_dir)

            self.assertEqual(result["submitter"], "static_sbatch")
            self.assertEqual(result["jobs"], {"normalize": ["981", "982"], "dedup": ["983"]})
            normalize_maps = [Path(path) for path in result["shard_maps"]["normalize"]]
            self.assertEqual(normalize_maps[0].read_text(encoding="utf-8"), "0\n1\n")
            self.assertEqual(normalize_maps[1].read_text(encoding="utf-8"), "2\n")
            shard_map = normalize_maps[0]
            submitted_script = Path(result["scripts"]["normalize"][0])
            self.assertTrue(submitted_script.is_relative_to(layout.root / "slurm" / "submissions"))
            call = submit.call_args_list[0]
            self.assertEqual(call.args[0], submitted_script)
            self.assertIn("--array=0-1%2", call.kwargs["sbatch_args"])
            self.assertEqual(call.kwargs["environment"]["MINT_SHARD_MAP"], str(shard_map))
            self.assertTrue(
                Path(call.kwargs["environment"]["MINT_RUNNER_SCRIPT"]).is_relative_to(
                    layout.root / "slurm" / "submissions"
                )
            )
            self.assertEqual(submit.call_args_list[1].args[1], [])
            self.assertIn("--array=0%2", submit.call_args_list[1].kwargs["sbatch_args"])
            self.assertEqual(submit.call_args_list[2].args[1], ["981", "982"])
            self.assertIn("--array=0%4", submit.call_args_list[2].kwargs["sbatch_args"])
            manifest = read_json(layout.manifest)
            audit = manifest["submissions"][0]["static_script_fingerprints"]
            self.assertEqual(audit["stages"]["normalize"]["submitted_path"], str(submitted_script))
            self.assertEqual(len(audit["stages"]["normalize"]["sha256"]), 64)

    def test_submission_lock_rejects_parallel_controller(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            layout = RunLayout(Path(temp) / "run")
            layout.root.mkdir()
            with _submission_lock(layout):
                with self.assertRaisesRegex(SlurmError, "run lock"):
                    with _submission_lock(layout):
                        self.fail("nested controller unexpectedly acquired the same run lock")


if __name__ == "__main__":
    unittest.main()
