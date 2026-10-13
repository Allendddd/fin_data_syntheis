from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .config_v2 import ConfigError, config_fingerprint, load_config
from .planner import create_plan, migrate_v1, migrate_v2, run_status, validate_run_structure
from .run_state import RunLayout, read_json
from .slurm import SlurmError, submit_run
from .stages import run_worker


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mint-synth",
        description="Production financial-data synthesis for SLURM and OpenAI-compatible inference services",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="Configuration operations")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_commands.add_parser("validate", help="Validate and resolve a v2 YAML configuration")
    config_validate.add_argument("--config", required=True)

    plan = commands.add_parser("plan", help="Freeze the input catalog and create an immutable run plan")
    plan.add_argument("--config", required=True)

    submit = commands.add_parser("submit", help="Plan a new run and submit its SLURM dependency graph")
    submit_source = submit.add_mutually_exclusive_group(required=True)
    submit_source.add_argument("--config", help="Plan a new run from this YAML before submission")
    submit_source.add_argument("--run-dir", help="Submit an already planned immutable run")
    submit.add_argument("--dry-run", action="store_true", help="Render scripts without calling sbatch")
    submit.add_argument(
        "--sbatch-dir",
        help="Use editable <stage>.sbatch files from this directory and its sibling scripts/run_stage.sh",
    )

    status = commands.add_parser("status", help="Report shard and publication status")
    status.add_argument("--run-dir", required=True)

    resume = commands.add_parser("resume", help="Submit only incomplete shards")
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--dry-run", action="store_true")
    resume.add_argument(
        "--sbatch-dir",
        help="Use editable <stage>.sbatch files from this directory and its sibling scripts/run_stage.sh",
    )
    resume.add_argument(
        "--force-resubmit",
        action="store_true",
        help="Resubmit incomplete shards without checking prior SLURM jobs (may duplicate active work)",
    )

    validate = commands.add_parser("validate", help="Validate run structure, signatures, and completion")
    validate.add_argument("--run-dir", required=True)

    export = commands.add_parser("export", help="Run the final quality gate and publish configured exports")
    export.add_argument("--run-dir", required=True)

    migrate = commands.add_parser("migrate-v1", help="Create a v2 YAML from a v1 input or output directory")
    migrate.add_argument("--input", required=True)
    migrate.add_argument("--output", required=True)

    migrate_v2_command = commands.add_parser("migrate-v2", help="Create a v3 SFT YAML from a v2 configuration")
    migrate_v2_command.add_argument("--input", required=True)
    migrate_v2_command.add_argument("--output", required=True)
    migrate_v2_command.add_argument("--cutoff", help="Override or provide the snapshot cutoff date (YYYY-MM-DD)")

    worker = commands.add_parser("worker", help="Internal entry point used by SLURM job arrays")
    worker.add_argument("--run-dir", required=True)
    worker.add_argument("--stage", required=True)
    worker.add_argument("--shard-id", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            resolved = load_config(args.config)
            _json(
                {
                    "valid": True,
                    "version": resolved["version"],
                    "fingerprint": config_fingerprint(resolved),
                    "input_paths": resolved["input"]["paths"],
                    "output_root": resolved["output"]["root"],
                }
            )
            return 0
        if args.command == "plan":
            layout = create_plan(args.config)
            _json({"run_dir": str(layout.root), "plan": read_json(layout.plan)})
            return 0
        if args.command == "submit":
            layout = create_plan(args.config) if args.config else RunLayout(args.run_dir)
            _json(submit_run(layout.root, dry_run=args.dry_run, script_dir=args.sbatch_dir))
            return 0
        if args.command == "status":
            _json(run_status(args.run_dir))
            return 0
        if args.command == "resume":
            _json(
                submit_run(
                    args.run_dir,
                    resume=True,
                    dry_run=args.dry_run,
                    force_resubmit=args.force_resubmit,
                    script_dir=args.sbatch_dir,
                )
            )
            return 0
        if args.command == "validate":
            report = validate_run_structure(args.run_dir)
            _json(report)
            return 0 if report["valid"] else 1
        if args.command == "export":
            _json(run_worker(args.run_dir, "export", 0))
            return 0
        if args.command == "migrate-v1":
            destination = migrate_v1(args.input, args.output)
            _json({"migrated": True, "config": str(destination)})
            return 0
        if args.command == "migrate-v2":
            destination = migrate_v2(args.input, args.output, cutoff=args.cutoff)
            _json({"migrated": True, "config": str(destination)})
            return 0
        if args.command == "worker":
            _json(run_worker(args.run_dir, args.stage, args.shard_id))
            return 0
        parser.error(f"unknown command: {args.command}")
    except (ConfigError, SlurmError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"mint-synth: error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
