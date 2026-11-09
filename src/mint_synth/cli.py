from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from mint_synth.core.config_v2 import ConfigError, config_fingerprint, load_config
from mint_synth.core.run_state import RunLayout, read_json
from mint_synth.workflow.local import run_plan
from mint_synth.workflow.planner import create_plan, migrate_v1, migrate_v2, run_status, validate_run_structure
from mint_synth.workflow.worker import run_worker


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mint-synth",
        description="Financial-data synthesis on a local machine or through an external scheduler",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="Configuration operations")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_commands.add_parser("validate", help="Validate and resolve a v2/v3 YAML configuration")
    config_validate.add_argument("--config", required=True)

    plan = commands.add_parser("plan", help="Freeze the input catalog and create an immutable run plan")
    plan.add_argument("--config", required=True)

    run = commands.add_parser("run", help="Execute or resume a synthesis plan on this machine")
    run_source = run.add_mutually_exclusive_group(required=True)
    run_source.add_argument("--config", help="Create a new immutable plan from this YAML")
    run_source.add_argument("--run-dir", help="Continue an existing immutable plan")
    run.add_argument("--workers", type=int, default=1, help="Maximum concurrent shard processes (default: 1)")

    status = commands.add_parser("status", help="Report shard and publication status")
    status.add_argument("--run-dir", required=True)

    validate = commands.add_parser("validate", help="Validate run structure, signatures, and completion")
    validate.add_argument("--run-dir", required=True)

    export = commands.add_parser("export", help="Run the final quality gate and publish configured exports")
    export.add_argument("--run-dir", required=True)

    migrate = commands.add_parser("migrate-v1", help="Create a v2 YAML from a v1 input or output directory")
    migrate.add_argument("--input", required=True)
    migrate.add_argument("--output", required=True)

    migrate_v2_command = commands.add_parser("migrate-v2", help="Create a v3 data-synthesis YAML from a v2 configuration")
    migrate_v2_command.add_argument("--input", required=True)
    migrate_v2_command.add_argument("--output", required=True)
    migrate_v2_command.add_argument("--cutoff", help="Override or provide the snapshot cutoff date (YYYY-MM-DD)")

    worker = commands.add_parser("worker", help="Internal single-shard entry point")
    worker.add_argument("--run-dir", required=True)
    worker.add_argument("--stage", required=True)
    worker.add_argument("--shard-id", required=True, type=int)

    trace_proxy = commands.add_parser("trace-proxy", help="Record model API calls through a local HTTP proxy")
    trace_proxy.add_argument("--upstream", required=True, help="vLLM/SGLang HTTP(S) origin, without /v1")
    trace_proxy.add_argument("--output-dir", required=True, help="Private directory for raw requests and responses")
    trace_proxy.add_argument("--run-id", required=True, help="Agent run identifier used to join framework events")
    trace_proxy.add_argument("--port", type=int, default=8765, help="Loopback listen port (default: 8765)")
    trace_proxy.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)

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
        if args.command == "run":
            if args.workers < 1:
                raise ValueError("workers must be at least 1")
            layout = create_plan(args.config) if args.config else RunLayout(args.run_dir)
            if args.config:
                print(f"mint-synth: planned run: {layout.root}", file=sys.stderr, flush=True)
            _json(run_plan(layout.root, workers=args.workers))
            return 0
        if args.command == "status":
            _json(run_status(args.run_dir))
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
        if args.command == "trace-proxy":
            from mint_synth.trace_proxy import TraceProxyConfig, serve_trace_proxy

            serve_trace_proxy(TraceProxyConfig(
                upstream=args.upstream,
                output_dir=args.output_dir,
                run_id=args.run_id,
                port=args.port,
                max_request_bytes=args.max_request_bytes,
            ))
            return 0
        parser.error(f"unknown command: {args.command}")
    except (ConfigError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"mint-synth: error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
