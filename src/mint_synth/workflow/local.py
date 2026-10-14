"""Execute an immutable synthesis plan on the current machine."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mint_synth.core.config_v2 import config_fingerprint, load_config
from mint_synth.core.provenance import auxiliary_fingerprints, package_fingerprint
from mint_synth.core.run_state import RunLayout, read_json, shard_complete
from mint_synth.workflow.planner import expected_stage_signature, run_status, verify_plan_fingerprint
from mint_synth.workflow.worker import run_worker


@contextmanager
def _run_lock(layout: RunLayout) -> Iterator[None]:
    path = layout.root / ".run.lock"
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"another run process is active: {layout.root}") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"another run process is active: {layout.root}") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _execute_stage(run_dir: Path, stage: str, shard_ids: list[int], workers: int) -> None:
    if workers == 1:
        for shard_id in shard_ids:
            run_worker(run_dir, stage, shard_id)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        remaining = iter(shard_ids)
        pending = {}
        for _ in range(min(workers, len(shard_ids))):
            shard_id = next(remaining)
            pending[pool.submit(run_worker, run_dir, stage, shard_id)] = shard_id
        while pending:
            finished = next(as_completed(pending))
            pending.pop(finished)
            finished.result()
            next_id = next(remaining, None)
            if next_id is not None:
                pending[pool.submit(run_worker, run_dir, stage, next_id)] = next_id


def run_plan(run_dir: str | Path, *, workers: int = 1) -> dict[str, Any]:
    """Run missing shards in plan order; completed shards remain immutable."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    layout = RunLayout(run_dir)
    with _run_lock(layout):
        plan = read_json(layout.plan)
        config = load_config(layout.config)
        if not verify_plan_fingerprint(plan):
            raise RuntimeError("plan fingerprint mismatch")
        if config_fingerprint(config) != plan.get("config_fingerprint"):
            raise RuntimeError("resolved configuration differs from the immutable plan")
        if package_fingerprint() != plan.get("code_fingerprint"):
            raise RuntimeError("running package differs from the code frozen in the plan")
        if auxiliary_fingerprints(config) != plan.get("auxiliary_fingerprints"):
            raise RuntimeError("formula, alias, seed, or evaluation file changed after planning")
        for stage in plan["stages"]:
            name = str(stage["name"])
            shard_ids = [
                shard_id
                for shard_id in range(int(stage["shard_count"]))
                if name == "export" or not shard_complete(
                    layout, name, shard_id, expected_stage_signature(plan, name, shard_id)
                )
            ]
            if not shard_ids:
                continue
            print(f"mint-synth: {name}: {len(shard_ids)} shard(s)", file=sys.stderr, flush=True)
            _execute_stage(layout.root, name, shard_ids, workers)
        return run_status(layout.root)
