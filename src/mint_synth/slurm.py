from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config_v2 import config_fingerprint, load_config
from .planner import expected_stage_signature, run_status, verify_plan_fingerprint
from .provenance import auxiliary_fingerprints, deployment_image_fingerprint, package_fingerprint, sha256_file
from .run_state import RunLayout, atomic_json, read_json, shard_complete, utc_now


class SlurmError(RuntimeError):
    pass


_STATIC_PARALLEL_RE = re.compile(
    r"^[ \t]*#[ \t]*MINT_MAX_PARALLEL_TASKS[ \t]*=[ \t]*([0-9]+)[ \t]*$",
)
_STATIC_PARALLEL_DECL_RE = re.compile(r"^[ \t]*#[ \t]*MINT_MAX_PARALLEL_TASKS\b")


def _array_spec(ids: Iterable[int], max_parallel: int) -> str:
    values = sorted(set(ids))
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return f"{','.join(ranges)}%{max_parallel}"


def _bind_paths(layout: RunLayout, config: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(layout.root),
            str(config["run"]["work_dir"]),
            str(config["output"]["root"]),
            *(str(path) for path in config["input"]["paths"]),
            *(str(path) for path in config["slurm"].get("bind_paths", [])),
            *(
                str(path)
                for path in (
                    config.get("entity_resolution", {}).get("alias_file"),
                    config.get("quality", {}).get("evaluation_file"),
                    config.get("dag", {}).get("formula_registry"),
                    config.get("query_realization", {}).get("seed_file"),
                )
                if path and path != "builtin"
            ),
        }
    )


def _static_parallel_limit(script: str | Path, fallback: int) -> int:
    if fallback < 1:
        raise SlurmError("SLURM array concurrency must be at least 1")
    path = Path(script)
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SlurmError(f"cannot read UTF-8 sbatch script {path}: {exc}") from exc
    declarations = [
        line for line in content.splitlines() if _STATIC_PARALLEL_DECL_RE.match(line)
    ]
    if len(declarations) > 1:
        raise SlurmError(f"sbatch script contains multiple MINT_MAX_PARALLEL_TASKS declarations: {path}")
    if not declarations:
        return fallback
    match = _STATIC_PARALLEL_RE.fullmatch(declarations[0])
    if match is None:
        raise SlurmError(
            f"invalid MINT_MAX_PARALLEL_TASKS declaration in {path}; expected '# MINT_MAX_PARALLEL_TASKS=<integer>'"
        )
    value = int(match.group(1))
    if value < 1:
        raise SlurmError(f"MINT_MAX_PARALLEL_TASKS must be at least 1 in {path}")
    return value


def _write_shard_map(path: str | Path, shard_ids: Iterable[int]) -> Path:
    destination = Path(path)
    values = list(shard_ids)
    if not values or any(not isinstance(value, int) or value < 0 for value in values):
        raise SlurmError("shard map must contain one or more non-negative integer shard IDs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(f"{value}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _atomic_copy_verified(source: Path, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output_stream, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            raise SlurmError(f"static submission file changed while it was being frozen: {source}")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _static_submission_files(
    script_dir: str | Path,
    stages: Iterable[Mapping[str, Any]],
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    directory = Path(script_dir).expanduser().resolve()
    if not directory.is_dir():
        raise SlurmError(f"sbatch script directory does not exist: {directory}")
    runner = directory.parent / "scripts" / "run_stage.sh"
    if not runner.is_file():
        raise SlurmError(
            f"static sbatch mode requires the sibling runner script scripts/run_stage.sh: {runner}"
        )
    scripts: dict[str, Path] = {}
    stage_fingerprints: dict[str, Any] = {}
    for stage in stages:
        name = str(stage["name"])
        script = directory / f"{name}.sbatch"
        if not script.is_file():
            raise SlurmError(f"missing sbatch script for stage {name}: {script}")
        scripts[name] = script
        stage_fingerprints[name] = {"path": str(script), "sha256": sha256_file(script)}
    fingerprints = {
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "stages": stage_fingerprints,
    }
    return runner, scripts, fingerprints


def _freeze_static_submission(
    layout: RunLayout,
    submission_id: str,
    runner: Path,
    scripts: Mapping[str, Path],
    fingerprints: Mapping[str, Any],
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    root = layout.root / "slurm" / "submissions" / submission_id
    frozen_runner = _atomic_copy_verified(
        runner,
        root / "scripts" / "run_stage.sh",
        str(fingerprints["runner"]["sha256"]),
    )
    frozen_scripts: dict[str, Path] = {}
    stage_fingerprints: dict[str, Any] = {}
    for name, source in scripts.items():
        digest = str(fingerprints["stages"][name]["sha256"])
        destination = _atomic_copy_verified(source, root / "jobs" / source.name, digest)
        frozen_scripts[name] = destination
        stage_fingerprints[name] = {
            "source_path": str(source),
            "submitted_path": str(destination),
            "sha256": digest,
        }
    frozen_fingerprints = {
        "runner": {
            "source_path": str(runner),
            "submitted_path": str(frozen_runner),
            "sha256": str(fingerprints["runner"]["sha256"]),
        },
        "stages": stage_fingerprints,
    }
    return frozen_runner, frozen_scripts, frozen_fingerprints


def render_script(
    layout: RunLayout,
    config: Mapping[str, Any],
    stage: Mapping[str, Any],
    shard_ids: Iterable[int] | None = None,
    part_index: int | None = None,
) -> Path:
    name = str(stage["name"])
    count = int(stage["shard_count"])
    ids = sorted(set(range(count) if shard_ids is None else shard_ids))
    if any(value < 0 or value >= count for value in ids):
        raise SlurmError(f"stage {name} contains a shard id outside [0, {count})")
    resources = config["slurm"].get("stage_resources", {}).get(name, {})
    stage_parallel = resources.get("max_parallel_tasks")
    array = _array_spec(
        range(len(ids)),
        int(stage_parallel) if stage_parallel is not None else int(config["slurm"]["max_parallel_tasks"]),
    )
    if not array:
        raise SlurmError(f"stage {name} has no shards to submit")
    log_root = layout.root / "slurm" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    job_name = f"mint-{name}" if part_index is None else f"mint-{name}-p{part_index:03d}"
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array}",
        f"#SBATCH --partition={config['slurm']['partition']}",
        f"#SBATCH --cpus-per-task={resources.get('cpus', 4)}",
        f"#SBATCH --mem={resources.get('memory', '16G')}",
        f"#SBATCH --time={resources.get('time', '04:00:00')}",
        "#SBATCH --export=ALL",
        f"#SBATCH --output={shlex.quote(f'{log_root.as_posix()}/{name}-%A_%a.out')}",
        f"#SBATCH --error={shlex.quote(f'{log_root.as_posix()}/{name}-%A_%a.err')}",
    ]
    if config["slurm"].get("account"):
        lines.append(f"#SBATCH --account={config['slurm']['account']}")
    if config["slurm"].get("qos"):
        lines.append(f"#SBATCH --qos={config['slurm']['qos']}")
    if int(resources.get("gpus", 0)) > 0:
        lines.append(f"#SBATCH --gpus={resources['gpus']}")
    for argument in config["slurm"].get("extra_sbatch_args", []):
        if "\n" in argument or "\r" in argument:
            raise SlurmError("slurm.extra_sbatch_args cannot contain newlines")
        lines.append(f"#SBATCH {argument}")

    binds = _bind_paths(layout, config)
    command = ["apptainer", "exec"]
    if binds:
        command.extend(["--bind", ",".join(binds)])
    command.extend(
        [
            str(config["slurm"]["apptainer_image"]),
            "mint-synth", "worker", "--run-dir", str(layout.root), "--stage", name,
            "--shard-id", "${MINT_SHARD_ID}",
        ]
    )
    quoted = [token if token == "${MINT_SHARD_ID}" else shlex.quote(token) for token in command]
    shard_mapping = " ".join(str(value) for value in ids)
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"MINT_SHARD_IDS=({shard_mapping})",
            'MINT_SHARD_ID="${MINT_SHARD_IDS[$SLURM_ARRAY_TASK_ID]}"',
            " ".join(quoted),
            "",
        ]
    )
    suffix = "" if part_index is None else f"-part-{part_index:03d}"
    path = layout.root / "slurm" / f"{name}{suffix}.sbatch"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def _submit_script(
    script: Path,
    dependency_ids: list[str],
    *,
    dry_run: bool,
    sbatch_args: Iterable[str] = (),
    environment: Mapping[str, str] | None = None,
) -> str:
    command = ["sbatch", "--parsable", *sbatch_args]
    if dependency_ids:
        command.append("--dependency=afterok:" + ":".join(dependency_ids))
    command.append(str(script))
    if dry_run:
        return "DRYRUN"
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=dict(environment) if environment is not None else None,
        )
    except FileNotFoundError as exc:
        raise SlurmError("sbatch was not found; run submit on a SLURM login node") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise SlurmError(f"sbatch failed for {script.name}: {detail}") from exc
    job_id = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise SlurmError(f"sbatch returned an invalid job id for {script.name}: {job_id!r}")
    return job_id


def _assert_no_active_jobs(manifest: Mapping[str, Any], *, force_resubmit: bool) -> None:
    if force_resubmit:
        return
    discovered: set[str] = set()
    for submission in manifest.get("submissions", []):
        if not isinstance(submission, Mapping) or submission.get("dry_run"):
            continue
        for raw in (submission.get("jobs", {}) or {}).values():
            values = raw if isinstance(raw, list) else [raw]
            for job_id in values:
                if re.fullmatch(r"[0-9]+", str(job_id)):
                    discovered.add(str(job_id))
    job_ids = sorted(discovered, key=int)
    if not job_ids:
        return
    command = ["squeue", "--noheader", "--jobs", ",".join(job_ids), "--format", "%A|%T"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SlurmError(
            "cannot verify whether previous jobs are active because squeue was not found; "
            "run resume on a SLURM login node or explicitly use --force-resubmit"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SlurmError(
            "cannot verify previous SLURM jobs before resume; use --force-resubmit only after checking them manually: "
            + exc.stderr.strip()
        ) from exc
    active: list[str] = []
    for line in result.stdout.splitlines():
        job_id, separator, state = line.strip().partition("|")
        # squeue lists jobs that have not fully left the live queue. Treat every returned
        # state (including site/version-specific REQUEUED variants) as unsafe to duplicate.
        if separator and job_id.strip() in discovered:
            active.append(f"{job_id}:{state.strip()}")
    if active:
        raise SlurmError(
            "refusing to duplicate still-active SLURM jobs ("
            + ", ".join(active[:20])
            + "); wait for them to finish, or use --force-resubmit after manual cancellation"
        )


@contextmanager
def _submission_lock(layout: RunLayout):
    path = layout.root / ".submission.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if os.name == "nt":  # pragma: no cover - SLURM itself is POSIX; this supports dry-run clients
            import msvcrt

            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SlurmError(f"another submit/resume process holds the run lock: {path}") from exc

            def unlock() -> None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise SlurmError(f"another submit/resume process holds the run lock: {path}") from exc

            def unlock() -> None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()
    finally:
        stream.close()


def _submit_run_locked(
    run_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    force_resubmit: bool = False,
    script_dir: str | Path | None = None,
) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    plan = read_json(layout.plan)
    config = load_config(layout.config)
    if not verify_plan_fingerprint(plan):
        raise SlurmError("plan fingerprint mismatch")
    if config_fingerprint(config) != plan.get("config_fingerprint"):
        raise SlurmError("resolved configuration changed after planning")
    if package_fingerprint() != plan.get("code_fingerprint"):
        raise SlurmError("local package differs from the code frozen in the plan")
    if auxiliary_fingerprints(config) != plan.get("auxiliary_fingerprints"):
        raise SlurmError("formula, alias, seed, or evaluation file changed after planning")
    if deployment_image_fingerprint(config) != plan.get("deployment_image_fingerprint"):
        raise SlurmError("Apptainer image changed after planning")
    static_runner: Path | None = None
    static_scripts: dict[str, Path] = {}
    static_fingerprints: dict[str, Any] | None = None
    if script_dir is not None:
        static_runner, static_scripts, static_fingerprints = _static_submission_files(
            script_dir, plan["stages"]
        )
        for script in static_scripts.values():
            _static_parallel_limit(script, int(config["slurm"]["max_parallel_tasks"]))
    if not dry_run:
        for name, endpoint in config["llm"]["endpoints"].items():
            env_name = str(endpoint["api_key_env"])
            if endpoint.get("require_api_key", True) and not os.getenv(env_name):
                raise SlurmError(f"required API key environment variable is not set for endpoint {name}: {env_name}")
    submitted: dict[str, list[str]] = {}
    rendered: dict[str, list[str]] = {}
    shard_maps: dict[str, list[str]] = {}
    manifest = read_json(layout.manifest)
    if not resume and manifest.get("state") in {"SUBMITTING", "SUBMITTED", "SUBMIT_FAILED"}:
        raise SlurmError("run was already submitted or partially submitted; use resume to avoid duplicate jobs")
    if resume and not dry_run:
        _assert_no_active_jobs(manifest, force_resubmit=force_resubmit)
    submission_id = uuid.uuid4().hex
    if static_runner is not None and static_fingerprints is not None:
        static_runner, static_scripts, static_fingerprints = _freeze_static_submission(
            layout,
            submission_id,
            static_runner,
            static_scripts,
            static_fingerprints,
        )
    submission = {
        "submission_id": submission_id,
        "submitted_at": utc_now(),
        "resume": resume,
        "dry_run": dry_run,
        "jobs": submitted,
        "scripts": rendered,
        "shard_maps": shard_maps,
        "submitter": "static_sbatch" if static_runner is not None else "generated_sbatch",
        "static_script_fingerprints": static_fingerprints,
        "state": "SUBMITTING",
    }
    manifest.setdefault("submissions", []).append(submission)
    manifest["state"] = "SUBMITTING"
    manifest["updated_at"] = utc_now()
    atomic_json(layout.manifest, manifest)
    try:
        for stage in plan["stages"]:
            name = stage["name"]
            count = int(stage["shard_count"])
            ids = (
                [
                    value
                    for value in range(count)
                    if not shard_complete(layout, name, value, expected_stage_signature(plan, name, value))
                ]
                if resume
                else list(range(count))
            )
            if not ids:
                continue
            dependency_ids = [
                job_id
                for dependency in stage.get("dependencies", [])
                for job_id in submitted.get(dependency, [])
            ]
            max_array_tasks = int(config["slurm"]["max_array_tasks"])
            id_chunks = [ids[index : index + max_array_tasks] for index in range(0, len(ids), max_array_tasks)]
            submitted[name] = []
            rendered[name] = []
            shard_maps[name] = []
            for part_index, id_chunk in enumerate(id_chunks):
                if static_runner is None:
                    script = render_script(
                        layout,
                        config,
                        stage,
                        id_chunk,
                        part_index=part_index if len(id_chunks) > 1 else None,
                    )
                    job_id = _submit_script(script, dependency_ids, dry_run=dry_run)
                else:
                    script = static_scripts[name]
                    map_path = _write_shard_map(
                        layout.root
                        / "slurm"
                        / "maps"
                        / submission["submission_id"]
                        / f"{name}-part-{part_index:03d}.txt",
                        id_chunk,
                    )
                    parallel = _static_parallel_limit(
                        script, int(config["slurm"]["max_parallel_tasks"])
                    )
                    array = _array_spec(range(len(id_chunk)), parallel)
                    log_root = layout.root / "slurm" / "logs"
                    log_root.mkdir(parents=True, exist_ok=True)
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "MINT_RUN_DIR": str(layout.root),
                            "MINT_STAGE": str(name),
                            "MINT_SHARD_MAP": str(map_path),
                            "MINT_APPTAINER_IMAGE": str(config["slurm"]["apptainer_image"]),
                            "MINT_BIND_SPEC": ",".join(_bind_paths(layout, config)),
                            "MINT_RUNNER_SCRIPT": str(static_runner),
                        }
                    )
                    job_id = _submit_script(
                        script,
                        dependency_ids,
                        dry_run=dry_run,
                        sbatch_args=(
                            f"--array={array}",
                            f"--output={log_root.as_posix()}/{name}-%A_%a.out",
                            f"--error={log_root.as_posix()}/{name}-%A_%a.err",
                        ),
                        environment=environment,
                    )
                    shard_maps[name].append(str(map_path))
                submitted[name].append(job_id)
                rendered[name].append(str(script))
                manifest["updated_at"] = utc_now()
                atomic_json(layout.manifest, manifest)
    except BaseException as exc:
        submission["state"] = "FAILED"
        submission["error"] = str(exc)
        manifest["state"] = "SUBMIT_FAILED"
        manifest["updated_at"] = utc_now()
        atomic_json(layout.manifest, manifest)
        raise
    submission["state"] = "DRY_RUN" if dry_run else "SUBMITTED"
    manifest["state"] = submission["state"]
    manifest["updated_at"] = utc_now()
    atomic_json(layout.manifest, manifest)
    return {"run_dir": str(layout.root), **submission, "status": run_status(layout.root)}


def submit_run(
    run_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    force_resubmit: bool = False,
    script_dir: str | Path | None = None,
) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    with _submission_lock(layout):
        return _submit_run_locked(
            layout.root,
            resume=resume,
            dry_run=dry_run,
            force_resubmit=force_resubmit,
            script_dir=script_dir,
        )
