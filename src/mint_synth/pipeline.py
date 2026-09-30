from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .atomic import AtomicSynthesizer, atomic_to_sft
from .audit import CAPABILITIES, audit_atomic, audit_long
from .chunking import chunk_sources
from .extract import FactExtractor, deduplicate_facts
from .jsonl import load_sources, read_jsonl, write_jsonl
from .llm import JsonLLM
from .long_horizon import (
    LongHorizonSynthesizer,
    load_evaluation_questions,
    load_seed_bank,
    sample_fact_bundles,
)
from .util import stable_id


Progress = Callable[[str], None]


@dataclass(frozen=True)
class PipelineConfig:
    input_path: str
    output_dir: str
    text_field: str = "text"
    id_field: str | None = None
    max_records: int | None = None
    chunk_chars: int = 6000
    overlap_chars: int = 400
    min_chunk_chars: int = 600
    max_facts_per_chunk: int = 30
    atomic_per_chunk: int = 2
    long_candidates: int = 40
    long_bundle_size: int = 12
    min_depth: int = 2
    min_sources: int = 2
    cutoff: str = ""
    seed_file: str | None = None
    evaluation_file: str | None = None
    language: str = "auto"
    overlap_threshold: float = 0.82
    random_seed: int = 42
    concurrency: int = 4
    skip_atomic: bool = False
    skip_long: bool = False
    resume: bool = False


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _config_fingerprint(
    config: PipelineConfig, model_metadata: dict[str, Any] | None = None
) -> str:
    material = asdict(config)
    for key in ("output_dir", "concurrency", "resume"):
        material.pop(key, None)
    material["input_sha256"] = _file_sha256(config.input_path)
    if config.seed_file:
        material["seed_sha256"] = _file_sha256(config.seed_file)
    if config.evaluation_file:
        material["evaluation_sha256"] = _file_sha256(config.evaluation_file)
    material["models"] = model_metadata or {}
    return stable_id("cfg", material, length=32)


def prepare_corpus(config: PipelineConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if config.max_records is not None and config.max_records < 1:
        raise ValueError("max_records must be at least 1")
    sources = load_sources(
        config.input_path,
        text_field=config.text_field,
        id_field=config.id_field,
        max_records=config.max_records,
    )
    chunks = chunk_sources(
        sources,
        max_chars=config.chunk_chars,
        overlap_chars=config.overlap_chars,
        min_chars=config.min_chunk_chars,
    )
    output = Path(config.output_dir)
    write_jsonl(output / "sources.jsonl", sources)
    write_jsonl(output / "chunks.jsonl", chunks)
    return sources, chunks


class SynthesisPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        extractor_llm: JsonLLM,
        constructor_llm: JsonLLM,
        auditor_llm: JsonLLM,
        progress: Progress | None = None,
        model_metadata: dict[str, Any] | None = None,
    ):
        if config.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if config.max_facts_per_chunk < 1:
            raise ValueError("max_facts_per_chunk must be at least 1")
        if config.atomic_per_chunk < 0 or config.long_candidates < 0:
            raise ValueError("candidate counts cannot be negative")
        if config.min_depth < 1 or config.min_sources < 1:
            raise ValueError("min_depth and min_sources must be at least 1")
        if config.long_bundle_size < config.min_sources:
            raise ValueError("long_bundle_size must be at least min_sources")
        if not 0.0 <= config.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in [0, 1]")
        if config.language not in {"auto", "zh", "en"}:
            raise ValueError("language must be auto, zh, or en")
        if not config.skip_long and not config.cutoff:
            raise ValueError("cutoff is required for long-horizon synthesis")
        self.config = config
        self.extractor_llm = extractor_llm
        self.constructor_llm = constructor_llm
        self.auditor_llm = auditor_llm
        self.progress = progress or (lambda _: None)
        self.model_metadata = model_metadata or {}
        self.output = Path(config.output_dir)
        self.rejections: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        fingerprint = _config_fingerprint(self.config, self.model_metadata)
        self._check_resume(fingerprint)
        rejection_path = self.output / "rejected.jsonl"
        if self.config.resume and rejection_path.exists():
            self.rejections = list(read_jsonl(rejection_path))
        self.progress("Preparing stable source snapshots and semantic chunks")
        sources, chunks = prepare_corpus(self.config)

        facts_path = self.output / "facts.jsonl"
        if self.config.resume and facts_path.exists():
            facts = list(read_jsonl(facts_path))
            self.progress(f"Resumed {len(facts)} facts")
        else:
            facts = self._extract_facts(chunks)
            write_jsonl(facts_path, facts)

        atomic_tasks: list[dict[str, Any]] = []
        atomic_path = self.output / "atomic_tasks.jsonl"
        if not self.config.skip_atomic:
            if self.config.resume and atomic_path.exists():
                atomic_tasks = list(read_jsonl(atomic_path))
                self.progress(f"Resumed {len(atomic_tasks)} atomic tasks")
                if not (self.output / "atomic_sft.jsonl").exists():
                    write_jsonl(self.output / "atomic_sft.jsonl", (atomic_to_sft(task) for task in atomic_tasks))
            else:
                atomic_tasks = self._synthesize_atomic(chunks, facts)
                write_jsonl(atomic_path, atomic_tasks)
                write_jsonl(self.output / "atomic_sft.jsonl", (atomic_to_sft(task) for task in atomic_tasks))

        long_tasks: list[dict[str, Any]] = []
        long_path = self.output / "long_tasks.jsonl"
        if not self.config.skip_long:
            if self.config.resume and long_path.exists():
                long_tasks = list(read_jsonl(long_path))
                self.progress(f"Resumed {len(long_tasks)} long-horizon tasks")
            else:
                long_tasks = self._synthesize_long(facts)
                write_jsonl(long_path, long_tasks)

        write_jsonl(self.output / "rejected.jsonl", self.rejections)
        manifest = {
            "pipeline": "mint-agent-data-engine-reproduction",
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint,
            "input_sha256": _file_sha256(self.config.input_path),
            "config": asdict(self.config),
            "models": self.model_metadata,
            "counts": {
                "sources": len(sources),
                "chunks": len(chunks),
                "facts": len(facts),
                "atomic_tasks": len(atomic_tasks),
                "long_tasks": len(long_tasks),
                "rejected": len(self.rejections),
            },
            "paper_contract": {
                "atomic_capabilities": sorted(CAPABILITIES),
                "long_min_operation_depth": self.config.min_depth,
                "long_min_source_breadth": self.config.min_sources,
                "graph_fixed_before_query": True,
                "exact_provenance_required": True,
                "deterministic_replay_required": True,
                "independent_semantic_audit_required": True,
            },
        }
        with (self.output / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        self.progress(f"Completed: {json.dumps(manifest['counts'], ensure_ascii=False)}")
        return manifest

    def _check_resume(self, fingerprint: str) -> None:
        manifest_path = self.output / "manifest.json"
        state_path = self.output / ".run-state.json"
        synthesis_files = [
            self.output / name
            for name in (
                "facts.jsonl",
                "atomic_tasks.jsonl",
                "atomic_sft.jsonl",
                "long_tasks.jsonl",
                "rejected.jsonl",
                "manifest.json",
            )
        ]
        if self.config.resume:
            previous_path = manifest_path if manifest_path.exists() else state_path
            if not previous_path.exists() and any(path.exists() for path in synthesis_files):
                raise ValueError("existing stage files have no run fingerprint and cannot be safely resumed")
            if previous_path.exists():
                with previous_path.open("r", encoding="utf-8") as stream:
                    previous = json.load(stream)
                if previous.get("config_fingerprint") != fingerprint:
                    raise ValueError("--resume configuration, input, or model differs from the existing run")
        elif state_path.exists() or any(path.exists() for path in synthesis_files):
            raise ValueError("output already contains a synthesis run; use a new directory or matching --resume")
        with state_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                {
                    "config_fingerprint": fingerprint,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")

    def _extract_facts(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        extractor = FactExtractor(self.extractor_llm, max_facts_per_chunk=self.config.max_facts_per_chunk)

        def work(chunk: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            try:
                return extractor.extract(chunk)
            except Exception as exc:  # API errors are preserved as auditable rejection records.
                return [], [{"stage": "fact_extraction_error", "chunk_id": chunk["chunk_id"], "source_id": chunk["source_id"], "reason": repr(exc)}]

        self.progress(f"Extracting facts from {len(chunks)} chunks")
        facts: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = [pool.submit(work, chunk) for chunk in chunks]
            for index, future in enumerate(futures, 1):
                accepted, rejected = future.result()
                facts.extend(accepted)
                self.rejections.extend(rejected)
                if index % 25 == 0 or index == len(futures):
                    self.progress(f"Fact extraction {index}/{len(futures)}")
        result = deduplicate_facts(facts)
        self.progress(f"Retained {len(result)} provenance-linked facts")
        return result

    @staticmethod
    def _facts_for_chunk(chunk: dict[str, Any], facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        start = int(chunk["locator"]["char_start"])
        end = int(chunk["locator"]["char_end"])
        return [
            fact
            for fact in facts
            if fact["source_id"] == chunk["source_id"]
            and int(fact["locator"]["char_start"]) >= start
            and int(fact["locator"]["char_end"]) <= end
        ]

    def _synthesize_atomic(
        self, chunks: list[dict[str, Any]], facts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        synthesizer = AtomicSynthesizer(
            self.constructor_llm, self.auditor_llm, language=self.config.language
        )
        capabilities = sorted(CAPABILITIES)
        jobs: list[tuple[dict[str, Any], list[dict[str, Any]], str, int]] = []
        for chunk in chunks:
            chunk_facts = self._facts_for_chunk(chunk, facts)
            if not chunk_facts:
                continue
            rng = random.Random(f"{self.config.random_seed}:{chunk['chunk_id']}")
            order = capabilities[:]
            rng.shuffle(order)
            for index in range(self.config.atomic_per_chunk):
                jobs.append((chunk, chunk_facts, order[index % len(order)], index))

        def work(job: tuple[dict[str, Any], list[dict[str, Any]], str, int]):
            chunk, chunk_facts, capability, index = job
            try:
                return synthesizer.synthesize(
                    chunk, chunk_facts, capability, candidate_index=index
                )
            except Exception as exc:
                return None, {"stage": "atomic_task_error", "chunk_id": chunk["chunk_id"], "capability": capability, "reason": repr(exc)}

        self.progress(f"Constructing {len(jobs)} atomic task candidates")
        tasks: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = [pool.submit(work, job) for job in jobs]
            for index, future in enumerate(futures, 1):
                task, rejection = future.result()
                if task:
                    tasks.append(task)
                if rejection:
                    self.rejections.append(rejection)
                if index % 20 == 0 or index == len(futures):
                    self.progress(f"Atomic synthesis {index}/{len(futures)}")
        return sorted(tasks, key=lambda task: task["task_id"])

    def _synthesize_long(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seeds = load_seed_bank(self.config.seed_file)
        evaluation_questions = load_evaluation_questions(self.config.evaluation_file)
        bundles = sample_fact_bundles(
            facts,
            count=self.config.long_candidates,
            bundle_size=self.config.long_bundle_size,
            min_sources=self.config.min_sources,
            random_seed=self.config.random_seed,
        )
        if not bundles:
            self.rejections.append(
                {
                    "stage": "long_task_setup",
                    "reason": f"not enough numeric facts across {self.config.min_sources} sources",
                }
            )
            self.progress("No eligible cross-source fact bundles; long-horizon output is empty")
            return []
        synthesizer = LongHorizonSynthesizer(
            self.constructor_llm, self.auditor_llm, language=self.config.language
        )
        jobs = [(seeds[index % len(seeds)], bundle, index) for index, bundle in enumerate(bundles)]

        def work(job: tuple[dict[str, Any], list[dict[str, Any]], int]):
            seed, bundle, index = job
            try:
                return synthesizer.synthesize(
                    seed,
                    bundle,
                    cutoff=self.config.cutoff,
                    min_depth=self.config.min_depth,
                    min_sources=self.config.min_sources,
                    evaluation_questions=evaluation_questions,
                    candidate_index=index,
                    overlap_threshold=self.config.overlap_threshold,
                )
            except Exception as exc:
                return None, {"stage": "long_task_error", "seed_id": seed["seed_id"], "reason": repr(exc)}

        self.progress(f"Constructing {len(jobs)} graph-first long-horizon candidates")
        tasks: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = [pool.submit(work, job) for job in jobs]
            for index, future in enumerate(futures, 1):
                task, rejection = future.result()
                if task:
                    tasks.append(task)
                if rejection:
                    self.rejections.append(rejection)
                if index % 10 == 0 or index == len(futures):
                    self.progress(f"Long-horizon synthesis {index}/{len(futures)}")
        return sorted(tasks, key=lambda task: task["task_id"])


def validate_output(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    sources = list(read_jsonl(root / "sources.jsonl"))
    chunks = list(read_jsonl(root / "chunks.jsonl"))
    facts = list(read_jsonl(root / "facts.jsonl")) if (root / "facts.jsonl").exists() else []
    atomic = list(read_jsonl(root / "atomic_tasks.jsonl")) if (root / "atomic_tasks.jsonl").exists() else []
    long_tasks = list(read_jsonl(root / "long_tasks.jsonl")) if (root / "long_tasks.jsonl").exists() else []
    source_index = {source["source_id"]: source for source in sources}
    chunk_index = {chunk["chunk_id"]: chunk for chunk in chunks}
    fact_index = {fact["fact_id"]: fact for fact in facts}
    errors: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    if (root / "manifest.json").exists():
        with (root / "manifest.json").open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    configured_depth = int(manifest.get("config", {}).get("min_depth", 1))
    configured_sources = int(manifest.get("config", {}).get("min_sources", 1))

    if len(source_index) != len(sources):
        errors.append({"kind": "source", "reason": "duplicate source_id"})
    if len(chunk_index) != len(chunks):
        errors.append({"kind": "chunk", "reason": "duplicate chunk_id"})
    if len(fact_index) != len(facts):
        errors.append({"kind": "fact", "reason": "duplicate fact_id"})
    for source in sources:
        content_hash = hashlib.sha256(source["text"].encode("utf-8")).hexdigest()
        if source.get("content_sha256") != content_hash:
            errors.append({"kind": "source", "id": source.get("source_id"), "reason": "content hash mismatch"})
    for chunk in chunks:
        source = source_index.get(chunk.get("source_id"))
        locator = chunk.get("locator", {})
        start, end = locator.get("char_start"), locator.get("char_end")
        if source is None:
            errors.append({"kind": "chunk", "id": chunk.get("chunk_id"), "reason": "unknown source"})
        elif not isinstance(start, int) or not isinstance(end, int) or source["text"][start:end] != chunk.get("text"):
            errors.append({"kind": "chunk", "id": chunk.get("chunk_id"), "reason": "text/span mismatch"})

    for fact in facts:
        source = source_index.get(fact.get("source_id"))
        if source is None:
            errors.append({"kind": "fact", "id": fact.get("fact_id"), "reason": "unknown source"})
            continue
        locator = fact.get("locator", {})
        start, end = locator.get("char_start"), locator.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int) or source["text"][start:end] != fact.get("quote"):
            errors.append({"kind": "fact", "id": fact.get("fact_id"), "reason": "provenance quote/span mismatch"})
        if fact.get("chunk_id") not in chunk_index:
            errors.append({"kind": "fact", "id": fact.get("fact_id"), "reason": "unknown chunk"})

    for task in atomic:
        result = audit_atomic(task, fact_index)
        if not result.accepted:
            errors.append({"kind": "atomic_task", "id": task.get("task_id"), "reason": list(result.reasons)})
    for task in long_tasks:
        result = audit_long(
            task,
            fact_index,
            min_depth=configured_depth,
            min_sources=configured_sources,
            seed_text=str(task.get("seed", {}).get("seed_query", "")),
        )
        if not result.accepted:
            errors.append({"kind": "long_task", "id": task.get("task_id"), "reason": list(result.reasons)})
    return {
        "valid": not errors,
        "counts": {
            "sources": len(sources),
            "chunks": len(chunks),
            "facts": len(facts),
            "atomic_tasks": len(atomic),
            "long_tasks": len(long_tasks),
        },
        "errors": errors,
    }
