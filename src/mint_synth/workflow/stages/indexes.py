from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mint_synth.core.run_state import RunLayout
from mint_synth.harness.snapshot import build_corpus_index, build_entity_partition
from mint_synth.workflow.planner import stage_by_name


def stage_corpus_index(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_id: int,
    output: Path,
) -> dict[str, Any]:
    del shard_id
    count = int(stage_by_name(plan, "facts")["shard_count"])
    paths = [layout.shard("facts", value) / "chunks.parquet" for value in range(count)]
    return build_corpus_index(
        paths, output / "snapshot.sqlite3",
        cutoff=str(config["run"]["cutoff"]),
        batch_rows=int(config["reader"]["batch_rows"]),
        available_at_field=str(config["input"]["available_at_field"]),
    )


def stage_retrieval_index(
    layout: RunLayout,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    shard_id: int,
    output: Path,
) -> dict[str, Any]:
    del plan
    return build_entity_partition(
        layout.shard("corpus_index", 0) / "snapshot.sqlite3",
        layout.shard("repartition", shard_id) / "facts.parquet",
        output / "snapshot.sqlite3",
        batch_rows=int(config["reader"]["batch_rows"]),
    )
