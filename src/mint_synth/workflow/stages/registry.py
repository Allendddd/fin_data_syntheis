from __future__ import annotations

from .common import StageHandler
from .ingest import stage_normalize, stage_id_validation, stage_dedup
from .facts import stage_facts, stage_entity_resolution, stage_repartition
from .questions import stage_atomic, stage_graph_planning, stage_query_realization, stage_validation
from .indexes import stage_corpus_index, stage_retrieval_index
from .query_export import stage_query_export
from .sft_data import stage_sft_prepare
from .publication import stage_export


HANDLERS: dict[str, StageHandler] = {
    "normalize": stage_normalize,
    "id_validation": stage_id_validation,
    "dedup": stage_dedup,
    "facts": stage_facts,
    "atomic": stage_atomic,
    "entity_resolution": stage_entity_resolution,
    "repartition": stage_repartition,
    "graph_planning": stage_graph_planning,
    "query_realization": stage_query_realization,
    "validation": stage_validation,
    "corpus_index": stage_corpus_index,
    "retrieval_index": stage_retrieval_index,
    "query_export": stage_query_export,
    "sft_prepare": stage_sft_prepare,
    "export": stage_export,
}
