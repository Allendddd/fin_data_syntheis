#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run_stage.sh <stage>}"
case "$stage" in
  normalize|id_validation|dedup|facts|atomic|entity_resolution|repartition|graph_planning|query_realization|validation|export)
    ;;
  *)
    echo "mint-synth: unsupported stage: $stage" >&2
    exit 64
    ;;
esac

if [[ -n "${MINT_STAGE:-}" && "$MINT_STAGE" != "$stage" ]]; then
  echo "mint-synth: stage mismatch: script=$stage environment=$MINT_STAGE" >&2
  exit 64
fi

: "${MINT_RUN_DIR:?MINT_RUN_DIR is not set}"
: "${MINT_SHARD_MAP:?MINT_SHARD_MAP is not set}"
: "${MINT_APPTAINER_IMAGE:?MINT_APPTAINER_IMAGE is not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

if [[ ! -r "$MINT_SHARD_MAP" ]]; then
  echo "mint-synth: shard map is not readable: $MINT_SHARD_MAP" >&2
  exit 66
fi
if [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "mint-synth: invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" >&2
  exit 64
fi

line_number=$((10#$SLURM_ARRAY_TASK_ID + 1))
shard_id="$(sed -n "${line_number}p" "$MINT_SHARD_MAP")"
shard_id="${shard_id%$'\r'}"
if [[ ! "$shard_id" =~ ^[0-9]+$ ]]; then
  echo "mint-synth: array index $SLURM_ARRAY_TASK_ID has no valid shard mapping" >&2
  exit 65
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "mint-synth: apptainer is not available on the compute node" >&2
  exit 69
fi

command=(apptainer exec)
if [[ -n "${MINT_BIND_SPEC:-}" ]]; then
  command+=(--bind "$MINT_BIND_SPEC")
fi
command+=(
  "$MINT_APPTAINER_IMAGE"
  mint-synth worker
  --run-dir "$MINT_RUN_DIR"
  --stage "$stage"
  --shard-id "$shard_id"
)
exec "${command[@]}"
