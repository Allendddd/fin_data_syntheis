#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: bash scripts/submit_slurm.sh <config.yaml|planned-run-dir> [--dry-run]" >&2
  exit 64
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_path="$1"
shift
mint_synth_bin="${MINT_SYNTH_BIN:-mint-synth}"

if [[ -d "$source_path" ]]; then
  source_args=(--run-dir "$source_path")
elif [[ -f "$source_path" ]]; then
  source_args=(--config "$source_path")
else
  echo "mint-synth: config or planned run does not exist: $source_path" >&2
  exit 66
fi

exec "$mint_synth_bin" submit \
  "${source_args[@]}" \
  --sbatch-dir "$repo_root/slurm" \
  "$@"
