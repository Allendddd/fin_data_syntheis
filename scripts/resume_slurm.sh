#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: bash scripts/resume_slurm.sh <planned-run-dir> [--dry-run|--force-resubmit]" >&2
  exit 64
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
run_dir="$1"
shift
mint_synth_bin="${MINT_SYNTH_BIN:-mint-synth}"

if [[ ! -d "$run_dir" ]]; then
  echo "mint-synth: planned run does not exist: $run_dir" >&2
  exit 66
fi

exec "$mint_synth_bin" resume \
  --run-dir "$run_dir" \
  --sbatch-dir "$repo_root/slurm" \
  "$@"
