#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: bash scripts/submit_slurm.sh <planned-run-dir> [sbatch options...]" >&2
  exit 64
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
run_dir="$1"
shift
if [[ ! -f "$run_dir/plan.json" ]]; then
  echo "mint-synth: plan.json was not found: $run_dir" >&2
  exit 66
fi
run_dir="$(cd -- "$run_dir" && pwd -P)"

exec sbatch --parsable \
  --chdir="$repo_root" \
  --output="$run_dir/logs/slurm-%j.out" \
  --error="$run_dir/logs/slurm-%j.err" \
  "$@" "$repo_root/slurm/mint-synth.sbatch" "$run_dir"
