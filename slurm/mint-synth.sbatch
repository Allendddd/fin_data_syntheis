#!/usr/bin/env bash
#SBATCH --job-name=mint-synth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --export=ALL

set -euo pipefail
: "${1:?planned run directory is required}"
python_bin="${MINT_PYTHON:-python}"
workers="${MINT_WORKERS:-1}"
exec "$python_bin" -m mint_synth run --run-dir "$1" --workers "$workers"
