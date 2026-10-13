#!/usr/bin/env bash
#SBATCH --job-name=mint-sft-prepare
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
# MINT_MAX_PARALLEL_TASKS=64

set -euo pipefail
exec bash "${MINT_RUNNER_SCRIPT:?MINT_RUNNER_SCRIPT is not set}" sft_prepare
