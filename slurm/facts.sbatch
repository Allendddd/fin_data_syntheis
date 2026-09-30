#!/usr/bin/env bash
#SBATCH --job-name=mint-facts
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
##SBATCH --account=replace-me
##SBATCH --qos=replace-me
# MINT_MAX_PARALLEL_TASKS=32

set -euo pipefail
exec bash "${MINT_RUNNER_SCRIPT:?MINT_RUNNER_SCRIPT is not set}" facts
