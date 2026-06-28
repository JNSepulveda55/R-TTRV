#!/bin/bash
#SBATCH --job-name=base-prepare
#SBATCH --account=nils
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=base-evals/logs/%x-%j.out
#SBATCH --error=base-evals/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export TRANSFORMERS_CACHE=/shared/models/huggingface/hub

mkdir -p base-evals/logs

ARGS=(
  --datasets "${DATASETS:-imagenet_a,dtd,seed,ai2d}"
  --out-dir "${OUT_DIR:-base-evals/data}"
)

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

python base-evals/prepare_data.py "${ARGS[@]}"
