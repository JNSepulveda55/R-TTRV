#!/bin/bash
#SBATCH --job-name=base-eval
#SBATCH --account=nils
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --chdir=/shared/home/juan.arias/R-TTRV
#SBATCH --output=base-evals/logs/%x-%A_%a.out
#SBATCH --error=base-evals/logs/%x-%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/shared/home/juan.arias/R-TTRV}"
cd "$REPO_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export TRANSFORMERS_CACHE=/shared/models/huggingface/hub
if [[ -n "${VLLM_USE_V1:-}" ]]; then
  export VLLM_USE_V1
else
  unset VLLM_USE_V1
fi

mkdir -p base-evals/logs

DATASETS_ENV="${DATASETS:-imagenet_a,dtd,seed,ai2d}"
DATASET_LIST=(${DATASETS_ENV//,/ })
DATASET="${DATASET_LIST[$SLURM_ARRAY_TASK_ID]}"

ARGS=(
  --dataset "$DATASET"
  --records "${DATA_DIR:-base-evals/data}/${DATASET}.jsonl"
  --model "${MODEL:-OpenGVLab/InternVL3-2B}"
  --batch-size "${BATCH_SIZE:-32}"
  --out-dir "${OUT_DIR:-base-evals/results}"
)

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

python base-evals/eval_vllm.py "${ARGS[@]}"
