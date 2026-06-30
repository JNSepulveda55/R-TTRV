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
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_DIR/base-evals/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-$CACHE_ROOT/vllm-flashinfer-autotune}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_ROOT/flashinfer-workspace}"
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE"
if [[ -n "${VLLM_USE_V1:-}" ]]; then
  export VLLM_USE_V1
else
  unset VLLM_USE_V1
fi

mkdir -p base-evals/logs
if [[ -n "${PROMPT_LOG_DIR:-}" ]]; then
  mkdir -p "$PROMPT_LOG_DIR"
fi

DATASETS_ENV="${DATASETS:-imagenet_a,dtd,seed,ai2d}"
DATASET_LIST=(${DATASETS_ENV//,/ })
DATASET="${DATASET_LIST[$SLURM_ARRAY_TASK_ID]}"

ARGS=(
  --dataset "$DATASET"
  --records "${DATA_DIR:-base-evals/data}/${DATASET}.jsonl"
  --model "${MODEL:-OpenGVLab/InternVL3-2B}"
  --batch-size "${BATCH_SIZE:-32}"
  --out-dir "${OUT_DIR:-base-evals/results}"
  --max-tokens "${MAX_TOKENS:-8}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
)

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ -n "${PROMPT_LOG_LIMIT:-}" ]]; then
  ARGS+=(--prompt-log-limit "$PROMPT_LOG_LIMIT")
fi
if [[ -n "${PROMPT_LOG_DIR:-}" ]]; then
  PROMPT_LOG_FILE="$PROMPT_LOG_DIR/${DATASET}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
  ARGS+=(--prompt-log-file "$PROMPT_LOG_FILE")
fi
if [[ "${LOG_PROMPTS_TO_CONSOLE:-0}" == "1" ]]; then
  ARGS+=(--log-prompts-to-console)
fi

python base-evals/eval_vllm.py "${ARGS[@]}"
