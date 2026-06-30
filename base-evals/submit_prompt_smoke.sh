#!/bin/bash
# Submit a tiny base-eval run that logs the prompt vLLM sends to the model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CONDA_ENV="${CONDA_ENV:-rttrv}"
export DATA_DIR="${DATA_DIR:-base-evals/data-ttrv-options}"
export DATASETS="${DATASETS:-dtd}"
export OUT_DIR="${OUT_DIR:-base-evals/results-prompt-smoke}"
export LIMIT="${LIMIT:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_TOKENS="${MAX_TOKENS:-8}"
export PROMPT_LOG_LIMIT="${PROMPT_LOG_LIMIT:-$LIMIT}"
export PROMPT_LOG_DIR="${PROMPT_LOG_DIR:-base-evals/logs/prompt-smoke}"
export LOG_PROMPTS_TO_CONSOLE="${LOG_PROMPTS_TO_CONSOLE:-1}"
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

DATASETS_SPACES="${DATASETS//,/ }"
read -r -a DATASET_LIST <<< "$DATASETS_SPACES"
if [[ "${#DATASET_LIST[@]}" -eq 0 ]]; then
  echo "No datasets selected." >&2
  exit 1
fi

PARTITION="${PARTITION:-debug}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-mbz-titan-3}"
SBATCH_ARGS=(
  "--partition=$PARTITION"
  "--time=$TIME_LIMIT"
  "--array=0-$((${#DATASET_LIST[@]} - 1))"
  "--export=ALL"
)
if [[ -n "$EXCLUDE_NODES" ]]; then
  SBATCH_ARGS+=("--exclude=$EXCLUDE_NODES")
fi

echo "Submitting prompt smoke for DATASETS=$DATASETS DATA_DIR=$DATA_DIR LIMIT=$LIMIT"
echo "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
if [[ -n "$EXCLUDE_NODES" ]]; then
  echo "Excluding nodes: $EXCLUDE_NODES"
fi
echo "Prompt JSONL logs will be written under $PROMPT_LOG_DIR"
sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/eval_array.sbatch"
