#!/usr/bin/env bash
# Submit author-code base-model validation without any TTRV training updates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VERL_DIR"
export TTRV_VERL_DIR="$VERL_DIR"

mkdir -p logs/author-base-eval results/author-base-eval

DATASETS="${DATASETS:-imagenet_a,dtd,seed,ai2d}"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
if [ "${#DATASET_LIST[@]}" -eq 0 ]; then
  echo "DATASETS is empty." >&2
  exit 1
fi

export EPOCHS="${EPOCHS:-0}"
export VAL_N="${VAL_N:-1}"
export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export RUN_LOG_DIR="${RUN_LOG_DIR:-$VERL_DIR/logs/author-base-eval}"
export RESULTS_DIR="${RESULTS_DIR:-$VERL_DIR/results/author-base-eval}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RESULTS_DIR/checkpoints}"
export EXPERIMENT="${EXPERIMENT:-author-base-eval}"
export WANDB_PROJECT="${WANDB_PROJECT:-TTRL-verl-author-base-eval}"

ACCOUNT="${ACCOUNT:-nils}"
PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
NO_GPU="${NO_GPU:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-180G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-mbz-titan-3}"

SBATCH_ARGS=(
  --job-name "ttrv-author-base"
  --account "$ACCOUNT"
  --partition "$PARTITION"
  --gres "gpu:$NO_GPU"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$MEM"
  --time "$TIME_LIMIT"
  --array "0-$((${#DATASET_LIST[@]} - 1))"
  --output "logs/author-base-eval/%x-%A_%a.out"
  --error "logs/author-base-eval/%x-%A_%a.err"
  --export=ALL
)

if [ -n "${QOS:-}" ]; then
  SBATCH_ARGS+=(--qos "$QOS")
fi
if [ -n "$EXCLUDE_NODES" ]; then
  SBATCH_ARGS+=(--exclude "$EXCLUDE_NODES")
fi

HYDRA_ARGS=(+trainer.val_only=True)
if [ "$#" -gt 0 ]; then
  HYDRA_ARGS+=("$@")
fi

echo "Submitting author-code base validation for DATASETS=$DATASETS on account=$ACCOUNT partition=$PARTITION gres=gpu:$NO_GPU"
echo "This runs initial validation only: EPOCHS=$EPOCHS, trainer.val_only=True, no training updates."
if [ -n "$EXCLUDE_NODES" ]; then
  echo "Excluding nodes: $EXCLUDE_NODES"
fi
sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/author_repro.sbatch" "${HYDRA_ARGS[@]}"
