#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VERL_DIR"
export TTRV_VERL_DIR="$VERL_DIR"

mkdir -p logs/author-repro results/author-repro

DATASETS="${DATASETS:-imagenet_a,dtd,seed,ai2d}"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
if [ "${#DATASET_LIST[@]}" -eq 0 ]; then
  echo "DATASETS is empty." >&2
  exit 1
fi

ACCOUNT="${ACCOUNT:-nils}"
PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NO_GPU="${NO_GPU:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-180G}"

SBATCH_ARGS=(
  --account "$ACCOUNT"
  --partition "$PARTITION"
  --gres "gpu:$NO_GPU"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$MEM"
  --time "$TIME_LIMIT"
  --array "0-$((${#DATASET_LIST[@]} - 1))"
  --export=ALL
)

if [ -n "${QOS:-}" ]; then
  SBATCH_ARGS+=(--qos "$QOS")
fi

echo "Submitting author-code TTRV jobs for DATASETS=$DATASETS on account=$ACCOUNT partition=$PARTITION gres=gpu:$NO_GPU"
sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/author_repro.sbatch"
