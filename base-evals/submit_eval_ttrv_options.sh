#!/bin/bash
# Submit evals against data files built from TTRV's original ABCD options.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV="${CONDA_ENV:-rttrv}"
export DATA_DIR="${DATA_DIR:-base-evals/data-ttrv-options}"
export DATASETS="${DATASETS:-dtd}"
export OUT_DIR="${OUT_DIR:-base-evals/results-ttrv-options}"

if [[ "${FULL:-0}" == "1" ]]; then
  unset LIMIT
else
  export LIMIT="${LIMIT:-8}"
fi

DATASETS_SPACES="${DATASETS//,/ }"
read -r -a DATASET_LIST <<< "$DATASETS_SPACES"
if [[ "${#DATASET_LIST[@]}" -eq 0 ]]; then
  echo "No datasets selected." >&2
  exit 1
fi

EXCLUDE_NODES="${EXCLUDE_NODES:-mbz-titan-3}"
SBATCH_ARGS=(
  "--array=0-$((${#DATASET_LIST[@]} - 1))"
  "--export=ALL"
)
if [[ -n "$EXCLUDE_NODES" ]]; then
  SBATCH_ARGS+=("--exclude=$EXCLUDE_NODES")
fi

if [[ -n "$EXCLUDE_NODES" ]]; then
  echo "Excluding nodes: $EXCLUDE_NODES"
fi
sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/eval_array.sbatch"
