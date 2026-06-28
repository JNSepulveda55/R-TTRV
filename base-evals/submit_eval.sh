#!/bin/bash
# Submit base-model eval with iteration-friendly defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV="${CONDA_ENV:-rttrv}"
export DATASETS="${DATASETS:-dtd}"

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

sbatch --array="0-$((${#DATASET_LIST[@]} - 1))" "$SCRIPT_DIR/eval_array.sbatch"
