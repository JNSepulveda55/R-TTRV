#!/bin/bash
# Submit data preparation with iteration-friendly defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV="${CONDA_ENV:-rttrv}"
export DATASETS="${DATASETS:-dtd}"
export OVERWRITE="${OVERWRITE:-1}"

if [[ "${FULL:-0}" == "1" ]]; then
  unset LIMIT
else
  export LIMIT="${LIMIT:-8}"
fi

sbatch "$SCRIPT_DIR/prepare.sbatch"
