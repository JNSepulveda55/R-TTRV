#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASETS="${DATASETS:-dtd}"
export ACCOUNT="${ACCOUNT:-nils}"
export PARTITION="${PARTITION:-debug}"
export TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
export NO_GPU="${NO_GPU:-1}"
export EPOCHS="${EPOCHS:-1}"
export NAME_SUFFIX="${NAME_SUFFIX:-_smoke}"
export LIMIT_TRAIN="${LIMIT_TRAIN:-2}"
export LIMIT_TEST="${LIMIT_TEST:-8}"
export PREPARE_OVERWRITE="${PREPARE_OVERWRITE:-1}"

export N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-4}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
export TEST_FREQ="${TEST_FREQ:-200000}"

exec "$SCRIPT_DIR/submit_author_repro.sh"
