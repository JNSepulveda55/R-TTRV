#!/usr/bin/env bash
#SBATCH --job-name=ttrv-author
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/author-repro/%x-%A_%a.out
#SBATCH --error=logs/author-repro/%x-%A_%a.err

set -euo pipefail

if [ -n "${TTRV_VERL_DIR:-}" ]; then
  VERL_DIR="$(cd "$TTRV_VERL_DIR" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/prepare_ttrv_author_data.py" ]; then
  VERL_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  SCRIPT_DIR_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  VERL_DIR="$(cd "$SCRIPT_DIR_FALLBACK/../.." && pwd)"
fi
SCRIPT_DIR="$VERL_DIR/examples/ttrv"
REPO_DIR="$(cd "$VERL_DIR/../.." && pwd)"
cd "$VERL_DIR"

if [ -z "${HOME:-}" ] || [ "$HOME" = "/var/spool" ] || [ ! -w "$HOME" ]; then
  export HOME="/shared/home/$USER"
fi
if [ -z "${XDG_CACHE_HOME:-}" ] || [[ "$XDG_CACHE_HOME" == /var/spool* ]] || [ ! -w "$(dirname "$XDG_CACHE_HOME")" ]; then
  export XDG_CACHE_HOME="$VERL_DIR/.cache"
fi
mkdir -p "$XDG_CACHE_HOME"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda was not found. Load conda before submitting or install the ttrv environment." >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-ttrv}"
conda activate "$CONDA_ENV"

export PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/shared/models/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/shared/models/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/shared/models/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$VERL_DIR/.cache/huggingface/datasets}"
mkdir -p "$HF_DATASETS_CACHE"

if [ -d /scratch ]; then
  export TMPDIR="${TMPDIR:-/scratch/$USER/ttrv-author-${SLURM_JOB_ID:-local}}"
  export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR/ray}"
  mkdir -p "$TMPDIR" "$RAY_TMPDIR"
fi

infer_dataset() {
  local task_name="$1"
  local candidate
  for candidate in imagenet_a dtd seed ai2d; do
    if [[ "$task_name" == "${candidate}_"* ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

DATASETS="${DATASETS:-imagenet_a,dtd,seed,ai2d}"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"

TASK_INPUT="${TASK:-}"
DATASET="${DATASET:-}"
if [ -z "$DATASET" ] && [ -n "$TASK_INPUT" ]; then
  if INFERRED_DATASET="$(infer_dataset "$TASK_INPUT")"; then
    DATASET="$INFERRED_DATASET"
  else
    echo "Could not infer DATASET from TASK=$TASK_INPUT. Set DATASET explicitly." >&2
    exit 1
  fi
fi
if [ -z "$DATASET" ]; then
  ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
  if [ "$ARRAY_INDEX" -ge "${#DATASET_LIST[@]}" ]; then
    echo "SLURM_ARRAY_TASK_ID=$ARRAY_INDEX is outside DATASETS=$DATASETS" >&2
    exit 1
  fi
  DATASET="${DATASET_LIST[$ARRAY_INDEX]}"
fi

TRAIN_SIZE="${TRAIN_SIZE:-20}"
NAME_SUFFIX="${NAME_SUFFIX:-}"
if [ -n "$TASK_INPUT" ]; then
  BASE_TASK="${DATASET}_${TRAIN_SIZE}"
  if [[ "$TASK_INPUT" == "$BASE_TASK"* ]] && [ "$TASK_INPUT" != "$BASE_TASK" ] && [ -z "$NAME_SUFFIX" ]; then
    NAME_SUFFIX="${TASK_INPUT#$BASE_TASK}"
  fi
fi
TASK="${TASK_INPUT:-${DATASET}_${TRAIN_SIZE}${NAME_SUFFIX}}"

MODEL="${MODEL:-OpenGVLab/InternVL3-2B}"
NO_GPU="${NO_GPU:-${SLURM_GPUS_ON_NODE:-2}}"
if ! [[ "$NO_GPU" =~ ^[0-9]+$ ]]; then
  NO_GPU=2
fi
EPOCHS="${EPOCHS:-2}"
ADVANTAGE="${ADVANTAGE:-grpo}"
K="${K:-3}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-7524}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
VAL_N="${VAL_N:-1}"
DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-$NO_GPU}"
N_VOTES_PER_PROMPT="${N_VOTES_PER_PROMPT:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
TEST_FREQ="${TEST_FREQ:-200000}"
SAVE_FREQ="${SAVE_FREQ:-20000000}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"

DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-$VERL_DIR/data-author-repro}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$VERL_DIR/logs/author-repro}"
RESULTS_DIR="${RESULTS_DIR:-$VERL_DIR/results/author-repro}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RESULTS_DIR/checkpoints}"
mkdir -p "$RUN_LOG_DIR" "$RESULTS_DIR" "$CHECKPOINT_ROOT"

DATA_DIR="$DATA_LOCAL_DIR/$TASK"
PREPARE_OVERWRITE="${PREPARE_OVERWRITE:-0}"
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ] || [ "$PREPARE_OVERWRITE" = "1" ]; then
  PREPARE_CMD=(python prepare_ttrv_author_data.py --datasets "$DATASET" --shot "$TRAIN_SIZE" --output-root "$DATA_LOCAL_DIR")
  if [ -n "$NAME_SUFFIX" ]; then
    PREPARE_CMD+=(--name-suffix "$NAME_SUFFIX")
  fi
  if [ -n "${LIMIT_TRAIN:-}" ]; then
    PREPARE_CMD+=(--limit-train "$LIMIT_TRAIN")
  fi
  if [ -n "${LIMIT_TEST:-}" ]; then
    PREPARE_CMD+=(--limit-test "$LIMIT_TEST")
  fi
  if [ "$PREPARE_OVERWRITE" = "1" ]; then
    PREPARE_CMD+=(--overwrite)
  fi
  "${PREPARE_CMD[@]}"
fi
TRAIN_ROW_COUNT="$(python -c 'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$DATA_DIR/train.json")"

BACKBONE_SAFE="$(printf '%s' "$MODEL" | tr '/:' '__')"
DATE_TAG="$(date +%m%d)"
TIME_TAG="$(date +%H%M%S)"
EXPERIMENT="${EXPERIMENT:-TTRL-Len@${K}k}"
WANDB_PROJECT="${WANDB_PROJECT:-TTRL-verl-author-repro}"
RUN_NAME="${RUN_NAME:-${TASK}-${BACKBONE_SAFE}-${ADVANTAGE}}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_ROOT/$TASK/$BACKBONE_SAFE/${SLURM_JOB_ID:-local}-${TIME_TAG}}"
LOG_FILE="${LOG_FILE:-$RUN_LOG_DIR/${TASK}_${BACKBONE_SAFE}_${EPOCHS}e_${SLURM_JOB_ID:-local}_${TIME_TAG}.log}"

unset VLLM_ATTENTION_BACKEND

echo "AUTHOR_REPRO dataset=$DATASET task=$TASK model=$MODEL train_size=$TRAIN_ROW_COUNT job_id=${SLURM_JOB_ID:-local} log_path=$LOG_FILE data_dir=$DATA_DIR output_dir=$OUTPUT_DIR"
echo "AUTHOR_REPRO_ENV conda_env=$CONDA_ENV no_gpu=$NO_GPU epochs=$EPOCHS limit_train=${LIMIT_TRAIN:-} limit_test=${LIMIT_TEST:-} vllm_use_v1=${VLLM_USE_V1:-unset}"

TRAIN_CMD=(
  python -m verl.trainer.main_ppo
  reward_model.reward_manager=ttrl
  reward_model.reward_kwargs.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT"
  reward_model.reward_kwargs.n_votes_per_prompt="$N_VOTES_PER_PROMPT"
  reward_model.reward_kwargs.mode=train
  "data.train_files=[$DATA_DIR/train.parquet]"
  "data.val_files=[$DATA_DIR/test.parquet]"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.train_batch_size="$DATA_TRAIN_BATCH_SIZE"
  data.filter_overlong_prompts=True
  data.truncation=error
  actor_rollout_ref.model.path="$MODEL"
  +actor_rollout_ref.model.trust_remote_code="$TRUST_REMOTE_CODE"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.optim.lr=5e-7
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03
  actor_rollout_ref.actor.optim.warmup_style=cosine
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.do_vote=True
  actor_rollout_ref.rollout.n_vote="$N_VOTES_PER_PROMPT"
  actor_rollout_ref.rollout.n="$N_SAMPLES_PER_PROMPT"
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N"
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
  actor_rollout_ref.rollout.max_num_batched_tokens="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
  critic.optim.lr=9e-6
  critic.model.use_remove_padding=True
  critic.model.path="$MODEL"
  +critic.model.trust_remote_code="$TRUST_REMOTE_CODE"
  critic.model.enable_gradient_checkpointing=True
  critic.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE"
  critic.model.fsdp_config.param_offload=False
  critic.model.fsdp_config.optimizer_offload=False
  algorithm.kl_ctrl.kl_coef=0.00
  algorithm.adv_estimator="$ADVANTAGE"
  "trainer.logger=['console']"
  trainer.project_name="$WANDB_PROJECT"
  trainer.experiment_name="$DATE_TAG-$EXPERIMENT-$RUN_NAME"
  trainer.n_gpus_per_node="$NO_GPU"
  trainer.nnodes=1
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.max_actor_ckpt_to_keep=0
  trainer.max_critic_ckpt_to_keep=0
  trainer.default_local_dir="$OUTPUT_DIR"
  trainer.total_epochs="$EPOCHS"
)

"${TRAIN_CMD[@]}" "$@" 2>&1 | tee "$LOG_FILE"
