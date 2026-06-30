# TTRV Author-Code Reproduction

This repo contains the authors' TTRV code under `TTRV/`. For reproducing the
four prior evals, use the authors' prompts, options, answers, IDs, and row order
as the source of truth. The only data change should be replacing each author
`image_path` with the matching local image already downloaded for the previous
evals under `base-evals/data/images/`.

The local paper copy is `docs/TTRV.pdf`. The implementation is a modified
`verl` trainer that uses `TTRLRewardManager` with GRPO-style rollouts for
vision-language prompts.

## Cluster Constraints

All GPU work must go through Slurm. Do not start GPU jobs on the login node.

Use the Nils account unless intentionally changing allocation:

```bash
srun -p debug --gres=gpu:1 -A nils --pty bash
sbatch --gres=gpu:2 -A nils path/to/script.sh
```

Cluster/runtime notes from the working environment:

- GPUs: RTX PRO 6000 Blackwell.
- NVIDIA driver: `590.48.01`.
- CUDA driver API: `13.1`.
- Compute capability: `(12, 0)`.
- Use the shared Hugging Face cache:
  - `HF_HOME=/shared/models/huggingface`
  - `HF_HUB_CACHE=/shared/models/huggingface/hub`
  - `TRANSFORMERS_CACHE=/shared/models/huggingface/hub`
- The Slurm wrapper defaults `HF_DATASETS_CACHE` to
  `TTRV/verl/.cache/huggingface/datasets`, which is ignored by git.
- Use `/scratch` only for temporary runtime files, not model caches.
- The current `ttrv` conda env imports the author rollout stack inside a GPU
  allocation. It has newer Blackwell-compatible packages than the author README
  pins, including `torch 2.8.0+cu128`, `vllm 0.10.2`, and
  `flash_attn 2.8.3.post1`.
- The Slurm wrappers exclude `mbz-titan-3` by default with
  `EXCLUDE_NODES=mbz-titan-3`, because job 9151 failed there while vLLM was
  initializing FlashInfer sampling on Blackwell. Override with
  `EXCLUDE_NODES=` only when intentionally testing that node.
- The wrappers default `VLLM_USE_FLASHINFER_SAMPLER=0` and keep vLLM and
  FlashInfer caches under ignored repo-local `.cache/` paths. This avoids the
  FlashInfer top-k/top-p sampler JIT path that failed on job 9151.

Do not force the authors' old README stack on this cluster. The README asks for
CUDA 12.4, `torch==2.5.1`, and `vllm==0.8.3`; those are not the minimum changes
needed here and are risky for Blackwell.

## Author Code Shape

Important files:

- `TTRV/README.md`: author setup and high-level TTRV description.
- `TTRV/verl/examples/ttrv/run.sh`: original single-run shell template.
- `TTRV/verl/data/preprocess.py`: author JSON-to-parquet conversion pattern.
- `TTRV/verl/verl/workers/reward_manager/ttrl.py`: `TTRLRewardManager`.
- `TTRV/verl/verl/utils/reward_score/ttrl/ttt_metrics.py`: majority,
  reward, pass@K, and post-pass metrics.
- `TTRV/verl/verl/trainer/ppo/ray_trainer.py`: PPO/GRPO training loop that
  switches rollout count to `n_votes_per_prompt`, computes TTRL rewards, selects
  the top `n_samples_per_prompt`, then updates the actor.

The author JSON rows have this relevant shape:

```json
{
  "id": "...",
  "source": "...",
  "prompt": "...",
  "answer": "A",
  "image_path": "/home/anirban/..."
}
```

The `verl` parquet rows produced for training use this shape:

```python
{
    "data_source": "GPQA-TTT",
    "prompt": [{"role": "user", "content": prompt}],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": answer},
    "extra_info": {"split": split, "index": f"{dataset}-{idx}", ...},
    "images": [{"image": local_image_path}],
}
```

`GPQA-TTT` is the data source accepted by the author reward manager for this
multiple-choice VQA/recognition path. The `answer` is preserved for validation
and metric reporting. During TTRL training, the reward is based on the model's
sampled answer distribution and entropy; the ground-truth answer is still logged
through metrics such as `train/label_accuracy`.

## Data Prep

Use the converter:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
conda run -n ttrv python prepare_ttrv_author_data.py \
  --datasets imagenet_a,dtd,seed,ai2d \
  --overwrite
```

Inputs:

- Author JSON: `TTRV/verl/data/<dataset>_20/{train,test}.json`
- Previous eval mapping: `base-evals/data-ttrv-options/<dataset>.jsonl`
- Local images: paths referenced by those JSONL files, under
  `base-evals/data/images/`

Outputs:

- Patched JSON and parquet:
  `TTRV/verl/data-author-repro/<dataset>_20/{train,test}.{json,parquet}`
- Manifest:
  `TTRV/verl/data-author-repro/<dataset>_20/manifest.json`

Matching rules:

- Default: exact `(original_image_path, prompt, answer)`.
- Fallback for `imagenet_a_20/train.json` and `dtd_20/train.json`: match by
  `original_image_path` only, because those author train prompts/options differ
  from the previous full test JSON while pointing to the same images.
- No fallback for `seed`, because original images can repeat and exact prompt
  matching disambiguates rows.

For a tiny smoke dataset:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
conda run -n ttrv python prepare_ttrv_author_data.py \
  --datasets dtd \
  --name-suffix _smoke \
  --limit-train 2 \
  --limit-test 8 \
  --overwrite
```

## Running Evals

The default reproduction scope is:

- `imagenet_a_20`
- `dtd_20`
- `seed_20`
- `ai2d_20`

Default model:

```bash
OpenGVLab/InternVL3-2B
```

Run a debug smoke job:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
bash examples/ttrv/submit_author_smoke.sh
```

The smoke launcher defaults to `dtd`, 2 train rows, 8 validation rows, 1 GPU,
1 epoch, and smaller rollout counts. It checks data, Slurm, imports, vLLM
rollout, reward wiring, and logging. It is not a reproduction metric.

Run the four full jobs as a Slurm array:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
bash examples/ttrv/submit_author_repro.sh
```

Useful overrides:

```bash
DATASETS=imagenet_a,dtd,seed,ai2d \
MODEL=OpenGVLab/InternVL3-2B \
NO_GPU=2 \
EPOCHS=2 \
bash examples/ttrv/submit_author_repro.sh
```

Single-task run:

```bash
DATASET=ai2d TASK=ai2d_20 NO_GPU=2 EPOCHS=2 \
sbatch -A nils --gres=gpu:2 examples/ttrv/author_repro.sbatch
```

Run author-code base validation with no training updates:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
bash examples/ttrv/submit_author_base_eval.sh
```

This uses the same author JSON/parquet data, `verl` dataloader, InternVL
processor/chat template, vLLM rollout path, and TTRL validation metrics as the
initial validation in a TTRV run. It sets `trainer.val_only=True` and
`EPOCHS=0`, so it exits after initial validation and does not run GRPO updates.
Logs are written to `TTRV/verl/logs/author-base-eval/`, and result/checkpoint
roots are under `TTRV/verl/results/author-base-eval/`.

Main exposed environment variables:

- `TASK`: dataset folder name under `data-author-repro`, for example `dtd_20`.
- `DATASET`: base dataset name, one of `imagenet_a`, `dtd`, `seed`, `ai2d`.
- `MODEL`: Hugging Face model path, default `OpenGVLab/InternVL3-2B`.
- `NO_GPU`: GPUs per node and training batch default.
- `EPOCHS`: trainer total epochs.
- `LIMIT_TRAIN`, `LIMIT_TEST`: optional converter row limits.
- `DATA_LOCAL_DIR`: parquet root, default `TTRV/verl/data-author-repro`.
- `RUN_LOG_DIR`: tee log root, default `TTRV/verl/logs/author-repro`.
- `RESULTS_DIR`: result root, default `TTRV/verl/results/author-repro`.
- `CHECKPOINT_ROOT`: checkpoint root, default
  `TTRV/verl/results/author-repro/checkpoints`.
- `OUTPUT_DIR`: exact trainer checkpoint path for a run.

The full wrapper preserves the author defaults `n_vote=32`,
`n_samples_per_prompt=16`, max prompt length `7524`, max response length `1024`,
and GRPO advantage estimation unless overridden.

## Metrics

Console logs contain the metrics needed for comparison:

- Initial validation accuracy: first `val-ttrl/label_accuracy`, usually at
  `step:0`.
- Final validation accuracy: last `val-ttrl/label_accuracy`.
- Core validation accuracy: `val-core/GPQA-TTT/acc/mean@N`.
- Training majority metric: `train/pass@32`.
- Post-selection training metric: `train/post_pass@16`.

Collect summary rows after jobs finish:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
conda run -n ttrv python summarize_ttrv_author_results.py \
  --logs 'logs/author-repro/*.log' 'logs/author-repro/*.out' \
  --out results/author-repro/summary.csv
```

For author-code base validation jobs, point the same summarizer at
`logs/author-base-eval/*.log` and write to
`results/author-base-eval/summary.csv`.

The CSV columns are:

```text
dataset,task,model,train_size,job_id,log_path,
initial_val_ttrl_label_accuracy,final_val_ttrl_label_accuracy,
initial_val_core_acc,final_val_core_acc,
final_train_pass_at_32,final_train_post_pass_at_16
```

## Validation Checklist

Data checks:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
conda run -n ttrv python prepare_ttrv_author_data.py \
  --datasets imagenet_a,dtd,seed,ai2d \
  --overwrite
```

Expected:

- All patched `image_path` values exist.
- Row counts match author JSON for train and test.
- Author fields `prompt`, `answer`, `source`, and `id` are unchanged in patched
  JSON.
- Parquet loads with `datasets.load_dataset` and includes `prompt`,
  `reward_model`, `extra_info`, and `images`.

Runtime import smoke, inside a Slurm GPU allocation:

```bash
srun -p debug --gres=gpu:1 -A nils --time=00:05:00 \
  bash -lc 'cd /shared/home/juan.arias/R-TTRV/TTRV/verl &&
  conda run -n ttrv python -c "import torch, vllm; import verl;
  from verl.workers.reward_manager import TTRLRewardManager;
  from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout;
  print(torch.__version__, vllm.__version__)"'
```

End-to-end smoke:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
bash examples/ttrv/submit_author_smoke.sh
```

Full reproduction:

```bash
cd /shared/home/juan.arias/R-TTRV/TTRV/verl
bash examples/ttrv/submit_author_repro.sh
```

Then collect `results/author-repro/summary.csv`.

## Known Runtime Notes

- `examples/ttrv/run.sh` is the author template, but it has a placeholder
  `DATA_LOCAL_DIR` and an undefined `BACKBONE` variable. Prefer the Slurm
  wrappers above for this cluster.
- The cluster adaptation is intentionally minimal: keep the author trainer and
  reward logic, but route runs through Slurm, use shared model caches, patch
  author JSON image paths, and use the installed Blackwell-compatible `ttrv`
  environment.
- The local code uses semantic vLLM version parsing so `vllm 0.10.2` selects
  the SPMD rollout path. Plain string comparison incorrectly treats `0.10.2` as
  older than `0.6.3`.
- InternVL runs need `trust_remote_code=True` for actor and critic model
  loading. The local tokenizer patch also exposes InternVL image/video token IDs
  expected by the processor path.
- The vLLM multimodal limit is image-only for these evals. Video prompts are
  disabled in rollout initialization to avoid vLLM 0.10 dummy video profiling on
  a benchmark that never supplies video.
- One successful smoke run logged a Ray/DataLoader worker-killed traceback
  during shutdown after final validation metrics and checkpoint save. Slurm
  still completed with exit code `0:0`; treat that pattern as a teardown warning,
  not a failed reproduction.
- Login-node imports of rollout code can fail or be misleading because GPU
  access is blocked there. Validate rollout imports inside `srun` or `sbatch`.
- `VLLM_USE_V1` is not forced by the wrappers. If a future vLLM issue appears,
  set it explicitly in the Slurm environment and record it with the job.
- `VLLM_USE_FLASHINFER_SAMPLER=0` is forced only as a runtime workaround for
  the sampler failure observed on `mbz-titan-3`; it can be overridden for
  targeted debugging.
- The generated parquet, logs, result CSVs, and checkpoints live under ignored
  paths in `TTRV/verl`.
