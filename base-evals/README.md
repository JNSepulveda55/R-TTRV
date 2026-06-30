# Base Evals

Standalone base-model evaluations for `OpenGVLab/InternVL3-2B` with vLLM.
No TTRV training, rewards, or model updates are used.

## Setup

Install the lightweight dependencies in your chosen environment:

```bash
conda activate rttrv
pip install -r base-evals/requirements.txt
```

The vLLM/Torch stack must support the cluster GPUs. The GPU nodes use RTX PRO
6000 Blackwell cards (`sm_120`), so old stacks such as `vllm==0.8.3` with
`torch==2.6.0+cu124` fail with `no kernel image is available for execution on
the device`.

The eval Slurm script does not force `VLLM_USE_V1`; it lets the installed vLLM
version choose its default engine. Set `VLLM_USE_V1` manually only for debugging.

## Prepare Data

Smoke test:

```bash
base-evals/submit_prepare.sh
```

Full Slurm job:

```bash
FULL=1 DATASETS=imagenet_a,dtd,seed,ai2d base-evals/submit_prepare.sh
```

## Prepare TTRV-Option Data

This keeps the local Hugging Face images from `base-evals/data/`, but uses the
exact ABCD prompts, options, and answers from the TTRV test JSONs. It still
evaluates only the base model; no TTRV training, rewards, or rollouts are used.

Smoke conversion:

```bash
python base-evals/prepare_ttrv_options_data.py --datasets dtd --limit 8 --overwrite
```

Full conversion:

```bash
python base-evals/prepare_ttrv_options_data.py --datasets imagenet_a,dtd,seed,ai2d --overwrite
```

Converted records are written to `base-evals/data-ttrv-options/`.

## Evaluate

Smoke test on the Slurm array:

```bash
base-evals/submit_eval.sh
```

Full eval:

```bash
FULL=1 DATASETS=imagenet_a,dtd,seed,ai2d base-evals/submit_eval.sh
```

Results are written to `base-evals/results-generated-options/` by default.

## Evaluate TTRV-Option Data

Smoke test:

```bash
base-evals/submit_eval_ttrv_options.sh
```

Full eval:

```bash
FULL=1 DATASETS=imagenet_a,dtd,seed,ai2d base-evals/submit_eval_ttrv_options.sh
```

This wrapper submits the same Slurm eval array as above with
`DATA_DIR=base-evals/data-ttrv-options` and writes to
`base-evals/results-ttrv-options/` by default.

## Prompt Smoke

To inspect exactly what the base-eval vLLM path exposes as the model prompt,
run:

```bash
base-evals/submit_prompt_smoke.sh
```

By default this submits a `debug` partition job for two `dtd` examples from
`base-evals/data-ttrv-options`. It writes normal eval outputs to
`base-evals/results-prompt-smoke/`, Slurm stdout/stderr to `base-evals/logs/`,
and prompt debug JSONL files to `base-evals/logs/prompt-smoke/`.

The base-eval Slurm path defaults to `VLLM_USE_FLASHINFER_SAMPLER=0` and the
submit wrappers exclude `mbz-titan-3`, because job 9151 failed there while vLLM
was initializing the FlashInfer sampler on the Blackwell GPU. Override with
`VLLM_USE_FLASHINFER_SAMPLER=1` or `EXCLUDE_NODES=` only when debugging that
runtime path.

Each prompt debug row includes the raw prompt submitted to vLLM, vLLM's returned
prompt string, decoded `prompt_token_ids` with special tokens preserved, token
IDs, image path, answer, prediction, and response. The same prompt blocks are
also printed to the Slurm stdout log by default.

## Plot Results

After evaluation writes `base-evals/results/summary.csv`, create a bar chart:

```bash
python base-evals/plot_results.py
```

The plot is saved to `base-evals/results/accuracy_bars.png`.
