# Base Evals

Standalone base-model evaluations for `OpenGVLab/InternVL3-2B` with vLLM.
No TTRV training, rewards, or model updates are used.

## Setup

Install the lightweight dependencies in your chosen environment:

```bash
pip install -r base-evals/requirements.txt
```

## Prepare Data

Smoke test:

```bash
python base-evals/prepare_data.py --datasets dtd --limit 8 --overwrite
```

Full Slurm job:

```bash
sbatch base-evals/prepare.sbatch
```

## Evaluate

Smoke test on the Slurm array:

```bash
LIMIT=8 sbatch base-evals/eval_array.sbatch
```

Full eval:

```bash
sbatch base-evals/eval_array.sbatch
```

Results are written to `base-evals/results/`.
