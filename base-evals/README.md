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
base-evals/submit_prepare.sh
```

Full Slurm job:

```bash
FULL=1 DATASETS=imagenet_a,dtd,seed,ai2d base-evals/submit_prepare.sh
```

## Evaluate

Smoke test on the Slurm array:

```bash
base-evals/submit_eval.sh
```

Full eval:

```bash
FULL=1 DATASETS=imagenet_a,dtd,seed,ai2d base-evals/submit_eval.sh
```

Results are written to `base-evals/results/`.
