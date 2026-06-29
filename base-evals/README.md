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

## Plot Results

After evaluation writes `base-evals/results/summary.csv`, create a bar chart:

```bash
python base-evals/plot_results.py
```

The plot is saved to `base-evals/results/accuracy_bars.png`.
