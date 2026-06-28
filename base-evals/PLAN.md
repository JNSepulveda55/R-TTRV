# Base InternVL3-2B vLLM Evaluation

This folder contains a standalone baseline evaluation path for
`OpenGVLab/InternVL3-2B` using vLLM. It intentionally does not use TTRV,
rollouts, GRPO, rewards, or model updates.

The implementation prepares ImageNet-A, DTD, SEED-Bench, and AI2D from
Hugging Face into ABCD multiple-choice JSONL files, then evaluates the base
model with greedy decoding.

Main files:

- `prepare_data.py`: downloads/prepares records and local image files.
- `eval_vllm.py`: runs vLLM inference and computes accuracy.
- `prepare.sbatch`: Slurm data-preparation job using account `nils`.
- `eval_array.sbatch`: Slurm GPU array job over the four datasets.
