## Coding Philosophy

- Optimize for readable research code, not production architecture.
- Use plain dictionaries, lists, and short functions unless a stronger structure
  clearly saves code.
- Avoid dataclasses, pydantic models, broad class hierarchies, excessive
  assertions, and redundant wrappers.
- Keep files short and direct. If a master's student in AI with limited codebase
  experience cannot follow it, simplify it.
- Prefer one config and one runnable loop until the first loop works.
- Use external libraries when they reduce code or save time, for example vLLM,
  PyYAML, or pytest.
- Keep scripts thin. Put reusable logic in a specific folder.

## Cluster Rules

- The local sandbox fails on this cluster. Run required shell commands with
  narrowly scoped elevated privileges and explain why.
- GPU training or evaluation must run through Slurm, not an interactive shell,
  unless the user explicitly asks for an interactive Slurm session.
- Use Slurm account `nils`.
- GPU check: `srun --gres=gpu:1 -A nils nvidia-smi`.
- Interactive GPU session: `srun --gres=gpu:1 -A nils --pty bash`.
- Batch jobs: use `sbatch`; place Slurm templates/jobs under `slurm/` and thin
  submission wrappers under `scripts/cluster/`.
- Slurm stdout/stderr files belong in ignored runtime output locations.
- Respect cluster limits: max 4 running jobs, max 20 queued plus running, max GPU
  job duration 3 days, max debug job duration 1 hour.
- Use `/scratch` only for node-local temporary files.
- Use `/shared/datasets`, `/shared/software`, and `/shared/models/huggingface`
  for shared datasets, tools, and HuggingFace caches.
- GPU jobs should set:
  - `HF_HOME=/shared/models/huggingface`
  - `HF_HUB_CACHE=/shared/models/huggingface/hub`
  - `TRANSFORMERS_CACHE=/shared/models/huggingface/hub`

## Data Hygiene And Safety

- Use canary-only secrets and sandboxed benchmark conditions.
- Never use real credentials, live private services, real people, or real private
  data in tasks.
- Do not add code that exfiltrates secrets, bypasses access controls, or executes
  untrusted external actions.
- Do not add network-dependent evaluators, cloud judges, or external data sources
  without explicit user approval.
- Keep raw datasets, model weights, checkpoints, generated trajectory corpora,
  and large artifacts out of git unless explicitly approved.
- Keep train/eval leakage boundaries visible in configs and saved outputs.

## Testing

- Local smoke tests should run without GPUs and without a live model server.
- Add Slurm/vLLM tests only after the fake local loop passes.
