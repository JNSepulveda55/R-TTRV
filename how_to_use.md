# Martin Cluster Quick Start

## Connecting

```bash
ssh <your_username>@10.67.33.23
```

## GPU Allocation Policy

Each professor's group has a **guaranteed GPU quota** based on their investment. The cluster has 24 GPUs total across 3 nodes.

| Group | Guaranteed GPUs | Account name | QoS name |
|-------|----------------|--------------|----------|
| Michalis Vazirgiannis | 10 | `michalis` | `guaranteed-michalis` |
| Mladen Kolar | 5 | `mladen` | `guaranteed-mladen` |
| Eric Moulines | 4 | `eric` | `guaranteed-eric` |
| Nils Lukas | 2 | `nils` | `guaranteed-nils` |
| Eduard Gorbunov | 2 | `eduard` | `guaranteed-eduard` |
| Raul Astudillo | 1 | `raul` | `guaranteed-raul` |

Allocations follow the Apr 2026 contributions table (24 GPUs total). SLURM picks your default QoS automatically from `--account=<prof>`, so you usually don't need to specify it explicitly.

**How it works:**
- By default, your jobs use your group's guaranteed quota and **cannot be preempted**.
- When GPUs are idle, you can use more than your quota by adding `--qos=normal`. These extra jobs **can be preempted** if the GPU owner needs them back.
- Preempted jobs are **requeued** (put back in the queue), not killed. You get a 5-minute grace period to save checkpoints before the job is requeued.
- The youngest preemptable jobs are preempted first, to minimize lost work.

## Quick Test: Verify GPU Access

```bash
srun --gres=gpu:1 -A <your_account> nvidia-smi
```

If this prints GPU info, you're all set.

## Submitting Jobs

All GPU work **must** go through SLURM. Direct GPU access is disabled.

```bash
# Interactive GPU session (1 GPU, 1 hour default)
srun --gres=gpu:1 -A <your_account> --pty bash

# Submit a batch job (uses your guaranteed quota)
sbatch --gres=gpu:2 -A <your_account> my_script.sh

# Submit with specific time limit
sbatch --gres=gpu:1 -A <your_account> --time=24:00:00 my_script.sh

# Burst beyond your quota (preemptable, uses idle GPUs)
sbatch --gres=gpu:4 -A <your_account> --qos=normal my_script.sh
```

Replace `<your_account>` with your professor's group: `nils`, `raul`, `eric`, `eduard`, `michalis`, or `mladen`.

## Example Batch Script

Save as `train.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --gres=gpu:1
#SBATCH --account=nils          # <-- your professor's account
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Your commands here
python train.py --epochs 100
```

Then: `sbatch train.sh`

**Tip for preemptable jobs:** Save checkpoints regularly so your work is not lost if the job is requeued.

## Monitoring

```bash
squeue                    # See all running/pending jobs
squeue -u $USER           # See your jobs
sinfo                     # Node status
sshare -a                 # Fair-share standings
scancel <jobid>           # Cancel a job
sacct -j <jobid>          # Job history/details
quota -s                  # Check your disk usage
```

## Per-User Limits

| Limit | Value |
|-------|-------|
| Max running jobs | 4 |
| Max queued + running | 20 |
| Max GPUs per user | Limited by your group's guaranteed quota (or 8 with `--qos=normal`) |
| Max job duration (gpu partition) | 3 days |
| Max job duration (debug partition) | 1 hour |

## Storage

| Path | Size | Purpose |
|------|------|---------|
| `/shared/home/<you>` | 500 GB quota | Your home directory (backed by NFS, visible on all nodes) |
| `/shared/datasets` | Shared | Shared datasets (please don't duplicate large datasets here) |
| `/shared/software` | Shared | Shared conda environments and tools |
| `/scratch` | 58 TB per node | Fast local scratch (not shared between nodes, not persistent) |

**Your disk quota is 500 GB.** Check your usage with: `quota -s`

## Tips

- Use `/scratch` on compute nodes for temporary large files (checkpoints, intermediate data). Copy final results back to `/shared/home/`.
- Use `--time` to set realistic time limits. Shorter jobs get scheduled faster.
- Use the `debug` partition for quick tests: `srun -p debug --gres=gpu:1 -A <account> --pty bash`
- Install conda environments in your home directory or in `/shared/software` if others need them too.
- **Save checkpoints frequently.** If you use `--qos=normal` to burst beyond your quota, your job may be requeued when the GPU owner needs it back.

## Need Help?

Contact your supervisor or the cluster admin.

## GPU Sharding (sharing a GPU between jobs)

If your job only needs a fraction of a GPU's VRAM (e.g. small inference, eval, debugging), you can request a **shard** instead of a whole GPU. Each physical GPU is split into **10 shards** (~9.6 GB VRAM each on 96 GB cards). Multiple shard jobs can run on the same GPU concurrently.

```bash
# Request 1 shard (~9.6 GB VRAM) instead of a whole GPU
srun --gres=shard:1 -A <your_account> --pty bash

# Batch job with 2 shards
sbatch --gres=shard:2 -A <your_account> my_script.sh
```

**Rules:**
- A physical GPU is either whole-reserved (`--gres=gpu:1`) **or** sharing shards — never both at the same time. Once any shard on a GPU is in use, that GPU cannot be claimed via `--gres=gpu:1` until all shards free up, and vice versa.
- VRAM is **not** isolated between shard jobs. Stay under your shard's budget (~9 GB) or you may OOM your neighbours on the same GPU.
- Use whole-GPU (`--gres=gpu:1`) for training; use shards for inference, evaluation, and debugging.

## Shared model cache (HuggingFace)

**The cluster has a shared HuggingFace model cache** at `/shared/models/huggingface/`. This is a single download point used by everyone, so you don't waste disk or bandwidth re-downloading the same Llama / Qwen / DeepSeek weights.

**Always set this in your job scripts (or your `~/.bashrc`):**

```bash
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export TRANSFORMERS_CACHE=/shared/models/huggingface/hub   # for older HF versions
```

Then `from_pretrained("meta-llama/Llama-3-8B")` and `vllm.LLM("...")` will pull from (or write to) the shared cache automatically.

**Why this matters:**
- Models live **outside your home quota** (your home is capped at 500 GB; models can be 100+ GB each).
- Cache is **shared across all 3 nodes** via NFS, so a download on one node is instantly visible to jobs landing on the others.
- Group-writable, so anyone in `researchers` can populate it. New files automatically inherit the `researchers` group (sgid bit).

**Do NOT use `/scratch` for HF models.** `/scratch` is **per-node local storage** — a model downloaded there is invisible to jobs running on a different node, and you'll end up with duplicate copies eating disk on every machine. Use `/scratch` for *transient* working files only (intermediate checkpoints, working data) and copy final results back to `/shared/home/` or wherever they belong.

**If your home quota is full because of an old `~/.cache/huggingface/`:**

```bash
# Move existing cache into shared cache (one-time)
rsync -aHAX ~/.cache/huggingface/ /shared/models/huggingface/
rm -rf ~/.cache/huggingface
ln -s /shared/models/huggingface ~/.cache/huggingface   # optional: keeps tools that hardcode the path happy
```
