# Reproducing Training Results

This document describes how to reproduce the training results with doom-arena.

## Setup

```bash
git clone https://github.com/wendlerc/doom-arena.git
cd doom-arena
uv sync
```

## Baseline: Pretrained Models

Download and evaluate the pretrained models to establish a baseline:

```bash
doom-download --model seed0
doom-eval --experiment 00_bots_128_fs2_narrow_see_0 --episodes 5
```

Expected results (seed0, 5 episodes):
```
Avg reward: 29.0 +/- 3.8
Avg fragcount: 24.6
Avg deathcount: 44.6
```

## Training from Scratch

### Quick Sanity Check (50M frames, ~30 min)

```bash
doom-train --experiment=sanity_check \
    --train_for_env_steps=50000000 \
    --with_wandb=True \
    --wandb_group=reproduce
```

Verified progression (reproduce_v1 run, Feb 28 2026, 12 workers x 8 envs, A6000):
| Frames | Reward | FPS | Notes |
|--------|--------|-----|-------|
| 1M | -6.2 | 23k | Random policy, mostly dying |
| 10M | -2.8 | 23k | Starting to move, occasionally shoots |
| 24M | -1.0 | 23k | Getting first kills, approaching break-even |
| 38M | +1.7 | 23k | Positive reward, consistent kills |
| 50M | +4.6 | 23k | ~6 frags/ep, 37 min wall time |

Evaluation at 50M (10 episodes): **avg reward 6.7, 5.8 frags/ep**.

### Full Training (100M frames, ~1 hour)

```bash
doom-train --experiment=full_run \
    --train_for_env_steps=100000000 \
    --with_wandb=True
```

Expected: ~9-10 avg reward at 100M frames, ~15 frags/ep.

### Extended Training (500M frames, ~5 hours)

```bash
doom-train --experiment=extended_run \
    --train_for_env_steps=500000000 \
    --with_wandb=True
```

Expected: ~75 avg reward at 500M frames, ~63 frags/ep. Far exceeds pretrained models.

### Monitor Progress

In a separate terminal:
```bash
doom-monitor --experiment full_run --interval 300
```

This logs periodic evaluation videos and reward curves to wandb.

## Evaluate Your Model

```bash
# Quick check
doom-eval --experiment full_run --episodes 5

# Thorough evaluation
doom-eval --experiment full_run --episodes 20

# Generate showcase video
doom-run --experiment full_run --episodes 3 --output results/full_run.mp4
```

## Hardware

All timings are for a single NVIDIA A6000 (48GB VRAM). Training uses:
- ~1-2 GB VRAM for the model + batches
- 12-16 CPU workers for parallel environment rollouts
- ~24-28k FPS throughput

Adjust `--num_workers` if you have fewer CPU cores or less RAM.

## Historical Results

### doom-arena reproduction (Feb 28 2026)

| Run | Frames | Reward | Frags/ep | Wall Time |
|-----|--------|--------|----------|-----------|
| reproduce_v1 | 50M | 6.7 | 5.8 | 37 min |

### Previous runs (doom-dashboard project, same algorithm)

| Run | Frames | Reward | Frags/ep | K/D |
|-----|--------|--------|----------|-----|
| sf_dm_train_v1 | 10M | -5.7 | ~3 | <0.5 |
| sf_dm_train_v1 | 100M | 9.15 | ~15 | ~1.2 |
| sf_dm_train_v1 | 155M | 29.8 | ~25 | ~2.5 |
| sf_dm_train_v1 | 500M | 75.3 | ~63 | ~3.2 |

The doom-arena reproduce_v1 run tracks the historical curve closely, confirming the migration is correct.
