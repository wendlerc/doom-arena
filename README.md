# Doom Arena

ViZDoom deathmatch training, evaluation, and human-vs-AI play using [Sample Factory](https://github.com/wendlerc/sample-factory/tree/doom-arena) APPO.

Train agents that learn to frag bots in Doom deathmatch, evaluate them in detail, and play against them yourself.

## Install

```bash
git clone https://github.com/wendlerc/doom-arena.git
cd doom-arena
uv sync
```

Requires Python 3.10-3.12. Uses a [patched fork of Sample Factory](https://github.com/wendlerc/sample-factory/tree/doom-arena) (installed automatically) with fixes for numpy 2.x, PyTorch 2.6+, and gymnasium compatibility.

## Quick Start

```bash
# Download pretrained models from HuggingFace
doom-download

# Run a pretrained agent and record a video
doom-run --episodes 5 --output results/showcase.mp4

# Train from scratch (~1.5h to match pretrained on A6000)
doom-train --experiment=my_run --with_wandb=True

# Evaluate (kills, deaths, damage stats)
doom-eval --experiment my_run --episodes 10

# Play against the AI yourself
doom-play --experiment my_run --num-bots 4 --timelimit 5
```

## Commands

| Command | Description |
|---------|-------------|
| `doom-train` | Train an APPO agent from scratch |
| `doom-run` | Run a trained agent, record video/frames |
| `doom-eval` | Detailed evaluation (frags, deaths, damage, JSON output) |
| `doom-monitor` | Monitor training: periodic eval + wandb video logging |
| `doom-play` | Play against a trained AI in multiplayer deathmatch |
| `doom-download` | Download pretrained models from HuggingFace |

See [docs/commands.md](docs/commands.md) for full usage details.

## Training Results

### From Scratch

Training from scratch with APPO on a single A6000 GPU:

| Frames | Avg Reward | Frags/ep | Time | Verified |
|--------|-----------|----------|------|----------|
| 10M | -2.8 | ~3 | ~6 min | reproduce_v1 |
| 50M | 6.7 | 5.8 | ~37 min | reproduce_v1 |
| 100M | 9.2 | ~15 | ~1 hr | sf_dm_train_v1 |
| 155M | 29.8 | ~25 | ~1.5 hr | sf_dm_train_v1 |
| 500M | 75.3 | ~63 | ~5 hr | sf_dm_train_v1 |

### Pretrained Models (HuggingFace)

| Model | Source | Avg Reward | Frags/ep |
|-------|--------|-----------|----------|
| seed0 | [andrewzhang505/doom_deathmatch_bots](https://huggingface.co/andrewzhang505/doom_deathmatch_bots) | 29.0 | 24.6 |
| seed2222 | [edbeeching/doom_deathmatch_bots_2222](https://huggingface.co/edbeeching/doom_deathmatch_bots_2222) | 24.0 | ~20 |
| seed3333 | [edbeeching/doom_deathmatch_bots_3333](https://huggingface.co/edbeeching/doom_deathmatch_bots_3333) | 25.1 | ~21 |

From-scratch training matches pretrained quality at ~150M frames and far surpasses it by 500M.

## Architecture

- **Algorithm**: APPO (Asynchronous PPO) via [Sample Factory](https://github.com/alex-petrenko/sample-factory)
- **Network**: ConvNet Simple -> 512 MLP -> LSTM 512
- **Observation**: 128x72 RGB, CHW format, frameskip=4
- **Actions**: 39 discrete (4 movement dirs, 7 weapons, attack, speed, discretized turning)
- **Environment**: ViZDoom `dwango5.wad` deathmatch, 7 built-in bots
- **Training throughput**: ~28k FPS with 16 workers x 8 envs on A6000

## Data Pipeline

The project includes a full pipeline for recording gameplay, encoding to latent representations, and preparing datasets for world model training.

### 1. Record Gameplay

Record AI-vs-AI multiplayer games (PvP with optional bots) or single-player bot games:

```bash
# Record PvP games (2 AI players + random bots), 100h target
python doom_arena/record.py --experiment sf_dm_train_v1 --mode pvp \
    --total-hours 100 --num-workers 4

# Multi-node recording (unique worker-id-offset per node to avoid shard collisions)
python doom_arena/record.py --experiment sf_dm_train_v1 --mode pvp \
    --total-hours 100 --num-workers 4 --worker-id-offset 4 --base-port 5600
```

Output: WebDataset shards in `datasets/pvp_recordings/` containing per-episode MP4 video, actions, rewards, and metadata for both players.

Features:
- **Per-player policy diversity**: Each player independently has a 20% chance of using random actions (vs trained policy)
- **Bot count variation**: PvP games randomly include 0-6 additional bots
- **Multi-node support**: `--worker-id-offset` and `--base-port` prevent shard/port collisions across nodes

### 2. Validate Autoencoder

Before encoding the full dataset, verify DC-AE encode-decode quality:

```bash
python preprocessing/validate_ae.py
```

Generates `preprocessing/ae_validation_report.html` with side-by-side original vs reconstructed frames, PSNR/SSIM metrics, and latent statistics.

### 3. Encode to Latents

Compress video frames to DC-AE-Lite latent representations (96x compression):

```bash
# Single GPU
python preprocessing/encode_dataset.py

# Multi-node parallel encoding (3 workers across 3 GPU nodes)
python preprocessing/encode_dataset.py --worker-id 0 --num-workers 3 --gpu 0  # node A
python preprocessing/encode_dataset.py --worker-id 1 --num-workers 3 --gpu 0  # node B
python preprocessing/encode_dataset.py --worker-id 2 --num-workers 3 --gpu 0  # node C
```

Each frame (480x640x3) is encoded to a (32, 15, 20) float16 latent (19.2 KB per frame). Output WebDataset shards in `datasets/pvp_latents/` contain paired latent-action data for both players.

Features:
- **Multi-node**: Workers use round-robin episode assignment with separate shard files (`latent-w{id}-*.tar`) and progress files (`progress-w{id}.json`)
- **Resume**: Progress tracked per worker; workers cross-check all progress files to avoid re-encoding
- **torch.compile**: Encoder compiled with `reduce-overhead` mode for ~2x throughput

### 4. Inspect & Validate

Inspect the recorded or encoded datasets:

```bash
# Inspect raw PvP recordings (side-by-side P1/P2 video, action heatmaps, downloadable MP4s)
python preprocessing/inspect_pvp.py --n-episodes 5

# Inspect latent dataset (decoded frames, original-vs-decoded PSNR, alignment checks)
python preprocessing/inspect_latents.py --n-episodes 5
```

### Latent Dataset Format

Each episode in the output WebDataset contains:

| File | Shape | Description |
|------|-------|-------------|
| `latents_p1.npy` | (N, 32, 15, 20) float16 | Player 1 encoded frames |
| `latents_p2.npy` | (N, 32, 15, 20) float16 | Player 2 encoded frames (PvP only) |
| `actions_p1.npy` | (N, 14) float32 | Player 1 actions (aligned by frame index) |
| `actions_p2.npy` | (N, 14) float32 | Player 2 actions (PvP only) |
| `rewards_p1.npy` | (N,) float32 | Player 1 rewards |
| `rewards_p2.npy` | (N,) float32 | Player 2 rewards (PvP only) |
| `meta.json` | — | Scenario, bot count, policy flags, latent metadata |

`latents[t]` + `actions[t]` gives the paired observation-action at timestep t.

## Project Structure

```
doom-arena/
  doom_arena/
    agent.py           # SFAgent class: model loading + inference utilities
    train.py           # APPO training with wandb
    run.py             # Run agent, record video/frames
    evaluate.py        # Detailed evaluation with game stats
    monitor.py         # Periodic checkpoint eval during training
    play.py            # Human-vs-AI multiplayer deathmatch
    record.py          # Multi-player gameplay recording pipeline
    fast_loader.py     # GPU-accelerated WebDataset loader (NVDEC)
    download_models.py # Download pretrained models from HuggingFace
    sample_frames.py   # Sample and compare frames across models
    log_wandb.py       # Log evaluation results to wandb
  preprocessing/
    validate_ae.py     # DC-AE encode-decode quality validation
    encode_dataset.py  # Full dataset encoding to latents (multi-node)
    inspect_pvp.py     # PvP recording inspection report
    inspect_latents.py # Latent dataset validation report
  pyproject.toml
  sf_train_dir/        # Experiment checkpoints (gitignored)
  datasets/            # Recorded & encoded datasets (gitignored)
  results/             # Eval results & videos (gitignored)
```

## Dependencies

This project depends on a [forked version of Sample Factory](https://github.com/wendlerc/sample-factory/tree/doom-arena) that includes compatibility patches for modern Python/PyTorch/numpy. The fork is installed automatically via `uv sync`.

## WandB

Training and evaluation metrics are logged to [Weights & Biases](https://wandb.ai):
- Project: `doom-deathmatch`
- Training group: `sf_training`
- Monitoring group: `sf_monitoring`

Pass `--with_wandb=True` to `doom-train` to enable logging.
