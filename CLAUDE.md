# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ViZDoom deathmatch training, evaluation, and human-vs-AI play using a [patched fork of Sample Factory](https://github.com/wendlerc/sample-factory/tree/doom-arena) APPO. Trains RL agents on deathmatch maps (primarily `dwango5.wad`, also `ssl2.wad`), records gameplay datasets for world model training, and supports human play against trained agents.

## Setup

```bash
uv sync                    # Install everything (Python 3.10-3.12)
doom-download              # Download pretrained HuggingFace models into sf_train_dir/
```

Requires a display (X11/Wayland) for `doom-play` and `doom-record-human`. Training/recording/eval work headless.

## CLI Commands

All installed as entry points via `pyproject.toml`:

```bash
doom-train --experiment=my_run --with_wandb=True   # Train APPO agent (~100M frames default)
doom-run --episodes 5 --output results/video.mp4   # Run agent, record video
doom-eval --experiment my_run --episodes 10         # Evaluate (frags, deaths, damage stats)
doom-monitor --experiment my_run --interval 300     # Poll for new checkpoints, log to wandb
doom-play --experiment my_run --num-bots 4          # Human-vs-AI multiplayer deathmatch
doom-download                                       # Download pretrained models
doom-record --experiment my_run --mode pvp --total-hours 10  # Record AI gameplay as WebDataset
doom-record-human --experiment my_run --num-bots 4           # Record human-vs-AI as WebDataset
```

All Sample Factory CLI args pass through to `doom-train` (run `doom-train --help`). CLI args override the defaults set in `train.py`.

## Testing

```bash
python test_deathmatch_fs4.py    # Integration test: 2-player multiplayer with frameskip=4
```

This is a standalone ViZDoom multiplayer smoke test (host + join on localhost, random actions, 1-minute episode). It verifies that `-deathmatch` mode works with the training frameskip.

## Architecture

### Core Abstraction: SFAgent (`doom_arena/agent.py`)

Central class for model loading and inference. Handles config loading from `sf_train_dir/<experiment>/cfg.json`, checkpoint discovery (tries `best_*` then `checkpoint_*`), observation preprocessing (numpy→torch), and LSTM state management. Used by `run.py`, `evaluate.py`, `play.py`, and `record.py`.

### Training Pipeline (`doom_arena/train.py`)

Thin wrapper around Sample Factory's `run_rl()`. Registers ViZDoom components, sets default hyperparameters (APPO, 16 workers × 8 envs, LSTM 512, batch 2048), then delegates to Sample Factory. Checkpoints land in `sf_train_dir/<experiment>/checkpoint_p0/`.

### Observation Pipeline

```
ViZDoom 160×120 CRCGCB → ResizeWrapper 128×72 → CHW → concat 23 measurements
(measurements: selected weapon, ammo, health, armor, weapon/ammo vectors for 8 slots)
```

**Critical sync constraint:** The `record.py` module replicates this pipeline manually (outside Sample Factory wrappers) via `preprocess_for_model()` and `extract_measurements()`. These functions **must stay in sync** with the training wrappers — if the observation pipeline changes in training, `record.py` must be updated to match, otherwise recorded actions will be wrong.

### Recording System (`doom_arena/record.py`)

Two modes for recording AI gameplay as WebDataset tar shards:

- **bots**: 1 AI + N bots, uses `.lmp` demo files replayed at 640×480 for video. More reliable.
- **pvp**: 2 AI players via ViZDoom multiplayer (ASYNC_PLAYER + threaded `make_action` to avoid sync deadlocks), frames captured at 160×120 and upscaled. Per-player 20% random policy chance for diversity.

Supports multiple scenarios with weighted random selection (defined in `SCENARIOS` dict): `dwango5.wad` (3min and 5min variants, 85% combined) and `ssl2.wad` (15%). Multi-node recording via `--worker-id-offset` and `--base-port` flags to avoid shard/port collisions.

Action conversion: model outputs 6 sub-space indices → `convert_action()` → 14 floats (13 binary buttons + 1 continuous turn delta). This matches `doom_action_space_full_discretized()` from Sample Factory.

### Human Recording (`doom_arena/record_human.py`)

Records human-vs-AI gameplay. AI runs as multiplayer host (ASYNC_PLAYER, 160×120, hidden) in a background thread; human joins in SPECTATOR mode (640×480, visible) in the main thread. Both players' frames, actions, and rewards are recorded independently and aligned afterward. Reuses constants and helpers from `record.py`.

### Human Play (`doom_arena/play.py`)

AI runs as multiplayer server (host) in a background thread via Sample Factory env wrappers. Human joins via a raw ViZDoom SPECTATOR mode window on localhost UDP. Bots added via `addbot` command on the host.

### Data Pipeline for World Model Training

```
record.py → WebDataset (MP4 + actions + rewards)
  → preprocessing/encode_dataset.py → DC-AE-Lite latents (32, 15, 20) float16
  → doom_arena/latent_loader.py for training
```

Encoding uses DC-AE-Lite model `mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers` (f32c32: 32× spatial compression, 32 latent channels). Each 480×640×3 frame → (32, 15, 20) float16 latent. Encoding uses `torch.compile` with `reduce-overhead` mode.

### Data Loaders

- **`doom_arena/loader.py`** — Random-access loader for raw WebDataset shards (MP4 video + actions). Provides `DoomDataset` with lazy video decoding and Jupyter visualization helpers (`show_frame()`, `plot_actions()`). Does not import vizdoom/torch/sample_factory.
- **`doom_arena/latent_loader.py`** — WebDataset streaming loader for encoded latent shards. `LatentDataset` for random-access exploration, `LatentTrainLoader` for training (clips episodes into fixed-length windows, yields batches). PvP episodes keep both player perspectives together. Performance: ~20k frames/s on local NVMe, ~2k frames/s on NFS.
- **`doom_arena/fast_loader.py`** — GPU-accelerated loader using NVDEC for video decoding.

**Performance tip:** For training, copy latent shards to local NVMe (`rsync -av datasets/pvp_latents/ /tmp/pvp_latents/`) — 10× throughput vs NFS.

### Preprocessing Tools

- **`preprocessing/encode_dataset.py`** — Encodes MP4 recordings to DC-AE latents. Supports multi-node parallel encoding via `--worker-id` / `--num-workers`, with per-worker progress tracking and resume.
- **`preprocessing/validate_ae.py`** — Generates HTML report with encode-decode quality metrics (PSNR/SSIM).
- **`preprocessing/inspect_pvp.py`** — HTML report for raw PvP recordings (side-by-side video, action heatmaps).
- **`preprocessing/inspect_latents.py`** — HTML report for latent dataset (decoded frames, alignment checks).
- **`preprocessing/explore_latents.py`** — Jupyter notebook (percent-cell format) for interactive latent dataset exploration.

## Key Dependencies

- **Sample Factory fork** (`doom-arena` branch): provides APPO training, ViZDoom env wrappers, action spaces, checkpoint management. The fork patches numpy 2.x, PyTorch 2.6+, and gymnasium compatibility.
- **ViZDoom**: Doom engine bindings. WAD files located in `sf_examples/vizdoom/doom/scenarios/`.
- **WebDataset**: tar-based sharded dataset format for recordings and latents.
- **DC-AE-Lite**: Autoencoder for latent encoding (`mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers`).

## Conventions

- Checkpoints: `sf_train_dir/<experiment>/checkpoint_p0/{best,checkpoint}_*.pth`
- Configs: `sf_train_dir/<experiment>/cfg.json`
- Gitignored: `sf_train_dir/`, `datasets/`, `results/`, `recordings/`, `wandb/`, `logs/`
- Game runs at 35 tics/sec. Training frameskip=4 (agent decides every 4th tic), recording frameskip=2 (`DECISION_INTERVAL`, higher-resolution actions for dataset).
- WandB project: `doom-deathmatch`, groups: `sf_training`, `sf_monitoring`
