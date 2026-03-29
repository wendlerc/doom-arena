# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ViZDoom deathmatch training, evaluation, and human-vs-AI play using a [patched fork of Sample Factory](https://github.com/wendlerc/sample-factory/tree/doom-arena) APPO. Trains RL agents on `dwango5.wad` deathmatch, records gameplay datasets for world model training, and supports human play against trained agents.

## Setup

```bash
uv sync                    # Install everything (Python 3.10-3.12)
doom-download              # Download pretrained HuggingFace models into sf_train_dir/
```

Requires a display (X11/Wayland) for `doom-play`. Training/recording/eval work headless.

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

All Sample Factory CLI args pass through to `doom-train` (run `doom-train --help`).

## Architecture

### Core Abstraction: SFAgent (`agent.py`)

Central class for model loading and inference. Handles config loading from `sf_train_dir/<experiment>/cfg.json`, checkpoint discovery (tries `best_*` then `checkpoint_*`), observation preprocessing (numpy→torch), and LSTM state management. Used by `run.py`, `evaluate.py`, `play.py`.

### Training Pipeline (`train.py`)

Thin wrapper around Sample Factory's `run_rl()`. Registers ViZDoom components, sets default hyperparameters (APPO, 16 workers × 8 envs, LSTM 512, batch 2048), then delegates to Sample Factory. Checkpoints land in `sf_train_dir/<experiment>/checkpoint_p0/`.

### Observation Pipeline

```
ViZDoom 160×120 CRCGCB → ResizeWrapper 128×72 → CHW → concat 23 measurements
(measurements: selected weapon, ammo, health, armor, weapon/ammo vectors for 8 slots)
```

The `record.py` module replicates this pipeline manually (outside Sample Factory wrappers) via `preprocess_for_model()` and `extract_measurements()` — these must stay in sync with the training wrappers.

### Recording System (`record.py`)

Two modes for recording AI gameplay as WebDataset tar shards:

- **bots**: 1 AI + N bots, uses `.lmp` demo files replayed at 640×480 for video. More reliable.
- **pvp**: 2 AI players via ViZDoom multiplayer (ASYNC_PLAYER + threaded `make_action` to avoid sync deadlocks), frames captured at 160×120 and upscaled. Per-player 20% random policy chance for diversity.

Action conversion: model outputs 6 sub-space indices → `convert_action()` → 14 floats (13 binary buttons + 1 continuous turn delta). This matches `doom_action_space_full_discretized()` from Sample Factory.

### Human Play (`play.py`)

AI runs as multiplayer server (host) in a background thread via Sample Factory env wrappers. Human joins via a raw ViZDoom SPECTATOR mode window on localhost UDP. Bots added via `addbot` command on the host.

### Data Pipeline for World Model Training

```
record.py → WebDataset (MP4 + actions + rewards)
  → encode_dataset.py → DC-AE-Lite latents (32, 15, 20) float16
  → latent_loader.py / fast_loader.py for training
```

Multi-node support via `--worker-id-offset` and `--base-port` flags. Encoding uses `torch.compile` with `reduce-overhead` mode.

## Key Dependencies

- **Sample Factory fork** (`doom-arena` branch): provides APPO training, ViZDoom env wrappers, action spaces, checkpoint management. The fork patches numpy 2.x, PyTorch 2.6+, and gymnasium compatibility.
- **ViZDoom**: Doom engine bindings. WAD files located in `sf_examples/vizdoom/doom/scenarios/`.
- **WebDataset**: tar-based sharded dataset format for recordings and latents.

## Conventions

- Checkpoints: `sf_train_dir/<experiment>/checkpoint_p0/{best,checkpoint}_*.pth`
- Configs: `sf_train_dir/<experiment>/cfg.json`
- Gitignored: `sf_train_dir/`, `datasets/`, `results/`, `wandb/`, `logs/`
- Game runs at 35 tics/sec. Training frameskip=4, recording frameskip=2 (`DECISION_INTERVAL`).
- WandB project: `doom-deathmatch`, groups: `sf_training`, `sf_monitoring`
