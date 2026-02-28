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
    download_models.py # Download pretrained models from HuggingFace
    sample_frames.py   # Sample and compare frames across models
    log_wandb.py       # Log evaluation results to wandb
  pyproject.toml
  sf_train_dir/        # Experiment checkpoints (gitignored)
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
