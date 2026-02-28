# Doom Arena

ViZDoom deathmatch training, evaluation, and human-vs-AI play using [Sample Factory](https://github.com/wendlerc/sample-factory) (APPO).

## Install

```bash
uv sync
```

## Quick Start

```bash
# Download pretrained models from HuggingFace
doom-download

# Run a pretrained agent (generates video)
doom-run --episodes 5 --output results/showcase.mp4

# Train from scratch
doom-train --experiment=my_run --with_wandb=True --train_for_env_steps=100000000

# Evaluate in detail (kills, deaths, damage)
doom-eval --experiment my_run --episodes 10

# Monitor training progress (periodic eval + wandb video logging)
doom-monitor --experiment my_run --interval 300

# Play against a trained AI
doom-play --experiment my_run --num-bots 4 --timelimit 5
```

## Architecture

- **Algorithm**: APPO (Asynchronous PPO)
- **Network**: ConvNet Simple -> 512 MLP -> LSTM 512
- **Input**: 128x72 RGB, frameskip=4
- **Actions**: 39 discrete (movement + shooting + weapon switching)
- **Environment**: dwango5.wad deathmatch with 7 bots

## Pretrained Models

| Model | Source | Avg Reward |
|-------|--------|------------|
| seed0 | andrewzhang505/doom_deathmatch_bots | ~25 |
| seed2222 | edbeeching/doom_deathmatch_bots_2222 | ~24 |
| seed3333 | edbeeching/doom_deathmatch_bots_3333 | ~25 |

Training from scratch reaches ~30 reward at 150M frames (~1.5h on A6000), and ~75 reward at 500M frames.
