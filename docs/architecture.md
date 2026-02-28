# Architecture

## Overview

Doom Arena trains RL agents to play ViZDoom deathmatch using Asynchronous PPO (APPO) from [Sample Factory](https://github.com/alex-petrenko/sample-factory). The agent learns to navigate, aim, switch weapons, and fight against 7 built-in bots on the classic `dwango5` deathmatch map.

## Algorithm: APPO

APPO (Asynchronous PPO) decouples environment rollouts from gradient updates, achieving much higher throughput than standard PPO:

- **Rollout workers** (16 by default) each run 8 environments, collecting experience asynchronously
- **A single learner** consumes batches and updates the policy
- **V-trace** can optionally correct for off-policy data (disabled by default, uses GAE)
- Achieves ~28k FPS on a single A6000 GPU

Key hyperparameters:
```
learning_rate: 0.0001
batch_size: 2048
num_epochs: 1
gamma: 0.99
gae_lambda: 0.95
ppo_clip_ratio: 0.1
exploration_loss: symmetric_kl (coeff 0.001)
max_grad_norm: 0.0 (no clipping)
```

## Neural Network

```
Observation (128x72 RGB)
    |
    v
ConvNet Simple (3 conv layers)
    |  [32, 64, 128 channels, 3x3/2x2 kernels]
    v
Flatten -> 512 FC
    |
    v
LSTM (512 hidden)
    |
    +--> Policy head -> 39 discrete actions
    |
    +--> Value head -> scalar value estimate
```

Additional inputs (health, ammo, weapon status — 23 values) are concatenated with the CNN output before the LSTM.

## Action Space

39 discrete actions, discretized from a multi-discrete space:

| Group | Actions | Description |
|-------|---------|-------------|
| Movement | forward/back/none | 3 options |
| Strafe | left/right/none | 3 options |
| Attack | attack, speed, combos | varies |
| Turning | 7 discretized turn angles | left/right delta |
| Weapons | select weapon 1-7 | 7 options |

The `Discretized` wrapper flattens these into a single categorical action (39 total).

## Environment

- **Game**: ViZDoom (Doom engine)
- **Map**: `dwango5.wad` — classic 8-player deathmatch arena
- **Opponents**: 7 built-in bots (default difficulty)
- **Resolution**: 128x72 RGB (downscaled from 160x120)
- **Frameskip**: 4 (agent sees every 4th frame)
- **Episode length**: 2 minutes (4200 steps at frameskip=4)
- **Respawn**: automatic after death (2 second delay)

### Reward Shaping

The environment uses `REWARD_SHAPING_DEATHMATCH_V0`:
- **Frags**: +1.0 per kill
- **Deaths**: -0.75 per death
- **Damage dealt**: scaled positive reward
- **Hit count**: small positive reward
- **Health/armor pickups**: small positive reward
- **Weapon pickups**: small positive reward

The true objective (used for checkpoint selection) is frag count.

## Observation Processing

Raw ViZDoom output goes through several wrappers:

1. **SetResolutionWrapper** — sets game to 160x120
2. **ResizeWrapper** — downscales to 128x72
3. **PixelFormatChwWrapper** — HWC -> CHW for PyTorch
4. **DoomAdditionalInput** — adds health, ammo, weapon status as separate obs key
5. **DoomRewardShapingWrapper** — applies reward shaping scheme

## Multiplayer (Human Play)

For human-vs-AI play (`doom-play`), we use ViZDoom's multiplayer networking:

1. **AI player** (host): Runs a `VizdoomEnvMultiplayer` server in a background thread with full observation processing matching training
2. **Human player** (client): Connects via a visible ViZDoom window in `SPECTATOR` mode — native keyboard/mouse input, no wrapper processing needed
3. **Bots**: Added by the server via `addbot` command
4. **Communication**: UDP on localhost, auto-detected port

## Code Organization

```
doom_arena/
  agent.py        # Core: SFAgent class handles model loading, obs processing, inference
                  # Also: run_episodes(), extract_frame(), is_done(), get_reward() helpers
  train.py        # Thin wrapper around Sample Factory's run_rl()
  run.py          # Uses SFAgent + run_episodes() for video generation
  evaluate.py     # Uses SFAgent + run_episodes() for game stats extraction
  monitor.py      # Polling loop that creates SFAgent for each new checkpoint
  play.py         # Multiplayer: AI thread (SFAgent) + human thread (raw ViZDoom SPECTATOR)
  download_models.py  # HuggingFace snapshot_download
  sample_frames.py    # Frame sampling + grid visualization
  log_wandb.py        # One-off wandb result logging
```

The `SFAgent` class in `agent.py` is the central abstraction — it handles config loading, model creation, checkpoint loading, observation preprocessing, and inference. All evaluation/inference scripts use it to avoid code duplication.
