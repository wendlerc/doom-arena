# Command Reference

All commands are installed as CLI entry points via `uv sync`.

## doom-train

Train an APPO agent from scratch on ViZDoom deathmatch.

```bash
# Basic training (100M frames, ~1hr on A6000)
doom-train --experiment=my_run

# With wandb logging
doom-train --experiment=my_run --with_wandb=True

# Resume a previous run
doom-train --experiment=my_run --restart_behavior=resume

# Custom hyperparameters (CLI args override defaults)
doom-train --experiment=my_run \
    --learning_rate=0.0003 \
    --num_workers=8 \
    --train_for_env_steps=200000000
```

**Default configuration:**
- Algorithm: APPO
- Environment: `doom_deathmatch_bots` (dwango5.wad, 7 bots)
- Workers: 16 x 8 envs = 128 parallel environments
- Batch size: 2048, 1 epoch per batch
- RNN: LSTM (512 hidden), rollout/recurrence length 32
- Learning rate: 0.0001
- Training budget: 100M environment steps
- Checkpoints saved every 120s, best checkpoint every 30s

All Sample Factory CLI arguments are supported. Run `doom-train --help` for the full list.

## doom-run

Run a trained agent and record video or save individual frames.

```bash
# Record video from pretrained model
doom-run --episodes 5 --output results/video.mp4

# Use a specific experiment
doom-run --experiment my_run --episodes 3 --output results/my_run.mp4

# Save individual frames (for dataset generation)
doom-run --experiment my_run --save-frames frames_dir/

# Use latest checkpoint instead of best
doom-run --experiment my_run --checkpoint latest

# Run on GPU
doom-run --experiment my_run --device cuda
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `00_bots_128_fs2_narrow_see_0` | Experiment name |
| `--train-dir` | `./sf_train_dir` | Directory containing experiments |
| `--checkpoint` | `best` | Which checkpoint: `best` or `latest` |
| `--episodes` | `3` | Number of episodes to run |
| `--output` | None | Save video to this path (.mp4) |
| `--save-frames` | None | Save individual frames to this directory |
| `--fps` | `35` | Video framerate |
| `--device` | `cpu` | Torch device (`cpu` or `cuda`) |

## doom-eval

Detailed evaluation with game statistics (frags, deaths, damage).

```bash
# Evaluate pretrained model
doom-eval --experiment 00_bots_128_fs2_narrow_see_0 --episodes 10

# Evaluate your trained model
doom-eval --experiment my_run --episodes 20

# Use GPU for faster inference
doom-eval --experiment my_run --device cuda
```

Results are saved to `results/eval_<experiment>.json` with per-episode stats and summary statistics.

**Output example:**
```
=== Summary (10 episodes) ===
  Experiment: my_run
  Checkpoint: best_000012345_50000000_reward_25.3.pth
  Avg reward: 25.3 +/- 4.1
  Min/Max reward: 18.2 / 32.1
  Avg fragcount: 21.0
  Avg deathcount: 12.5
  Avg hitcount: 340.0
  Avg damagecount: 3500.0
```

## doom-monitor

Monitor training by periodically evaluating the latest checkpoint and logging videos to wandb.

```bash
# Monitor a training run (check every 5 min)
doom-monitor --experiment my_run --interval 300

# Faster polling with more eval episodes
doom-monitor --experiment my_run --interval 120 --episodes 5

# Limit total evaluations
doom-monitor --experiment my_run --max-evals 20
```

Run this alongside `doom-train` to track training progress in wandb. The monitor detects new checkpoints and only evaluates when the checkpoint changes.

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `sf_dm_train_v1` | Experiment to monitor |
| `--train-dir` | `./sf_train_dir` | Directory containing experiments |
| `--interval` | `300` | Seconds between evaluation attempts |
| `--episodes` | `3` | Episodes per evaluation |
| `--max-evals` | `50` | Maximum number of evaluations |

## doom-play

Play against a trained AI in ViZDoom deathmatch. The AI runs as the multiplayer host and you join via a visible game window.

```bash
# Play against pretrained model with 4 bots
doom-play --experiment 00_bots_128_fs2_narrow_see_0 --num-bots 4

# Play against your trained model, 5 minute match
doom-play --experiment my_run --timelimit 5

# Record your gameplay
doom-play --experiment my_run --record results/vs_ai.mp4

# No bots, just you vs the AI
doom-play --experiment my_run --num-bots 0
```

**How it works:**
1. The AI agent starts as a multiplayer server (hidden window)
2. A visible ViZDoom window opens for you to join
3. You play using keyboard/mouse directly in the game window
4. Match results (frags, deaths) are printed at the end

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `00_bots_128_fs2_narrow_see_0` | AI model to play against |
| `--train-dir` | `./sf_train_dir` | Directory containing experiments |
| `--checkpoint` | `best` | Which checkpoint: `best` or `latest` |
| `--num-bots` | `4` | Number of built-in bots in the match |
| `--timelimit` | `5.0` | Match duration in minutes |
| `--device` | `cpu` | Torch device for AI inference |
| `--record` | None | Save video of your view to this path |

**Requirements:** Needs a display (X11/Wayland). Won't work over SSH without X forwarding.

## doom-download

Download pretrained models from HuggingFace.

```bash
# Download all 3 pretrained models
doom-download

# Download a specific model
doom-download --model seed0

# Custom download directory
doom-download --train-dir /path/to/models
```

**Available models:**
| Key | HuggingFace Repo | Reward |
|-----|-----------------|--------|
| `seed0` | andrewzhang505/doom_deathmatch_bots | ~29 |
| `seed2222` | edbeeching/doom_deathmatch_bots_2222 | ~24 |
| `seed3333` | edbeeching/doom_deathmatch_bots_3333 | ~25 |
