#!/usr/bin/env python3
"""
Monitor training: periodically evaluate the latest checkpoint and log videos to wandb.

Usage:
    doom-monitor --experiment my_run --interval 300
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import wandb

from doom_arena.agent import SFAgent, run_episodes, load_cfg

from sample_factory.algo.learning.learner import Learner


def _get_latest_checkpoint(cfg) -> str | None:
    """Get the name of the latest available checkpoint."""
    for prefix in ["best_*", "checkpoint_*"]:
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(cfg, 0), prefix
        )
        if checkpoints:
            return os.path.basename(checkpoints[-1])
    return None


def _save_video(frames: list, path: str, fps: int = 35):
    """Save frames as video."""
    if not frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="sf_dm_train_v1")
    ap.add_argument("--train-dir", default="./sf_train_dir")
    ap.add_argument("--interval", type=int, default=300, help="Seconds between evaluations")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-evals", type=int, default=50)
    args = ap.parse_args()

    from doom_arena.agent import _ensure_registered
    _ensure_registered()

    run = wandb.init(
        project="doom-deathmatch",
        name=f"monitor_{args.experiment}",
        group="sf_monitoring",
        tags=["monitoring", "sample_factory"],
        config={"experiment": args.experiment, "eval_episodes": args.episodes},
    )

    eval_count = 0
    last_checkpoint = None

    while eval_count < args.max_evals:
        print(f"\n--- Evaluation {eval_count + 1} ---")

        # Check if config exists yet
        try:
            cfg = load_cfg(args.experiment, args.train_dir)
        except FileNotFoundError:
            print("No config available yet, waiting...")
            time.sleep(args.interval)
            continue

        # Check for new checkpoint
        current_ckpt = _get_latest_checkpoint(cfg)
        if current_ckpt is None:
            print("No checkpoint available yet, waiting...")
            time.sleep(args.interval)
            continue

        if current_ckpt == last_checkpoint:
            print(f"Same checkpoint ({current_ckpt}), waiting...")
            time.sleep(args.interval)
            continue

        last_checkpoint = current_ckpt
        eval_count += 1

        # Load agent and evaluate
        agent = SFAgent(args.experiment, args.train_dir, device="cpu")
        results = run_episodes(agent, args.episodes, collect_frames=True)
        agent.close()

        rewards = [ep.reward for ep in results]
        all_frames = [f for ep in results for f in ep.frames]
        avg_reward = float(np.mean(rewards))

        print(f"  Checkpoint: {current_ckpt}")
        print(f"  Avg reward: {avg_reward:.1f}")

        # Save and log video
        video_path = f"results/training_progress/{args.experiment}_eval_{eval_count:03d}.mp4"
        _save_video(all_frames, video_path)

        log_data = {
            "eval/avg_reward": avg_reward,
            "eval/min_reward": min(rewards),
            "eval/max_reward": max(rewards),
            "eval/checkpoint": current_ckpt,
            "eval/eval_count": eval_count,
        }

        if os.path.exists(video_path):
            log_data["eval/gameplay_video"] = wandb.Video(
                video_path, fps=35, format="mp4",
                caption=f"Eval {eval_count}: reward={avg_reward:.1f}",
            )

        wandb.log(log_data)
        print(f"  Logged to wandb (eval {eval_count})")

        if eval_count < args.max_evals:
            time.sleep(args.interval)

    wandb.finish()
    print(f"\nMonitoring complete ({eval_count} evaluations)")


if __name__ == "__main__":
    main()
