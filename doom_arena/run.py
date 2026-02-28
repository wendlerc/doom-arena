#!/usr/bin/env python3
"""
Run a trained agent and record video/frames.

Usage:
    doom-run --episodes 5 --output results/showcase.mp4
    doom-run --experiment my_run --save-frames dataset/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from doom_arena.agent import SFAgent, run_episodes


def main():
    ap = argparse.ArgumentParser(description="Run trained deathmatch agent")
    ap.add_argument("--train-dir", default="./sf_train_dir")
    ap.add_argument("--experiment", default="00_bots_128_fs2_narrow_see_0")
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--output", "-o", default=None, help="Save video to this path")
    ap.add_argument("--save-frames", default=None, help="Save frames to this directory")
    ap.add_argument("--fps", type=int, default=35)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    agent = SFAgent(args.experiment, args.train_dir, args.checkpoint, args.device)
    results = run_episodes(agent, args.episodes, collect_frames=True)

    # Print per-episode stats
    for i, ep in enumerate(results):
        print(f"  Ep {i+1}: reward={ep.reward:.1f}, steps={ep.steps}, frames={len(ep.frames)}")

    all_frames = [f for ep in results for f in ep.frames]
    rewards = [ep.reward for ep in results]

    # Save video
    if args.output and all_frames:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for f in all_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        duration = len(all_frames) / args.fps
        print(f"\nSaved video: {args.output} ({len(all_frames)} frames, {duration:.1f}s)")

    # Save individual frames
    if args.save_frames and all_frames:
        out_dir = Path(args.save_frames)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(all_frames):
            cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        print(f"\nSaved {len(all_frames)} frames to {args.save_frames}")

    # Summary
    print(f"\n=== Summary ({args.episodes} episodes) ===")
    print(f"  Avg reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Total frames: {len(all_frames)}")

    agent.close()


if __name__ == "__main__":
    main()
