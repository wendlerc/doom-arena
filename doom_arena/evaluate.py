#!/usr/bin/env python3
"""
Detailed evaluation: extract kills, deaths, damage from game variables.

Usage:
    doom-eval --experiment my_run --episodes 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from doom_arena.agent import SFAgent, run_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="00_bots_128_fs2_narrow_see_0")
    ap.add_argument("--train-dir", default="./sf_train_dir")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    agent = SFAgent(args.experiment, args.train_dir, device=args.device)
    results = run_episodes(agent, args.episodes)

    # Print per-episode stats
    for i, ep in enumerate(results):
        print(f"  Ep {i+1}: reward={ep.reward:.1f}, steps={ep.steps}, vars={ep.game_vars}")

    rewards = [ep.reward for ep in results]

    print(f"\n=== Summary ({args.episodes} episodes) ===")
    print(f"  Experiment: {args.experiment}")
    print(f"  Checkpoint: {agent.checkpoint_name}")
    print(f"  Avg reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Min/Max reward: {np.min(rewards):.1f} / {np.max(rewards):.1f}")

    # Print game stats if available
    for key in ["fragcount", "deathcount", "hitcount", "damagecount"]:
        vals = [ep.game_vars.get(key) for ep in results if key in ep.game_vars]
        if vals:
            print(f"  Avg {key}: {np.mean(vals):.1f}")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"eval_{args.experiment}.json"

    episode_stats = [
        {"episode": i, "reward": ep.reward, "steps": ep.steps, **ep.game_vars}
        for i, ep in enumerate(results)
    ]
    with open(results_path, "w") as f:
        json.dump({
            "experiment": args.experiment,
            "checkpoint": agent.checkpoint_name,
            "episodes": args.episodes,
            "stats": episode_stats,
            "summary": {
                "avg_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
            }
        }, f, indent=2)
    print(f"\nSaved results to {results_path}")

    agent.close()


if __name__ == "__main__":
    main()
