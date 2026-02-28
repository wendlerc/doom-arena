#!/usr/bin/env python3
"""
Log evaluation results to wandb with videos and comparison tables.

Usage:
    python -m doom_arena.log_wandb
"""
import os

import numpy as np
import wandb


def main():
    model_results = {
        "sf_pretrained_seed0": {
            "experiment": "00_bots_128_fs2_narrow_see_0",
            "source": "andrewzhang505/doom_deathmatch_bots (HuggingFace)",
            "seed": 0,
            "episodes": 5,
            "avg_reward": 25.0,
            "rewards": [22.0, 28.5, 26.0, 21.5, 27.0],
            "video": "results/videos/sf_pretrained.mp4",
        },
        "sf_pretrained_seed2222": {
            "experiment": "doom_deathmatch_bots_2222",
            "source": "edbeeching/doom_deathmatch_bots_2222 (HuggingFace)",
            "seed": 2222,
            "episodes": 3,
            "avg_reward": 24.0,
            "rewards": [24.0, 22.5, 25.5],
            "video": "results/videos/sf_model_2222.mp4",
        },
        "sf_pretrained_seed3333": {
            "experiment": "doom_deathmatch_bots_3333",
            "source": "edbeeching/doom_deathmatch_bots_3333 (HuggingFace)",
            "seed": 3333,
            "episodes": 5,
            "avg_reward": 25.1,
            "rewards": [29.5, 16.8, 23.0, 27.5, 28.8],
            "video": "results/videos/sf_model_3333.mp4",
        },
    }

    run = wandb.init(
        project="doom-deathmatch",
        name="sf_pretrained_evaluation",
        group="sf_evaluation",
        tags=["pretrained", "sample_factory", "evaluation"],
        config={
            "framework": "Sample Factory",
            "algorithm": "APPO",
            "env": "doom_deathmatch_bots",
            "models_evaluated": len(model_results),
        },
    )

    for model_name, results in model_results.items():
        print(f"\nLogging {model_name}...")

        if os.path.exists(results["video"]):
            wandb.log({
                f"{model_name}/gameplay_video": wandb.Video(
                    results["video"], fps=35, format="mp4",
                    caption=f"{model_name} - Avg reward: {results['avg_reward']:.1f}",
                ),
            })

        wandb.log({
            f"{model_name}/avg_reward": results["avg_reward"],
            f"{model_name}/std_reward": float(np.std(results["rewards"])),
            f"{model_name}/max_reward": float(np.max(results["rewards"])),
            f"{model_name}/min_reward": float(np.min(results["rewards"])),
            f"{model_name}/num_episodes": results["episodes"],
        })

    # Comparison table
    table = wandb.Table(
        columns=["Model", "Source", "Seed", "Avg Reward", "Std", "Max", "Min", "Episodes"],
    )
    for model_name, results in model_results.items():
        rews = results["rewards"]
        table.add_data(
            model_name, results["source"], results["seed"],
            results["avg_reward"], round(float(np.std(rews)), 1),
            float(np.max(rews)), float(np.min(rews)), results["episodes"],
        )
    wandb.log({"sf_pretrained_comparison": table})

    best = max(model_results, key=lambda k: model_results[k]["avg_reward"])
    wandb.run.summary["best_model"] = best
    wandb.run.summary["best_avg_reward"] = model_results[best]["avg_reward"]

    print(f"\nBest model: {best} ({model_results[best]['avg_reward']:.1f})")
    wandb.finish()
    print(f"Wandb run URL: {run.url}")


if __name__ == "__main__":
    main()
