#!/usr/bin/env python3
"""
Sample frames from trained models for visual comparison.

Usage:
    python -m doom_arena.sample_frames
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from doom_arena.agent import SFAgent, run_episodes


def create_frame_grid(frames_dict: dict[str, list], output_path: str, scale: int = 3):
    """Create a grid of sampled frames from multiple models."""
    num_samples = max(len(f) for f in frames_dict.values())
    sample_frame = list(frames_dict.values())[0][0]
    h, w = sample_frame.shape[:2]
    sh, sw = h * scale, w * scale

    label_h = 30
    grid = np.zeros(
        (len(frames_dict) * (sh + label_h) + label_h, num_samples * sw, 3),
        dtype=np.uint8,
    )

    for i, (model_name, frames) in enumerate(frames_dict.items()):
        y_offset = i * (sh + label_h) + label_h
        cv2.putText(
            grid, model_name, (10, y_offset - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        for j, frame in enumerate(frames):
            x_offset = j * sw
            scaled = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_NEAREST)
            scaled = cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR)
            grid[y_offset:y_offset + sh, x_offset:x_offset + sw] = scaled

    cv2.imwrite(output_path, grid)
    return output_path


def main():
    output_dir = Path("results/frame_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "SF Seed 0": "00_bots_128_fs2_narrow_see_0",
        "SF Seed 2222": "doom_deathmatch_bots_2222",
        "SF Seed 3333": "doom_deathmatch_bots_3333",
    }

    all_frames = {}
    for label, experiment in models.items():
        print(f"Sampling frames from {label} ({experiment})...")
        agent = SFAgent(experiment)
        results = run_episodes(agent, 1, collect_frames=True)
        agent.close()

        frames = results[0].frames
        # Sample at regular intervals
        num_samples = 8
        if len(frames) >= num_samples:
            indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
            frames = [frames[i] for i in indices]

        all_frames[label] = frames
        print(f"  Got {len(frames)} sample frames")

        # Save individual frames
        for i, f in enumerate(frames):
            fname = f"{experiment}_sample_{i:02d}.png"
            cv2.imwrite(str(output_dir / fname), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    # Create comparison grid
    grid_path = str(output_dir / "sf_models_comparison_grid.png")
    create_frame_grid(all_frames, grid_path)
    print(f"\nSaved comparison grid: {grid_path}")

    print(f"All frames saved to {output_dir}")


if __name__ == "__main__":
    main()
