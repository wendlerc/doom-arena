"""Benchmark latent loader throughput with wandb logging."""
import argparse
import time

import wandb

from doom_arena.latent_loader import LatentTrainLoader


def benchmark(args):
    config = dict(
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=42,
    )
    run = wandb.init(
        project="doom-arena",
        name=f"loader-bench_bs{args.batch_size}_w{args.num_workers}_cl{args.clip_len}",
        tags=["benchmark", "latent-loader"],
        config=config,
    )

    loader = LatentTrainLoader(
        args.root,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=42,
    )
    print(f"num_batches={loader.num_batches}")

    t_start = time.time()
    t0 = t_start
    total_clips = 0
    total_frames = 0

    for i, batch in enumerate(loader):
        t1 = time.time()
        dt = t1 - t0
        bs = batch["latents_p1"].shape[0]
        cl = batch["latents_p1"].shape[1]
        total_clips += bs
        total_frames += bs * cl

        wall = t1 - t_start
        clips_per_sec = total_clips / wall if wall > 0 else 0
        frames_per_sec = total_frames / wall if wall > 0 else 0

        wandb.log({
            "batch_idx": i,
            "batch_time_s": dt,
            "cumulative_clips": total_clips,
            "cumulative_frames": total_frames,
            "clips_per_sec": clips_per_sec,
            "frames_per_sec": frames_per_sec,
            "wall_time_s": wall,
        })

        if i % 50 == 0 or i < 5:
            print(f"Batch {i:3d} | dt={dt:6.2f}s | wall={wall:.1f}s | "
                  f"{clips_per_sec:.0f} clips/s | {frames_per_sec:.0f} frames/s",
                  flush=True)
        t0 = t1

    wall_total = time.time() - t_start

    wandb.summary.update({
        "total_wall_s": wall_total,
        "total_clips": total_clips,
        "total_frames": total_frames,
        "avg_clips_per_sec": total_clips / wall_total,
        "avg_frames_per_sec": total_frames / wall_total,
    })

    print(f"\n--- Summary ---")
    print(f"  Total: {total_clips} clips, {total_frames} frames in {wall_total:.1f}s")
    print(f"  Avg: {total_clips/wall_total:.0f} clips/s, {total_frames/wall_total:.0f} frames/s")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/pvp_latents")
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()
    benchmark(args)
