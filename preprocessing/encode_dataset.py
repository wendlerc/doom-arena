#!/usr/bin/env python3
"""
Encode Doom video dataset into DC-AE latent representations.

Iterates all episodes in the WebDataset, encodes each frame using
DC-AE-Lite (f32c32: 32x spatial, 32 latent channels), and writes
paired (latent, action, reward) data as new WebDataset shards.

For PvP episodes (detected by presence of video_p2.mp4), encodes both
players' perspectives into separate latent arrays.

Output per episode (PvP):
  {key}.latents_p1.npy   (N, 32, 15, 20) float16
  {key}.latents_p2.npy   (N, 32, 15, 20) float16
  {key}.actions_p1.npy   (N, 14) float32
  {key}.actions_p2.npy   (N, 14) float32
  {key}.rewards_p1.npy   (N,) float32
  {key}.rewards_p2.npy   (N,) float32
  {key}.meta.json         augmented metadata

Output per episode (single-player bots):
  {key}.latents_p1.npy   (N, 32, 15, 20) float16
  {key}.actions_p1.npy   (N, 14) float32
  {key}.rewards_p1.npy   (N,) float32
  {key}.meta.json         augmented metadata

Resume support: tracks completed episode IDs in progress.json.

Usage:
    python preprocessing/encode_dataset.py --data-root datasets/pvp_recordings --output-dir datasets/pvp_latents
    python preprocessing/encode_dataset.py --batch-size 32 --sanity 3
"""
import sys, os, io, json, time, argparse
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
import webdataset as wds

DATA_ROOT = "datasets/pvp_recordings"
OUTPUT_DIR = "datasets/pvp_latents"
MODEL_ID = "mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers"
SHARD_SIZE_MB = 4096  # ~4 GB per shard
ENCODE_BATCH = 64     # frames per encode batch (fp16 fits bs=64 in 36 GB)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def load_progress(output_dir: str, worker_id: int | None = None) -> set[str]:
    """Load all progress files to get the union of completed episode IDs."""
    from pathlib import Path
    done = set()
    # Load main progress file
    main_path = os.path.join(output_dir, "progress.json")
    if os.path.exists(main_path):
        with open(main_path) as f:
            done.update(json.load(f))
    # Load all worker progress files
    for p in Path(output_dir).glob("progress-w*.json"):
        with open(p) as f:
            done.update(json.load(f))
    return done


def save_progress(output_dir: str, done_ids: set[str], worker_id: int | None = None):
    if worker_id is not None:
        path = os.path.join(output_dir, f"progress-w{worker_id}.json")
    else:
        path = os.path.join(output_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(sorted(done_ids), f)


def encode_video_to_latents(dc_ae, mp4_bytes: bytes, batch_size: int,
                            device: str = "cuda") -> np.ndarray:
    """Encode all frames of an MP4 video into latents.

    Decodes video to CPU (avoids GPU VRAM contention with torch.compile
    CUDA graphs), then sends batches to GPU for encoding.

    Returns:
        (N, 32, 15, 20) float16 numpy array
    """
    from doom_arena.fast_loader import decode_video

    video_cpu = decode_video(mp4_bytes, device="cpu")  # (N, 3, H, W) uint8 CPU
    n_frames = video_cpu.shape[0]
    latent_parts = []

    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        batch = video_cpu[start:end].to(device).half().div_(255.0).mul_(2.0).sub_(1.0)

        with torch.no_grad():
            latent = dc_ae.encode(batch).latent  # (B, 32, 15, 20) fp16

        latent_parts.append(latent.cpu().numpy())
        del batch, latent

    del video_cpu
    return np.concatenate(latent_parts, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Encode Doom dataset to DC-AE latents")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=ENCODE_BATCH)
    parser.add_argument("--shard-size-mb", type=int, default=SHARD_SIZE_MB)
    parser.add_argument("--sanity", type=int, default=0,
                        help="Only process N episodes for sanity check (0=all)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (for debugging)")
    parser.add_argument("--worker-id", type=int, default=None,
                        help="Worker ID for multi-node encoding (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of encoding workers")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index to use")
    args = parser.parse_args()

    if args.worker_id is not None and args.worker_id >= args.num_workers:
        print(f"ERROR: --worker-id {args.worker_id} >= --num-workers {args.num_workers}")
        sys.exit(1)

    from diffusers import AutoencoderDC
    from doom_arena.fast_loader import DoomDataset

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_root}...")
    ds = DoomDataset(args.data_root, verbose=True)

    # Load model in fp16 for 2x VRAM savings
    device = f"cuda:{args.gpu}"
    print(f"Loading DC-AE: {args.model_id} (fp16) on {device}...")
    t0 = time.perf_counter()
    dc_ae = AutoencoderDC.from_pretrained(args.model_id, torch_dtype=torch.float16)
    dc_ae = dc_ae.to(device).eval()

    if not args.no_compile:
        print("  Compiling encoder with torch.compile...")
        dc_ae.encoder = torch.compile(dc_ae.encoder, mode="reduce-overhead")
        # Warmup compile with a dummy batch
        dummy = torch.randn(args.batch_size, 3, 480, 640, device=device, dtype=torch.float16)
        for _ in range(3):
            with torch.no_grad():
                _ = dc_ae.encode(dummy).latent
        del dummy
        torch.cuda.empty_cache()

    print(f"  Ready in {time.perf_counter() - t0:.1f}s, "
          f"VRAM: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

    # Resume — load ALL progress files to avoid re-encoding
    all_done_ids = load_progress(args.output_dir, args.worker_id)
    # Track only this worker's progress for saving
    done_ids = set()
    if args.worker_id is not None:
        # Load this worker's own progress file for incremental saves
        wpath = os.path.join(args.output_dir, f"progress-w{args.worker_id}.json")
        if os.path.exists(wpath):
            with open(wpath) as f:
                done_ids = set(json.load(f))
    else:
        done_ids = set(all_done_ids)
    print(f"  Previously completed: {len(all_done_ids)} episodes (global), "
          f"{len(done_ids)} (this worker)")

    n_total = len(ds)
    if args.sanity > 0:
        n_total = min(args.sanity, n_total)
        print(f"  Sanity mode: processing {n_total} episodes only")

    # Write shards — use worker-specific shard prefix to avoid collisions
    from pathlib import Path
    if args.worker_id is not None:
        shard_prefix = f"latent-w{args.worker_id}"
    else:
        shard_prefix = "latent"
    shard_pat = os.path.join(args.output_dir, shard_prefix + "-%06d.tar")
    existing_shards = sorted(Path(args.output_dir).glob(shard_prefix + "-*.tar"))
    if existing_shards:
        last_num = max(int(s.stem.split("-")[-1]) for s in existing_shards)
        start_shard = last_num + 1
        print(f"  Found {len(existing_shards)} existing shards ({shard_prefix}), "
              f"continuing from shard {start_shard:06d}")
    else:
        start_shard = 0

    # Build episode indices for this worker
    all_indices = list(range(n_total))
    if args.worker_id is not None:
        all_indices = [i for i in all_indices if i % args.num_workers == args.worker_id]
        print(f"  Worker {args.worker_id}/{args.num_workers}: "
              f"processing {len(all_indices)} of {n_total} episodes")

    t_start = time.perf_counter()
    n_done = 0
    n_skipped = 0
    total_frames = 0

    with wds.ShardWriter(shard_pat, maxsize=args.shard_size_mb * 1024 * 1024,
                         start_shard=start_shard) as writer:
        for idx, i in enumerate(all_indices):
            ep = None
            try:
                ep = ds[i]
                ep_id = ep.meta.get("episode_id", f"idx_{i}")

                if ep_id in all_done_ids:
                    n_skipped += 1
                    continue

                is_pvp = "video_p2.mp4" in ep._member_names
                t_ep = time.perf_counter()

                # --- Encode P1 video ---
                mp4_p1 = ep._extract_file("video_p1.mp4")
                latents_p1 = encode_video_to_latents(dc_ae, mp4_p1, args.batch_size, device)
                del mp4_p1

                # P1 actions/rewards from Episode (already loaded)
                actions_p1 = ep.actions.numpy()
                rewards_p1 = ep.rewards.numpy()

                # Verify frame-action alignment for P1
                n_frames = latents_p1.shape[0]
                if n_frames != len(actions_p1):
                    print(f"  WARNING: ep {i} P1 frame-action mismatch: "
                          f"{n_frames} latents vs {len(actions_p1)} actions. Truncating.")
                    n_min = min(n_frames, len(actions_p1))
                    latents_p1 = latents_p1[:n_min]
                    actions_p1 = actions_p1[:n_min]
                    rewards_p1 = rewards_p1[:n_min]
                    n_frames = n_min

                # Build output sample
                sample = {
                    "__key__": f"ep_{ep_id}",
                    "latents_p1.npy": _npy_bytes(latents_p1),
                    "actions_p1.npy": _npy_bytes(actions_p1),
                    "rewards_p1.npy": _npy_bytes(rewards_p1),
                }

                # --- Encode P2 video (PvP only) ---
                if is_pvp:
                    mp4_p2 = ep._extract_file("video_p2.mp4")
                    latents_p2 = encode_video_to_latents(dc_ae, mp4_p2, args.batch_size, device)
                    del mp4_p2

                    # Extract P2 actions/rewards manually from tar
                    actions_p2 = np.load(io.BytesIO(ep._extract_file("actions_p2.npy")))
                    rewards_p2 = np.load(io.BytesIO(ep._extract_file("rewards_p2.npy")))

                    # Verify alignment for P2
                    n_frames_p2 = latents_p2.shape[0]
                    if n_frames_p2 != len(actions_p2):
                        print(f"  WARNING: ep {i} P2 frame-action mismatch: "
                              f"{n_frames_p2} latents vs {len(actions_p2)} actions. Truncating.")
                        n_min = min(n_frames_p2, len(actions_p2))
                        latents_p2 = latents_p2[:n_min]
                        actions_p2 = actions_p2[:n_min]
                        rewards_p2 = rewards_p2[:n_min]

                    # P1 and P2 should have same frame count (same game)
                    if latents_p1.shape[0] != latents_p2.shape[0]:
                        print(f"  WARNING: ep {i} P1/P2 frame count mismatch: "
                              f"{latents_p1.shape[0]} vs {latents_p2.shape[0]}. Truncating.")
                        n_min = min(latents_p1.shape[0], latents_p2.shape[0])
                        latents_p1 = latents_p1[:n_min]
                        actions_p1 = actions_p1[:n_min]
                        rewards_p1 = rewards_p1[:n_min]
                        latents_p2 = latents_p2[:n_min]
                        actions_p2 = actions_p2[:n_min]
                        rewards_p2 = rewards_p2[:n_min]
                        n_frames = n_min
                        # Re-write P1 with truncated version
                        sample["latents_p1.npy"] = _npy_bytes(latents_p1)
                        sample["actions_p1.npy"] = _npy_bytes(actions_p1)
                        sample["rewards_p1.npy"] = _npy_bytes(rewards_p1)

                    sample["latents_p2.npy"] = _npy_bytes(latents_p2)
                    sample["actions_p2.npy"] = _npy_bytes(actions_p2)
                    sample["rewards_p2.npy"] = _npy_bytes(rewards_p2)

                    total_frames += n_frames  # count P2 frames too (same video, encoded twice)
                    del latents_p2

                # Augment metadata
                meta = dict(ep.meta)
                meta["latent_shape_p1"] = list(latents_p1.shape)
                meta["latent_dtype"] = "float16"
                meta["latent_spatial_compression"] = 32
                meta["latent_channels"] = 32
                meta["ae_model"] = args.model_id
                meta["n_latent_frames"] = n_frames
                meta["is_pvp"] = is_pvp
                sample["meta.json"] = json.dumps(meta).encode()

                writer.write(sample)

                dt_ep = time.perf_counter() - t_ep
                total_frames += n_frames
                n_done += 1
                done_ids.add(ep_id)
                all_done_ids.add(ep_id)

                # Save progress every episode (safe for kill/restart)
                save_progress(args.output_dir, done_ids, args.worker_id)

                elapsed = time.perf_counter() - t_start
                fps = total_frames / elapsed
                remaining = len(all_indices) - idx - 1
                eta_min = remaining * (elapsed / max(n_done, 1)) / 60

                sc = ep.meta.get("scenario", "?")
                pvp_tag = " [PvP]" if is_pvp else ""
                wid_tag = f" W{args.worker_id}" if args.worker_id is not None else ""
                print(f"  [{n_done}/{len(all_indices) - n_skipped}]{wid_tag} ep={i} {sc}{pvp_tag} "
                      f"{n_frames} frames "
                      f"latent={latents_p1.shape} "
                      f"{dt_ep:.1f}s "
                      f"({fps:.0f} fps, ETA {eta_min:.0f}min)")

                del latents_p1

            except Exception as e:
                print(f"  ERROR ep={i}: {e}")
                import traceback
                traceback.print_exc()

            finally:
                if ep is not None:
                    del ep
                torch.cuda.empty_cache()
                # Periodically reload other workers' progress to avoid duplicates
                if n_done % 50 == 0 and args.worker_id is not None:
                    all_done_ids = load_progress(args.output_dir, args.worker_id)

    # Final progress save
    save_progress(args.output_dir, done_ids, args.worker_id)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {n_done} episodes, {total_frames:,} frames in {elapsed/60:.1f}min")
    print(f"  Avg throughput: {total_frames/elapsed:.0f} fps")
    print(f"  Output: {args.output_dir}")

    # Print output shard stats
    shards = sorted(Path(args.output_dir).glob("latent-*.tar"))
    total_bytes = sum(s.stat().st_size for s in shards)
    print(f"  {len(shards)} shards, {total_bytes/1e9:.1f} GB total")


if __name__ == "__main__":
    main()
