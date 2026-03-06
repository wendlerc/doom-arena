"""
WebDataset-based data loader for pre-encoded latent video shards.

Loads DC-AE latent tensors (N, 32, 15, 20) float16 with paired actions
and rewards from WebDataset tar shards. Supports random-access indexing
into frames within episodes, deterministic shuffling for distributed
training, and clip-based iteration for video model training.

PvP episodes keep both player perspectives (P1 and P2) together as a
single sample — their frames are temporally aligned from the same game.

Performance note:
    Shards are ~4GB each. On NFS this yields ~6 frames/s; on local NVMe
    ~85 frames/s (14x faster). For training, copy shards to local storage:

        rsync -av datasets/pvp_latents/ /tmp/pvp_latents/

    Then point the loader at /tmp/pvp_latents.

Usage:
    from doom_arena.latent_loader import LatentDataset, LatentTrainLoader

    # Interactive / exploration
    ds = LatentDataset("datasets/pvp_latents")
    ep = ds[42]
    ep.latents_p1   # (N, 32, 15, 20) float16 torch tensor
    ep.latents_p2   # (N, 32, 15, 20) float16 torch tensor (PvP only)
    ep.actions_p1   # (N, 14) float32
    ep.rewards_p1   # (N,) float32

    # Training pipeline (WebDataset streaming)
    loader = LatentTrainLoader(
        "datasets/pvp_latents",
        clip_len=16,
        batch_size=32,
        num_workers=4,
    )
    for batch in loader:
        latents_p1 = batch["latents_p1"]   # (B, T, 32, 15, 20)
        latents_p2 = batch["latents_p2"]   # (B, T, 32, 15, 20)
        actions_p1 = batch["actions_p1"]   # (B, T, 14)
        actions_p2 = batch["actions_p2"]   # (B, T, 14)
        rewards_p1 = batch["rewards_p1"]   # (B, T)
        rewards_p2 = batch["rewards_p2"]   # (B, T)
"""
from __future__ import annotations

import io
import json
import math
import random
import tarfile
from multiprocessing import Value
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset, get_worker_info
from webdataset.filters import _shuffle

GAME_FPS = 35
LATENT_CHANNELS = 32
LATENT_H = 15
LATENT_W = 20


# ---------------------------------------------------------------------------
# Interactive random-access dataset
# ---------------------------------------------------------------------------

class LatentEpisode:
    """A single episode from the latent dataset.

    PvP episodes have both P1 and P2 data; single-player episodes only have P1.
    """

    def __init__(self, latents_p1, actions_p1, rewards_p1,
                 latents_p2, actions_p2, rewards_p2, meta):
        self.latents_p1 = torch.from_numpy(latents_p1)  # (N, 32, 15, 20) fp16
        self.actions_p1 = torch.from_numpy(actions_p1)   # (N, 14)
        self.rewards_p1 = torch.from_numpy(rewards_p1)   # (N,)
        self.meta = meta
        self.is_pvp = latents_p2 is not None

        if self.is_pvp:
            self.latents_p2 = torch.from_numpy(latents_p2)
            self.actions_p2 = torch.from_numpy(actions_p2)
            self.rewards_p2 = torch.from_numpy(rewards_p2)
        else:
            self.latents_p2 = None
            self.actions_p2 = None
            self.rewards_p2 = None

    @property
    def n_frames(self) -> int:
        return self.latents_p1.shape[0]

    def __repr__(self):
        sc = self.meta.get("scenario", "?")
        pvp = " PvP" if self.is_pvp else ""
        return f"LatentEpisode({sc}{pvp}, {self.n_frames} frames)"

    def __getitem__(self, idx):
        """Index into frames: ep[10:26] returns a slice of all arrays."""
        result = {
            "latents_p1": self.latents_p1[idx],
            "actions_p1": self.actions_p1[idx],
            "rewards_p1": self.rewards_p1[idx],
        }
        if self.is_pvp:
            result["latents_p2"] = self.latents_p2[idx]
            result["actions_p2"] = self.actions_p2[idx]
            result["rewards_p2"] = self.rewards_p2[idx]
        return result


class LatentDataset:
    """Random-access dataset over latent WebDataset shards."""

    def __init__(self, root: str, verbose: bool = True):
        self._root = Path(root)
        self._index: list[dict] = []
        self._build_index(verbose)

    def _build_index(self, verbose: bool):
        shards = sorted(self._root.glob("latent-*.tar"))
        if verbose:
            print(f"Scanning {len(shards)} latent shards in {self._root}...")

        for shard_path in shards:
            shard_str = str(shard_path)
            try:
                with tarfile.open(shard_path, "r") as tar:
                    groups: dict[str, dict[str, str]] = {}
                    for member in tar.getmembers():
                        if member.isdir():
                            continue
                        parts = member.name.split(".", 1)
                        if len(parts) != 2:
                            continue
                        key, ext = parts[0], parts[1]
                        groups.setdefault(key, {})[ext] = member.name

                    for key, members in groups.items():
                        if "latents_p1.npy" not in members:
                            continue
                        # Load metadata
                        meta = {}
                        if "meta.json" in members:
                            meta_bytes = tar.extractfile(
                                tar.getmember(members["meta.json"])
                            ).read()
                            meta = json.loads(meta_bytes)

                        self._index.append({
                            "shard_path": shard_str,
                            "key": key,
                            "members": members,
                            "meta": meta,
                            "is_pvp": "latents_p2.npy" in members,
                            "n_frames": meta.get("n_latent_frames", 0),
                        })
            except (tarfile.TarError, OSError) as e:
                if verbose:
                    print(f"  Warning: skipping {shard_path.name}: {e}")

        if verbose:
            total_frames = sum(e["n_frames"] for e in self._index)
            total_hours = total_frames / GAME_FPS / 3600
            n_pvp = sum(1 for e in self._index if e["is_pvp"])
            print(f"Indexed {len(self._index)} episodes ({total_hours:.1f}h), "
                  f"{n_pvp} PvP, {total_frames:,} frames")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx) -> LatentEpisode:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        entry = self._index[idx]
        shard_path = entry["shard_path"]
        members = entry["members"]

        with tarfile.open(shard_path, "r") as tar:
            def _load_npy(ext):
                if ext not in members:
                    return None
                return np.load(io.BytesIO(
                    tar.extractfile(tar.getmember(members[ext])).read()
                ))

            latents_p1 = _load_npy("latents_p1.npy")
            actions_p1 = _load_npy("actions_p1.npy")
            rewards_p1 = _load_npy("rewards_p1.npy")
            latents_p2 = _load_npy("latents_p2.npy")
            actions_p2 = _load_npy("actions_p2.npy")
            rewards_p2 = _load_npy("rewards_p2.npy")

        if actions_p1 is None:
            actions_p1 = np.empty((latents_p1.shape[0], 14), dtype=np.float32)
        if rewards_p1 is None:
            rewards_p1 = np.empty(latents_p1.shape[0], dtype=np.float32)

        return LatentEpisode(
            latents_p1, actions_p1, rewards_p1,
            latents_p2, actions_p2, rewards_p2,
            entry["meta"],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sample(self) -> LatentEpisode:
        return self[random.randint(0, len(self) - 1)]

    def summary(self):
        n = len(self._index)
        if n == 0:
            print("LatentDataset: empty")
            return
        total_frames = sum(e["n_frames"] for e in self._index)
        total_hours = total_frames / GAME_FPS / 3600
        n_pvp = sum(1 for e in self._index if e["is_pvp"])
        scenarios: dict[str, int] = {}
        for e in self._index:
            sc = e["meta"].get("scenario", "unknown")
            scenarios[sc] = scenarios.get(sc, 0) + 1
        print(f"LatentDataset: {n} episodes, {total_hours:.1f}h, {total_frames:,} frames")
        print(f"  PvP: {n_pvp} | Single-player: {n - n_pvp}")
        print(f"  Scenarios:")
        for sc, count in sorted(scenarios.items(), key=lambda x: -x[1]):
            print(f"    {sc}: {count}")

    def __repr__(self):
        return f"LatentDataset({self._root}, {len(self)} episodes)"


# ---------------------------------------------------------------------------
# WebDataset streaming training loader
# ---------------------------------------------------------------------------

_SHARD_SHUFFLE_SIZE = 100
_SHARD_SHUFFLE_INITIAL = 10


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


class _DetShuffle(wds.PipelineStage):
    """Deterministic shuffle with epoch-aware seeding for reproducibility."""

    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            worker_info = get_worker_info()
            seed = worker_info.seed if worker_info else 0
            seed += epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class _ResampledShards(IterableDataset):
    """Sample shards with replacement for infinite streaming."""

    def __init__(self, urls, epoch=-1, deterministic=True):
        super().__init__()
        self.urls = list(urls)
        self.epoch = epoch
        self.deterministic = deterministic
        self.rng = random.Random()

    def __iter__(self):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            worker_info = get_worker_info()
            seed = (worker_info.seed if worker_info else 0) + epoch
            self.rng.seed(seed)
        while True:
            yield dict(url=self.rng.choice(self.urls))


def _decode_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


def _extract_clip(sample: dict, clip_len: int, rng: random.Random) -> dict:
    """Extract a random clip of clip_len frames from an episode sample.

    Keeps P1 and P2 temporally aligned.
    """
    latents_p1 = sample["latents_p1.npy"]
    n_frames = latents_p1.shape[0]

    if n_frames <= clip_len:
        start = 0
    else:
        start = rng.randint(0, n_frames - clip_len)
    end = min(start + clip_len, n_frames)
    actual_len = end - start

    def _slice_and_pad(arr):
        clip = arr[start:end]
        if actual_len < clip_len:
            pad_shape = (clip_len - actual_len,) + clip.shape[1:]
            clip = np.concatenate([clip, np.zeros(pad_shape, dtype=clip.dtype)])
        return clip

    n = latents_p1.shape[0]
    result = {
        "latents_p1": _slice_and_pad(latents_p1),
        "actions_p1": _slice_and_pad(sample["actions_p1.npy"]) if "actions_p1.npy" in sample else np.zeros((clip_len, 14), dtype=np.float32),
        "rewards_p1": _slice_and_pad(sample["rewards_p1.npy"]) if "rewards_p1.npy" in sample else np.zeros(clip_len, dtype=np.float32),
        "n_frames": actual_len,
    }

    if "latents_p2.npy" in sample:
        result["latents_p2"] = _slice_and_pad(sample["latents_p2.npy"])
        result["actions_p2"] = _slice_and_pad(sample["actions_p2.npy"]) if "actions_p2.npy" in sample else np.zeros((clip_len, 14), dtype=np.float32)
        result["rewards_p2"] = _slice_and_pad(sample["rewards_p2.npy"]) if "rewards_p2.npy" in sample else np.zeros(clip_len, dtype=np.float32)
    else:
        # Pad with zeros for non-PvP episodes so batches have uniform keys
        result["latents_p2"] = np.zeros_like(result["latents_p1"])
        result["actions_p2"] = np.zeros_like(result["actions_p1"])
        result["rewards_p2"] = np.zeros_like(result["rewards_p1"])

    return result


def _collate_clips(batch: list[dict]) -> dict:
    """Stack a list of clip dicts into batched tensors."""
    return {
        "latents_p1": torch.from_numpy(np.stack([b["latents_p1"] for b in batch])),
        "latents_p2": torch.from_numpy(np.stack([b["latents_p2"] for b in batch])),
        "actions_p1": torch.from_numpy(np.stack([b["actions_p1"] for b in batch])),
        "actions_p2": torch.from_numpy(np.stack([b["actions_p2"] for b in batch])),
        "rewards_p1": torch.from_numpy(np.stack([b["rewards_p1"] for b in batch])),
        "rewards_p2": torch.from_numpy(np.stack([b["rewards_p2"] for b in batch])),
        "n_frames": torch.tensor([b["n_frames"] for b in batch]),
    }


def log_and_continue(exn):
    import logging
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


class LatentTrainLoader:
    """WebDataset-based streaming loader for latent video training.

    Features (following best practices from OpenCLIP/latent_clip):
    - Deterministic shuffling at shard and sample level with epoch seeding
    - Shard resampling for infinite streaming (no epoch boundaries)
    - Multi-worker data loading with proper shard splitting
    - Random clip extraction with P1/P2 temporal alignment
    - Uniform batch keys (non-PvP episodes get zero-padded P2)

    Args:
        root: path to latent shard directory
        clip_len: number of frames per training clip
        batch_size: batch size
        num_workers: dataloader workers
        seed: random seed for shuffling
        epoch: initial epoch (use set_epoch() to update)
        resampled: if True, sample shards with replacement (infinite)
        num_samples: total samples per epoch (required if resampled)
        world_size: for distributed training
        rank: for distributed training
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 16,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        epoch: int = 0,
        resampled: bool = True,
        num_samples: int | None = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.root = Path(root)
        self.clip_len = clip_len
        self.batch_size = batch_size
        self.world_size = world_size
        self._shared_epoch = SharedEpoch(epoch)

        # Find all shards
        shard_paths = sorted(self.root.glob("latent-*.tar"))
        shard_urls = [str(p) for p in shard_paths]
        assert len(shard_urls) > 0, f"No latent-*.tar shards found in {root}"

        # Clip extraction RNG (per-worker seeded)
        clip_rng = random.Random(seed)

        # Build pipeline
        if resampled:
            pipeline = [_ResampledShards(
                shard_urls,
                epoch=self._shared_epoch,
                deterministic=True,
            )]
        else:
            pipeline = [wds.SimpleShardList(shard_urls)]

        if not resampled:
            pipeline.extend([
                _DetShuffle(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=seed,
                    epoch=self._shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])

        pipeline.extend([
            wds.tarfile_to_samples(handler=log_and_continue),
            # Decode all numpy arrays in one pass (handles both P1 and P2)
            wds.map(_decode_all_npy, handler=log_and_continue),
            # Extract random clip (randomness provides within-episode shuffling)
            wds.map(lambda sample: _extract_clip(sample, clip_len, clip_rng), handler=log_and_continue),
            # Batch
            wds.batched(batch_size, partial=False, collation_fn=_collate_clips),
        ])

        dataset = wds.DataPipeline(*pipeline)

        # Compute epoch length
        if resampled:
            if num_samples is None:
                # Estimate from shard count (rough: ~13 episodes/shard)
                num_samples = len(shard_urls) * 13
            global_batch_size = batch_size * world_size
            num_batches = math.ceil(num_samples / global_batch_size)
            num_workers_actual = max(1, num_workers)
            num_worker_batches = math.ceil(num_batches / num_workers_actual)
            num_batches = num_worker_batches * num_workers_actual
            num_samples = num_batches * global_batch_size
            dataset = dataset.with_epoch(num_worker_batches)

        self.num_samples = num_samples
        self.num_batches = num_batches if resampled else None

        self._dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    def set_epoch(self, epoch: int):
        self._shared_epoch.set_value(epoch)

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        if self.num_batches is not None:
            return self.num_batches
        raise TypeError("Length unknown for non-resampled loader; iterate to exhaust.")


def _decode_all_npy(sample: dict) -> dict:
    """Decode all .npy byte buffers in a sample to numpy arrays."""
    for key in list(sample.keys()):
        if key.endswith(".npy") and isinstance(sample[key], bytes):
            sample[key] = _decode_npy(sample[key])
    return sample
