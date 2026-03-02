"""
GPU-accelerated data loader for Doom gameplay WebDataset shards.

Uses PyNvVideoCodec (NVIDIA NVDEC hardware) for video decoding at ~3800 fps.
Falls back to cv2 if PyNvVideoCodec is unavailable.

Usage:
    from doom_arena.fast_loader import DoomDataset, DoomTrainLoader

    # Interactive (same API as loader.py, but returns GPU tensors)
    ds = DoomDataset("datasets/mp_recordings")
    ep = ds[42]
    ep.video          # (N, 3, H, W) float32 torch tensor on cuda
    ep.show_frame(0)  # matplotlib still works

    # Training pipeline
    loader = DoomTrainLoader("datasets/mp_recordings", clip_len=16, batch_size=32)
    for batch in loader:
        video = batch["video"]    # (B, T, 3, H, W) float32 cuda
        actions = batch["actions"] # (B, T, 14) float32
        rewards = batch["rewards"] # (B, T) float32
"""
from __future__ import annotations

import io
import json
import os
import random
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

# Use RAM-backed tmpdir if available (avoids disk I/O for temp video files)
_TMPDIR = "/dev/shm" if os.path.isdir("/dev/shm") else None

try:
    import PyNvVideoCodec as nvc
    HAS_NVCODEC = True
except ImportError:
    HAS_NVCODEC = False

# Constants (no vizdoom/sample_factory import needed)
GAME_FPS = 35
BUTTON_NAMES = [
    "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_RIGHT", "MOVE_LEFT",
    "SELECT_WEAPON1", "SELECT_WEAPON2", "SELECT_WEAPON3", "SELECT_WEAPON4",
    "SELECT_WEAPON5", "SELECT_WEAPON6", "SELECT_WEAPON7",
    "ATTACK", "SPEED", "TURN_LEFT_RIGHT_DELTA",
]


# --- Video Decoding ---

def decode_video_gpu(mp4_bytes: bytes, gpu_id: int = 0) -> torch.Tensor:
    """Decode MP4 bytes to (N, 3, H, W) uint8 tensor on GPU via NVDEC.

    Uses a temp file since PyNvVideoCodec SimpleDecoder requires a file path.
    The decode itself is hardware-accelerated (~3800 fps on A6000).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=_TMPDIR)
    tmp.write(mp4_bytes)
    tmp.close()
    try:
        dec = nvc.SimpleDecoder(
            tmp.name,
            gpu_id=gpu_id,
            output_color_type=nvc.OutputColorType.RGBP,  # (3, H, W) planar
        )
        meta = dec.get_stream_metadata()
        frames = dec.get_batch_frames(batch_size=meta.num_frames)
        if not frames:
            return torch.empty((0, 3, 480, 640), dtype=torch.uint8, device=f"cuda:{gpu_id}")
        tensors = [torch.from_dlpack(f) for f in frames]
        return torch.stack(tensors)
    finally:
        os.unlink(tmp.name)


def decode_video_cpu(mp4_bytes: bytes) -> torch.Tensor:
    """Fallback: decode MP4 bytes to (N, 3, H, W) uint8 tensor on CPU via cv2."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=_TMPDIR)
    tmp.write(mp4_bytes)
    tmp.close()
    try:
        cap = cv2.VideoCapture(tmp.name)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))  # HWC → CHW
        cap.release()
        if not frames:
            return torch.empty((0, 3, 480, 640), dtype=torch.uint8)
        return torch.stack(frames)
    finally:
        os.unlink(tmp.name)


def decode_video(mp4_bytes: bytes, device: str = "cuda", gpu_id: int = 0) -> torch.Tensor:
    """Decode MP4 bytes to (N, 3, H, W) uint8 tensor. GPU if available, else CPU."""
    if device != "cpu" and HAS_NVCODEC and torch.cuda.is_available():
        return decode_video_gpu(mp4_bytes, gpu_id)
    return decode_video_cpu(mp4_bytes)


def decode_single_frame_gpu(mp4_bytes: bytes, frame_idx: int, gpu_id: int = 0) -> torch.Tensor:
    """Decode a single frame via GPU seek. Returns (3, H, W) uint8 on GPU."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=_TMPDIR)
    tmp.write(mp4_bytes)
    tmp.close()
    try:
        dec = nvc.SimpleDecoder(
            tmp.name,
            gpu_id=gpu_id,
            output_color_type=nvc.OutputColorType.RGBP,
        )
        frames = dec.get_batch_frames_by_index([frame_idx])
        if not frames:
            raise IndexError(f"Frame {frame_idx} not found")
        return torch.from_dlpack(frames[0]).clone()  # clone: dlpack memory freed when decoder exits
    finally:
        os.unlink(tmp.name)


def decode_single_frame_cpu(mp4_bytes: bytes, frame_idx: int) -> torch.Tensor:
    """Fallback: decode single frame via cv2 seek. Returns (3, H, W) uint8 CPU."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=_TMPDIR)
    tmp.write(mp4_bytes)
    tmp.close()
    try:
        cap = cv2.VideoCapture(tmp.name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IndexError(f"Frame {frame_idx} not found")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1)
    finally:
        os.unlink(tmp.name)


# --- Episode ---

class Episode:
    """A single Doom gameplay episode with GPU-accelerated video decoding.

    Attributes:
        actions: (N, 14) float32 torch tensor
        rewards: (N,) float32 torch tensor
        meta: dict with episode metadata
        video_uint8: lazy (N, 3, H, W) uint8 on GPU (cached, ~5.8GB per 3-min ep)
        video: lazy (N, 3, H, W) float32 [0,1] on CPU (not cached)
        numpy_video: (N, H, W, 3) uint8 numpy for matplotlib
    """

    def __init__(self, actions, rewards, meta, shard_path, key, member_names, device="cuda"):
        self.actions = torch.from_numpy(actions)
        self.rewards = torch.from_numpy(rewards)
        self.meta = meta
        self._shard_path = shard_path
        self._key = key
        self._member_names = member_names
        self._device = device
        self._video_cache = None

    def _extract_file(self, ext: str) -> bytes:
        name = self._member_names.get(ext)
        if name is None:
            raise KeyError(f"No {ext} in episode {self._key}")
        with tarfile.open(self._shard_path, "r") as tar:
            return tar.extractfile(tar.getmember(name)).read()

    @property
    def n_frames(self) -> int:
        return self.meta.get("n_frames", len(self.actions))

    @property
    def video_uint8(self) -> torch.Tensor:
        """Lazy-load (N, 3, H, W) uint8 tensor (GPU if available)."""
        if self._video_cache is None:
            mp4_bytes = self._extract_file("video_p1.mp4")
            self._video_cache = decode_video(mp4_bytes, device=self._device)
        return self._video_cache

    @property
    def video(self) -> torch.Tensor:
        """(N, 3, H, W) float32 [0,1] on CPU. Full float32 videos are too large for VRAM
        (~23GB for a 3-min episode), so this returns a CPU tensor. For GPU training,
        use DoomTrainLoader which slices small clips from cached uint8."""
        return self.video_uint8.cpu().float().div_(255.0)

    @property
    def numpy_video(self) -> np.ndarray:
        """(N, H, W, 3) uint8 numpy for matplotlib compatibility."""
        v = self.video_uint8.cpu()
        return v.permute(0, 2, 3, 1).numpy()

    @property
    def demo(self) -> bytes:
        return self._extract_file("demo_p1.lmp")

    def get_frame(self, i: int) -> torch.Tensor:
        """Single frame (3, H, W) uint8 tensor without full video decode."""
        if self._video_cache is not None:
            return self._video_cache[i]
        mp4_bytes = self._extract_file("video_p1.mp4")
        if self._device != "cpu" and HAS_NVCODEC and torch.cuda.is_available():
            return decode_single_frame_gpu(mp4_bytes, i)
        return decode_single_frame_cpu(mp4_bytes, i)

    def show_frame(self, i: int = 0, ax=None):
        """Display frame i using matplotlib."""
        import matplotlib.pyplot as plt

        frame = self.get_frame(i).cpu().permute(1, 2, 0).numpy()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
        ax.imshow(frame)
        sc = self.meta.get("scenario", "?")
        frags = self.meta.get("frag_p1", 0)
        ax.set_title(f"Frame {i}/{self.n_frames} | {sc} | frags={frags:.0f}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def play(self, start: int = 0, end: int | None = None, fps: int | None = None,
             max_frames: int = 300):
        """Play video segment in notebook as HTML5 animation."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if fps is None:
            fps = GAME_FPS
        if end is None:
            end = min(start + max_frames, self.n_frames)
        end = min(end, self.n_frames)

        frames = self.numpy_video[start:end]
        if len(frames) > max_frames:
            step = len(frames) // max_frames
            frames = frames[::step]

        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.axis("off")
        im = ax.imshow(frames[0])

        def update(i):
            im.set_data(frames[i])
            ax.set_title(f"Frame {start + i}")
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frames),
                             interval=1000 / fps, blit=True)
        plt.close(fig)

        try:
            from IPython.display import HTML, display
            display(HTML(anim.to_html5_video()))
        except ImportError:
            plt.show()

    def plot_actions(self, figsize=(14, 6)):
        """Plot action channels: heatmap + activation frequency bar chart."""
        import matplotlib.pyplot as plt

        names = self.meta.get("button_names", BUTTON_NAMES)
        acts = self.actions.numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                        gridspec_kw={"width_ratios": [3, 1]})
        ax1.imshow(acts.T, aspect="auto", interpolation="nearest", cmap="viridis")
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("Frame")
        ax1.set_title("Action channels over time")

        freq = np.mean(np.abs(acts), axis=0)
        ax2.barh(range(len(names)), freq)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_title("Mean |activation|")
        plt.tight_layout()
        plt.show()

    def plot_rewards(self, figsize=(10, 4)):
        """Plot per-step and cumulative rewards."""
        import matplotlib.pyplot as plt

        r = self.rewards.numpy()
        t = np.arange(len(r)) / GAME_FPS
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.plot(t, r, linewidth=0.5)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Reward")
        ax1.set_title("Per-step reward")
        ax2.plot(t, np.cumsum(r))
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Cumulative reward")
        ax2.set_title("Cumulative reward")
        plt.tight_layout()
        plt.show()

    def summary(self):
        m = self.meta
        nf = self.n_frames
        backend = "NVDEC" if (HAS_NVCODEC and self._device != "cpu") else "cv2"
        print(f"Episode: {m.get('episode_id', '?')[:12]}...")
        print(f"  Scenario: {m.get('scenario')} ({m.get('map')})")
        print(f"  Mode: {m.get('mode')} | Bots: {m.get('n_bots')}")
        print(f"  Frames: {nf} ({nf / GAME_FPS:.1f}s)")
        print(f"  Frags: {m.get('frag_p1', 0):.0f} | Deaths: {m.get('death_p1', 0):.0f}")
        print(f"  Total reward: {m.get('total_reward_p1', 0):.1f}")
        print(f"  Random policy: {m.get('random_policy', False)}")
        print(f"  Backend: {backend} | Device: {self._device}")

    def __repr__(self):
        sc = self.meta.get("scenario", "?")
        return f"Episode({sc}, {self.n_frames} frames, {self.meta.get('frag_p1', 0):.0f} frags)"


# --- Dataset ---

class DoomDataset:
    """GPU-accelerated random-access dataset over Doom WebDataset shards.

    Same API as doom_arena.loader.DoomDataset but returns torch tensors
    and uses NVDEC hardware for video decoding.
    """

    def __init__(self, root: str, device: str = "cuda", verbose: bool = True):
        self._root = Path(root)
        self._device = device
        self._index: list[dict] = []
        self._build_index(verbose)

    @classmethod
    def _from_index(cls, root: Path, index: list[dict], device: str) -> DoomDataset:
        obj = cls.__new__(cls)
        obj._root = root
        obj._device = device
        obj._index = index
        return obj

    def _build_index(self, verbose: bool):
        shards = sorted(self._root.glob("mp-*.tar"))
        if verbose:
            backend = "NVDEC" if (HAS_NVCODEC and self._device != "cpu") else "cv2"
            print(f"Scanning {len(shards)} shards (backend={backend})...")

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
                        meta_name = members.get("meta.json")
                        if meta_name is None:
                            continue
                        meta_member = tar.getmember(meta_name)
                        meta_bytes = tar.extractfile(meta_member).read()
                        meta = json.loads(meta_bytes)
                        self._index.append({
                            "shard_path": shard_str,
                            "key": key,
                            "members": members,
                            "meta": meta,
                        })
            except (tarfile.TarError, OSError) as e:
                if verbose:
                    print(f"  Warning: skipping {shard_path.name}: {e}")

        if verbose:
            total_hours = sum(
                e["meta"].get("n_frames", 0) for e in self._index
            ) / GAME_FPS / 3600
            print(f"Indexed {len(self._index)} episodes ({total_hours:.1f}h)")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx) -> Episode | list[Episode]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        entry = self._index[idx]
        shard_path = entry["shard_path"]
        members = entry["members"]

        with tarfile.open(shard_path, "r") as tar:
            actions_name = members.get("actions_p1.npy")
            rewards_name = members.get("rewards_p1.npy")
            actions = np.load(io.BytesIO(
                tar.extractfile(tar.getmember(actions_name)).read()
            )) if actions_name else np.empty((0, 14), dtype=np.float32)
            rewards = np.load(io.BytesIO(
                tar.extractfile(tar.getmember(rewards_name)).read()
            )) if rewards_name else np.empty(0, dtype=np.float32)

        return Episode(
            actions=actions,
            rewards=rewards,
            meta=entry["meta"],
            shard_path=shard_path,
            key=entry["key"],
            member_names=members,
            device=self._device,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sample(self) -> Episode:
        return self[random.randint(0, len(self) - 1)]

    def filter(self, **kwargs) -> DoomDataset:
        filtered = [
            e for e in self._index
            if all(e["meta"].get(k) == v for k, v in kwargs.items())
        ]
        return DoomDataset._from_index(self._root, filtered, self._device)

    def summary(self):
        n = len(self._index)
        if n == 0:
            print("DoomDataset: empty")
            return

        total_frames = sum(e["meta"].get("n_frames", 0) for e in self._index)
        total_hours = total_frames / GAME_FPS / 3600
        scenarios: dict[str, int] = {}
        total_frags = total_deaths = 0.0
        random_count = 0
        for e in self._index:
            m = e["meta"]
            sc = m.get("scenario", "unknown")
            scenarios[sc] = scenarios.get(sc, 0) + 1
            total_frags += m.get("frag_p1", 0)
            total_deaths += m.get("death_p1", 0)
            random_count += int(m.get("random_policy", False))

        backend = "NVDEC" if (HAS_NVCODEC and self._device != "cpu") else "cv2"
        print(f"DoomDataset: {n} episodes, {total_hours:.1f}h ({backend} on {self._device})")
        print(f"  Total frames: {total_frames:,}")
        print(f"  Scenarios:")
        for sc, count in sorted(scenarios.items(), key=lambda x: -x[1]):
            print(f"    {sc}: {count} ({count / n * 100:.1f}%)")
        print(f"  Avg frags/ep: {total_frags / n:.1f}")
        print(f"  Avg deaths/ep: {total_deaths / n:.1f}")
        print(f"  Random policy: {random_count} ({random_count / n * 100:.1f}%)")

    def __repr__(self):
        return f"DoomDataset({self._root}, {len(self)} episodes, {self._device})"


# --- Training Data Loader ---

class _ClipIndex:
    """Pre-computed index of (episode_idx, start_frame) clips for training."""

    def __init__(self, dataset: DoomDataset, clip_len: int, stride: int):
        self.clips: list[tuple[int, int]] = []
        for i, entry in enumerate(dataset._index):
            n = entry["meta"].get("n_frames", 0)
            for start in range(0, max(1, n - clip_len + 1), stride):
                self.clips.append((i, start))

    def __len__(self):
        return len(self.clips)

    def shuffle(self):
        random.shuffle(self.clips)


class DoomTrainLoader:
    """PyTorch-compatible training data loader with GPU video decoding.

    Yields batches of fixed-length video clips with synchronized actions/rewards.

    Usage:
        loader = DoomTrainLoader("datasets/mp_recordings", clip_len=16, batch_size=32)
        for batch in loader:
            video = batch["video"]    # (B, T, 3, H, W) float32 [0,1] on cuda
            actions = batch["actions"] # (B, T, 14) float32
            rewards = batch["rewards"] # (B, T) float32
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 16,
        stride: int = 8,
        batch_size: int = 32,
        device: str = "cuda",
        shuffle: bool = True,
        max_cache: int = 32,
        verbose: bool = True,
        **filter_kwargs,
    ):
        self._ds = DoomDataset(root, device=device, verbose=verbose)
        if filter_kwargs:
            self._ds = self._ds.filter(**filter_kwargs)
        self._clip_len = clip_len
        self._stride = stride
        self._batch_size = batch_size
        self._device = device
        self._shuffle = shuffle
        self._clip_index = _ClipIndex(self._ds, clip_len, stride)
        if verbose:
            print(f"DoomTrainLoader: {len(self._clip_index)} clips "
                  f"(clip_len={clip_len}, stride={stride}, batch={batch_size})")

        # LRU cache in CPU RAM: episode_idx → (video_uint8_cpu, actions, rewards)
        # Decode on GPU (NVDEC), immediately move to CPU for caching.
        # Only clip batches are moved to GPU, keeping VRAM free for the model.
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._cache_order: list[int] = []
        self._max_cache = max_cache

        # Pre-allocate pinned buffers for fast CPU→GPU transfer (~8.7x speedup)
        self._vid_buf = torch.empty(
            (batch_size, clip_len, 3, 480, 640), dtype=torch.uint8,
        ).pin_memory()
        self._act_buf = torch.zeros(
            (batch_size, clip_len, 14), dtype=torch.float32,
        ).pin_memory()
        self._rew_buf = torch.zeros(
            (batch_size, clip_len,), dtype=torch.float32,
        ).pin_memory()

    def _get_episode_data(self, ep_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get (video_uint8_cpu, actions, rewards) for episode, using LRU cache."""
        if ep_idx in self._cache:
            self._cache_order.remove(ep_idx)
            self._cache_order.append(ep_idx)
            return self._cache[ep_idx]

        ep = self._ds[ep_idx]
        video = ep.video_uint8.cpu()  # decode on GPU, cache on CPU

        # Evict oldest if cache full
        while len(self._cache) >= self._max_cache:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        data = (video, ep.actions, ep.rewards)
        self._cache[ep_idx] = data
        self._cache_order.append(ep_idx)
        return data

    def __len__(self):
        return (len(self._clip_index) + self._batch_size - 1) // self._batch_size

    def _make_batch(self, batch_clips: list[tuple[int, int]]) -> dict:
        """Assemble one batch dict from a list of (ep_idx, start_frame) pairs."""
        B = len(batch_clips)
        T = self._clip_len

        # Use pre-allocated pinned buffers (8.7x faster CPU→GPU transfer)
        vid = self._vid_buf[:B]
        act = self._act_buf[:B]
        rew = self._rew_buf[:B]
        act.zero_()
        rew.zero_()
        metas = []

        for i, (ep_idx, start) in enumerate(batch_clips):
            video, ep_actions, ep_rewards = self._get_episode_data(ep_idx)
            end = min(start + T, video.shape[0])
            n = end - start

            vid[i, :n] = video[start:end]
            if n < T:
                vid[i, n:] = 0
            act[i, :n] = ep_actions[start:end]
            rew[i, :n] = ep_rewards[start:end]
            metas.append(self._ds._index[ep_idx]["meta"])

        dev = self._device
        return {
            # Pinned → GPU non-blocking transfer + float cast (~53ms vs 460ms unpinned)
            "video": vid.to(dev, dtype=torch.float32, non_blocking=True).div_(255.0),
            "actions": act.to(dev, non_blocking=True),
            "rewards": rew.to(dev, non_blocking=True),
            "meta": metas,
        }

    def __iter__(self):
        # Group clips by episode for cache-friendly access
        ep_clips: dict[int, list[int]] = {}
        for ep_idx, start in self._clip_index.clips:
            ep_clips.setdefault(ep_idx, []).append(start)

        ep_order = list(ep_clips.keys())
        if self._shuffle:
            random.shuffle(ep_order)

        # Process episodes in groups of max_cache — decode group, yield all
        # clips from group, then move to next group. Clips within each group
        # are shuffled for training diversity.
        for g_start in range(0, len(ep_order), self._max_cache):
            group_eps = ep_order[g_start:g_start + self._max_cache]

            # Collect and shuffle clips within this group
            group_clips = []
            for ep_idx in group_eps:
                for start in ep_clips[ep_idx]:
                    group_clips.append((ep_idx, start))
            if self._shuffle:
                random.shuffle(group_clips)

            # Yield batches from this group
            for b_start in range(0, len(group_clips), self._batch_size):
                yield self._make_batch(group_clips[b_start:b_start + self._batch_size])

            # Evict this group from cache before loading next
            self._cache.clear()
            self._cache_order.clear()
