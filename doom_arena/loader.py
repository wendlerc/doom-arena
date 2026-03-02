"""
Data loader for Doom gameplay WebDataset shards.

Provides random-access Episode objects with lazy video loading
and interactive visualization helpers for Jupyter/VS Code notebooks.

Usage:
    from doom_arena.loader import DoomDataset

    ds = DoomDataset("datasets/mp_recordings")
    ds.summary()
    ep = ds[42]
    ep.show_frame(100)
    ep.plot_actions()
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

# Constants (duplicated from record.py to avoid importing vizdoom/torch/sample_factory)
GAME_FPS = 35
BUTTON_NAMES = [
    "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_RIGHT", "MOVE_LEFT",
    "SELECT_WEAPON1", "SELECT_WEAPON2", "SELECT_WEAPON3", "SELECT_WEAPON4",
    "SELECT_WEAPON5", "SELECT_WEAPON6", "SELECT_WEAPON7",
    "ATTACK", "SPEED", "TURN_LEFT_RIGHT_DELTA",
]


def decode_video_bytes(mp4_bytes: bytes) -> np.ndarray:
    """Decode MP4 bytes to (n_frames, H, W, 3) uint8 RGB numpy array."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(mp4_bytes)
    tmp.close()
    try:
        cap = cv2.VideoCapture(tmp.name)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return np.empty((0, 480, 640, 3), dtype=np.uint8)
        return np.stack(frames)
    finally:
        os.unlink(tmp.name)


def _decode_single_frame(mp4_bytes: bytes, frame_idx: int) -> np.ndarray:
    """Decode a single frame from MP4 bytes without loading all frames."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(mp4_bytes)
    tmp.close()
    try:
        cap = cv2.VideoCapture(tmp.name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IndexError(f"Frame {frame_idx} not found in video")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        os.unlink(tmp.name)


class Episode:
    """A single Doom gameplay episode with lazy video loading.

    Attributes:
        actions: (n_frames, 14) float32 numpy array
        rewards: (n_frames,) float32 numpy array
        meta: dict with all metadata from meta.json
        video: lazy-loaded (n_frames, 480, 640, 3) uint8 RGB array
        demo: lazy-loaded raw bytes of .lmp demo file
    """

    def __init__(self, actions, rewards, meta, shard_path, key, member_names):
        self.actions = actions
        self.rewards = rewards
        self.meta = meta
        self._shard_path = shard_path
        self._key = key
        self._member_names = member_names  # {ext: member_name_in_tar}
        self._video_cache = None
        self._demo_cache = None

    def _extract_file(self, ext: str) -> bytes:
        """Extract a single file from the tar shard by extension."""
        name = self._member_names.get(ext)
        if name is None:
            raise KeyError(f"No {ext} in episode {self._key}")
        with tarfile.open(self._shard_path, "r") as tar:
            member = tar.getmember(name)
            return tar.extractfile(member).read()

    @property
    def n_frames(self) -> int:
        return self.meta.get("n_frames", len(self.actions))

    @property
    def video(self) -> np.ndarray:
        """Lazy-load and cache all video frames as (n_frames, H, W, 3) uint8 RGB."""
        if self._video_cache is None:
            mp4_bytes = self._extract_file("video_p1.mp4")
            self._video_cache = decode_video_bytes(mp4_bytes)
        return self._video_cache

    @property
    def demo(self) -> bytes:
        """Lazy-load the .lmp demo file bytes."""
        if self._demo_cache is None:
            self._demo_cache = self._extract_file("demo_p1.lmp")
        return self._demo_cache

    def get_frame(self, i: int) -> np.ndarray:
        """Get a single frame (H, W, 3) without loading the full video."""
        if self._video_cache is not None:
            return self._video_cache[i]
        mp4_bytes = self._extract_file("video_p1.mp4")
        return _decode_single_frame(mp4_bytes, i)

    def show_frame(self, i: int = 0, ax=None):
        """Display frame i using matplotlib."""
        import matplotlib.pyplot as plt

        frame = self.get_frame(i)
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
        """Play video segment in notebook as HTML5 animation.

        Args:
            start: first frame index
            end: last frame index (default: all)
            fps: playback speed (default: GAME_FPS)
            max_frames: cap to avoid huge HTML blobs (default: 300 = ~8.5s)
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if fps is None:
            fps = GAME_FPS
        if end is None:
            end = min(start + max_frames, self.n_frames)
        end = min(end, self.n_frames)

        frames = self.video[start:end]
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
        """Plot action channels: heatmap over time + activation frequency bar chart."""
        import matplotlib.pyplot as plt

        names = self.meta.get("button_names", BUTTON_NAMES)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                        gridspec_kw={"width_ratios": [3, 1]})

        acts = self.actions.T  # (14, n_frames)
        ax1.imshow(acts, aspect="auto", interpolation="nearest", cmap="viridis")
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("Frame")
        ax1.set_title("Action channels over time")

        freq = np.mean(np.abs(self.actions), axis=0)
        ax2.barh(range(len(names)), freq)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_title("Mean |activation|")

        plt.tight_layout()
        plt.show()

    def plot_rewards(self, figsize=(10, 4)):
        """Plot per-step and cumulative rewards over time."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        t = np.arange(len(self.rewards)) / GAME_FPS
        ax1.plot(t, self.rewards, linewidth=0.5)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Reward")
        ax1.set_title("Per-step reward")

        ax2.plot(t, np.cumsum(self.rewards))
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Cumulative reward")
        ax2.set_title("Cumulative reward")

        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print episode statistics."""
        m = self.meta
        nf = self.n_frames
        print(f"Episode: {m.get('episode_id', '?')[:12]}...")
        print(f"  Scenario: {m.get('scenario')} ({m.get('map')})")
        print(f"  Mode: {m.get('mode')} | Bots: {m.get('n_bots')}")
        print(f"  Frames: {nf} ({nf / GAME_FPS:.1f}s)")
        print(f"  Frags: {m.get('frag_p1', 0):.0f} | Deaths: {m.get('death_p1', 0):.0f}")
        print(f"  Total reward: {m.get('total_reward_p1', 0):.1f}")
        print(f"  Random policy: {m.get('random_policy', False)}")
        print(f"  Checkpoint: {m.get('checkpoint_p1', '?')}")

    def __repr__(self):
        sc = self.meta.get("scenario", "?")
        nf = self.n_frames
        frags = self.meta.get("frag_p1", 0)
        return f"Episode({sc}, {nf} frames, {frags:.0f} frags)"


class DoomDataset:
    """Indexed random-access dataset over Doom gameplay WebDataset shards.

    Usage:
        ds = DoomDataset("datasets/mp_recordings")
        ep = ds[42]           # random access
        ep = ds.sample()      # random episode
        for ep in ds:          # iterate
            ...
    """

    def __init__(self, root: str, verbose: bool = True):
        self._root = Path(root)
        self._index: list[dict] = []
        self._build_index(verbose)

    @classmethod
    def _from_index(cls, root: Path, index: list[dict]) -> DoomDataset:
        """Create a filtered dataset from an existing index (no re-scan)."""
        obj = cls.__new__(cls)
        obj._root = root
        obj._index = index
        return obj

    def _build_index(self, verbose: bool):
        """Scan all shards, group members by key, eagerly load meta.json."""
        shards = sorted(self._root.glob("mp-*.tar"))
        if verbose:
            print(f"Scanning {len(shards)} shards in {self._root}...")

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
            print(f"Indexed {len(self._index)} episodes ({total_hours:.1f}h) "
                  f"from {len(shards)} shards")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx) -> Episode | list[Episode]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        entry = self._index[idx]
        shard_path = entry["shard_path"]
        members = entry["members"]
        meta = entry["meta"]

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
            meta=meta,
            shard_path=shard_path,
            key=entry["key"],
            member_names=members,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sample(self) -> Episode:
        """Return a random episode."""
        return self[random.randint(0, len(self) - 1)]

    def filter(self, **kwargs) -> DoomDataset:
        """Return a filtered dataset. Example: ds.filter(scenario="dwango5_3min")"""
        filtered = [
            entry for entry in self._index
            if all(entry["meta"].get(k) == v for k, v in kwargs.items())
        ]
        return DoomDataset._from_index(self._root, filtered)

    def summary(self):
        """Print dataset statistics."""
        n = len(self._index)
        if n == 0:
            print("DoomDataset: empty")
            return

        total_frames = sum(e["meta"].get("n_frames", 0) for e in self._index)
        total_hours = total_frames / GAME_FPS / 3600

        scenarios: dict[str, int] = {}
        total_frags = 0.0
        total_deaths = 0.0
        random_count = 0

        for e in self._index:
            m = e["meta"]
            sc = m.get("scenario", "unknown")
            scenarios[sc] = scenarios.get(sc, 0) + 1
            total_frags += m.get("frag_p1", 0)
            total_deaths += m.get("death_p1", 0)
            random_count += int(m.get("random_policy", False))

        print(f"DoomDataset: {n} episodes, {total_hours:.1f}h of gameplay")
        print(f"  Total frames: {total_frames:,}")
        print(f"  Scenarios:")
        for sc, count in sorted(scenarios.items(), key=lambda x: -x[1]):
            print(f"    {sc}: {count} ({count / n * 100:.1f}%)")
        print(f"  Avg frags/ep: {total_frags / n:.1f}")
        print(f"  Avg deaths/ep: {total_deaths / n:.1f}")
        print(f"  Random policy: {random_count} ({random_count / n * 100:.1f}%)")

    def __repr__(self):
        return f"DoomDataset({self._root}, {len(self)} episodes)"
