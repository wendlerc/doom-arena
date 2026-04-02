"""Save game session recordings to WebDataset shards.

Matches the format used by doom_arena/record.py so existing loaders work.
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid

import numpy as np

from doom_arena.record import (
    BUTTON_NAMES,
    DECISION_INTERVAL,
    GAME_FPS,
    encode_video,
)

logger = logging.getLogger("doom_arena.web")

OUTPUT_DIR = "datasets/web_recordings"


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def save_session_recording(
    session_id: str,
    session_config: dict,
    episode_recordings: list[dict],
    output_dir: str = OUTPUT_DIR,
) -> str | None:
    """Save all episodes from a session to a WebDataset tar shard.

    Args:
        session_id: Unique session identifier.
        session_config: Session configuration dict (wad, map, timelimit, etc.).
        episode_recordings: List of episode data dicts from GameRunner.
        output_dir: Output directory for shards.

    Returns:
        Path to the saved tar shard, or None on failure.
    """
    import webdataset as wds

    if not episode_recordings:
        logger.warning("No episodes to save")
        return None

    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"web-{session_id}.tar")

    try:
        with wds.TarWriter(shard_path) as sink:
            for ep_idx, ep_data in enumerate(episode_recordings):
                if not ep_data or "players" not in ep_data:
                    continue

                ep_id = ep_data.get("episode_id", uuid.uuid4().hex[:16])
                key = f"ep_{ep_id}"

                sample = {"__key__": key}

                # Per-player data
                player_metas = []
                for p_idx, (slot_id, pdata) in enumerate(sorted(ep_data["players"].items()), start=1):
                    suffix = f"p{p_idx}"

                    # Video (prefer pre-encoded bytes from GameRunner)
                    video_bytes = pdata.get("video_bytes")
                    if video_bytes:
                        sample[f"video_{suffix}.mp4"] = video_bytes
                    else:
                        frames = pdata.get("frames", [])
                        if frames:
                            video_bytes = encode_video(frames, fps=GAME_FPS)
                            sample[f"video_{suffix}.mp4"] = video_bytes

                    # Actions
                    actions = pdata.get("actions")
                    if actions is not None and len(actions) > 0:
                        sample[f"actions_{suffix}.npy"] = _npy_bytes(actions)

                    # Rewards
                    rewards = pdata.get("rewards")
                    if rewards is not None and len(rewards) > 0:
                        sample[f"rewards_{suffix}.npy"] = _npy_bytes(rewards)

                    player_metas.append({
                        "slot_id": slot_id,
                        "player_index": p_idx,
                        "name": pdata.get("name", f"Player_{p_idx}"),
                        "type": pdata.get("type", "unknown"),
                        "frags": pdata.get("frags", 0.0),
                        "deaths": pdata.get("deaths", 0.0),
                        "n_frames": pdata.get("n_frames", len(frames)),
                        "total_reward": float(np.sum(pdata["rewards"])) if pdata.get("rewards") is not None and len(pdata["rewards"]) > 0 else 0.0,
                    })

                # Metadata
                meta = {
                    "episode_id": ep_id,
                    "session_id": session_id,
                    "mode": "web",
                    "scenario": session_config.get("wad", "unknown"),
                    "map": session_config.get("doom_map", "map01"),
                    "timelimit_min": session_config.get("timelimit", 5.0),
                    "n_bots": session_config.get("num_bots", 0),
                    "num_players": len(ep_data["players"]),
                    "players": player_metas,
                    "game_tics": ep_data.get("tic", 0),
                    "duration_s": ep_data.get("duration_s", 0.0),
                    "button_names": BUTTON_NAMES,
                    "decision_interval": DECISION_INTERVAL,
                    "fps": GAME_FPS,
                    "video_resolution": "640x480",
                    "timestamp": time.time(),
                }
                sample["meta.json"] = json.dumps(meta, indent=2).encode()

                sink.write(sample)

        logger.info(f"Saved recording: {shard_path} ({len(episode_recordings)} episodes)")
        return shard_path

    except Exception as e:
        logger.exception(f"Failed to save recording: {e}")
        return None
