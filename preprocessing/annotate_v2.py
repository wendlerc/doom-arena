#!/usr/bin/env python3
"""
Hybrid annotation pipeline: action-based detection + Gemini image confirmation.

Phase 1: Detect candidate events from action/reward data (frame-exact)
  - Weapon switches: SELECT_WEAPONx transitions
  - Shooting bursts: ATTACK=1 regions grouped into bursts
Phase 2: Send short frame sequences around candidates to Gemini for
  visual confirmation and enrichment (weapon type, context, etc.)
Phase 3: Segment-based detection for visual-only events (health/armor
  pickups, deaths, frags) by sending frames at ~2fps to Gemini

This bypasses Gemini's ~1fps video sampling by sending images directly.

Requires: google-genai, opencv-python-headless, Pillow
          (uv pip install google-genai Pillow)

Usage:
    python preprocessing/annotate_v2.py --data-root recordings
    python preprocessing/annotate_v2.py --data-root recordings --player p1
"""
import sys, os, io, json, re, time, argparse, tarfile, tempfile, base64
from datetime import datetime, timezone
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from google import genai
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAME_FPS = 35
BUTTON_NAMES = [
    "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_RIGHT", "MOVE_LEFT",
    "SELECT_WEAPON1", "SELECT_WEAPON2", "SELECT_WEAPON3", "SELECT_WEAPON4",
    "SELECT_WEAPON5", "SELECT_WEAPON6", "SELECT_WEAPON7",
    "ATTACK", "SPEED", "TURN_LEFT_RIGHT_DELTA",
]
WEAPON_NAMES = {
    1: "fist_chainsaw", 2: "pistol", 3: "shotgun",
    4: "super_shotgun", 5: "chaingun", 6: "rocket_launcher", 7: "plasma_rifle",
}
WEAPON_SELECT_INDICES = list(range(4, 11))  # SELECT_WEAPON1..7
ATTACK_INDEX = 11

# Frame extraction settings
CONTEXT_SECONDS = 1.5   # seconds of context around each candidate
SAMPLE_FPS = 5          # fps for frame extraction around candidates
SEGMENT_LENGTH = 15     # seconds per segment for visual-only detection
SEGMENT_SAMPLE_FPS = 2  # fps for segment-based scanning


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
def call_gemini(client, model_name, contents, max_retries=5, base_delay=10.0):
    """Call Gemini with retry and backoff."""
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            return response.text
        except Exception as e:
            err = str(e)
            if attempt == max_retries:
                raise
            if any(code in err for code in ["429", "ResourceExhausted", "500", "503"]):
                m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s?['\"]", err)
                server_delay = float(m.group(1)) if m else 0
                delay = max(server_delay, base_delay * (2 ** attempt))
                print(f"      Retrying in {delay:.0f}s...", flush=True)
                time.sleep(delay)
            else:
                raise


def parse_gemini_json(raw_text):
    """Parse Gemini JSON response with fallbacks for markdown fences and truncation."""
    text = raw_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # Try extracting complete array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try recovering truncated array
    match = re.search(r"\[.*", text, re.DOTALL)
    if match:
        fragment = match.group()
        last_brace = fragment.rfind("}")
        if last_brace > 0:
            try:
                result = json.loads(fragment[:last_brace + 1] + "]")
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Cannot parse JSON: {text[:200]}...")


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------
def scan_episodes(data_root, shard_pattern="*.tar"):
    """Scan shards and return episode dicts."""
    shards = sorted(Path(data_root).glob(shard_pattern))
    episodes = []
    for shard_path in shards:
        try:
            with tarfile.open(shard_path, "r") as tar:
                groups = {}
                for member in tar.getmembers():
                    if member.isdir():
                        continue
                    parts = member.name.split(".", 1)
                    if len(parts) != 2:
                        continue
                    groups.setdefault(parts[0], {})[parts[1]] = member.name
                for key, members in groups.items():
                    if "meta.json" not in members:
                        continue
                    meta = json.loads(tar.extractfile(tar.getmember(members["meta.json"])).read())
                    episodes.append({"shard": str(shard_path), "key": key, "members": members, "meta": meta})
        except (tarfile.TarError, OSError) as e:
            print(f"  Skipping {shard_path}: {e}")
    return episodes


def load_episode_data(shard_path, members, player):
    """Load actions, rewards arrays for a player from a shard."""
    with tarfile.open(shard_path, "r") as tar:
        actions_key = f"actions_{player}.npy"
        rewards_key = f"rewards_{player}.npy"
        actions = np.load(io.BytesIO(tar.extractfile(tar.getmember(members[actions_key])).read()))
        rewards = np.load(io.BytesIO(tar.extractfile(tar.getmember(members[rewards_key])).read()))
    return actions, rewards


def extract_frames_from_video(shard_path, member_name, frame_indices):
    """Extract specific frames from an MP4 inside a tar shard. Returns list of PIL Images."""
    import cv2

    # Extract video to temp file
    with tarfile.open(shard_path, "r") as tar:
        video_data = tar.extractfile(tar.getmember(member_name)).read()

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_data)
    tmp.close()

    frames = []
    try:
        cap = cv2.VideoCapture(tmp.name)
        frame_set = set(frame_indices)
        max_frame = max(frame_indices)
        idx = 0
        result = {}
        while cap.isOpened() and idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in frame_set:
                # BGR -> RGB -> PIL
                result[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        frames = [result.get(i) for i in frame_indices if i in result]
    finally:
        os.unlink(tmp.name)

    return frames


# ---------------------------------------------------------------------------
# Phase 1: Action-based candidate detection
# ---------------------------------------------------------------------------
def detect_weapon_switches(actions):
    """Detect weapon switch events from action data. Returns list of (frame, weapon_slot)."""
    events = []
    for slot_idx, col_idx in enumerate(WEAPON_SELECT_INDICES):
        col = actions[:, col_idx]
        # Find 0->1 transitions
        transitions = np.where((col[1:] == 1.0) & (col[:-1] == 0.0))[0] + 1
        weapon_slot = slot_idx + 1
        for frame in transitions:
            events.append({"frame": int(frame), "weapon_slot": weapon_slot})
    events.sort(key=lambda e: e["frame"])
    return events


def detect_shooting_bursts(actions, min_gap=10, min_length=2):
    """Detect shooting bursts from ATTACK action. Returns list of (frame_start, frame_end)."""
    attack = actions[:, ATTACK_INDEX] == 1.0
    if not attack.any():
        return []

    # Find contiguous regions, merge if gap < min_gap
    changes = np.diff(attack.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases
    if attack[0]:
        starts = np.concatenate([[0], starts])
    if attack[-1]:
        ends = np.concatenate([ends, [len(attack)]])

    if len(starts) == 0:
        return []

    # Merge close bursts
    merged = [(int(starts[0]), int(ends[0]))]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], int(e))
        else:
            merged.append((int(s), int(e)))

    # Filter short bursts
    return [{"frame_start": s, "frame_end": e} for s, e in merged if e - s >= min_length]


def detect_reward_events(rewards, pos_thresh=0.5, neg_thresh=-0.5):
    """Detect reward spikes (frags/deaths). Returns list of (frame, reward)."""
    events = []
    pos = np.where(rewards > pos_thresh)[0]
    neg = np.where(rewards < neg_thresh)[0]
    for f in pos:
        events.append({"frame": int(f), "reward": float(rewards[f]), "type": "positive"})
    for f in neg:
        events.append({"frame": int(f), "reward": float(rewards[f]), "type": "negative"})
    events.sort(key=lambda e: e["frame"])
    return events


# ---------------------------------------------------------------------------
# Phase 2: Gemini confirmation of action-detected events
# ---------------------------------------------------------------------------
CONFIRM_WEAPON_SWITCH_PROMPT = """\
These frames (at ~5fps) show a moment in a Doom deathmatch game where the \
player switches weapons. The frames are in chronological order.

Look at the weapon visible in the player's hands and the HUD at the bottom \
(health bottom-left, armor bottom-right, ammo center).

For this weapon switch event, tell me:
1. What weapon the player switched FROM (visible in early frames)
2. What weapon the player switched TO (visible in later frames)
3. Is this actually a weapon PICKUP (new weapon from the ground) rather than a switch?
4. Current health and armor values visible in the HUD

Doom weapons: fist(1), pistol(2), shotgun(3), super_shotgun(4), chaingun(5), \
rocket_launcher(6), plasma_rifle(7), bfg(7), chainsaw(1)

Return a single JSON object (no array) with:
{"event_type": "weapon_switch" or "weapon_pickup",
 "weapon_from": "shotgun", "weapon_to": "super_shotgun",
 "health": 85, "armor": 40,
 "description": "brief description",
 "confidence": 0.0-1.0}

Return ONLY the JSON object. No markdown fences.\
"""

CONFIRM_SHOOTING_PROMPT = """\
These frames (at ~5fps) show a moment in a Doom deathmatch game where the \
player is shooting. The frames are in chronological order.

Look at the weapon, muzzle flash, projectiles, and whether enemies are visible.

For this shooting event, tell me:
1. What weapon is being fired
2. Is a target/enemy visible on screen
3. Does the player hit or kill anyone
4. Current health and armor values

Doom weapons: fist(1), pistol(2), shotgun(3), super_shotgun(4), chaingun(5), \
rocket_launcher(6), plasma_rifle(7), bfg(7), chainsaw(1)

Return a single JSON object:
{"event_type": "shooting",
 "weapon_type": "rocket_launcher",
 "target_visible": true,
 "hit_or_kill": false,
 "health": 85, "armor": 40,
 "description": "brief description",
 "confidence": 0.0-1.0}

Return ONLY the JSON object. No markdown fences.\
"""


# ---------------------------------------------------------------------------
# Phase 3: Segment-based scanning for visual-only events
# ---------------------------------------------------------------------------
SEGMENT_SCAN_PROMPT = """\
These frames (at ~2fps, timestamps shown) are from a Doom deathmatch game. \
Scan them carefully for these events:

1. **Health pickup**: health number (bottom-left HUD) increases between frames. \
   Green bottles (+1), medkits (+10/+25), soulsphere (blue orb, +100).
2. **Armor pickup**: armor number (bottom-right HUD) increases between frames. \
   Armor helmets (+1), green armor (+100), blue armor (+200).
3. **Death**: health drops to 0, screen goes red, view tilts/falls.
4. **Respawn**: sudden location change, health resets to 100, holding pistol/fist.
5. **Frag (kill)**: enemy dies/gibs, frag message in top-left corner.
6. **Other notable events**: powerup pickups, explosions, anything interesting.

IMPORTANT: Compare health/armor numbers between consecutive frames to detect \
pickups. Even small changes (+1 from bottles) count.

The timestamp under each frame shows MM:SS.s format.

Return a JSON array of events. Each event MUST have:
- "timestamp": "MM:SS.s" (the frame timestamp where the event occurs)
- "event_type": one of "health_pickup", "armor_pickup", "death", "respawn", "frag", "other"
- "confidence": float 0.0-1.0
- "description": brief description
- "details": {"health_before": 80, "health_after": 90, "item": "medkit"} or similar

If no events found in this segment, return: []

Return ONLY the JSON array. No markdown fences.\
"""


def frame_to_ts(frame, fps=GAME_FPS):
    t = frame / fps
    return f"{int(t//60):02d}:{t%60:04.1f}"


def ts_to_frame(ts, fps=GAME_FPS):
    parts = ts.strip().split(":")
    if len(parts) == 2:
        secs = float(parts[0]) * 60 + float(parts[1])
    else:
        secs = float(parts[0])
    return int(round(secs * fps))


def pil_to_part(img):
    """Convert PIL Image to a Gemini Part with inline image data."""
    from google.genai import types
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buf.getvalue()))


def annotate_with_frames(client, model_name, frames, frame_indices, prompt, delay=2.0):
    """Send frames + prompt to Gemini. Returns parsed JSON."""
    # Build contents: alternating images with timestamp labels
    contents = []
    for frame, idx in zip(frames, frame_indices):
        ts = frame_to_ts(idx)
        contents.append(f"[{ts}]")
        contents.append(pil_to_part(frame))
    contents.append(prompt)

    raw = call_gemini(client, model_name, contents, base_delay=delay)
    return parse_gemini_json(raw)


# ---------------------------------------------------------------------------
# Main annotation pipeline
# ---------------------------------------------------------------------------
def annotate_episode(client, model_name, shard_path, members, player, meta, delay=2.0):
    """Full hybrid annotation for one episode+player."""
    n_frames = meta.get("n_frames", 0)
    video_key = f"video_{player}.mp4"

    # Load action/reward data
    actions, rewards = load_episode_data(shard_path, members, player)
    print(f"    Loaded actions {actions.shape}, rewards {rewards.shape}")

    all_events = []

    # ===== Phase 1: Action-based detection =====
    weapon_switches = detect_weapon_switches(actions)
    shooting_bursts = detect_shooting_bursts(actions)
    reward_events = detect_reward_events(rewards)

    print(f"    Phase 1: {len(weapon_switches)} weapon switches, "
          f"{len(shooting_bursts)} shooting bursts, {len(reward_events)} reward events")

    # ===== Phase 2: Confirm weapon switches with Gemini =====
    if weapon_switches:
        print(f"    Phase 2a: Confirming {len(weapon_switches)} weapon switches...", flush=True)
        # Group nearby switches (within 1s) to reduce API calls
        groups = []
        current_group = [weapon_switches[0]]
        for ws in weapon_switches[1:]:
            if ws["frame"] - current_group[-1]["frame"] < GAME_FPS:
                current_group.append(ws)
            else:
                groups.append(current_group)
                current_group = [ws]
        groups.append(current_group)

        for gi, group in enumerate(groups):
            center = group[len(group) // 2]["frame"]
            ctx = int(CONTEXT_SECONDS * GAME_FPS)
            f_start = max(0, center - ctx)
            f_end = min(n_frames - 1, center + ctx)
            step = max(1, GAME_FPS // SAMPLE_FPS)
            indices = list(range(f_start, f_end + 1, step))

            frames = extract_frames_from_video(shard_path, members[video_key], indices)
            if not frames:
                continue

            try:
                result = annotate_with_frames(
                    client, model_name, frames, indices, CONFIRM_WEAPON_SWITCH_PROMPT, delay
                )
                if isinstance(result, list):
                    result = result[0]
                result["frame_start"] = group[0]["frame"]
                result["frame_end"] = group[-1]["frame"]
                result["timestamp_start"] = frame_to_ts(group[0]["frame"])
                result["timestamp_end"] = frame_to_ts(group[-1]["frame"])
                result["source"] = "action+gemini"
                result["action_data"] = {"weapon_slots": [g["weapon_slot"] for g in group]}
                all_events.append(result)
                print(f"      [{gi+1}/{len(groups)}] {result.get('event_type','?')}: {result.get('description','')[:60]}")
            except Exception as e:
                print(f"      [{gi+1}/{len(groups)}] FAILED: {e}")

            if gi < len(groups) - 1:
                time.sleep(delay)

    # ===== Phase 2b: Confirm shooting bursts (sample a few representative ones) =====
    if shooting_bursts:
        # Only confirm up to 15 longest/most interesting bursts to save API calls
        bursts_sorted = sorted(shooting_bursts, key=lambda b: b["frame_end"] - b["frame_start"], reverse=True)
        bursts_to_confirm = bursts_sorted[:15]
        print(f"    Phase 2b: Confirming {len(bursts_to_confirm)}/{len(shooting_bursts)} shooting bursts...", flush=True)

        for bi, burst in enumerate(bursts_to_confirm):
            mid = (burst["frame_start"] + burst["frame_end"]) // 2
            ctx = int(CONTEXT_SECONDS * GAME_FPS)
            f_start = max(0, mid - ctx)
            f_end = min(n_frames - 1, mid + ctx)
            step = max(1, GAME_FPS // SAMPLE_FPS)
            indices = list(range(f_start, f_end + 1, step))

            frames = extract_frames_from_video(shard_path, members[video_key], indices)
            if not frames:
                continue

            try:
                result = annotate_with_frames(
                    client, model_name, frames, indices, CONFIRM_SHOOTING_PROMPT, delay
                )
                if isinstance(result, list):
                    result = result[0]
                result["frame_start"] = burst["frame_start"]
                result["frame_end"] = burst["frame_end"]
                result["timestamp_start"] = frame_to_ts(burst["frame_start"])
                result["timestamp_end"] = frame_to_ts(burst["frame_end"])
                result["source"] = "action+gemini"
                result["action_data"] = {"burst_length_frames": burst["frame_end"] - burst["frame_start"]}
                all_events.append(result)
                print(f"      [{bi+1}/{len(bursts_to_confirm)}] {result.get('weapon_type','?')}: {result.get('description','')[:60]}")
            except Exception as e:
                print(f"      [{bi+1}/{len(bursts_to_confirm)}] FAILED: {e}")

            if bi < len(bursts_to_confirm) - 1:
                time.sleep(delay)

        # Add remaining bursts as action-only events (no Gemini confirmation)
        for burst in bursts_sorted[15:]:
            all_events.append({
                "event_type": "shooting",
                "frame_start": burst["frame_start"],
                "frame_end": burst["frame_end"],
                "timestamp_start": frame_to_ts(burst["frame_start"]),
                "timestamp_end": frame_to_ts(burst["frame_end"]),
                "confidence": 0.7,
                "description": f"Shooting burst ({burst['frame_end']-burst['frame_start']} frames)",
                "details": {},
                "source": "action_only",
            })

    # ===== Phase 3: Segment-based scanning for visual events =====
    n_segments = max(1, n_frames // (SEGMENT_LENGTH * GAME_FPS))
    print(f"    Phase 3: Scanning {n_segments} segments ({SEGMENT_LENGTH}s each) for visual events...", flush=True)

    for seg_idx in range(n_segments):
        seg_start = seg_idx * SEGMENT_LENGTH * GAME_FPS
        seg_end = min((seg_idx + 1) * SEGMENT_LENGTH * GAME_FPS, n_frames - 1)
        step = max(1, GAME_FPS // SEGMENT_SAMPLE_FPS)
        indices = list(range(int(seg_start), int(seg_end), step))

        if not indices:
            continue

        frames = extract_frames_from_video(shard_path, members[video_key], indices)
        if not frames:
            continue

        try:
            results = annotate_with_frames(
                client, model_name, frames, indices, SEGMENT_SCAN_PROMPT, delay
            )
            for ev in results:
                ts = ev.get("timestamp", "")
                if ts:
                    frame = ts_to_frame(ts)
                    ev["frame_start"] = frame
                    ev["frame_end"] = min(frame + GAME_FPS, n_frames - 1)
                    ev["timestamp_start"] = ts
                    ev["timestamp_end"] = frame_to_ts(ev["frame_end"])
                ev["source"] = "visual_scan"
            all_events.extend(results)
            n_ev = len(results)
            seg_ts = frame_to_ts(int(seg_start))
            print(f"      Segment {seg_idx+1}/{n_segments} [{seg_ts}]: {n_ev} events")
        except Exception as e:
            print(f"      Segment {seg_idx+1}/{n_segments}: FAILED: {e}")

        if seg_idx < n_segments - 1:
            time.sleep(delay)

    # Sort all events by frame
    all_events.sort(key=lambda e: e.get("frame_start", 0))
    return all_events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid annotation: action data + Gemini vision")
    parser.add_argument("--data-root", default="recordings")
    parser.add_argument("--shard-pattern", default="*.tar")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--player", default="all", choices=["p1", "p2", "all"])
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls")
    parser.add_argument("--sanity", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    output_dir = args.output_dir or os.path.join(args.data_root, "annotations_v2")
    os.makedirs(output_dir, exist_ok=True)

    players = ["p1", "p2"] if args.player == "all" else [args.player]

    print(f"Scanning {args.data_root}...")
    episodes = scan_episodes(args.data_root, args.shard_pattern)
    if args.sanity > 0:
        episodes = episodes[:args.sanity]
    print(f"Found {len(episodes)} episode(s)")

    total_events = 0
    for ep_idx, ep in enumerate(episodes):
        ep_id = ep["meta"].get("episode_id", ep["key"])
        meta = ep["meta"]
        n_frames = meta.get("n_frames", 0)
        print(f"\nEpisode {ep_idx+1}/{len(episodes)}: {ep_id[:16]}... ({n_frames} frames, {n_frames/GAME_FPS:.0f}s)")

        for player in players:
            if f"video_{player}.mp4" not in ep["members"]:
                continue

            out_path = os.path.join(output_dir, f"{ep_id}_{player}.json")
            if os.path.exists(out_path):
                with open(out_path) as f:
                    existing = json.load(f)
                n_ev = len(existing.get("events", []))
                print(f"  {player}: already annotated ({n_ev} events), skipping")
                total_events += n_ev
                continue

            print(f"  {player}: annotating...")
            events = annotate_episode(
                client, args.model, ep["shard"], ep["members"], player, meta, args.delay
            )

            annotation = {
                "version": "2.0",
                "pipeline": "hybrid_action_gemini",
                "episode_id": ep_id,
                "source_shard": ep["shard"],
                "player": player,
                "model": args.model,
                "n_frames": n_frames,
                "fps": GAME_FPS,
                "duration_s": round(n_frames / GAME_FPS, 1),
                "annotated_at": datetime.now(timezone.utc).isoformat(),
                "meta": meta,
                "events": events,
            }
            with open(out_path, "w") as f:
                json.dump(annotation, f, indent=2)

            total_events += len(events)
            print(f"  {player}: {len(events)} events -> {out_path}")

    print(f"\nDone. {total_events} total events.")


if __name__ == "__main__":
    main()
