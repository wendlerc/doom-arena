# %% [markdown]
# # Extract Targeted Evaluation Clips
# Curated short clips (1-3s) from PvP recordings for world model evaluation.
# Each clip is annotated with what happens and verified by Gemini.
#
# Categories:
# - weapon_switch: player switches from weapon A to B
# - attack_firing: player fires weapon (Gemini-verified muzzle flash/projectile)
# - attack_not_firing: player holds weapon without firing
# - kill: player frags an enemy (Gemini-detected)
# - death_respawn: player dies and respawns (Gemini-detected)

# %%
import sys, os, io, json, re, time, random, tarfile, tempfile
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import cv2
from PIL import Image
from google import genai
from google.genai import types

# %%
# --- Config ---
GAME_FPS = 35
DATA_ROOT = "datasets/pvp_recordings"
MAX_SHARDS = 5
OUTPUT_DIR = "datasets/eval_clips"
MODEL_NAME = "gemini-2.5-flash"
DELAY = 2.0
SEED = 42

N_SWITCH_PER_WEAPON = 20
N_ATTACK_FIRING = 50
N_ATTACK_NOT_FIRING = 50
N_KILL_TARGET = 30
N_DEATH_TARGET = 30

CLIP_FRAMES = 70  # 2 seconds at 35fps
CLIP_CONTEXT = 35  # 1s before event, 1s after

WEAPON_NAMES = {
    1: "fist_chainsaw", 2: "pistol", 3: "shotgun",
    4: "super_shotgun", 5: "chaingun", 6: "rocket_launcher", 7: "plasma_rifle",
}
WEAPON_SELECT_INDICES = list(range(4, 11))
ATTACK_INDEX = 11

# %%
# --- Helpers (from sanity_check_actions.py) ---

def _extract_retry_delay(err_str):
    m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s?['\"]", err_str)
    return float(m.group(1)) if m else None


def call_gemini(client, model_name, contents, max_retries=5, base_delay=10.0):
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model_name, contents=contents).text
        except Exception as e:
            err = str(e)
            if attempt == max_retries:
                raise
            if any(c in err for c in ["429", "ResourceExhausted", "500", "503"]):
                d = max(_extract_retry_delay(err) or 0, base_delay * (2 ** attempt))
                print(f"      Retrying in {d:.0f}s...", flush=True)
                time.sleep(d)
            else:
                raise


def parse_gemini_json(raw):
    text = re.sub(r"```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    for attempt in [
        lambda: json.loads(text),
        lambda: json.loads(re.search(r"\{.*\}", text, re.DOTALL).group()),
        lambda: json.loads(re.search(r"\[.*\]", text, re.DOTALL).group()),
    ]:
        try:
            return attempt()
        except (json.JSONDecodeError, AttributeError):
            continue
    # Truncated array recovery
    m = re.search(r"\[.*", text, re.DOTALL)
    if m:
        frag = m.group()
        lb = frag.rfind("}")
        if lb > 0:
            try:
                return json.loads(frag[:lb + 1] + "]")
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Cannot parse JSON: {text[:200]}...")


def pil_to_part(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buf.getvalue()))


def scan_episodes(data_root, shard_pattern="*.tar", max_shards=None):
    shards = sorted(Path(data_root).glob(shard_pattern))
    if max_shards:
        shards = shards[:max_shards]
    episodes = []
    for sp in shards:
        try:
            with tarfile.open(sp, "r") as tar:
                groups = {}
                for m in tar.getmembers():
                    if m.isdir(): continue
                    parts = m.name.split(".", 1)
                    if len(parts) == 2:
                        groups.setdefault(parts[0], {})[parts[1]] = m.name
                for key, members in groups.items():
                    if "meta.json" not in members: continue
                    meta = json.loads(tar.extractfile(tar.getmember(members["meta.json"])).read())
                    episodes.append({"shard": str(sp), "key": key, "members": members, "meta": meta})
        except (tarfile.TarError, OSError):
            continue
    return episodes


def load_actions(shard_path, members, player):
    with tarfile.open(shard_path, "r") as tar:
        return np.load(io.BytesIO(tar.extractfile(tar.getmember(members[f"actions_{player}.npy"])).read()))


@contextmanager
def open_video(shard_path, member_name):
    with tarfile.open(shard_path, "r") as tar:
        data = tar.extractfile(tar.getmember(member_name)).read()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(data); tmp.close()
    try:
        yield tmp.name
    finally:
        os.unlink(tmp.name)


def extract_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    fset = set(frame_indices)
    mx = max(frame_indices)
    result = {}
    idx = 0
    while cap.isOpened() and idx <= mx:
        ret, frame = cap.read()
        if not ret: break
        if idx in fset:
            result[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return result


def save_clip_video(frames_dict, frame_start, frame_end, out_path):
    """Save frames as MP4 clip."""
    indices = sorted(k for k in frames_dict if frame_start <= k <= frame_end)
    if not indices: return False
    first = frames_dict[indices[0]]
    w, h = first.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, GAME_FPS, (w, h))
    for i in indices:
        if i in frames_dict:
            writer.write(cv2.cvtColor(np.array(frames_dict[i]), cv2.COLOR_RGB2BGR))
    writer.release()
    return True


# %%
# --- Event detection ---

def detect_weapon_changes(actions):
    prev_w = None
    events = []
    for i in range(actions.shape[0]):
        active = np.where(actions[i, 4:11] == 1.0)[0]
        if len(active) > 0:
            w = int(active[0]) + 1
            if prev_w is not None and w != prev_w:
                events.append({"frame": i, "weapon_from": prev_w, "weapon_to": w})
            prev_w = w
    return events


def detect_attack_bursts(actions, min_gap=5, min_length=2):
    attack = actions[:, ATTACK_INDEX] == 1.0
    if not attack.any(): return []
    changes = np.diff(attack.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if attack[0]: starts = np.concatenate([[0], starts])
    if attack[-1]: ends = np.concatenate([ends, [len(attack)]])
    merged = [(int(starts[0]), int(ends[0]))]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] < min_gap: merged[-1] = (merged[-1][0], int(e))
        else: merged.append((int(s), int(e)))
    return [(s, e) for s, e in merged if e - s >= min_length]


def find_not_firing_frames(actions, n_samples, rng):
    """Find frames where a weapon (W3-W7) is held but ATTACK=0."""
    candidates = []
    for i in range(actions.shape[0]):
        active_w = np.where(actions[i, 4:11] == 1.0)[0]
        if len(active_w) > 0 and (active_w[0] + 1) >= 3 and actions[i, ATTACK_INDEX] == 0:
            candidates.append((i, int(active_w[0]) + 1))
    if len(candidates) > n_samples:
        candidates = rng.sample(candidates, n_samples)
    return candidates


def frame_to_ts(f):
    t = f / GAME_FPS
    return f"{int(t//60):02d}:{t%60:04.1f}"


# %%
# --- Gemini prompts ---

VERIFY_WEAPON_SWITCH = """\
These frames show a Doom deathmatch clip at 35fps. According to action data, \
the player switches from {weapon_from} to {weapon_to} around the middle frames.

Is a weapon switch visually occurring? Look for the weapon changing in the player's hands.

Doom weapons: fist(1), pistol(2), shotgun(3), super_shotgun(4), chaingun(5), \
rocket_launcher(6), plasma_rifle(7)

Return ONLY a JSON object:
{{"verified": true or false, "weapon_from": "name or null", "weapon_to": "name or null", "confidence": 0.0-1.0}}
"""

VERIFY_ATTACK_FIRING = """\
These frames show a Doom deathmatch clip at 35fps. According to action data, \
the player is pressing the ATTACK button during these frames.

Is the weapon actually FIRING? Look for: muzzle flash, recoil animation, \
projectiles (rockets, plasma balls), shotgun blast, or bullet tracers. \
The weapon must show a clear firing animation, not just be held.

Return ONLY a JSON object:
{{"verified": true or false, "weapon_type": "name or null", "firing_evidence": "description of what you see", "confidence": 0.0-1.0}}
"""

VERIFY_NOT_FIRING = """\
These frames show a Doom deathmatch clip at 35fps. The player should be \
holding a weapon but NOT firing it.

Confirm: Is a weapon visible in the player's hands? Is it idle (no firing animation)?

Return ONLY a JSON object:
{{"verified": true or false, "weapon_type": "name or null", "is_idle": true or false, "confidence": 0.0-1.0}}
"""

DETECT_KILLS_DEATHS = """\
These frames (at ~2fps, timestamps shown) are from a Doom deathmatch game.
Look specifically for these events:

1. **Kill/Frag**: Enemy dies — look for: enemy falling/gibbing into pieces, \
blood explosion, frag message appearing in the top-left corner of the screen.

2. **Death**: Player dies — look for: screen flashing red, view tilting/dropping, \
health going to 0, everything going dark.

3. **Respawn**: Player respawns — look for: bright green circular flame/teleport \
animation at spawn point, sudden location change, health resetting to 100%, \
holding pistol or fist.

Return a JSON array of events. Each event MUST have:
- "timestamp": "MM:SS.s" of the event
- "event_type": "kill", "death", or "respawn"
- "description": brief description
- "details": {{"weapon_used": "shotgun"}} for kills, {{"cause": "rocket"}} for deaths, {{}} for respawns
- "confidence": 0.0-1.0

If no events found, return: []
Return ONLY the JSON array. No markdown fences.
"""

# %%
# --- Setup ---
print("Scanning episodes...")
episodes = scan_episodes(DATA_ROOT, max_shards=MAX_SHARDS)
print(f"Found {len(episodes)} episodes")

rng = random.Random(SEED)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

os.makedirs(OUTPUT_DIR, exist_ok=True)
for cat in ["weapon_switch", "attack_firing", "attack_not_firing", "kill", "death_respawn"]:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

all_clips = []  # master index

# %%
# --- Collect candidate events across all episodes ---
print("\nPhase 1: Collecting action-based events...")

ws_candidates = []  # (ep_idx, player, event_dict)
af_candidates = []  # (ep_idx, player, mid_frame)
anf_candidates = []  # (ep_idx, player, frame, weapon)

for ep_idx, ep in enumerate(episodes):
    meta = ep["meta"]
    for player in ["p1", "p2"]:
        if meta.get(f"random_policy_{player}", False):
            continue
        actions = load_actions(ep["shard"], ep["members"], player)

        # Weapon changes
        for ev in detect_weapon_changes(actions):
            ws_candidates.append((ep_idx, player, ev))

        # Attack bursts
        for s, e in detect_attack_bursts(actions):
            af_candidates.append((ep_idx, player, (s + e) // 2))

        # Not firing
        for frame, weapon in find_not_firing_frames(actions, 20, rng):
            anf_candidates.append((ep_idx, player, frame, weapon))

print(f"  Weapon changes: {len(ws_candidates)}")
print(f"  Attack bursts: {len(af_candidates)}")
print(f"  Not-firing candidates: {len(anf_candidates)}")

# %%
# --- Sample events ---
# Weapon switches: up to N per weapon_to
ws_by_weapon = defaultdict(list)
for item in ws_candidates:
    ws_by_weapon[item[2]["weapon_to"]].append(item)

sampled_ws = []
for w in sorted(ws_by_weapon.keys()):
    pool = ws_by_weapon[w]
    n = min(N_SWITCH_PER_WEAPON, len(pool))
    sampled_ws.extend(rng.sample(pool, n))
    print(f"  W{w} ({WEAPON_NAMES[w]}): sampled {n}/{len(pool)}")

sampled_af = rng.sample(af_candidates, min(N_ATTACK_FIRING, len(af_candidates)))
sampled_anf = rng.sample(anf_candidates, min(N_ATTACK_NOT_FIRING, len(anf_candidates)))

print(f"\nSampled: {len(sampled_ws)} switches, {len(sampled_af)} firing, {len(sampled_anf)} not-firing")

# %% [markdown]
# ## Phase 1: Extract and verify weapon switch clips

# %%
print("\n=== Extracting weapon switch clips ===")
ws_count = 0

# Group by (ep_idx, player) for video reuse
from itertools import groupby
sampled_ws.sort(key=lambda x: (x[0], x[1]))

for (ep_idx, player), group in groupby(sampled_ws, key=lambda x: (x[0], x[1])):
    items = list(group)
    ep = episodes[ep_idx]
    n_frames = ep["meta"]["n_frames"]
    video_key = f"video_{player}.mp4"

    # Collect all needed frames
    clip_ranges = []
    for _, _, ev in items:
        f_start = max(0, ev["frame"] - CLIP_CONTEXT)
        f_end = min(n_frames - 1, ev["frame"] + CLIP_CONTEXT)
        clip_ranges.append((ev, f_start, f_end))

    all_indices = set()
    for ev, fs, fe in clip_ranges:
        all_indices.update(range(fs, fe + 1))
    # Also add sparse frames for Gemini verification
    for ev, fs, fe in clip_ranges:
        for off in [-5, -2, 0, 3, 7]:
            fi = max(0, min(n_frames - 1, ev["frame"] + off))
            all_indices.add(fi)

    with open_video(ep["shard"], ep["members"][video_key]) as vpath:
        frame_cache = extract_frames(vpath, sorted(all_indices))
        actions = load_actions(ep["shard"], ep["members"], player)

        for ev, f_start, f_end in clip_ranges:
            wn_from = WEAPON_NAMES[ev["weapon_from"]]
            wn_to = WEAPON_NAMES[ev["weapon_to"]]

            # Gemini verification
            verify_indices = sorted(set(
                max(0, min(n_frames - 1, ev["frame"] + o)) for o in [-5, -2, 0, 3, 7]
            ))
            verify_frames = [frame_cache[i] for i in verify_indices if i in frame_cache]
            if len(verify_frames) < 3:
                continue

            prompt = VERIFY_WEAPON_SWITCH.format(weapon_from=wn_from, weapon_to=wn_to)
            contents = [pil_to_part(f) for f in verify_frames] + [prompt]

            try:
                raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
                result = parse_gemini_json(raw)
                if isinstance(result, list): result = result[0]
            except Exception as e:
                print(f"    ERR: {e}"); time.sleep(DELAY); continue

            verified = result.get("verified", False)
            tag = "OK" if verified else "SKIP"
            print(f"  [{ws_count}] {tag} {wn_from}->{wn_to} conf={result.get('confidence','?')}")

            if not verified:
                time.sleep(DELAY); continue

            # Save clip
            clip_id = f"ws_{wn_to}_{ws_count:03d}"
            clip_dir = os.path.join(OUTPUT_DIR, "weapon_switch")
            save_clip_video(frame_cache, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
            np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])

            annotation = {
                "clip_id": clip_id,
                "category": "weapon_switch",
                "source_episode": ep["meta"]["episode_id"],
                "source_player": player,
                "source_shard": ep["shard"],
                "frame_start": f_start,
                "frame_end": f_end,
                "duration_frames": f_end - f_start + 1,
                "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                "fps": GAME_FPS,
                "ground_truth": {
                    "weapon_from": wn_from,
                    "weapon_to": wn_to,
                    "switch_frame": ev["frame"],
                },
                "gemini_verified": True,
                "gemini_response": result,
            }
            with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                json.dump(annotation, f, indent=2)
            all_clips.append(annotation)
            ws_count += 1
            time.sleep(DELAY)

print(f"\nWeapon switch clips: {ws_count} verified")

# %% [markdown]
# ## Phase 1b: Extract and verify attack firing clips

# %%
print("\n=== Extracting attack firing clips ===")
af_count = 0

sampled_af.sort(key=lambda x: (x[0], x[1]))
for (ep_idx, player), group in groupby(sampled_af, key=lambda x: (x[0], x[1])):
    items = list(group)
    ep = episodes[ep_idx]
    n_frames = ep["meta"]["n_frames"]
    video_key = f"video_{player}.mp4"

    all_indices = set()
    clip_ranges = []
    for _, _, mid in items:
        f_start = max(0, mid - CLIP_CONTEXT)
        f_end = min(n_frames - 1, mid + CLIP_CONTEXT)
        clip_ranges.append((mid, f_start, f_end))
        all_indices.update(range(f_start, f_end + 1))
        for off in [-2, -1, 0, 1, 2]:
            all_indices.add(max(0, min(n_frames - 1, mid + off)))

    with open_video(ep["shard"], ep["members"][video_key]) as vpath:
        frame_cache = extract_frames(vpath, sorted(all_indices))
        actions = load_actions(ep["shard"], ep["members"], player)

        for mid, f_start, f_end in clip_ranges:
            verify_indices = sorted(set(
                max(0, min(n_frames - 1, mid + o)) for o in [-2, -1, 0, 1, 2]
            ))
            verify_frames = [frame_cache[i] for i in verify_indices if i in frame_cache]
            if len(verify_frames) < 3: continue

            contents = [pil_to_part(f) for f in verify_frames] + [VERIFY_ATTACK_FIRING]
            try:
                raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
                result = parse_gemini_json(raw)
                if isinstance(result, list): result = result[0]
            except Exception as e:
                print(f"    ERR: {e}"); time.sleep(DELAY); continue

            verified = result.get("verified", False)
            wtype = result.get("weapon_type", "?")
            evidence = result.get("firing_evidence", "")[:50]
            tag = "OK" if verified else "SKIP"
            print(f"  [{af_count}] {tag} {wtype} — {evidence}")

            if not verified:
                time.sleep(DELAY); continue

            clip_id = f"af_{af_count:03d}"
            clip_dir = os.path.join(OUTPUT_DIR, "attack_firing")
            save_clip_video(frame_cache, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
            np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])

            annotation = {
                "clip_id": clip_id, "category": "attack_firing",
                "source_episode": ep["meta"]["episode_id"],
                "source_player": player, "source_shard": ep["shard"],
                "frame_start": f_start, "frame_end": f_end,
                "duration_frames": f_end - f_start + 1,
                "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                "fps": GAME_FPS,
                "ground_truth": {"weapon_type": wtype, "burst_mid": mid},
                "gemini_verified": True, "gemini_response": result,
            }
            with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                json.dump(annotation, f, indent=2)
            all_clips.append(annotation)
            af_count += 1
            time.sleep(DELAY)

print(f"\nAttack firing clips: {af_count} verified")

# %% [markdown]
# ## Phase 1c: Extract and verify attack-not-firing clips

# %%
print("\n=== Extracting attack-not-firing clips ===")
anf_count = 0

sampled_anf.sort(key=lambda x: (x[0], x[1]))
for (ep_idx, player), group in groupby(sampled_anf, key=lambda x: (x[0], x[1])):
    items = list(group)
    ep = episodes[ep_idx]
    n_frames = ep["meta"]["n_frames"]
    video_key = f"video_{player}.mp4"

    all_indices = set()
    clip_ranges = []
    for _, _, frame, weapon in items:
        f_start = max(0, frame - CLIP_CONTEXT)
        f_end = min(n_frames - 1, frame + CLIP_CONTEXT)
        clip_ranges.append((frame, weapon, f_start, f_end))
        all_indices.update(range(f_start, f_end + 1))
        for off in [-2, -1, 0, 1, 2]:
            all_indices.add(max(0, min(n_frames - 1, frame + off)))

    with open_video(ep["shard"], ep["members"][video_key]) as vpath:
        frame_cache = extract_frames(vpath, sorted(all_indices))
        actions = load_actions(ep["shard"], ep["members"], player)

        for frame, weapon, f_start, f_end in clip_ranges:
            verify_indices = sorted(set(
                max(0, min(n_frames - 1, frame + o)) for o in [-2, -1, 0, 1, 2]
            ))
            verify_frames = [frame_cache[i] for i in verify_indices if i in frame_cache]
            if len(verify_frames) < 3: continue

            contents = [pil_to_part(f) for f in verify_frames] + [VERIFY_NOT_FIRING]
            try:
                raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
                result = parse_gemini_json(raw)
                if isinstance(result, list): result = result[0]
            except Exception as e:
                print(f"    ERR: {e}"); time.sleep(DELAY); continue

            verified = result.get("verified", False) and result.get("is_idle", False)
            tag = "OK" if verified else "SKIP"
            print(f"  [{anf_count}] {tag} {WEAPON_NAMES.get(weapon, '?')}")

            if not verified:
                time.sleep(DELAY); continue

            clip_id = f"anf_{anf_count:03d}"
            clip_dir = os.path.join(OUTPUT_DIR, "attack_not_firing")
            save_clip_video(frame_cache, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
            np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])

            annotation = {
                "clip_id": clip_id, "category": "attack_not_firing",
                "source_episode": ep["meta"]["episode_id"],
                "source_player": player, "source_shard": ep["shard"],
                "frame_start": f_start, "frame_end": f_end,
                "duration_frames": f_end - f_start + 1,
                "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                "fps": GAME_FPS,
                "ground_truth": {"weapon_type": WEAPON_NAMES.get(weapon, "unknown"), "held_weapon_slot": weapon},
                "gemini_verified": True, "gemini_response": result,
            }
            with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                json.dump(annotation, f, indent=2)
            all_clips.append(annotation)
            anf_count += 1
            time.sleep(DELAY)

print(f"\nAttack not-firing clips: {anf_count} verified")

# %% [markdown]
# ## Phase 2: Gemini-detected kills and deaths

# %%
print("\n=== Detecting kills and deaths via Gemini ===")
kill_count = 0
death_count = 0

SEGMENT_LEN = 15  # seconds
SEGMENT_FPS = 2

for ep_idx, ep in enumerate(episodes):
    meta = ep["meta"]
    for player in ["p1", "p2"]:
        if meta.get(f"random_policy_{player}", False):
            continue
        if kill_count >= N_KILL_TARGET and death_count >= N_DEATH_TARGET:
            break

        n_frames = meta["n_frames"]
        video_key = f"video_{player}.mp4"
        ep_id = meta["episode_id"][:12]
        n_segments = max(1, n_frames // (SEGMENT_LEN * GAME_FPS))

        print(f"\n  {ep_id}... {player}: scanning {n_segments} segments")

        with open_video(ep["shard"], ep["members"][video_key]) as vpath:
            actions = load_actions(ep["shard"], ep["members"], player)

            for seg_idx in range(n_segments):
                if kill_count >= N_KILL_TARGET and death_count >= N_DEATH_TARGET:
                    break

                seg_start = seg_idx * SEGMENT_LEN * GAME_FPS
                seg_end = min((seg_idx + 1) * SEGMENT_LEN * GAME_FPS, n_frames - 1)
                step = max(1, GAME_FPS // SEGMENT_FPS)
                indices = list(range(int(seg_start), int(seg_end), step))
                if not indices: continue

                frame_cache = extract_frames(vpath, indices)
                frames = [frame_cache[i] for i in indices if i in frame_cache]
                if len(frames) < 5: continue

                contents = []
                for f, i in zip(frames, indices):
                    contents.append(f"[{frame_to_ts(i)}]")
                    contents.append(pil_to_part(f))
                contents.append(DETECT_KILLS_DEATHS)

                try:
                    raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
                    events = parse_gemini_json(raw)
                    if not isinstance(events, list): events = [events]
                except Exception as e:
                    print(f"    Segment {seg_idx}: ERR {e}")
                    time.sleep(DELAY); continue

                for ev in events:
                    etype = ev.get("event_type", "")
                    ts = ev.get("timestamp", "")
                    if not ts: continue

                    # Parse timestamp to frame
                    parts = ts.strip().split(":")
                    try:
                        secs = float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[0])
                    except (ValueError, IndexError):
                        continue
                    event_frame = int(round(secs * GAME_FPS))
                    event_frame = max(0, min(n_frames - 1, event_frame))

                    f_start = max(0, event_frame - CLIP_CONTEXT)
                    f_end = min(n_frames - 1, event_frame + CLIP_CONTEXT)

                    # Extract clip frames
                    clip_indices = list(range(f_start, f_end + 1))
                    clip_frames = extract_frames(vpath, clip_indices)

                    if etype == "kill" and kill_count < N_KILL_TARGET:
                        clip_id = f"kill_{kill_count:03d}"
                        clip_dir = os.path.join(OUTPUT_DIR, "kill")
                        save_clip_video(clip_frames, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
                        np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])
                        annotation = {
                            "clip_id": clip_id, "category": "kill",
                            "source_episode": meta["episode_id"],
                            "source_player": player, "source_shard": ep["shard"],
                            "frame_start": f_start, "frame_end": f_end,
                            "duration_frames": f_end - f_start + 1,
                            "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                            "fps": GAME_FPS,
                            "ground_truth": {
                                "description": ev.get("description", ""),
                                "weapon_used": ev.get("details", {}).get("weapon_used", "unknown"),
                            },
                            "gemini_verified": True, "gemini_response": ev,
                        }
                        with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                            json.dump(annotation, f, indent=2)
                        all_clips.append(annotation)
                        kill_count += 1
                        print(f"    KILL [{kill_count}]: {ev.get('description','')[:60]}")

                    elif etype in ("death", "respawn") and death_count < N_DEATH_TARGET:
                        clip_id = f"dr_{death_count:03d}"
                        clip_dir = os.path.join(OUTPUT_DIR, "death_respawn")
                        save_clip_video(clip_frames, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
                        np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])
                        annotation = {
                            "clip_id": clip_id, "category": "death_respawn",
                            "source_episode": meta["episode_id"],
                            "source_player": player, "source_shard": ep["shard"],
                            "frame_start": f_start, "frame_end": f_end,
                            "duration_frames": f_end - f_start + 1,
                            "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                            "fps": GAME_FPS,
                            "ground_truth": {
                                "event_type": etype,
                                "description": ev.get("description", ""),
                                "cause": ev.get("details", {}).get("cause", ""),
                            },
                            "gemini_verified": True, "gemini_response": ev,
                        }
                        with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                            json.dump(annotation, f, indent=2)
                        all_clips.append(annotation)
                        death_count += 1
                        print(f"    DEATH/RESPAWN [{death_count}]: {ev.get('description','')[:60]}")

                time.sleep(DELAY)

    if kill_count >= N_KILL_TARGET and death_count >= N_DEATH_TARGET:
        break

print(f"\nKills: {kill_count}, Deaths/Respawns: {death_count}")

# %% [markdown]
# ## Save master index

# %%
# Summary
from collections import Counter
cat_counts = Counter(c["category"] for c in all_clips)
print(f"\n=== FINAL SUMMARY ===")
print(f"Total clips: {len(all_clips)}")
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat}: {count}")

index = {
    "total_clips": len(all_clips),
    "categories": dict(cat_counts),
    "data_root": DATA_ROOT,
    "max_shards": MAX_SHARDS,
    "model": MODEL_NAME,
    "seed": SEED,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "clips": all_clips,
}
index_path = os.path.join(OUTPUT_DIR, "index.json")
with open(index_path, "w") as f:
    json.dump(index, f, indent=2)
print(f"\nIndex saved to {index_path}")
print("Done!")
