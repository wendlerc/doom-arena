# %% [markdown]
# # Eval Clip Extraction v2: Reference-Frame-Guided
#
# Improvement over v1: uses reference frames of each weapon as visual examples
# in Gemini prompts, so it can accurately identify weapons and verify events.
#
# Steps:
# 1. Resolve weapon slot → visual weapon mapping from human gameplay
# 2. Extract clear reference frames for each weapon
# 3. Use references to guide clip extraction with strict verification

# %%
import sys, os, io, json, re, time, random, tarfile, tempfile
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from itertools import groupby

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
HUMAN_SHARDS = sorted(Path("recordings").glob("human-*.tar"))
OUTPUT_DIR = "datasets/eval_clips_v2"
REF_DIR = os.path.join(OUTPUT_DIR, "reference_frames")
MODEL_NAME = "gemini-2.5-flash"
DELAY = 2.0
SEED = 42

N_SWITCH_PER_WEAPON = 15
N_ATTACK_FIRING = 40
N_ATTACK_NOT_FIRING = 40
N_KILL_TARGET = 25
N_DEATH_TARGET = 25

CLIP_CONTEXT = 35  # 1s each side of event

WEAPON_SELECT_INDICES = list(range(4, 11))
ATTACK_INDEX = 11

# %%
# --- Helpers ---

def _retry_delay(err):
    m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s?['\"]", err)
    return float(m.group(1)) if m else None

def call_gemini(client, model_name, contents, max_retries=5, base_delay=10.0):
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model_name, contents=contents).text
        except Exception as e:
            err = str(e)
            if attempt == max_retries: raise
            if any(c in err for c in ["429", "ResourceExhausted", "500", "503"]):
                d = max(_retry_delay(err) or 0, base_delay * (2 ** attempt))
                print(f"      Retry in {d:.0f}s...", flush=True); time.sleep(d)
            else: raise

def parse_json(raw):
    text = re.sub(r"```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    for fn in [
        lambda: json.loads(text),
        lambda: json.loads(re.search(r"\{.*\}", text, re.DOTALL).group()),
        lambda: json.loads(re.search(r"\[.*\]", text, re.DOTALL).group()),
    ]:
        try: return fn()
        except: continue
    # Truncated array
    m = re.search(r"\[.*", text, re.DOTALL)
    if m:
        lb = m.group().rfind("}")
        if lb > 0:
            try: return json.loads(m.group()[:lb+1] + "]")
            except: pass
    raise ValueError(f"Cannot parse: {text[:200]}...")

def pil_to_part(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buf.getvalue()))

def scan_episodes(shards):
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
        except: continue
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
    try: yield tmp.name
    finally: os.unlink(tmp.name)

def extract_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    fset, mx = set(frame_indices), max(frame_indices)
    result = {}; idx = 0
    while cap.isOpened() and idx <= mx:
        ret, frame = cap.read()
        if not ret: break
        if idx in fset:
            result[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return result

def save_clip_video(frames_dict, f_start, f_end, out_path):
    indices = sorted(k for k in frames_dict if f_start <= k <= f_end)
    if not indices: return False
    first = frames_dict[indices[0]]
    w, h = first.size
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), GAME_FPS, (w, h))
    for i in indices:
        if i in frames_dict:
            writer.write(cv2.cvtColor(np.array(frames_dict[i]), cv2.COLOR_RGB2BGR))
    writer.release()
    return True

def detect_weapon_changes(actions):
    prev_w = None; events = []
    for i in range(actions.shape[0]):
        active = np.where(actions[i, 4:11] == 1.0)[0]
        if len(active) > 0:
            w = int(active[0]) + 1
            if prev_w is not None and w != prev_w:
                events.append({"frame": i, "weapon_from": prev_w, "weapon_to": w})
            prev_w = w
    return events

def detect_attack_bursts(actions, min_gap=5, min_length=3):
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

def frame_to_ts(f):
    t = f / GAME_FPS
    return f"{int(t//60):02d}:{t%60:04.1f}"

def ts_to_frame(ts):
    parts = ts.strip().split(":")
    secs = float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[0])
    return int(round(secs * GAME_FPS))

# %%
# --- Setup ---
os.makedirs(REF_DIR, exist_ok=True)
for cat in ["weapon_switch", "attack_firing", "attack_not_firing", "kill", "death_respawn"]:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

print(f"Human shards: {len(HUMAN_SHARDS)}")
episodes = scan_episodes(HUMAN_SHARDS)
print(f"Episodes: {len(episodes)}")

rng = random.Random(SEED)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# %% [markdown]
# ## Step 1: Resolve weapon slots and extract reference frames
#
# For each weapon slot (1-7), sample frames from human gameplay where that slot
# is active, send to Gemini to identify the weapon, and save the clearest frame.

# %%
print("\n=== Step 1: Weapon reference frames ===")

# Collect frames per weapon slot across all episodes
slot_frames = defaultdict(list)  # slot → [(ep_idx, frame_idx), ...]

for ep_idx, ep in enumerate(episodes):
    actions = load_actions(ep["shard"], ep["members"], "p1")
    # Track which weapon is held (most recent SELECT)
    current_weapon = None
    weapon_stretches = []  # (slot, start, end)
    for i in range(len(actions)):
        active = np.where(actions[i, 4:11] == 1.0)[0]
        if len(active) > 0:
            w = int(active[0]) + 1
            if w != current_weapon:
                if current_weapon is not None and weapon_stretches:
                    weapon_stretches[-1] = (current_weapon, weapon_stretches[-1][1], i - 1)
                weapon_stretches.append((w, i, None))
                current_weapon = w
    if weapon_stretches and weapon_stretches[-1][2] is None:
        weapon_stretches[-1] = (current_weapon, weapon_stretches[-1][1], len(actions) - 1)

    # For each stretch, sample frames where NOT attacking (clearest weapon view)
    for slot, start, end in weapon_stretches:
        if end is None: continue
        for i in range(start, min(end + 1, len(actions))):
            if actions[i, ATTACK_INDEX] == 0:
                slot_frames[slot].append((ep_idx, i))

print("Weapon slot frame counts:")
for slot in sorted(slot_frames.keys()):
    print(f"  Slot {slot}: {len(slot_frames[slot])} non-attack frames")

# %%
# For each slot, sample up to 10 frames and ask Gemini to identify + pick best
SLOT_IDENTIFY_PROMPT = """\
These frames are from a first-person Doom deathmatch game. The player is \
holding a weapon in each frame (not firing).

For each frame, identify the weapon being held. Then pick the SINGLE frame \
where the weapon is most clearly visible.

Doom weapons: fist, chainsaw, pistol, shotgun (pump-action), \
super_shotgun (double-barrel), chaingun (heavy rotary gun), \
rocket_launcher (large tube), plasma_rifle (blue energy weapon), bfg

Return ONLY a JSON object:
{"weapon_name": "the weapon name", "best_frame_index": 0, "confidence": 0.0-1.0}

Frame indices are 0-based in the order shown.
"""

weapon_mapping = {}  # slot → {"name": str, "ref_frame": PIL.Image, "ref_path": str}

for slot in sorted(slot_frames.keys()):
    candidates = slot_frames[slot]
    sampled = rng.sample(candidates, min(10, len(candidates)))

    # Extract frames
    frames_by_ep = defaultdict(list)
    for ep_idx, frame_idx in sampled:
        frames_by_ep[ep_idx].append(frame_idx)

    all_frames = []
    for ep_idx, frame_indices in frames_by_ep.items():
        ep = episodes[ep_idx]
        with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
            fc = extract_frames(vpath, frame_indices)
            for fi in frame_indices:
                if fi in fc:
                    all_frames.append(fc[fi])

    if not all_frames:
        print(f"  Slot {slot}: no frames extracted, skipping")
        continue

    # Ask Gemini
    contents = []
    for i, img in enumerate(all_frames):
        contents.append(f"Frame {i}:")
        contents.append(pil_to_part(img))
    contents.append(SLOT_IDENTIFY_PROMPT)

    try:
        raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
        result = parse_json(raw)
        if isinstance(result, list): result = result[0]
        name = result.get("weapon_name", "unknown").lower().replace(" ", "_").replace("-", "_")
        best_idx = result.get("best_frame_index", 0)
        best_idx = max(0, min(best_idx, len(all_frames) - 1))
        conf = result.get("confidence", 0)

        # Save reference frame
        ref_path = os.path.join(REF_DIR, f"slot{slot}_{name}.jpg")
        all_frames[best_idx].save(ref_path, quality=90)

        weapon_mapping[slot] = {
            "name": name,
            "slot": slot,
            "ref_path": ref_path,
            "ref_image": all_frames[best_idx],
            "confidence": conf,
        }
        print(f"  Slot {slot} = {name} (conf={conf}, saved {ref_path})")
    except Exception as e:
        print(f"  Slot {slot}: FAILED ({e})")

    time.sleep(DELAY)

# Save mapping
mapping_json = {s: {k: v for k, v in d.items() if k != "ref_image"} for s, d in weapon_mapping.items()}
with open(os.path.join(REF_DIR, "reference.json"), "w") as f:
    json.dump(mapping_json, f, indent=2)
print(f"\nWeapon mapping saved. {len(weapon_mapping)} weapons identified.")

# %% [markdown]
# ## Step 2: Extract clips with reference-frame-guided verification

# %%
all_clips = []

# Build reference content parts for each weapon
ref_parts = {}
for slot, info in weapon_mapping.items():
    ref_parts[slot] = pil_to_part(info["ref_image"])

def weapon_name(slot):
    return weapon_mapping.get(slot, {}).get("name", f"weapon_{slot}")

# %% [markdown]
# ### 2a: Weapon switch clips

# %%
print("\n=== Extracting weapon switch clips ===")

SWITCH_VERIFY_PROMPT = """\
REFERENCE — {weapon_from}:
[shown above as Reference A]

REFERENCE — {weapon_to}:
[shown above as Reference B]

The following gameplay frames (in order) should show the player switching \
from {weapon_from} to {weapon_to}. Compare with the reference images.

Questions:
1. In the EARLY frames, is the weapon matching Reference A ({weapon_from})?
2. In the LATER frames, is the weapon matching Reference B ({weapon_to})?
3. Is there a visible weapon transition (lowering + raising animation)?

Return ONLY a JSON object:
{{"verified": true or false, "weapon_from_visible": true or false, \
"weapon_to_visible": true or false, "confidence": 0.0-1.0, \
"notes": "brief description"}}
"""

# Collect weapon changes from all human episodes
ws_candidates = []
for ep_idx, ep in enumerate(episodes):
    actions = load_actions(ep["shard"], ep["members"], "p1")
    for ev in detect_weapon_changes(actions):
        ws_candidates.append((ep_idx, ev))

# Sample per weapon_to
ws_by_weapon = defaultdict(list)
for item in ws_candidates:
    ws_by_weapon[item[1]["weapon_to"]].append(item)

sampled_ws = []
for w in sorted(ws_by_weapon.keys()):
    pool = ws_by_weapon[w]
    n = min(N_SWITCH_PER_WEAPON, len(pool))
    sampled_ws.extend(rng.sample(pool, n))
    print(f"  {weapon_name(w)}: {n}/{len(pool)} sampled")

ws_count = 0
sampled_ws.sort(key=lambda x: x[0])
for ep_idx, group in groupby(sampled_ws, key=lambda x: x[0]):
    items = list(group)
    ep = episodes[ep_idx]
    n_frames = ep["meta"]["n_frames"]

    all_indices = set()
    clip_data = []
    for _, ev in items:
        f_start = max(0, ev["frame"] - CLIP_CONTEXT)
        f_end = min(n_frames - 1, ev["frame"] + CLIP_CONTEXT)
        # Sparse frames for verification: before, during, after switch
        verify = sorted(set(max(0, min(n_frames-1, ev["frame"] + o)) for o in [-10, -5, -2, 0, 2, 5, 10, 15]))
        clip_data.append((ev, f_start, f_end, verify))
        all_indices.update(range(f_start, f_end + 1))
        all_indices.update(verify)

    with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
        frame_cache = extract_frames(vpath, sorted(all_indices))
        actions = load_actions(ep["shard"], ep["members"], "p1")

        for ev, f_start, f_end, verify in clip_data:
            wf, wt = ev["weapon_from"], ev["weapon_to"]
            wf_name, wt_name = weapon_name(wf), weapon_name(wt)

            verify_frames = [frame_cache[i] for i in verify if i in frame_cache]
            if len(verify_frames) < 4: continue

            # Build prompt with reference frames
            contents = [f"Reference A ({wf_name}):"]
            if wf in ref_parts: contents.append(ref_parts[wf])
            contents.append(f"Reference B ({wt_name}):")
            if wt in ref_parts: contents.append(ref_parts[wt])
            contents.append("Gameplay frames (chronological):")
            for img in verify_frames:
                contents.append(pil_to_part(img))
            contents.append(SWITCH_VERIFY_PROMPT.format(weapon_from=wf_name, weapon_to=wt_name))

            try:
                raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
                result = parse_json(raw)
                if isinstance(result, list): result = result[0]
            except Exception as e:
                print(f"    ERR: {e}"); time.sleep(DELAY); continue

            verified = (result.get("verified", False) and
                       result.get("weapon_from_visible", False) and
                       result.get("weapon_to_visible", False))
            tag = "OK" if verified else "SKIP"
            print(f"  [{ws_count}] {tag} {wf_name}->{wt_name} conf={result.get('confidence','?')} {result.get('notes','')[:40]}")

            if not verified:
                time.sleep(DELAY); continue

            clip_id = f"ws_{wt_name}_{ws_count:03d}"
            clip_dir = os.path.join(OUTPUT_DIR, "weapon_switch")
            save_clip_video(frame_cache, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
            np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])

            annotation = {
                "clip_id": clip_id, "category": "weapon_switch",
                "source_episode": ep["meta"]["episode_id"], "source_player": "p1",
                "source_shard": ep["shard"],
                "frame_start": f_start, "frame_end": f_end,
                "duration_frames": f_end - f_start + 1,
                "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                "fps": GAME_FPS,
                "ground_truth": {"weapon_from": wf_name, "weapon_to": wt_name, "switch_frame": ev["frame"]},
                "gemini_verified": True, "gemini_response": result,
            }
            with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                json.dump(annotation, f, indent=2)
            all_clips.append(annotation)
            ws_count += 1
            time.sleep(DELAY)

print(f"\nWeapon switch clips: {ws_count}")

# %% [markdown]
# ### 2b: Attack firing clips

# %%
print("\n=== Extracting attack firing clips ===")

ATTACK_VERIFY_PROMPT = """\
REFERENCE — {weapon_name}:
[shown above]

The following gameplay frames show the player pressing ATTACK. \
Compare the weapon with the reference image.

Questions:
1. Is the player holding the SAME weapon as in the reference ({weapon_name})?
2. Is there a visible firing animation? Look for: muzzle flash, bullet/projectile \
leaving the weapon, recoil kick, shotgun pump, rocket trail, plasma bolt, explosion.

Return ONLY a JSON object:
{{"verified": true or false, "weapon_matches": true or false, \
"firing_visible": true or false, "weapon_type": "name", \
"firing_evidence": "what you see", "confidence": 0.0-1.0}}
"""

# Collect attack bursts and determine which weapon is held
af_candidates = []
for ep_idx, ep in enumerate(episodes):
    actions = load_actions(ep["shard"], ep["members"], "p1")
    bursts = detect_attack_bursts(actions)
    # For each burst, find which weapon is active
    for s, e in bursts:
        mid = (s + e) // 2
        # Find most recent weapon select before burst
        held = None
        for i in range(mid, max(0, mid - 100), -1):
            active = np.where(actions[i, 4:11] == 1.0)[0]
            if len(active) > 0: held = int(active[0]) + 1; break
        if held and held >= 2:  # skip fist
            af_candidates.append((ep_idx, mid, held, s, e))

sampled_af = rng.sample(af_candidates, min(N_ATTACK_FIRING * 2, len(af_candidates)))  # oversample to account for failures
af_count = 0

sampled_af.sort(key=lambda x: x[0])
for ep_idx, group in groupby(sampled_af, key=lambda x: x[0]):
    if af_count >= N_ATTACK_FIRING: break
    items = list(group)
    ep = episodes[ep_idx]
    n_frames = ep["meta"]["n_frames"]

    all_indices = set()
    clip_data = []
    for _, mid, held, bs, be in items:
        f_start = max(0, mid - CLIP_CONTEXT)
        f_end = min(n_frames - 1, mid + CLIP_CONTEXT)
        verify = sorted(set(max(0, min(n_frames-1, mid + o)) for o in [-3, -1, 0, 1, 3]))
        clip_data.append((mid, held, f_start, f_end, verify))
        all_indices.update(range(f_start, f_end + 1))
        all_indices.update(verify)

    with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
        frame_cache = extract_frames(vpath, sorted(all_indices))
        actions = load_actions(ep["shard"], ep["members"], "p1")

        for mid, held, f_start, f_end, verify in clip_data:
            if af_count >= N_ATTACK_FIRING: break
            wname = weapon_name(held)
            verify_frames = [frame_cache[i] for i in verify if i in frame_cache]
            if len(verify_frames) < 3: continue

            contents = [f"Reference ({wname}):"]
            if held in ref_parts: contents.append(ref_parts[held])
            contents.append("Gameplay frames during ATTACK:")
            for img in verify_frames:
                contents.append(pil_to_part(img))
            contents.append(ATTACK_VERIFY_PROMPT.format(weapon_name=wname))

            try:
                raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
                result = parse_json(raw)
                if isinstance(result, list): result = result[0]
            except Exception as e:
                print(f"    ERR: {e}"); time.sleep(DELAY); continue

            verified = result.get("verified", False) and result.get("firing_visible", False)
            tag = "OK" if verified else "SKIP"
            evidence = result.get("firing_evidence", "")[:50]
            print(f"  [{af_count}] {tag} {wname} — {evidence}")

            if not verified:
                time.sleep(DELAY); continue

            clip_id = f"af_{wname}_{af_count:03d}"
            clip_dir = os.path.join(OUTPUT_DIR, "attack_firing")
            save_clip_video(frame_cache, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
            np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])

            annotation = {
                "clip_id": clip_id, "category": "attack_firing",
                "source_episode": ep["meta"]["episode_id"], "source_player": "p1",
                "source_shard": ep["shard"],
                "frame_start": f_start, "frame_end": f_end,
                "duration_frames": f_end - f_start + 1,
                "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                "fps": GAME_FPS,
                "ground_truth": {"weapon_type": wname, "burst_mid": mid},
                "gemini_verified": True, "gemini_response": result,
            }
            with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                json.dump(annotation, f, indent=2)
            all_clips.append(annotation)
            af_count += 1
            time.sleep(DELAY)

print(f"\nAttack firing clips: {af_count}")

# %% [markdown]
# ### 2c: Attack not-firing clips

# %%
print("\n=== Extracting attack-not-firing clips ===")

IDLE_VERIFY_PROMPT = """\
REFERENCE — {weapon_name}:
[shown above]

The following frames should show the player holding {weapon_name} without firing.

Questions:
1. Is the weapon matching the reference ({weapon_name})?
2. Is the weapon IDLE (no firing animation, no muzzle flash, no projectile)?

Return ONLY a JSON object:
{{"verified": true or false, "weapon_matches": true or false, \
"is_idle": true or false, "confidence": 0.0-1.0}}
"""

anf_candidates = []
for ep_idx, ep in enumerate(episodes):
    actions = load_actions(ep["shard"], ep["members"], "p1")
    current_w = None
    for i in range(len(actions)):
        active = np.where(actions[i, 4:11] == 1.0)[0]
        if len(active) > 0: current_w = int(active[0]) + 1
        if current_w and current_w >= 2 and actions[i, ATTACK_INDEX] == 0:
            anf_candidates.append((ep_idx, i, current_w))

sampled_anf = rng.sample(anf_candidates, min(N_ATTACK_NOT_FIRING * 2, len(anf_candidates)))
anf_count = 0

sampled_anf.sort(key=lambda x: x[0])
for ep_idx, group in groupby(sampled_anf, key=lambda x: x[0]):
    if anf_count >= N_ATTACK_NOT_FIRING: break
    items = list(group)
    ep = episodes[ep_idx]
    n_frames = ep["meta"]["n_frames"]

    all_indices = set()
    clip_data = []
    for _, frame, held in items:
        f_start = max(0, frame - CLIP_CONTEXT)
        f_end = min(n_frames - 1, frame + CLIP_CONTEXT)
        verify = sorted(set(max(0, min(n_frames-1, frame + o)) for o in [-2, -1, 0, 1, 2]))
        clip_data.append((frame, held, f_start, f_end, verify))
        all_indices.update(range(f_start, f_end + 1))
        all_indices.update(verify)

    with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
        frame_cache = extract_frames(vpath, sorted(all_indices))
        actions = load_actions(ep["shard"], ep["members"], "p1")

        for frame, held, f_start, f_end, verify in clip_data:
            if anf_count >= N_ATTACK_NOT_FIRING: break
            wname = weapon_name(held)
            verify_frames = [frame_cache[i] for i in verify if i in frame_cache]
            if len(verify_frames) < 3: continue

            contents = [f"Reference ({wname}):"]
            if held in ref_parts: contents.append(ref_parts[held])
            contents.append("Gameplay frames (should be idle):")
            for img in verify_frames: contents.append(pil_to_part(img))
            contents.append(IDLE_VERIFY_PROMPT.format(weapon_name=wname))

            try:
                raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
                result = parse_json(raw)
                if isinstance(result, list): result = result[0]
            except Exception as e:
                print(f"    ERR: {e}"); time.sleep(DELAY); continue

            verified = result.get("verified", False) and result.get("is_idle", False)
            tag = "OK" if verified else "SKIP"
            print(f"  [{anf_count}] {tag} {wname}")

            if not verified:
                time.sleep(DELAY); continue

            clip_id = f"anf_{wname}_{anf_count:03d}"
            clip_dir = os.path.join(OUTPUT_DIR, "attack_not_firing")
            save_clip_video(frame_cache, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
            np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])

            annotation = {
                "clip_id": clip_id, "category": "attack_not_firing",
                "source_episode": ep["meta"]["episode_id"], "source_player": "p1",
                "source_shard": ep["shard"],
                "frame_start": f_start, "frame_end": f_end,
                "duration_frames": f_end - f_start + 1,
                "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                "fps": GAME_FPS,
                "ground_truth": {"weapon_type": wname, "held_weapon_slot": held},
                "gemini_verified": True, "gemini_response": result,
            }
            with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                json.dump(annotation, f, indent=2)
            all_clips.append(annotation)
            anf_count += 1
            time.sleep(DELAY)

print(f"\nAttack not-firing clips: {anf_count}")

# %% [markdown]
# ### 2d: Kill and death/respawn clips (Gemini segment scan)

# %%
print("\n=== Detecting kills and deaths ===")

KILL_DEATH_PROMPT = """\
These frames (at ~2fps, timestamps shown) are from a Doom deathmatch game.
Look specifically for:

1. **Kill/Frag**: Enemy dies — enemy falls/gibs into red chunks, frag message \
in the top-left corner, score increasing.

2. **Death**: Player dies — screen flashes bright RED, view tilts and drops to \
the ground, health goes to 0%.

3. **Respawn**: Player respawns — a bright GREEN circular flame/teleport \
animation appears, the player suddenly appears at a new location holding a pistol, \
health resets to 100%.

Return a JSON array. Each event MUST have:
- "timestamp": "MM:SS.s"
- "event_type": "kill", "death", or "respawn"
- "description": what you see
- "details": {{"weapon_used": "name"}} for kills, {{"cause": "description"}} for deaths
- "confidence": 0.0-1.0

If none found, return: []
Return ONLY the JSON array. No markdown.
"""

kill_count = 0
death_count = 0
SEGMENT_LEN = 15
SEGMENT_FPS = 2

for ep_idx, ep in enumerate(episodes):
    if kill_count >= N_KILL_TARGET and death_count >= N_DEATH_TARGET: break
    meta = ep["meta"]
    n_frames = meta["n_frames"]
    n_segments = max(1, n_frames // (SEGMENT_LEN * GAME_FPS))
    ep_id = meta["episode_id"][:12]

    print(f"\n  {ep_id}...: {n_segments} segments, frags={meta.get('frag_p1')}, deaths={meta.get('death_p1')}")

    with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
        actions = load_actions(ep["shard"], ep["members"], "p1")

        for seg_idx in range(n_segments):
            if kill_count >= N_KILL_TARGET and death_count >= N_DEATH_TARGET: break

            seg_start = seg_idx * SEGMENT_LEN * GAME_FPS
            seg_end = min((seg_idx + 1) * SEGMENT_LEN * GAME_FPS, n_frames - 1)
            step = max(1, GAME_FPS // SEGMENT_FPS)
            indices = list(range(int(seg_start), int(seg_end), step))
            if not indices: continue

            fc = extract_frames(vpath, indices)
            frames = [fc[i] for i in indices if i in fc]
            if len(frames) < 5: continue

            contents = []
            for img, i in zip(frames, indices):
                contents.append(f"[{frame_to_ts(i)}]")
                contents.append(pil_to_part(img))
            contents.append(KILL_DEATH_PROMPT)

            try:
                raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
                events = parse_json(raw)
                if not isinstance(events, list): events = [events]
            except Exception as e:
                print(f"    Seg {seg_idx}: ERR"); time.sleep(DELAY); continue

            for ev in events:
                etype = ev.get("event_type", "")
                ts = ev.get("timestamp", "")
                if not ts: continue
                try: event_frame = ts_to_frame(ts)
                except: continue
                event_frame = max(0, min(n_frames - 1, event_frame))

                f_start = max(0, event_frame - CLIP_CONTEXT)
                f_end = min(n_frames - 1, event_frame + CLIP_CONTEXT)
                clip_frames = extract_frames(vpath, list(range(f_start, f_end + 1)))

                if etype == "kill" and kill_count < N_KILL_TARGET:
                    clip_id = f"kill_{kill_count:03d}"
                    clip_dir = os.path.join(OUTPUT_DIR, "kill")
                    save_clip_video(clip_frames, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
                    np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])
                    annotation = {
                        "clip_id": clip_id, "category": "kill",
                        "source_episode": meta["episode_id"], "source_player": "p1",
                        "source_shard": ep["shard"],
                        "frame_start": f_start, "frame_end": f_end,
                        "duration_frames": f_end - f_start + 1,
                        "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                        "fps": GAME_FPS,
                        "ground_truth": {"description": ev.get("description",""), "weapon_used": ev.get("details",{}).get("weapon_used","")},
                        "gemini_verified": True, "gemini_response": ev,
                    }
                    with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                        json.dump(annotation, f, indent=2)
                    all_clips.append(annotation)
                    kill_count += 1
                    print(f"    KILL [{kill_count}]: {ev.get('description','')[:50]}")

                elif etype in ("death", "respawn") and death_count < N_DEATH_TARGET:
                    clip_id = f"dr_{death_count:03d}"
                    clip_dir = os.path.join(OUTPUT_DIR, "death_respawn")
                    save_clip_video(clip_frames, f_start, f_end, os.path.join(clip_dir, f"{clip_id}.mp4"))
                    np.save(os.path.join(clip_dir, f"{clip_id}_actions.npy"), actions[f_start:f_end + 1])
                    annotation = {
                        "clip_id": clip_id, "category": "death_respawn",
                        "source_episode": meta["episode_id"], "source_player": "p1",
                        "source_shard": ep["shard"],
                        "frame_start": f_start, "frame_end": f_end,
                        "duration_frames": f_end - f_start + 1,
                        "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                        "fps": GAME_FPS,
                        "ground_truth": {"event_type": etype, "description": ev.get("description",""), "cause": ev.get("details",{}).get("cause","")},
                        "gemini_verified": True, "gemini_response": ev,
                    }
                    with open(os.path.join(clip_dir, f"{clip_id}.json"), "w") as f:
                        json.dump(annotation, f, indent=2)
                    all_clips.append(annotation)
                    death_count += 1
                    print(f"    DEATH/RESPAWN [{death_count}]: {ev.get('description','')[:50]}")

            time.sleep(DELAY)

print(f"\nKills: {kill_count}, Deaths/Respawns: {death_count}")

# %% [markdown]
# ## Save master index

# %%
from collections import Counter
cat_counts = Counter(c["category"] for c in all_clips)
print(f"\n=== FINAL SUMMARY ===")
print(f"Total clips: {len(all_clips)}")
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat}: {count}")

index = {
    "total_clips": len(all_clips),
    "categories": dict(cat_counts),
    "data_root": str(HUMAN_SHARDS[0].parent),
    "model": MODEL_NAME,
    "seed": SEED,
    "weapon_mapping": {str(s): {k:v for k,v in d.items() if k != "ref_image"} for s, d in weapon_mapping.items()},
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "clips": all_clips,
}
with open(os.path.join(OUTPUT_DIR, "index.json"), "w") as f:
    json.dump(index, f, indent=2)
print(f"\nIndex saved to {OUTPUT_DIR}/index.json")

# Re-encode clips to H.264 for browser playback
print("\nRe-encoding clips to H.264...")
ffmpeg = "/tmp/ffmpeg"
if os.path.exists(ffmpeg):
    count = 0
    for mp4 in Path(OUTPUT_DIR).rglob("*.mp4"):
        os.system(f'{ffmpeg} -y -i "{mp4}" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart "{mp4}.tmp" 2>/dev/null && mv "{mp4}.tmp" "{mp4}"')
        count += 1
    print(f"  Re-encoded {count} clips")
else:
    print("  ffmpeg not found at /tmp/ffmpeg, skipping re-encode")

print("\nDone!")
