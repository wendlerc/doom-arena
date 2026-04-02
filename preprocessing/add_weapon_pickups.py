# %% [markdown]
# # Add weapon pickup clips to eval_clips_v2
# Uses the weapon timeline (already built by extract_eval_clips_v2.py) to detect
# moments where a NEW weapon first appears — indicating a weapon pickup.
#
# Detection: scan the visual weapon timeline per episode. After a respawn
# (FIST/PISTOL), track which weapons appear. Each first appearance of a new
# weapon is a pickup event. Extract clips around those transitions.

# %%
import sys, os, io, json, re, time, random, tarfile, tempfile
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from itertools import groupby

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

import numpy as np, cv2
from PIL import Image
from google import genai
from google.genai import types

# %%
GAME_FPS = 35
OUTPUT_DIR = "datasets/eval_clips_v2"
REF_DIR = os.path.join(OUTPUT_DIR, "reference_frames")
MODEL = "gemini-2.5-flash"
DELAY = 2.0
SEED = 42
CLIP_CONTEXT = 35
SCAN_FPS = 2
BATCH_SIZE = 15
N_PICKUP_PER_WEAPON = 5

HUMAN_SHARDS = sorted(Path("recordings").glob("human-*.tar"))
rng = random.Random(SEED)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# %%
# --- Reuse helpers from v2 script ---
def call_gemini(contents, retries=5, base=10.0):
    for att in range(retries + 1):
        try: return client.models.generate_content(model=MODEL, contents=contents).text
        except Exception as e:
            err = str(e)
            if att == retries: raise
            if any(c in err for c in ["429","ResourceExhausted","500","503"]):
                time.sleep(base * (2**att))
            else: raise

def parse_json(raw):
    text = re.sub(r"```(?:json)?\s*","",raw.strip())
    text = re.sub(r"```\s*$","",text,flags=re.MULTILINE).strip()
    for fn in [lambda:json.loads(text), lambda:json.loads(re.search(r"\{.*\}",text,re.DOTALL).group()),
               lambda:json.loads(re.search(r"\[.*\]",text,re.DOTALL).group())]:
        try: return fn()
        except: pass
    raise ValueError(f"Parse fail: {text[:200]}")

def pil_to_part(img):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    return types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buf.getvalue()))

def scan_episodes(shards):
    eps = []
    for sp in shards:
        try:
            with tarfile.open(sp,"r") as tar:
                groups = {}
                for m in tar.getmembers():
                    if m.isdir(): continue
                    p = m.name.split(".",1)
                    if len(p)==2: groups.setdefault(p[0],{})[p[1]] = m.name
                for k,mem in groups.items():
                    if "meta.json" not in mem: continue
                    meta = json.loads(tar.extractfile(tar.getmember(mem["meta.json"])).read())
                    eps.append({"shard":str(sp),"key":k,"members":mem,"meta":meta})
        except: pass
    return eps

def load_actions(shard, members, player):
    with tarfile.open(shard,"r") as tar:
        return np.load(io.BytesIO(tar.extractfile(tar.getmember(members[f"actions_{player}.npy"])).read()))

@contextmanager
def open_video(shard, member):
    with tarfile.open(shard,"r") as tar: data = tar.extractfile(tar.getmember(member)).read()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(data); tmp.close()
    try: yield tmp.name
    finally: os.unlink(tmp.name)

def extract_frames(vpath, indices):
    cap = cv2.VideoCapture(vpath)
    fset, mx = set(indices), max(indices)
    out = {}; idx = 0
    while cap.isOpened() and idx <= mx:
        ret, f = cap.read()
        if not ret: break
        if idx in fset: out[idx] = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release(); return out

def save_clip(frames_dict, f_start, f_end, path):
    indices = sorted(k for k in frames_dict if f_start <= k <= f_end)
    if not indices: return False
    first = frames_dict[indices[0]]; w,h = first.size
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), GAME_FPS, (w,h))
    for i in indices:
        if i in frames_dict: wr.write(cv2.cvtColor(np.array(frames_dict[i]), cv2.COLOR_RGB2BGR))
    wr.release(); return True

def ft(f): t=f/GAME_FPS; return f"{int(t//60):02d}:{t%60:04.1f}"

# %%
# Load reference images
ref_images = {}
for f in os.listdir(REF_DIR):
    if f.endswith(".jpg"):
        ref_images[f.replace(".jpg","").upper()] = Image.open(os.path.join(REF_DIR, f))
print(f"References: {sorted(ref_images.keys())}")

episodes = scan_episodes(HUMAN_SHARDS)
print(f"Episodes: {len(episodes)}")

os.makedirs(os.path.join(OUTPUT_DIR, "weapon_pickup"), exist_ok=True)

# %%
# --- Step 1: Build weapon timeline (same as v2 main script) ---
TIMELINE_PROMPT = """\
For each frame, identify the first-person weapon at the bottom.
Weapons: FIST, CHAINSAW, PISTOL, SHOTGUN, SUPER_SHOTGUN, CHAINGUN, \
ROCKET_LAUNCHER, PLASMA_RIFLE, BFG, NONE
Return ONLY a JSON array of weapon names.
"""

weapon_timelines = {}
for ep_idx, ep in enumerate(episodes):
    n_frames = ep["meta"]["n_frames"]
    step = max(1, GAME_FPS // SCAN_FPS)
    scan_indices = list(range(0, n_frames, step))
    ep_id = ep["meta"]["episode_id"][:12]
    print(f"\nEp {ep_idx}: {ep_id}... scanning {len(scan_indices)} frames")

    timeline = []
    with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
        frame_cache = extract_frames(vpath, scan_indices)
        for batch_start in range(0, len(scan_indices), BATCH_SIZE):
            batch_idx = scan_indices[batch_start:batch_start + BATCH_SIZE]
            batch_frames = [frame_cache[i] for i in batch_idx if i in frame_cache]
            if not batch_frames: continue
            contents = [pil_to_part(img) for img in batch_frames] + [TIMELINE_PROMPT]
            try:
                weapons = parse_json(call_gemini(contents))
                if not isinstance(weapons, list): weapons = [weapons]
                while len(weapons) < len(batch_frames): weapons.append("NONE")
            except:
                weapons = ["NONE"] * len(batch_frames)
            for fi, w in zip(batch_idx, weapons):
                timeline.append((fi, str(w).upper().replace(" ","_")))
            bn = batch_start//BATCH_SIZE + 1
            tb = (len(scan_indices)+BATCH_SIZE-1)//BATCH_SIZE
            if bn % 10 == 0 or bn == tb: print(f"    Batch {bn}/{tb}")
            time.sleep(DELAY)
    weapon_timelines[ep_idx] = timeline

# %%
# --- Step 2: Detect weapon pickups ---
# A pickup is when a weapon appears for the first time (or first time after respawn).
# After FIST/PISTOL (respawn weapons), any new weapon = pickup.

RESPAWN_WEAPONS = {"FIST", "PISTOL", "NONE"}

pickup_events = []  # (ep_idx, frame_before, frame_after, weapon_picked_up)

for ep_idx, timeline in weapon_timelines.items():
    seen_weapons = set()
    prev_weapon = None
    prev_frame = None

    for fi, w in timeline:
        if w in RESPAWN_WEAPONS or w == "NONE":
            if w in ("FIST", "PISTOL"):
                # Possible respawn — reset seen weapons
                if prev_weapon and prev_weapon not in RESPAWN_WEAPONS:
                    seen_weapons = {"FIST", "PISTOL"}
            prev_weapon = w; prev_frame = fi
            continue

        if w not in seen_weapons and prev_frame is not None:
            pickup_events.append((ep_idx, prev_frame, fi, w))
            print(f"  Ep {ep_idx}: pickup {w} at frame {fi} ({ft(fi)})")

        seen_weapons.add(w)
        prev_weapon = w; prev_frame = fi

print(f"\nTotal pickup events: {len(pickup_events)}")
by_weapon = defaultdict(list)
for ev in pickup_events:
    by_weapon[ev[3]].append(ev)
for w in sorted(by_weapon):
    print(f"  {w}: {len(by_weapon[w])}")

# %%
# --- Step 3: Sample and verify pickup clips ---

PICKUP_VERIFY = """\
Reference — {weapon}:
[shown above]

These gameplay frames should show the player picking up a {weapon}. \
A weapon pickup looks like: a weapon sprite on the ground, the player walks over it, \
and the weapon appears in their hands (matching the reference).

Look for:
1. Is the weapon visible on the ground in early frames?
2. Does the player's held weapon change TO {weapon} (matching the reference)?

Return ONLY: {{"verified": true/false, "pickup_visible": true/false, \
"weapon_matches_ref": true/false, "confidence": 0.0-1.0}}
"""

sampled_pickups = []
for w in sorted(by_weapon):
    pool = by_weapon[w]
    n = min(N_PICKUP_PER_WEAPON, len(pool))
    sampled_pickups.extend(rng.sample(pool, n))
    print(f"  {w}: {n}/{len(pool)} sampled")

pickup_count = 0
new_clips = []

sampled_pickups.sort(key=lambda x: x[0])
for ep_idx, group in groupby(sampled_pickups, key=lambda x: x[0]):
    items = list(group)
    ep = episodes[ep_idx]
    n_frames = ep["meta"]["n_frames"]

    all_indices = set()
    clip_data = []
    for _, f_before, f_after, weapon in items:
        center = (f_before + f_after) // 2
        f_start = max(0, center - CLIP_CONTEXT)
        f_end = min(n_frames - 1, center + CLIP_CONTEXT)
        verify = sorted(set(max(0,min(n_frames-1,center+o)) for o in [-15,-8,-3,0,3,8,15]))
        clip_data.append((f_before, f_after, weapon, f_start, f_end, verify))
        all_indices.update(range(f_start, f_end+1))
        all_indices.update(verify)

    with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
        fc = extract_frames(vpath, sorted(all_indices))
        actions = load_actions(ep["shard"], ep["members"], "p1")

        for f_before, f_after, weapon, f_start, f_end, verify in clip_data:
            vframes = [fc[i] for i in verify if i in fc]
            if len(vframes) < 4: continue

            contents = [f"Reference ({weapon}):"]
            if weapon in ref_images: contents.append(pil_to_part(ref_images[weapon]))
            contents.append("Gameplay (chronological):")
            for img in vframes: contents.append(pil_to_part(img))
            contents.append(PICKUP_VERIFY.format(weapon=weapon))

            try:
                r = parse_json(call_gemini(contents))
                if isinstance(r, list): r = r[0]
            except Exception as e:
                print(f"    ERR: {e}"); time.sleep(DELAY); continue

            ok = r.get("verified", False) and r.get("weapon_matches_ref", False)
            tag = "OK" if ok else "SKIP"
            print(f"  [{pickup_count}] {tag} {weapon} conf={r.get('confidence','?')}")

            if ok:
                cid = f"wp_{weapon.lower()}_{pickup_count:03d}"
                cdir = os.path.join(OUTPUT_DIR, "weapon_pickup")
                save_clip(fc, f_start, f_end, os.path.join(cdir, f"{cid}.mp4"))
                np.save(os.path.join(cdir, f"{cid}_actions.npy"), actions[f_start:f_end+1])
                ann = {
                    "clip_id": cid, "category": "weapon_pickup",
                    "source_episode": ep["meta"]["episode_id"], "source_player": "p1",
                    "source_shard": ep["shard"],
                    "frame_start": f_start, "frame_end": f_end,
                    "duration_frames": f_end - f_start + 1,
                    "duration_sec": round((f_end - f_start + 1) / GAME_FPS, 2),
                    "fps": GAME_FPS,
                    "ground_truth": {"weapon_picked_up": weapon,
                                     "pickup_frame_before": f_before, "pickup_frame_after": f_after},
                    "gemini_verified": True, "gemini_response": r,
                }
                with open(os.path.join(cdir, f"{cid}.json"), "w") as f:
                    json.dump(ann, f, indent=2)
                new_clips.append(ann)
                pickup_count += 1
            time.sleep(DELAY)

print(f"\nWeapon pickup clips: {pickup_count}")

# %%
# --- Update master index ---
with open(os.path.join(OUTPUT_DIR, "index.json")) as f:
    index = json.load(f)

index["clips"].extend(new_clips)
index["total_clips"] = len(index["clips"])
from collections import Counter
index["categories"] = dict(Counter(c["category"] for c in index["clips"]))

with open(os.path.join(OUTPUT_DIR, "index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"\nUpdated index: {index['total_clips']} total clips")
for cat, n in sorted(index["categories"].items()):
    print(f"  {cat}: {n}")

# Re-encode new clips
ffmpeg = "/tmp/ffmpeg"
if os.path.exists(ffmpeg):
    for mp4 in Path(os.path.join(OUTPUT_DIR, "weapon_pickup")).glob("*.mp4"):
        os.system(f'{ffmpeg} -y -i "{mp4}" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart "{mp4}.tmp" 2>/dev/null && mv "{mp4}.tmp" "{mp4}"')
    print("Re-encoded pickup clips")

print("Done!")
