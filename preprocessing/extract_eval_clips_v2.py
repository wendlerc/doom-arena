# %% [markdown]
# # Eval Clip Extraction v2: Visual Weapon Timeline
#
# Approach:
# 1. Scan human gameplay at ~2fps — Gemini identifies weapon per frame → weapon timeline
# 2. Detect visual weapon switches from timeline → extract clips
# 3. Cross-ref ATTACK action + visual weapon → firing / not-firing clips
# 4. Segment scan for kills and deaths (Gemini-detected)
# 5. All clips verified with reference frame comparison

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
HUMAN_SHARDS = sorted(Path("recordings").glob("human-*.tar"))
OUTPUT_DIR = "datasets/eval_clips_v2"
REF_DIR = os.path.join(OUTPUT_DIR, "reference_frames")
MODEL = "gemini-2.5-flash"
DELAY = 2.0
SEED = 42
CLIP_CONTEXT = 35  # 1s each side
ATTACK_INDEX = 11
SCAN_FPS = 2  # frames per second for weapon timeline scan
BATCH_SIZE = 15  # frames per Gemini call for timeline

N_SWITCH_PER_WEAPON = 10
N_ATTACK_FIRING = 30
N_ATTACK_NOT_FIRING = 30
N_KILL = 25
N_DEATH = 25

rng = random.Random(SEED)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# %%
# --- Helpers ---
def _retry_delay(err):
    m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s?['\"]", err)
    return float(m.group(1)) if m else None

def call_gemini(contents, retries=5, base=10.0):
    for att in range(retries + 1):
        try:
            return client.models.generate_content(model=MODEL, contents=contents).text
        except Exception as e:
            err = str(e)
            if att == retries: raise
            if any(c in err for c in ["429","ResourceExhausted","500","503"]):
                d = max(_retry_delay(err) or 0, base * (2**att))
                print(f"      Retry {d:.0f}s...", flush=True); time.sleep(d)
            else: raise

def parse_json(raw):
    text = re.sub(r"```(?:json)?\s*","",raw.strip())
    text = re.sub(r"```\s*$","",text,flags=re.MULTILINE).strip()
    for fn in [lambda:json.loads(text), lambda:json.loads(re.search(r"\{.*\}",text,re.DOTALL).group()),
               lambda:json.loads(re.search(r"\[.*\]",text,re.DOTALL).group())]:
        try: return fn()
        except: pass
    m = re.search(r"\[.*",text,re.DOTALL)
    if m:
        lb = m.group().rfind("}")
        if lb>0:
            try: return json.loads(m.group()[:lb+1]+"]")
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
def tf(ts):
    p=ts.strip().split(":"); s=float(p[0])*60+float(p[1]) if len(p)==2 else float(p[0])
    return int(round(s*GAME_FPS))

# %%
# --- Load reference frames ---
ref_images = {}
for f in os.listdir(REF_DIR):
    if f.endswith(".jpg"):
        name = f.replace(".jpg","").upper()
        ref_images[name] = Image.open(os.path.join(REF_DIR, f))
print(f"Reference weapons: {sorted(ref_images.keys())}")

episodes = scan_episodes(HUMAN_SHARDS)
print(f"Episodes: {len(episodes)}")

for cat in ["weapon_switch","attack_firing","attack_not_firing","kill","death_respawn"]:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

# %% [markdown]
# ## Step 1: Build weapon timeline per episode
# Scan at ~2fps, send batches of frames to Gemini, get weapon label per frame.

# %%
TIMELINE_PROMPT = """\
For each frame, identify the first-person weapon at the bottom of the screen.
Return a JSON array of weapon names, one per frame, in order.

Weapons: FIST, CHAINSAW, PISTOL, SHOTGUN, SUPER_SHOTGUN, CHAINGUN, \
ROCKET_LAUNCHER, PLASMA_RIFLE, BFG, NONE
If the view is obscured (death screen, menu, respawn flash), say NONE.

Return ONLY the JSON array. Example: ["SHOTGUN", "SHOTGUN", "PISTOL", "NONE"]
"""

# weapon_timelines[ep_idx] = [(frame_idx, weapon_name), ...]
weapon_timelines = {}

for ep_idx, ep in enumerate(episodes):
    n_frames = ep["meta"]["n_frames"]
    step = max(1, GAME_FPS // SCAN_FPS)
    scan_indices = list(range(0, n_frames, step))
    ep_id = ep["meta"]["episode_id"][:12]

    print(f"\nEp {ep_idx}: {ep_id}... scanning {len(scan_indices)} frames at {SCAN_FPS}fps")

    timeline = []
    with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
        # Extract ALL scan frames in one pass
        frame_cache = extract_frames(vpath, scan_indices)

        # Send in batches to Gemini
        for batch_start in range(0, len(scan_indices), BATCH_SIZE):
            batch_indices = scan_indices[batch_start:batch_start + BATCH_SIZE]
            batch_frames = [frame_cache[i] for i in batch_indices if i in frame_cache]
            if not batch_frames: continue

            contents = []
            for img in batch_frames:
                contents.append(pil_to_part(img))
            contents.append(TIMELINE_PROMPT)

            try:
                raw = call_gemini(contents)
                weapons = parse_json(raw)
                if not isinstance(weapons, list): weapons = [weapons]
                # Pad if shorter
                while len(weapons) < len(batch_frames): weapons.append("NONE")
            except Exception as e:
                print(f"    Batch {batch_start//BATCH_SIZE}: ERR {e}")
                weapons = ["NONE"] * len(batch_frames)
                time.sleep(DELAY); continue

            for fi, w in zip(batch_indices, weapons):
                w = str(w).upper().replace(" ","_").replace("-","_")
                timeline.append((fi, w))

            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(scan_indices) + BATCH_SIZE - 1) // BATCH_SIZE
            if batch_num % 5 == 0 or batch_num == total_batches:
                print(f"    Batch {batch_num}/{total_batches}")
            time.sleep(DELAY)

    weapon_timelines[ep_idx] = timeline
    # Quick summary
    from collections import Counter
    wc = Counter(w for _,w in timeline)
    print(f"    Weapons: {dict(wc.most_common(5))}")

print(f"\nTimeline built for {len(weapon_timelines)} episodes")

# %% [markdown]
# ## Step 2a: Detect weapon switches from visual timeline

# %%
print("\n=== Detecting weapon switches from visual timeline ===")

# Find transitions where weapon changes (ignoring NONE)
switch_events = []  # (ep_idx, frame_before, frame_after, weapon_from, weapon_to)
for ep_idx, timeline in weapon_timelines.items():
    prev_weapon = None
    prev_frame = None
    for fi, w in timeline:
        if w == "NONE": continue
        if prev_weapon is not None and w != prev_weapon:
            switch_events.append((ep_idx, prev_frame, fi, prev_weapon, w))
        prev_weapon = w
        prev_frame = fi

print(f"Total visual weapon switches: {len(switch_events)}")
by_weapon_to = defaultdict(list)
for ev in switch_events:
    by_weapon_to[ev[4]].append(ev)
for w in sorted(by_weapon_to):
    print(f"  -> {w}: {len(by_weapon_to[w])}")

# Sample
sampled_switches = []
for w in sorted(by_weapon_to):
    pool = by_weapon_to[w]
    n = min(N_SWITCH_PER_WEAPON, len(pool))
    sampled_switches.extend(rng.sample(pool, n))

print(f"Sampled {len(sampled_switches)} switches for extraction")

# %% [markdown]
# ## Step 2b: Extract and verify weapon switch clips

# %%
SWITCH_VERIFY = """\
Reference A ({wf}):
[shown above]

Reference B ({wt}):
[shown above]

These gameplay frames should show a switch from {wf} to {wt}.
Compare the weapon in early vs late frames with the references.

Return ONLY: {{"verified": true/false, "weapon_from_matches": true/false, \
"weapon_to_matches": true/false, "confidence": 0.0-1.0}}
"""

all_clips = []
ws_count = 0

sampled_switches.sort(key=lambda x: x[0])
for ep_idx, group in groupby(sampled_switches, key=lambda x: x[0]):
    items = list(group)
    ep = episodes[ep_idx]
    n_frames = ep["meta"]["n_frames"]

    # Collect all needed frames
    all_indices = set()
    clip_data = []
    for _, f_before, f_after, wf, wt in items:
        center = (f_before + f_after) // 2
        f_start = max(0, center - CLIP_CONTEXT)
        f_end = min(n_frames - 1, center + CLIP_CONTEXT)
        verify = sorted(set(max(0,min(n_frames-1,center+o)) for o in [-15,-8,-3,0,3,8,15]))
        clip_data.append((f_before, f_after, wf, wt, f_start, f_end, verify))
        all_indices.update(range(f_start, f_end+1))
        all_indices.update(verify)

    with open_video(ep["shard"], ep["members"]["video_p1.mp4"]) as vpath:
        fc = extract_frames(vpath, sorted(all_indices))
        actions = load_actions(ep["shard"], ep["members"], "p1")

        for f_before, f_after, wf, wt, f_start, f_end, verify in clip_data:
            vframes = [fc[i] for i in verify if i in fc]
            if len(vframes) < 4: continue

            contents = [f"Reference A ({wf}):"]
            if wf in ref_images: contents.append(pil_to_part(ref_images[wf]))
            contents.append(f"Reference B ({wt}):")
            if wt in ref_images: contents.append(pil_to_part(ref_images[wt]))
            contents.append("Gameplay (chronological):")
            for img in vframes: contents.append(pil_to_part(img))
            contents.append(SWITCH_VERIFY.format(wf=wf, wt=wt))

            try:
                r = parse_json(call_gemini(contents))
                if isinstance(r,list): r=r[0]
            except Exception as e:
                print(f"    ERR: {e}"); time.sleep(DELAY); continue

            ok = r.get("verified",False) and r.get("weapon_from_matches",False) and r.get("weapon_to_matches",False)
            tag = "OK" if ok else "SKIP"
            print(f"  [{ws_count}] {tag} {wf}->{wt} conf={r.get('confidence','?')}")

            if ok:
                cid = f"ws_{wt.lower()}_{ws_count:03d}"
                cdir = os.path.join(OUTPUT_DIR,"weapon_switch")
                save_clip(fc, f_start, f_end, os.path.join(cdir,f"{cid}.mp4"))
                np.save(os.path.join(cdir,f"{cid}_actions.npy"), actions[f_start:f_end+1])
                ann = {"clip_id":cid,"category":"weapon_switch",
                       "source_episode":ep["meta"]["episode_id"],"source_player":"p1",
                       "source_shard":ep["shard"],"frame_start":f_start,"frame_end":f_end,
                       "duration_frames":f_end-f_start+1,"duration_sec":round((f_end-f_start+1)/GAME_FPS,2),
                       "fps":GAME_FPS,"ground_truth":{"weapon_from":wf,"weapon_to":wt,
                       "switch_frame_before":f_before,"switch_frame_after":f_after},
                       "gemini_verified":True,"gemini_response":r}
                with open(os.path.join(cdir,f"{cid}.json"),"w") as f: json.dump(ann,f,indent=2)
                all_clips.append(ann)
                ws_count += 1
            time.sleep(DELAY)

print(f"\nWeapon switches: {ws_count}")

# %% [markdown]
# ## Step 3: Attack firing and not-firing clips
# Use ATTACK action data + visual weapon from timeline.

# %%
print("\n=== Attack firing clips ===")

FIRING_VERIFY = """\
Reference ({wn}):
[shown above]

These frames show the player pressing ATTACK with {wn}.
Is there visible firing? (muzzle flash, projectile, recoil, shotgun pump)
Does the weapon match the reference?

Return ONLY: {{"verified": true/false, "weapon_matches": true/false, \
"firing_visible": true/false, "evidence": "brief", "confidence": 0.0-1.0}}
"""

def detect_bursts(actions, min_len=3):
    a = actions[:,ATTACK_INDEX]==1.0
    if not a.any(): return []
    ch = np.diff(a.astype(int))
    ss = np.where(ch==1)[0]+1; ee = np.where(ch==-1)[0]+1
    if a[0]: ss=np.concatenate([[0],ss])
    if a[-1]: ee=np.concatenate([ee,[len(a)]])
    mg=[(int(ss[0]),int(ee[0]))]
    for s,e in zip(ss[1:],ee[1:]):
        if s-mg[-1][1]<5: mg[-1]=(mg[-1][0],int(e))
        else: mg.append((int(s),int(e)))
    return [(s,e) for s,e in mg if e-s>=min_len]

def get_visual_weapon(ep_idx, frame):
    """Look up weapon from timeline for given frame."""
    tl = weapon_timelines.get(ep_idx, [])
    if not tl: return "UNKNOWN"
    # Find closest timeline entry
    best = "UNKNOWN"
    best_dist = float('inf')
    for fi, w in tl:
        if w == "NONE": continue
        d = abs(fi - frame)
        if d < best_dist: best_dist = d; best = w
    return best

af_candidates = []
for ep_idx, ep in enumerate(episodes):
    actions = load_actions(ep["shard"], ep["members"], "p1")
    for s,e in detect_bursts(actions):
        mid = (s+e)//2
        w = get_visual_weapon(ep_idx, mid)
        if w not in ("UNKNOWN","NONE","FIST"):
            af_candidates.append((ep_idx, mid, w, s, e))

sampled_af = rng.sample(af_candidates, min(N_ATTACK_FIRING*2, len(af_candidates)))
af_count = 0

sampled_af.sort(key=lambda x: x[0])
for ep_idx, group in groupby(sampled_af, key=lambda x: x[0]):
    if af_count >= N_ATTACK_FIRING: break
    items = list(group)
    ep = episodes[ep_idx]; n_frames = ep["meta"]["n_frames"]

    all_idx = set()
    cdata = []
    for _,mid,w,bs,be in items:
        fs=max(0,mid-CLIP_CONTEXT); fe=min(n_frames-1,mid+CLIP_CONTEXT)
        v=sorted(set(max(0,min(n_frames-1,mid+o)) for o in [-3,-1,0,1,3]))
        cdata.append((mid,w,fs,fe,v)); all_idx.update(range(fs,fe+1)); all_idx.update(v)

    with open_video(ep["shard"],ep["members"]["video_p1.mp4"]) as vp:
        fc=extract_frames(vp,sorted(all_idx))
        actions=load_actions(ep["shard"],ep["members"],"p1")
        for mid,w,fs,fe,v in cdata:
            if af_count>=N_ATTACK_FIRING: break
            vf=[fc[i] for i in v if i in fc]
            if len(vf)<3: continue
            contents=[f"Reference ({w}):"]
            if w in ref_images: contents.append(pil_to_part(ref_images[w]))
            contents.append("Gameplay during ATTACK:")
            for img in vf: contents.append(pil_to_part(img))
            contents.append(FIRING_VERIFY.format(wn=w))
            try:
                r=parse_json(call_gemini(contents));
                if isinstance(r,list):r=r[0]
            except Exception as e: print(f"    ERR:{e}"); time.sleep(DELAY); continue
            ok=r.get("verified",False) and r.get("firing_visible",False)
            tag="OK" if ok else "SKIP"
            print(f"  [{af_count}] {tag} {w} — {r.get('evidence','')[:40]}")
            if ok:
                cid=f"af_{w.lower()}_{af_count:03d}"
                cdir=os.path.join(OUTPUT_DIR,"attack_firing")
                save_clip(fc,fs,fe,os.path.join(cdir,f"{cid}.mp4"))
                np.save(os.path.join(cdir,f"{cid}_actions.npy"),actions[fs:fe+1])
                ann={"clip_id":cid,"category":"attack_firing","source_episode":ep["meta"]["episode_id"],
                     "source_player":"p1","source_shard":ep["shard"],"frame_start":fs,"frame_end":fe,
                     "duration_frames":fe-fs+1,"duration_sec":round((fe-fs+1)/GAME_FPS,2),"fps":GAME_FPS,
                     "ground_truth":{"weapon_type":w},"gemini_verified":True,"gemini_response":r}
                with open(os.path.join(cdir,f"{cid}.json"),"w") as f: json.dump(ann,f,indent=2)
                all_clips.append(ann); af_count+=1
            time.sleep(DELAY)

print(f"\nAttack firing: {af_count}")

# %%
print("\n=== Attack not-firing clips ===")

IDLE_VERIFY = """\
Reference ({wn}):
[shown above]

These frames should show the player holding {wn} without firing (idle).
Does the weapon match? Is it idle (no muzzle flash, no projectile)?

Return ONLY: {{"verified": true/false, "weapon_matches": true/false, \
"is_idle": true/false, "confidence": 0.0-1.0}}
"""

anf_candidates = []
for ep_idx, ep in enumerate(episodes):
    actions = load_actions(ep["shard"], ep["members"], "p1")
    step = max(1, GAME_FPS // SCAN_FPS)
    for fi in range(0, len(actions), step * 5):
        if actions[fi, ATTACK_INDEX] == 0:
            w = get_visual_weapon(ep_idx, fi)
            if w not in ("UNKNOWN","NONE","FIST"):
                anf_candidates.append((ep_idx, fi, w))

sampled_anf = rng.sample(anf_candidates, min(N_ATTACK_NOT_FIRING*2, len(anf_candidates)))
anf_count = 0

sampled_anf.sort(key=lambda x: x[0])
for ep_idx, group in groupby(sampled_anf, key=lambda x: x[0]):
    if anf_count >= N_ATTACK_NOT_FIRING: break
    items = list(group)
    ep = episodes[ep_idx]; n_frames = ep["meta"]["n_frames"]
    all_idx = set(); cdata = []
    for _,fi,w in items:
        fs=max(0,fi-CLIP_CONTEXT);fe=min(n_frames-1,fi+CLIP_CONTEXT)
        v=sorted(set(max(0,min(n_frames-1,fi+o)) for o in [-2,-1,0,1,2]))
        cdata.append((fi,w,fs,fe,v)); all_idx.update(range(fs,fe+1)); all_idx.update(v)

    with open_video(ep["shard"],ep["members"]["video_p1.mp4"]) as vp:
        fc=extract_frames(vp,sorted(all_idx))
        actions=load_actions(ep["shard"],ep["members"],"p1")
        for fi,w,fs,fe,v in cdata:
            if anf_count>=N_ATTACK_NOT_FIRING: break
            vf=[fc[i] for i in v if i in fc]
            if len(vf)<3: continue
            contents=[f"Reference ({w}):"]
            if w in ref_images: contents.append(pil_to_part(ref_images[w]))
            contents.append("Gameplay (should be idle):")
            for img in vf: contents.append(pil_to_part(img))
            contents.append(IDLE_VERIFY.format(wn=w))
            try:
                r=parse_json(call_gemini(contents));
                if isinstance(r,list):r=r[0]
            except: time.sleep(DELAY); continue
            ok=r.get("verified",False) and r.get("is_idle",False)
            if ok:
                cid=f"anf_{w.lower()}_{anf_count:03d}"
                cdir=os.path.join(OUTPUT_DIR,"attack_not_firing")
                save_clip(fc,fs,fe,os.path.join(cdir,f"{cid}.mp4"))
                np.save(os.path.join(cdir,f"{cid}_actions.npy"),actions[fs:fe+1])
                ann={"clip_id":cid,"category":"attack_not_firing","source_episode":ep["meta"]["episode_id"],
                     "source_player":"p1","source_shard":ep["shard"],"frame_start":fs,"frame_end":fe,
                     "duration_frames":fe-fs+1,"duration_sec":round((fe-fs+1)/GAME_FPS,2),"fps":GAME_FPS,
                     "ground_truth":{"weapon_type":w},"gemini_verified":True,"gemini_response":r}
                with open(os.path.join(cdir,f"{cid}.json"),"w") as f: json.dump(ann,f,indent=2)
                all_clips.append(ann); anf_count+=1
                print(f"  [{anf_count}] OK {w}")
            time.sleep(DELAY)

print(f"\nAttack not-firing: {anf_count}")

# %% [markdown]
# ## Step 4: Kill and death/respawn (segment scan)

# %%
print("\n=== Kill / death / respawn ===")

KILL_DEATH_PROMPT = """\
These frames (~2fps, timestamps shown) are from a Doom deathmatch.
Look for:
1. **Kill**: enemy falls/gibs, frag message in top-left corner
2. **Death**: screen flashes bright RED, view drops
3. **Respawn**: bright GREEN circular flame/teleport, new location, pistol in hand

Return JSON array of events. Each: {{"timestamp":"MM:SS.s","event_type":"kill"/"death"/"respawn","description":"...","details":{{}},"confidence":0.0-1.0}}
If none: []
Return ONLY the JSON array.
"""

kill_count = death_count = 0
SEG_LEN = 15

for ep_idx, ep in enumerate(episodes):
    if kill_count>=N_KILL and death_count>=N_DEATH: break
    meta=ep["meta"]; n_frames=meta["n_frames"]; ep_id=meta["episode_id"][:12]
    n_seg = max(1, n_frames//(SEG_LEN*GAME_FPS))
    print(f"\n  {ep_id}...: {n_seg} segments, frags={meta.get('frag_p1')}, deaths={meta.get('death_p1')}")

    with open_video(ep["shard"],ep["members"]["video_p1.mp4"]) as vp:
        actions=load_actions(ep["shard"],ep["members"],"p1")
        for si in range(n_seg):
            if kill_count>=N_KILL and death_count>=N_DEATH: break
            ss=si*SEG_LEN*GAME_FPS; se=min((si+1)*SEG_LEN*GAME_FPS,n_frames-1)
            step=max(1,GAME_FPS//SCAN_FPS)
            indices=list(range(int(ss),int(se),step))
            if not indices: continue
            fc=extract_frames(vp,indices)
            frames=[fc[i] for i in indices if i in fc]
            if len(frames)<5: continue

            contents=[]
            for img,i in zip(frames,indices):
                contents.append(f"[{ft(i)}]"); contents.append(pil_to_part(img))
            contents.append(KILL_DEATH_PROMPT)

            try: evts=parse_json(call_gemini(contents))
            except: time.sleep(DELAY); continue
            if not isinstance(evts,list): evts=[evts]

            for ev in evts:
                et=ev.get("event_type",""); ts=ev.get("timestamp","")
                if not ts: continue
                try: ef=tf(ts)
                except: continue
                ef=max(0,min(n_frames-1,ef))
                fs=max(0,ef-CLIP_CONTEXT); fe=min(n_frames-1,ef+CLIP_CONTEXT)
                cf=extract_frames(vp,list(range(fs,fe+1)))

                if et=="kill" and kill_count<N_KILL:
                    cid=f"kill_{kill_count:03d}"; cdir=os.path.join(OUTPUT_DIR,"kill")
                    save_clip(cf,fs,fe,os.path.join(cdir,f"{cid}.mp4"))
                    np.save(os.path.join(cdir,f"{cid}_actions.npy"),actions[fs:fe+1])
                    ann={"clip_id":cid,"category":"kill","source_episode":meta["episode_id"],
                         "source_player":"p1","source_shard":ep["shard"],"frame_start":fs,"frame_end":fe,
                         "duration_frames":fe-fs+1,"duration_sec":round((fe-fs+1)/GAME_FPS,2),"fps":GAME_FPS,
                         "ground_truth":{"description":ev.get("description",""),"weapon_used":ev.get("details",{}).get("weapon_used","")},"gemini_verified":True,"gemini_response":ev}
                    with open(os.path.join(cdir,f"{cid}.json"),"w") as f: json.dump(ann,f,indent=2)
                    all_clips.append(ann); kill_count+=1
                    print(f"    KILL [{kill_count}]: {ev.get('description','')[:50]}")
                elif et in ("death","respawn") and death_count<N_DEATH:
                    cid=f"dr_{death_count:03d}"; cdir=os.path.join(OUTPUT_DIR,"death_respawn")
                    save_clip(cf,fs,fe,os.path.join(cdir,f"{cid}.mp4"))
                    np.save(os.path.join(cdir,f"{cid}_actions.npy"),actions[fs:fe+1])
                    ann={"clip_id":cid,"category":"death_respawn","source_episode":meta["episode_id"],
                         "source_player":"p1","source_shard":ep["shard"],"frame_start":fs,"frame_end":fe,
                         "duration_frames":fe-fs+1,"duration_sec":round((fe-fs+1)/GAME_FPS,2),"fps":GAME_FPS,
                         "ground_truth":{"event_type":et,"description":ev.get("description",""),"cause":ev.get("details",{}).get("cause","")},"gemini_verified":True,"gemini_response":ev}
                    with open(os.path.join(cdir,f"{cid}.json"),"w") as f: json.dump(ann,f,indent=2)
                    all_clips.append(ann); death_count+=1
                    print(f"    DEATH/RESPAWN [{death_count}]: {ev.get('description','')[:50]}")
            time.sleep(DELAY)

print(f"\nKills: {kill_count}, Deaths: {death_count}")

# %% [markdown]
# ## Save index + re-encode

# %%
from collections import Counter
cc=Counter(c["category"] for c in all_clips)
print(f"\n=== FINAL ===\nTotal: {len(all_clips)}")
for c,n in sorted(cc.items()): print(f"  {c}: {n}")

index={"total_clips":len(all_clips),"categories":dict(cc),"model":MODEL,"seed":SEED,
       "weapon_references":sorted(ref_images.keys()),"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
       "clips":all_clips}
with open(os.path.join(OUTPUT_DIR,"index.json"),"w") as f: json.dump(index,f,indent=2)

# Re-encode
ffmpeg="/tmp/ffmpeg"
if os.path.exists(ffmpeg):
    n=0
    for mp4 in Path(OUTPUT_DIR).rglob("*.mp4"):
        os.system(f'{ffmpeg} -y -i "{mp4}" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart "{mp4}.tmp" 2>/dev/null && mv "{mp4}.tmp" "{mp4}"')
        n+=1
    print(f"Re-encoded {n} clips")

print("Done!")
