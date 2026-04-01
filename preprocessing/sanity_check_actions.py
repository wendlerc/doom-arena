# %% [markdown]
# # Sanity Check: Action Data vs Video Content
# Verify that recorded actions (weapon switches, attacks) match what's
# visually happening in the video by asking Gemini to confirm.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# %%
# --- Constants ---
GAME_FPS = 35
WEAPON_SELECT_INDICES = list(range(4, 11))
ATTACK_INDEX = 11
WEAPON_NAMES = {
    1: "fist_chainsaw", 2: "pistol", 3: "shotgun",
    4: "super_shotgun", 5: "chaingun", 6: "rocket_launcher", 7: "plasma_rifle",
}
BUTTON_NAMES = [
    "FWD", "BACK", "RIGHT", "LEFT",
    "W1", "W2", "W3", "W4", "W5", "W6", "W7",
    "ATTACK", "SPEED", "TURN",
]

DATA_ROOT = "datasets/pvp_recordings"
MAX_SHARDS = 3
N_SWITCH_PER_WEAPON = 20
N_ATTACK_SAMPLES = 100
MODEL_NAME = "gemini-2.5-flash"
DELAY = 2.0
SEED = 42
OUTPUT_DIR = "preprocessing"

# %% [markdown]
# ## Helpers

# %%
def _extract_retry_delay(err_str):
    m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s?['\"]", err_str)
    return float(m.group(1)) if m else None


def call_gemini(client, model_name, contents, max_retries=5, base_delay=10.0):
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            return response.text
        except Exception as e:
            err = str(e)
            if attempt == max_retries:
                raise
            if any(code in err for code in ["429", "ResourceExhausted", "500", "503"]):
                server_delay = _extract_retry_delay(err)
                delay = max(server_delay or 0, base_delay * (2 ** attempt))
                print(f"    Retrying in {delay:.0f}s...", flush=True)
                time.sleep(delay)
            else:
                raise


def parse_gemini_json(raw_text):
    text = raw_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON: {text[:200]}...")


def pil_to_part(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buf.getvalue()))


# %%
# --- Episode scanning and data loading ---

def scan_episodes(data_root, shard_pattern="*.tar", max_shards=None):
    shards = sorted(Path(data_root).glob(shard_pattern))
    if max_shards:
        shards = shards[:max_shards]
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
                    episodes.append({
                        "shard": str(shard_path), "key": key,
                        "members": members, "meta": meta,
                    })
        except (tarfile.TarError, OSError) as e:
            print(f"  Skipping {shard_path}: {e}")
    return episodes


def load_actions(shard_path, members, player):
    """Load action array for a player. Returns (N, 14) float32."""
    key = f"actions_{player}.npy"
    with tarfile.open(shard_path, "r") as tar:
        return np.load(io.BytesIO(tar.extractfile(tar.getmember(members[key])).read()))


@contextmanager
def open_video(shard_path, member_name):
    """Extract video to temp file, yield path. Cleaned up on exit."""
    with tarfile.open(shard_path, "r") as tar:
        video_data = tar.extractfile(tar.getmember(member_name)).read()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_data)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.unlink(tmp.name)


def extract_frames(video_path, frame_indices):
    """Extract specific frames from a video file. Returns dict {frame_idx: PIL.Image}."""
    cap = cv2.VideoCapture(video_path)
    frame_set = set(frame_indices)
    max_frame = max(frame_indices)
    result = {}
    idx = 0
    while cap.isOpened() and idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_set:
            result[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return result

# %% [markdown]
# ## Event Detection

# %%
def detect_weapon_changes(actions):
    """Detect frames where the active weapon changes.

    Returns list of {"frame": int, "weapon_from": int, "weapon_to": int}.
    """
    prev_weapon = None
    events = []
    for i in range(actions.shape[0]):
        active = np.where(actions[i, 4:11] == 1.0)[0]
        if len(active) > 0:
            w = int(active[0]) + 1  # 1-indexed
            if prev_weapon is not None and w != prev_weapon:
                events.append({"frame": i, "weapon_from": prev_weapon, "weapon_to": w})
            prev_weapon = w
    return events


def detect_attack_bursts(actions, min_gap=5, min_length=2):
    """Detect contiguous ATTACK=1 regions. Returns list of (start, end) tuples."""
    attack = actions[:, ATTACK_INDEX] == 1.0
    if not attack.any():
        return []
    changes = np.diff(attack.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if attack[0]:
        starts = np.concatenate([[0], starts])
    if attack[-1]:
        ends = np.concatenate([ends, [len(attack)]])
    merged = [(int(starts[0]), int(ends[0]))]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], int(e))
        else:
            merged.append((int(s), int(e)))
    return [(s, e) for s, e in merged if e - s >= min_length]


# %% [markdown]
# ## Gemini Prompts

# %%
WEAPON_SWITCH_PROMPT = """\
These 5 frames (in chronological order) are from a first-person Doom deathmatch at 35fps.
According to the recorded action data, the player switched from {weapon_from} to {weapon_to} at the middle frame.

Look at the weapon visible in the player's hands across these frames.

Return ONLY a JSON object:
{{"switch_occurred": true or false, "weapon_to": "weapon name or null", "confidence": 0.0 to 1.0}}
"""

ATTACK_PROMPT = """\
These 5 frames (in chronological order) are from a first-person Doom deathmatch at 35fps.
According to the recorded action data, the player is pressing ATTACK at the middle frame.

Look at the weapon in the player's hands for muzzle flash, recoil, or projectiles.

Return ONLY a JSON object:
{{"holding_weapon": true or false, "weapon_type": "weapon name or null", "weapon_firing": true or false, "confidence": 0.0 to 1.0}}
"""

# %% [markdown]
# ## Load episodes and sample events

# %%
print(f"Scanning {DATA_ROOT} (max {MAX_SHARDS} shards)...")
episodes = scan_episodes(DATA_ROOT, max_shards=MAX_SHARDS)
print(f"Found {len(episodes)} episodes")

# Filter to non-random-policy players
rng = random.Random(SEED)

# Collect weapon switch and attack events across episodes
all_switch_events = []  # (ep_idx, player, event_dict)
all_attack_events = []  # (ep_idx, player, frame)

for ep_idx, ep in enumerate(episodes):
    meta = ep["meta"]
    for player in ["p1", "p2"]:
        if meta.get(f"random_policy_{player}", False):
            print(f"  Skipping {meta['episode_id'][:12]}... {player} (random policy)")
            continue

        actions = load_actions(ep["shard"], ep["members"], player)

        # Weapon changes
        changes = detect_weapon_changes(actions)
        for ev in changes:
            all_switch_events.append((ep_idx, player, ev))

        # Attack bursts
        bursts = detect_attack_bursts(actions)
        for s, e in bursts:
            mid = (s + e) // 2
            all_attack_events.append((ep_idx, player, mid))

        print(f"  {meta['episode_id'][:12]}... {player}: {len(changes)} weapon changes, {len(bursts)} attack bursts")

print(f"\nTotal: {len(all_switch_events)} weapon changes, {len(all_attack_events)} attack bursts")

# %%
# Sample weapon switches: up to N_SWITCH_PER_WEAPON per weapon_to
by_weapon = defaultdict(list)
for item in all_switch_events:
    w = item[2]["weapon_to"]
    by_weapon[w].append(item)

sampled_switches = []
for w in sorted(by_weapon.keys()):
    pool = by_weapon[w]
    n = min(N_SWITCH_PER_WEAPON, len(pool))
    sampled_switches.extend(rng.sample(pool, n))
    print(f"  W{w} ({WEAPON_NAMES[w]}): sampled {n}/{len(pool)}")

# Sample attack events
sampled_attacks = rng.sample(all_attack_events, min(N_ATTACK_SAMPLES, len(all_attack_events)))
print(f"\nSampled {len(sampled_switches)} weapon switches, {len(sampled_attacks)} attack events")

# %% [markdown]
# ## Run Gemini Verification

# %%
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# %% [markdown]
# ### Check 1: Weapon Switches

# %%
switch_results = []

# Group by (ep_idx, player) to reuse extracted video
from itertools import groupby

sampled_switches.sort(key=lambda x: (x[0], x[1]))
for (ep_idx, player), group in groupby(sampled_switches, key=lambda x: (x[0], x[1])):
    items = list(group)
    ep = episodes[ep_idx]
    ep_id = ep["meta"]["episode_id"][:12]
    video_key = f"video_{player}.mp4"
    n_frames = ep["meta"]["n_frames"]

    print(f"\n  {ep_id}... {player}: {len(items)} switch checks")

    # Collect all needed frames
    all_indices = set()
    clips = []
    for _, _, ev in items:
        offsets = [-5, -2, 0, 3, 7]
        indices = [max(0, min(n_frames - 1, ev["frame"] + o)) for o in offsets]
        indices = sorted(set(indices))
        clips.append((ev, indices))
        all_indices.update(indices)

    # Extract all frames in one pass
    with open_video(ep["shard"], ep["members"][video_key]) as vpath:
        frame_cache = extract_frames(vpath, sorted(all_indices))

    for i, (ev, indices) in enumerate(clips):
        frames = [frame_cache[idx] for idx in indices if idx in frame_cache]
        if not frames:
            continue

        prompt = WEAPON_SWITCH_PROMPT.format(
            weapon_from=WEAPON_NAMES[ev["weapon_from"]],
            weapon_to=WEAPON_NAMES[ev["weapon_to"]],
        )
        contents = [pil_to_part(f) for f in frames]
        contents.append(prompt)

        try:
            raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
            result = parse_gemini_json(raw)
            if isinstance(result, list):
                result = result[0]
        except Exception as e:
            result = {"error": str(e)[:100]}

        result["expected_from"] = WEAPON_NAMES[ev["weapon_from"]]
        result["expected_to"] = WEAPON_NAMES[ev["weapon_to"]]
        result["frame"] = ev["frame"]
        result["episode"] = ep["meta"]["episode_id"]
        result["player"] = player
        switch_results.append(result)

        status = "OK" if result.get("switch_occurred") else "NO" if "error" not in result else "ERR"
        wto = result.get("weapon_to", "?")
        print(f"    [{i+1}/{len(clips)}] {status} expected={WEAPON_NAMES[ev['weapon_to']]} gemini={wto}")

        time.sleep(DELAY)

print(f"\nWeapon switch checks complete: {len(switch_results)} results")

# %% [markdown]
# ### Check 2: Attack Verification

# %%
attack_results = []

sampled_attacks.sort(key=lambda x: (x[0], x[1]))
for (ep_idx, player), group in groupby(sampled_attacks, key=lambda x: (x[0], x[1])):
    items = list(group)
    ep = episodes[ep_idx]
    ep_id = ep["meta"]["episode_id"][:12]
    video_key = f"video_{player}.mp4"
    n_frames = ep["meta"]["n_frames"]

    print(f"\n  {ep_id}... {player}: {len(items)} attack checks")

    all_indices = set()
    clips = []
    for _, _, frame in items:
        offsets = [-2, -1, 0, 1, 2]
        indices = [max(0, min(n_frames - 1, frame + o)) for o in offsets]
        indices = sorted(set(indices))
        clips.append((frame, indices))
        all_indices.update(indices)

    with open_video(ep["shard"], ep["members"][video_key]) as vpath:
        frame_cache = extract_frames(vpath, sorted(all_indices))

    for i, (frame, indices) in enumerate(clips):
        frames = [frame_cache[idx] for idx in indices if idx in frame_cache]
        if not frames:
            continue

        contents = [pil_to_part(f) for f in frames]
        contents.append(ATTACK_PROMPT)

        try:
            raw = call_gemini(client, MODEL_NAME, contents, base_delay=DELAY)
            result = parse_gemini_json(raw)
            if isinstance(result, list):
                result = result[0]
        except Exception as e:
            result = {"error": str(e)[:100]}

        result["frame"] = frame
        result["episode"] = ep["meta"]["episode_id"]
        result["player"] = player
        attack_results.append(result)

        if (i + 1) % 20 == 0:
            firing = sum(1 for r in attack_results if r.get("weapon_firing"))
            print(f"    [{i+1}/{len(clips)}] firing_rate_so_far={firing}/{len(attack_results)}")

        time.sleep(DELAY)

print(f"\nAttack checks complete: {len(attack_results)} results")

# %% [markdown]
# ## Save Results

# %%
# Compute summary stats
switch_by_weapon = defaultdict(lambda: {"total": 0, "confirmed": 0})
for r in switch_results:
    if "error" in r:
        continue
    w = r["expected_to"]
    switch_by_weapon[w]["total"] += 1
    if r.get("switch_occurred"):
        switch_by_weapon[w]["confirmed"] += 1

attack_total = sum(1 for r in attack_results if "error" not in r)
attack_holding = sum(1 for r in attack_results if r.get("holding_weapon"))
attack_firing = sum(1 for r in attack_results if r.get("weapon_firing"))

summary = {
    "weapon_switch": {
        "total_checked": len([r for r in switch_results if "error" not in r]),
        "total_confirmed": sum(1 for r in switch_results if r.get("switch_occurred")),
        "per_weapon": {
            w: {"checked": d["total"], "confirmed": d["confirmed"],
                "rate": round(d["confirmed"] / max(1, d["total"]), 3)}
            for w, d in sorted(switch_by_weapon.items())
        },
    },
    "attack": {
        "total_checked": attack_total,
        "holding_weapon": attack_holding,
        "holding_rate": round(attack_holding / max(1, attack_total), 3),
        "weapon_firing": attack_firing,
        "firing_rate": round(attack_firing / max(1, attack_total), 3),
    },
}

results_data = {
    "metadata": {
        "data_root": DATA_ROOT,
        "max_shards": MAX_SHARDS,
        "model": MODEL_NAME,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    },
    "summary": summary,
    "weapon_switch_results": switch_results,
    "attack_results": attack_results,
}

out_path = os.path.join(OUTPUT_DIR, "sanity_check_results.json")
with open(out_path, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"Results saved to {out_path}")

print("\n=== SUMMARY ===")
print(f"Weapon switches: {summary['weapon_switch']['total_confirmed']}/{summary['weapon_switch']['total_checked']} confirmed")
for w, d in sorted(summary["weapon_switch"]["per_weapon"].items()):
    print(f"  {w}: {d['confirmed']}/{d['checked']} ({d['rate']*100:.0f}%)")
print(f"Attack: holding={summary['attack']['holding_rate']*100:.0f}%, firing={summary['attack']['firing_rate']*100:.0f}%")

# %% [markdown]
# ## Plots

# %%
# Plot 1: Weapon switch confirmation rate per weapon type
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#0f0f1e")

weapons = sorted(switch_by_weapon.keys())
rates = [switch_by_weapon[w]["confirmed"] / max(1, switch_by_weapon[w]["total"]) * 100 for w in weapons]
totals = [switch_by_weapon[w]["total"] for w in weapons]
colors = ["#2ecc71" if r >= 70 else "#f39c12" if r >= 40 else "#e74c3c" for r in rates]

bars = ax.bar(range(len(weapons)), rates, color=colors, edgecolor="#333", width=0.6)
ax.set_xticks(range(len(weapons)))
ax.set_xticklabels([f"{w}\n(n={t})" for w, t in zip(weapons, totals)], color="#ccc", fontsize=9)
ax.set_ylabel("Gemini Confirmation Rate (%)", color="#ccc")
ax.set_title("Weapon Switch Verification", color="#eee", fontsize=14)
ax.set_ylim(0, 110)
ax.tick_params(colors="#888")
for spine in ax.spines.values():
    spine.set_color("#333")
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{rate:.0f}%", ha="center", va="bottom", color="#eee", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sanity_check_weapon_switch.png"), dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print("Saved sanity_check_weapon_switch.png")

# %%
# Plot 2: Attack confirmation rate
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#0f0f1e")

categories = ["Holding\nWeapon", "Weapon\nFiring"]
vals = [
    attack_holding / max(1, attack_total) * 100,
    attack_firing / max(1, attack_total) * 100,
]
bar_colors = ["#3498db", "#e67e22"]

bars = ax.bar(categories, vals, color=bar_colors, edgecolor="#333", width=0.5)
ax.set_ylabel("Confirmation Rate (%)", color="#ccc")
ax.set_title(f"Attack Verification (n={attack_total})", color="#eee", fontsize=14)
ax.set_ylim(0, 110)
ax.tick_params(colors="#888")
for spine in ax.spines.values():
    spine.set_color("#333")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.0f}%", ha="center", va="bottom", color="#eee", fontsize=13, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sanity_check_attack.png"), dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print("Saved sanity_check_attack.png")

# %%
# Plot 3: Weapon confusion matrix (expected vs Gemini-identified)
weapon_list = sorted(WEAPON_NAMES.values())
n_weapons = len(weapon_list)
w2i = {w: i for i, w in enumerate(weapon_list)}

matrix = np.zeros((n_weapons, n_weapons), dtype=int)
unmatched = 0
for r in switch_results:
    if "error" in r or not r.get("switch_occurred"):
        continue
    expected = r.get("expected_to", "").lower().replace(" ", "_")
    identified = (r.get("weapon_to") or "").lower().replace(" ", "_")
    if expected in w2i and identified in w2i:
        matrix[w2i[expected], w2i[identified]] += 1
    else:
        unmatched += 1

if matrix.sum() > 0:
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0f0f1e")

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_weapons))
    ax.set_yticks(range(n_weapons))
    short_names = [w.replace("_", "\n") for w in weapon_list]
    ax.set_xticklabels(short_names, rotation=45, ha="right", color="#ccc", fontsize=8)
    ax.set_yticklabels(short_names, color="#ccc", fontsize=8)
    ax.set_xlabel("Gemini Identified", color="#ccc")
    ax.set_ylabel("Expected (Action Data)", color="#ccc")
    ax.set_title("Weapon Confusion Matrix", color="#eee", fontsize=14)
    ax.tick_params(colors="#888")

    for i in range(n_weapons):
        for j in range(n_weapons):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "#222",
                        fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sanity_check_confusion.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved sanity_check_confusion.png (unmatched: {unmatched})")
else:
    print("No confirmed switches with identified weapons — skipping confusion matrix")

# %%
print("\nDone! Check preprocessing/sanity_check_*.png for plots.")
