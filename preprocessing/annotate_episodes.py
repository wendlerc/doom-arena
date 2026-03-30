#!/usr/bin/env python3
"""
Annotate Doom gameplay episodes with event timestamps using Google Gemini vision.

Uses Gemini 2.5's native video understanding to detect gameplay events
(weapon pickups, health/armor pickups, weapon switches, shooting, frags,
deaths/respawns) across multiple focused annotation passes.

Output per episode+player:
  {output-dir}/{episode_id}_{player}.json

Requires: google-genai  (uv pip install google-genai)

Usage:
    export GEMINI_API_KEY=...
    python preprocessing/annotate_episodes.py --data-root recordings
    python preprocessing/annotate_episodes.py --data-root recordings --player p1 --passes pickup_weapons,combat_frags
    python preprocessing/annotate_episodes.py --data-root datasets/pvp_recordings --shard-pattern "mp-*.tar"
"""
import sys, os, io, json, re, time, argparse, tarfile, tempfile
from datetime import datetime, timezone
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from google import genai

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

SYSTEM_CONTEXT = """\
You are an expert Doom gameplay analyst annotating a first-person deathmatch \
video recorded at 35 fps, 640x480 resolution.

HUD layout (bottom of screen):
- Bottom-left: health percentage (number)
- Bottom-right: armor percentage (number)
- Bottom-center: ammo count for current weapon
- The currently held weapon is visible in the player's hands (first-person view)

Doom weapons by slot:
1. Fist / Chainsaw
2. Pistol
3. Shotgun
4. Super Shotgun (double-barrel)
5. Chaingun
6. Rocket Launcher
7. Plasma Rifle / BFG 9000

When a player picks up a weapon, it appears on the ground as a sprite, \
the player walks over it, and the weapon may appear in their hands. \
Health pickups are green/blue bottles or medkits. Armor is green/blue armor \
vests or helmets. Frag messages appear in the top-left corner of the screen.

The video is approximately 5 minutes long. Be thorough — scan the entire video.\
"""

# ---------------------------------------------------------------------------
# Pass definitions
# ---------------------------------------------------------------------------
PASSES = {
    "pickup_weapons": {
        "event_types": ["weapon_pickup"],
        "prompt": """\
Watch the entire video carefully and find every moment where the player picks \
up a NEW weapon they did not previously have. Key indicators:
- A weapon sprite is visible on the ground and the player walks over it
- The weapon in the player's hands changes to something they haven't used before
- A new weapon appears for the first time in the game

For each pickup, identify the weapon type (shotgun, super_shotgun, chaingun, \
rocket_launcher, plasma_rifle, chainsaw, bfg).

Return a JSON array of events. Each event MUST have these exact fields:
- "timestamp_start": start time as "MM:SS.s" (e.g. "01:23.5")
- "timestamp_end": end time as "MM:SS.s"
- "event_type": "weapon_pickup"
- "confidence": float 0.0-1.0
- "description": brief description of what happens
- "details": {"weapon_type": "shotgun"} (the weapon picked up)

Return ONLY the JSON array. No markdown fences, no explanation text.\
""",
    },
    "pickup_health_armor": {
        "event_types": ["health_pickup", "armor_pickup"],
        "prompt": """\
Watch the entire video carefully and find every moment where the player picks \
up health or armor items. Key indicators:
- Health pickups: green bottles (+1 HP), medkits (+10 or +25 HP), soulsphere (+100 HP blue sphere)
- The health number in the bottom-left HUD increases
- Armor pickups: green armor vests, blue armor vests, armor helmets (+1 armor)
- The armor number in the bottom-right HUD increases

For each pickup, note whether it's health or armor and estimate the amount gained \
if you can see the HUD numbers change.

Return a JSON array of events. Each event MUST have these exact fields:
- "timestamp_start": start time as "MM:SS.s"
- "timestamp_end": end time as "MM:SS.s"
- "event_type": "health_pickup" or "armor_pickup"
- "confidence": float 0.0-1.0
- "description": brief description
- "details": {"item_type": "medkit", "amount_estimate": "+25"} or similar

Return ONLY the JSON array. No markdown fences, no explanation text.\
""",
    },
    "weapon_switch": {
        "event_types": ["weapon_switch"],
        "prompt": """\
Watch the entire video carefully and find every moment where the player \
SWITCHES between weapons they already have in their inventory (NOT picking \
up a new weapon). Key indicators:
- The weapon in the player's hands changes (lowering animation then raising new weapon)
- This is a deliberate weapon change, not from picking up a new weapon
- The player cycles through weapons they already possess

For each switch, identify the weapon switched FROM and TO.

Return a JSON array of events. Each event MUST have these exact fields:
- "timestamp_start": start time as "MM:SS.s"
- "timestamp_end": end time as "MM:SS.s"
- "event_type": "weapon_switch"
- "confidence": float 0.0-1.0
- "description": brief description
- "details": {"weapon_from": "shotgun", "weapon_to": "rocket_launcher"}

Return ONLY the JSON array. No markdown fences, no explanation text.\
""",
    },
    "shooting": {
        "event_types": ["shooting"],
        "prompt": """\
Watch the entire video carefully and find sequences where the player is \
actively SHOOTING/FIRING their weapon. Key indicators:
- Muzzle flash / weapon firing animation
- Projectiles visible (rockets, plasma balls)
- Shotgun/chaingun recoil animation
- The attack action is happening

For each shooting sequence, identify the weapon being used and roughly how \
long the burst lasts. Group continuous firing into single events rather than \
marking each individual shot.

Return a JSON array of events. Each event MUST have these exact fields:
- "timestamp_start": start time as "MM:SS.s"
- "timestamp_end": end time as "MM:SS.s"
- "event_type": "shooting"
- "confidence": float 0.0-1.0
- "description": brief description
- "details": {"weapon_type": "rocket_launcher", "target_visible": true/false}

Return ONLY the JSON array. No markdown fences, no explanation text.\
""",
    },
    "combat_frags": {
        "event_types": ["combat", "frag"],
        "prompt": """\
Watch the entire video carefully and find every combat encounter and frag \
(kill). Key indicators:
- Enemy players/bots visible on screen
- Player shooting at enemies
- Enemies taking damage (blood splatter, staggering)
- Enemy dying (falling down, gibbing into pieces)
- Frag notification messages in the top-left corner of the screen
- Score/frag count incrementing

Mark combat sequences (active fighting) and highlight frags (confirmed kills) \
separately. A combat sequence may contain zero or more frags.

Return a JSON array of events. Each event MUST have these exact fields:
- "timestamp_start": start time as "MM:SS.s"
- "timestamp_end": end time as "MM:SS.s"
- "event_type": "combat" or "frag"
- "confidence": float 0.0-1.0
- "description": brief description
- "details": {"weapon_used": "shotgun"} (for frags, include the weapon that got the kill)

Return ONLY the JSON array. No markdown fences, no explanation text.\
""",
    },
    "death_respawn": {
        "event_types": ["death", "respawn"],
        "prompt": """\
Watch the entire video carefully and find every moment where the player DIES \
and then RESPAWNS. Key indicators for death:
- Screen flashes red
- Player's view tilts / falls to the ground
- Health drops to 0
- A death message may appear

Key indicators for respawn:
- Screen suddenly changes to a completely different location
- Player appears with default weapon (pistol or fist)
- Health resets to 100%
- Brief invulnerability period (player may flash)

Mark death and respawn as separate events (they happen in sequence). \
Note the approximate cause of death if visible (e.g. "killed by rocket").

Return a JSON array of events. Each event MUST have these exact fields:
- "timestamp_start": start time as "MM:SS.s"
- "timestamp_end": end time as "MM:SS.s"
- "event_type": "death" or "respawn"
- "confidence": float 0.0-1.0
- "description": brief description
- "details": {"cause": "rocket"} for deaths, {} for respawns

Return ONLY the JSON array. No markdown fences, no explanation text.\
""",
    },
    "other": {
        "event_types": ["other"],
        "prompt": """\
Watch the entire video carefully and identify any notable gameplay moments \
NOT covered by these categories: weapon pickups, health/armor pickups, \
weapon switches, shooting, combat/frags, deaths/respawns.

Examples of what to look for:
- Picking up powerups (invulnerability, invisibility, berserk, megasphere)
- Environmental hazards (lava, crushing ceilings, barrels exploding)
- Near-death escapes (surviving at very low health)
- Notable movement/navigation (jumping gaps, using lifts, secret areas)
- Picking up ammo (backpack, ammo boxes)
- Any other visually interesting or important moments

Return a JSON array of events. Each event MUST have these exact fields:
- "timestamp_start": start time as "MM:SS.s"
- "timestamp_end": end time as "MM:SS.s"
- "event_type": "other"
- "confidence": float 0.0-1.0
- "description": detailed description of what happens
- "details": {"category": "powerup"} or similar

If nothing notable is found beyond the other categories, return an empty array: []

Return ONLY the JSON array. No markdown fences, no explanation text.\
""",
    },
}

PASS_NAMES = list(PASSES.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def timestamp_to_frame(ts: str, fps: int = GAME_FPS, n_frames: int | None = None) -> int:
    """Convert 'MM:SS.s' or 'HH:MM:SS.s' timestamp to frame number."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        total_seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        total_seconds = float(parts[0]) * 60 + float(parts[1])
    else:
        total_seconds = float(parts[0])
    frame = int(round(total_seconds * fps))
    if n_frames is not None:
        frame = max(0, min(frame, n_frames - 1))
    return frame


def frame_to_timestamp(frame: int, fps: int = GAME_FPS) -> str:
    """Convert frame number to 'MM:SS.s' string."""
    total_seconds = frame / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:04.1f}"


def parse_gemini_json(raw_text: str) -> list[dict]:
    """Parse Gemini response, handling markdown fences, truncated arrays, and extra text."""
    text = raw_text.strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        pass
    # Try extracting JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Handle truncated JSON array: find last complete object and close the array
    match = re.search(r"\[.*", text, re.DOTALL)
    if match:
        fragment = match.group()
        # Find last complete JSON object (ends with })
        last_brace = fragment.rfind("}")
        if last_brace > 0:
            truncated = fragment[:last_brace + 1] + "]"
            try:
                result = json.loads(truncated)
                if isinstance(result, list):
                    print(f"(recovered {len(result)} events from truncated response) ", end="")
                    return result
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Cannot parse Gemini response as JSON: {text[:300]}...")


def _extract_retry_delay(err_str: str) -> float | None:
    """Extract retryDelay from Gemini error message if present."""
    # Matches "retryDelay': '25s'" or "retryDelay': '1.17758818s'"
    m = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s?['\"]", err_str)
    if m:
        return float(m.group(1))
    return None


def call_gemini(
    client: genai.Client, model_name: str, contents, max_retries: int = 5, base_delay: float = 10.0
) -> str:
    """Call Gemini generate_content with exponential backoff retry."""
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            return response.text
        except Exception as e:
            err = str(e)
            if attempt == max_retries:
                raise
            if any(code in err for code in ["429", "ResourceExhausted", "500", "503"]):
                # Use server-suggested delay if available, otherwise exponential backoff
                server_delay = _extract_retry_delay(err)
                delay = max(server_delay or 0, base_delay * (2 ** attempt))
                print(f"    Retrying in {delay:.0f}s ({err[:80]})...")
                time.sleep(delay)
            else:
                raise


def wait_for_file_active(client: genai.Client, file, timeout: float = 300.0, poll_interval: float = 5.0):
    """Poll until an uploaded Gemini file is in ACTIVE state."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = client.files.get(name=file.name)
        if f.state.name == "ACTIVE":
            return f
        if f.state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {file.name}")
        time.sleep(poll_interval)
    raise TimeoutError(f"File {file.name} not active after {timeout}s")


def scan_episodes(data_root: str, shard_pattern: str = "*.tar") -> list[dict]:
    """Scan shards and return list of episode dicts with shard path, key, members, meta."""
    shards = sorted(Path(data_root).glob(shard_pattern))
    episodes = []
    for shard_path in shards:
        try:
            with tarfile.open(shard_path, "r") as tar:
                groups: dict[str, dict[str, str]] = {}
                for member in tar.getmembers():
                    if member.isdir():
                        continue
                    parts = member.name.split(".", 1)
                    if len(parts) != 2:
                        continue
                    key, ext = parts
                    groups.setdefault(key, {})[ext] = member.name
                for key, members in groups.items():
                    meta_name = members.get("meta.json")
                    if not meta_name:
                        continue
                    meta = json.loads(tar.extractfile(tar.getmember(meta_name)).read())
                    episodes.append({
                        "shard": str(shard_path),
                        "key": key,
                        "members": members,
                        "meta": meta,
                    })
        except (tarfile.TarError, OSError) as e:
            print(f"  Skipping {shard_path}: {e}")
            continue
    return episodes


def extract_video(shard_path: str, member_name: str) -> str:
    """Extract MP4 from tar to a temp file. Returns path. Caller must delete."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    with tarfile.open(shard_path, "r") as tar:
        f = tar.extractfile(tar.getmember(member_name))
        tmp.write(f.read())
    tmp.close()
    return tmp.name


def load_existing_annotations(path: str) -> dict | None:
    """Load existing annotation JSON if it exists."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Core annotation
# ---------------------------------------------------------------------------
def annotate_video(
    client: genai.Client,
    model_name: str,
    video_path: str,
    meta: dict,
    n_frames: int,
    passes: list[str],
    max_retries: int = 3,
    delay: float = 5.0,
) -> tuple[list[dict], list[str]]:
    """
    Run annotation passes on one video file.
    Returns (events, passes_completed).
    """
    # Upload video
    print(f"    Uploading video ({os.path.getsize(video_path) / 1e6:.0f} MB)...")
    t0 = time.time()
    uploaded = client.files.upload(file=video_path)
    print(f"    Upload done in {time.time() - t0:.1f}s, waiting for processing...")
    uploaded = wait_for_file_active(client, uploaded)
    print(f"    File active. Running {len(passes)} passes...")

    all_events = []
    completed = []

    try:
        for i, pass_name in enumerate(passes):
            pass_def = PASSES[pass_name]
            prompt = SYSTEM_CONTEXT + "\n\n" + pass_def["prompt"]

            print(f"    Pass {i+1}/{len(passes)}: {pass_name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                raw = call_gemini(client, model_name, [uploaded, prompt], max_retries=max_retries, base_delay=delay)
                events = parse_gemini_json(raw)

                # Enrich events with frame numbers and pass tag
                for ev in events:
                    ev["pass"] = pass_name
                    if "timestamp_start" in ev:
                        ev["frame_start"] = timestamp_to_frame(
                            ev["timestamp_start"], n_frames=n_frames
                        )
                    if "timestamp_end" in ev:
                        ev["frame_end"] = timestamp_to_frame(
                            ev["timestamp_end"], n_frames=n_frames
                        )

                all_events.extend(events)
                completed.append(pass_name)
                dt = time.time() - t0
                print(f"{len(events)} events ({dt:.1f}s)")

            except Exception as e:
                print(f"FAILED: {e}")
                continue

            # Rate limit between passes
            if i < len(passes) - 1:
                time.sleep(delay)

    finally:
        # Clean up uploaded file
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

    # Sort events by frame_start
    all_events.sort(key=lambda e: e.get("frame_start", 0))
    return all_events, completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Annotate Doom episodes with Gemini vision")
    parser.add_argument("--data-root", default="recordings", help="Directory with WebDataset shards")
    parser.add_argument("--shard-pattern", default="*.tar", help="Glob pattern for shard files")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: {data-root}/annotations)")
    parser.add_argument("--player", default="all", choices=["p1", "p2", "all"],
                        help="Which player perspective to annotate")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--passes", default="all",
                        help="Comma-separated pass names or 'all'")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per API call")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay (seconds) between API calls")
    parser.add_argument("--sanity", type=int, default=0, help="Only process N episodes (0=all)")
    args = parser.parse_args()

    # API: prefer API key, fall back to Vertex AI if project is set
    api_key = os.environ.get("GEMINI_API_KEY")
    gcp_project = os.environ.get("GCP_PROJECT")
    if api_key:
        client = genai.Client(api_key=api_key)
    elif gcp_project:
        location = os.environ.get("GCP_LOCATION", "us-central1")
        print(f"Using Vertex AI (project={gcp_project}, location={location})")
        client = genai.Client(vertexai=True, project=gcp_project, location=location)
    else:
        print("ERROR: Set GEMINI_API_KEY or GCP_PROJECT env variable")
        sys.exit(1)

    # Output directory
    output_dir = args.output_dir or os.path.join(args.data_root, "annotations")
    os.makedirs(output_dir, exist_ok=True)

    # Passes
    if args.passes == "all":
        passes = PASS_NAMES
    else:
        passes = [p.strip() for p in args.passes.split(",")]
        for p in passes:
            if p not in PASSES:
                print(f"ERROR: Unknown pass '{p}'. Available: {', '.join(PASS_NAMES)}")
                sys.exit(1)

    # Players
    players = ["p1", "p2"] if args.player == "all" else [args.player]

    # Scan episodes
    print(f"Scanning {args.data_root} for {args.shard_pattern}...")
    episodes = scan_episodes(args.data_root, args.shard_pattern)
    if args.sanity > 0:
        episodes = episodes[:args.sanity]
    print(f"Found {len(episodes)} episode(s)")

    if not episodes:
        print("No episodes found. Check --data-root and --shard-pattern.")
        sys.exit(1)

    total_events = 0
    for ep_idx, ep in enumerate(episodes):
        ep_id = ep["meta"].get("episode_id", ep["key"])
        meta = ep["meta"]
        n_frames = meta.get("n_frames", 0)

        print(f"\nEpisode {ep_idx+1}/{len(episodes)}: {ep_id[:16]}... "
              f"({n_frames} frames, {n_frames/GAME_FPS:.0f}s)")

        for player in players:
            video_key = f"video_{player}.mp4"
            if video_key not in ep["members"]:
                print(f"  {player}: no video, skipping")
                continue

            out_path = os.path.join(output_dir, f"{ep_id}_{player}.json")

            # Resume: check existing annotations
            existing = load_existing_annotations(out_path)
            if existing:
                done = set(existing.get("passes_completed", []))
                remaining = [p for p in passes if p not in done]
                if not remaining:
                    n_ev = len(existing.get("events", []))
                    print(f"  {player}: already complete ({n_ev} events), skipping")
                    total_events += n_ev
                    continue
                print(f"  {player}: resuming ({len(remaining)} passes remaining)")
                passes_to_run = remaining
                prior_events = existing.get("events", [])
                prior_passes = list(done)
            else:
                passes_to_run = passes
                prior_events = []
                prior_passes = []

            # Extract video
            print(f"  {player}: extracting video from shard...")
            video_path = extract_video(ep["shard"], ep["members"][video_key])

            try:
                events, completed = annotate_video(
                    client, args.model, video_path, meta, n_frames,
                    passes_to_run, args.max_retries, args.delay,
                )
            finally:
                os.unlink(video_path)

            # Merge with prior results
            all_events = prior_events + events
            all_events.sort(key=lambda e: e.get("frame_start", 0))
            all_passes = prior_passes + completed

            # Save
            annotation = {
                "version": "1.0",
                "episode_id": ep_id,
                "source_shard": ep["shard"],
                "player": player,
                "model": args.model,
                "n_frames": n_frames,
                "fps": GAME_FPS,
                "duration_s": round(n_frames / GAME_FPS, 1),
                "annotated_at": datetime.now(timezone.utc).isoformat(),
                "meta": meta,
                "passes_completed": all_passes,
                "events": all_events,
            }
            with open(out_path, "w") as f:
                json.dump(annotation, f, indent=2)

            n_new = len(events)
            total_events += len(all_events)
            print(f"  {player}: {n_new} new events, {len(all_events)} total -> {out_path}")

    print(f"\nDone. {total_events} total events across all episodes.")


if __name__ == "__main__":
    main()
