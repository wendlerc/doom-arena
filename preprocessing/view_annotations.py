#!/usr/bin/env python3
"""
NLE-style interactive annotation viewer for Doom gameplay episodes.

Generates a standalone HTML viewer modeled after video editing software:
- Top: episode tabs + video player with details panel
- Bottom: zoomable/scrollable timeline with filmstrip thumbnails and
  annotation tracks per event type

Usage:
    python preprocessing/view_annotations.py --data-root recordings --ann-dir recordings/annotations_v2
"""
import sys, os, io, json, argparse, tarfile, base64
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

GAME_FPS = 35
THUMB_INTERVAL = 3   # seconds between thumbnails
THUMB_W = 160        # thumbnail width in px
THUMB_H = 90         # thumbnail height in px (16:9-ish for 640x480)

EVENT_COLORS = {
    "weapon_pickup": "#e74c3c",
    "health_pickup": "#2ecc71",
    "armor_pickup": "#3498db",
    "weapon_switch": "#f39c12",
    "shooting": "#e67e22",
    "combat": "#9b59b6",
    "frag": "#ff4757",
    "death": "#c0392b",
    "respawn": "#1abc9c",
    "other": "#636e72",
}

EVENT_LABELS = {
    "weapon_pickup": "Weapon Pickup",
    "health_pickup": "Health Pickup",
    "armor_pickup": "Armor Pickup",
    "weapon_switch": "Weapon Switch",
    "shooting": "Shooting",
    "combat": "Combat",
    "frag": "Frag (Kill)",
    "death": "Death",
    "respawn": "Respawn",
    "other": "Other",
}


def scan_annotations(data_root, ann_dir=None):
    """Scan annotations, group by episode+player, return latest version of each.

    Files follow the naming convention:
      {episode_id}_{player}.json          (original, v0)
      {episode_id}_{player}_v1.json       (first edit)
      {episode_id}_{player}_v2.json       (second edit)
      {episode_id}_{player}_edited.json   (legacy, treated as v1)
    """
    import re as _re
    if ann_dir is None:
        ann_dir = os.path.join(data_root, "annotations")
    if not os.path.isdir(ann_dir):
        return []

    # Group files by (episode_id, player) → [(version_num, path, data)]
    groups = {}
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(ann_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        ep_id = data.get("episode_id", "")
        player = data.get("player", "")
        if not ep_id or not player:
            continue

        # Determine version number from filename
        base = fname.removesuffix(".json")
        m = _re.search(r"_v(\d+)$", base)
        if m:
            ver = int(m.group(1))
        elif base.endswith("_edited"):
            ver = 1  # legacy naming
        else:
            ver = 0  # original

        key = (ep_id, player)
        data["_version"] = ver
        data["_filename"] = fname
        groups.setdefault(key, []).append((ver, data))

    # Pick latest version for each episode+player, attach version history
    results = []
    for key, versions in groups.items():
        versions.sort(key=lambda x: x[0])
        latest = versions[-1][1]
        latest["_all_versions"] = [{"version": v, "filename": d["_filename"]} for v, d in versions]
        latest["_next_version"] = versions[-1][0] + 1
        results.append(latest)

    results.sort(key=lambda a: (a.get("episode_id", ""), a.get("player", "")))
    return results


def extract_video_file(shard_path, member_name, output_dir):
    out_name = member_name.replace("/", "_")
    out_path = os.path.join(output_dir, out_name)
    if not os.path.exists(out_path):
        with tarfile.open(shard_path, "r") as tar:
            data = tar.extractfile(tar.getmember(member_name)).read()
        with open(out_path, "wb") as f:
            f.write(data)
    return out_name


def extract_thumbnails(shard_path, member_name, output_dir, ep_id, player, duration):
    """Extract thumbnails every THUMB_INTERVAL seconds. Returns list of base64 strings."""
    import cv2
    import tempfile

    thumb_dir = os.path.join(output_dir, f"thumbs_{ep_id[:8]}_{player}")
    n_thumbs = int(duration / THUMB_INTERVAL) + 1

    # Check if already extracted
    if os.path.isdir(thumb_dir) and len([f for f in os.listdir(thumb_dir) if f.endswith('.jpg')]) >= n_thumbs:
        thumbs = []
        for i in range(n_thumbs):
            p = os.path.join(thumb_dir, f"{i:04d}.jpg")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    thumbs.append(base64.b64encode(f.read()).decode())
        if len(thumbs) == n_thumbs:
            return thumbs

    os.makedirs(thumb_dir, exist_ok=True)

    with tarfile.open(shard_path, "r") as tar:
        video_data = tar.extractfile(tar.getmember(member_name)).read()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_data)
    tmp.close()

    thumbs = []
    try:
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or GAME_FPS
        for i in range(n_thumbs):
            target_frame = int(i * THUMB_INTERVAL * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (THUMB_W, THUMB_H))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf.tobytes()).decode()
            thumbs.append(b64)
            with open(os.path.join(thumb_dir, f"{i:04d}.jpg"), "wb") as f:
                f.write(buf.tobytes())
        cap.release()
    finally:
        os.unlink(tmp.name)

    print(f"    Extracted {len(thumbs)} thumbnails (every {THUMB_INTERVAL}s)")
    return thumbs


def extract_actions(shard_path, members, player):
    """Extract action array from shard. Returns list of 14-element lists, one per frame."""
    import numpy as np
    actions_key = f"actions_{player}.npy"
    if actions_key not in members:
        return []
    with tarfile.open(shard_path, "r") as tar:
        data = tar.extractfile(tar.getmember(members[actions_key])).read()
    actions = np.load(io.BytesIO(data))  # (N, 14) float32
    # Downsample to ~5fps to keep JS payload small (every 7th frame at 35fps)
    step = max(1, GAME_FPS // 5)
    downsampled = actions[::step]
    # Convert to compact format: only store nonzero/significant values
    # For binary buttons (0-12): round to 0/1
    # For turn delta (13): round to 1 decimal
    result = []
    for row in downsampled:
        compact = [int(round(row[i])) for i in range(13)]
        compact.append(round(float(row[13]), 1))
        result.append(compact)
    return result


def generate_html(annotations, data_root, output_dir):
    """Generate NLE-style HTML viewer."""

    episodes_data = []
    for ann in annotations:
        ep_id = ann["episode_id"]
        player = ann["player"]
        n_frames = ann["n_frames"]
        duration = n_frames / GAME_FPS
        events = ann.get("events", [])
        meta = ann.get("meta", {})

        shard = ann["source_shard"]
        video_key = f"video_{player}.mp4"
        ep_key_prefix = None
        try:
            with tarfile.open(shard, "r") as tar:
                for m in tar.getnames():
                    if m.endswith(video_key):
                        ep_key_prefix = m
                        break
        except Exception:
            continue
        if not ep_key_prefix:
            continue

        video_filename = extract_video_file(shard, ep_key_prefix, output_dir)

        print(f"  Extracting thumbnails for {ep_id[:12]}... {player}")
        thumbs = extract_thumbnails(shard, ep_key_prefix, output_dir, ep_id, player, duration)

        # Extract action data for overlay
        tar_members = {}
        try:
            with tarfile.open(shard, "r") as tar:
                for m in tar.getnames():
                    parts = m.split(".", 1)
                    if len(parts) == 2:
                        tar_members[parts[1]] = m
        except Exception:
            pass
        actions_data = extract_actions(shard, tar_members, player)
        print(f"    Actions: {len(actions_data)} samples (downsampled to ~5fps)")

        # Build track data
        tracks_data = {}
        event_types_present = []
        seen = set()
        for ev in events:
            t = ev["event_type"]
            if t not in seen:
                event_types_present.append(t)
                seen.add(t)
        for etype in event_types_present:
            tracks_data[etype] = [{
                "start": e.get("frame_start", 0) / GAME_FPS,
                "end": e.get("frame_end", e.get("frame_start", 0)) / GAME_FPS,
                "description": e.get("description", ""),
                "details": e.get("details", {}),
                "confidence": e.get("confidence", 0),
                "timestamp_start": e.get("timestamp_start", ""),
                "timestamp_end": e.get("timestamp_end", ""),
                "source": e.get("source", ""),
            } for e in events if e["event_type"] == etype]

        frag = meta.get(f"frag_{player}", "?")
        death = meta.get(f"death_{player}", "?")
        scenario = meta.get("scenario", "?")
        is_human = meta.get(f"is_human_{player}", False)
        player_label = f"{player.upper()} ({'human' if is_human else 'AI'})"

        vid_id = f"vid_{ep_id[:8]}_{player}".replace("-", "")

        episodes_data.append({
            "vid_id": vid_id,
            "ep_id": ep_id,
            "player": player,
            "player_label": player_label,
            "scenario": scenario,
            "frag": frag,
            "death": death,
            "duration": duration,
            "n_events": len(events),
            "video_filename": video_filename,
            "thumbs": thumbs,
            "tracks": tracks_data,
            "ann_meta": {
                "version": ann.get("version", "1.0"),
                "pipeline": ann.get("pipeline", ""),
                "episode_id": ep_id,
                "source_shard": ann.get("source_shard", ""),
                "player": player,
                "model": ann.get("model", ""),
                "n_frames": n_frames,
                "fps": ann.get("fps", GAME_FPS),
                "duration_s": round(duration, 1),
                "annotated_at": ann.get("annotated_at", ""),
                "meta": meta,
            },
            "actions": actions_data,
            "current_version": ann.get("_version", 0),
            "next_version": ann.get("_next_version", 1),
            "all_versions": ann.get("_all_versions", []),
        })

    # Serialize episode data for JS (without thumbs in the main JSON — too large)
    # Instead, embed thumbs as a separate JS array per episode
    episodes_json_list = []
    thumbs_js_parts = []
    actions_js_parts = []
    for ep in episodes_data:
        ep_copy = {k: v for k, v in ep.items() if k not in ("thumbs", "actions")}
        episodes_json_list.append(ep_copy)
        thumbs_js_parts.append(
            f'THUMBS["{ep["vid_id"]}"] = {json.dumps(ep["thumbs"])};'
        )
        actions_js_parts.append(
            f'ACTIONS["{ep["vid_id"]}"] = {json.dumps(ep["actions"])};'
        )

    episodes_json = json.dumps(episodes_json_list)
    colors_json = json.dumps(EVENT_COLORS)
    labels_json = json.dumps(EVENT_LABELS)
    thumbs_js = "\n".join(thumbs_js_parts)
    actions_js = "\n".join(actions_js_parts)

    # Build tab HTML
    tabs_html = ""
    for i, ep in enumerate(episodes_data):
        active = "active" if i == 0 else ""
        tabs_html += f'<div class="tab {active}" data-index="{i}">{ep["player_label"]} — {ep["scenario"]} — {ep["duration"]:.0f}s</div>\n'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Doom Annotation Editor</title>
<style>
:root {{
    --bg-dark: #1a1a2e;
    --bg-darker: #0f0f1e;
    --bg-panel: #16213e;
    --bg-track: #0a0a18;
    --border: #2a2a4a;
    --text: #e0e0e0;
    --text-dim: #8888aa;
    --playhead: #ff4757;
    --ruler-bg: #1e1e38;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ height: 100%; overflow: hidden; }}
body {{
    background: var(--bg-darker);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 13px;
    display: flex;
    flex-direction: column;
}}

/* --- Tab bar --- */
.tab-bar {{
    display: flex;
    background: var(--bg-dark);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    overflow-x: auto;
}}
.tab {{
    padding: 8px 18px;
    cursor: pointer;
    color: var(--text-dim);
    border-right: 1px solid var(--border);
    white-space: nowrap;
    font-size: 12px;
    transition: background 0.15s;
}}
.tab:hover {{ background: var(--bg-panel); }}
.tab.active {{
    background: var(--bg-panel);
    color: var(--text);
    border-bottom: 2px solid var(--playhead);
}}

/* --- Main layout --- */
.main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
.episode-container {{ display: none; flex: 1; flex-direction: column; overflow: hidden; }}
.episode-container.active {{ display: flex; }}

/* --- Top row: details + player --- */
.top-row {{
    display: flex;
    flex-shrink: 0;
    border-bottom: 1px solid var(--border);
    height: 340px;
    min-height: 280px;
}}
.details-panel {{
    width: 320px;
    flex-shrink: 0;
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    padding: 12px;
    overflow-y: auto;
}}
.details-panel h3 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 8px;
}}
.detail-row {{
    margin-bottom: 6px;
}}
.detail-label {{
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.detail-value {{
    font-size: 13px;
    color: var(--text);
    word-break: break-word;
}}
.detail-value.type-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 600;
    font-size: 12px;
}}
.no-selection {{
    color: var(--text-dim);
    font-style: italic;
    margin-top: 40px;
    text-align: center;
}}
.player-panel {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    min-width: 0;
    position: relative;
}}
.player-panel video {{
    max-width: 100%;
    max-height: 100%;
    display: block;
}}
.action-overlay {{
    position: absolute;
    top: 8px;
    right: 8px;
    font-family: 'Consolas', 'SF Mono', monospace;
    font-size: 16px;
    font-weight: bold;
    line-height: 1.3;
    color: #fff;
    text-shadow: 0 0 4px #000, 0 0 8px #000, 1px 1px 2px #000;
    pointer-events: none;
    text-align: right;
    white-space: pre;
    z-index: 5;
    background: rgba(0,0,0,0.35);
    padding: 6px 10px;
    border-radius: 6px;
}}
.action-overlay .act-active {{
    color: #ff4757;
}}
.action-overlay .act-move {{
    color: #2ed573;
}}
.action-overlay .act-weapon {{
    color: #ffa502;
}}
.action-overlay .act-turn {{
    color: #70a1ff;
}}

/* --- Timeline area --- */
.timeline-area {{
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--bg-darker);
}}

/* Controls bar */
.controls-bar {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 12px;
    background: var(--bg-dark);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    height: 32px;
}}
.controls-bar label {{
    font-size: 11px;
    color: var(--text-dim);
}}
.zoom-slider {{
    width: 140px;
    accent-color: var(--playhead);
}}
.time-display {{
    font-family: 'Consolas', 'SF Mono', monospace;
    font-size: 12px;
    color: var(--text);
    margin-left: auto;
}}

/* Scroll container for ruler + filmstrip + tracks */
.timeline-scroll {{
    flex: 1;
    overflow-x: auto;
    overflow-y: auto;
    position: relative;
}}
.timeline-content {{
    position: relative;
    min-height: 100%;
}}

/* Track label gutter (fixed left) */
.track-labels {{
    position: sticky;
    left: 0;
    z-index: 15;
    width: 110px;
    flex-shrink: 0;
}}

/* Time ruler */
.ruler {{
    height: 22px;
    background: var(--ruler-bg);
    border-bottom: 1px solid var(--border);
    position: relative;
    cursor: pointer;
}}
.ruler-tick {{
    position: absolute;
    top: 0;
    height: 100%;
    border-left: 1px solid #333;
    font-size: 9px;
    color: var(--text-dim);
    padding-left: 3px;
    padding-top: 2px;
    pointer-events: none;
    white-space: nowrap;
}}

/* Filmstrip */
.filmstrip {{
    height: {THUMB_H + 4}px;
    background: var(--bg-track);
    border-bottom: 1px solid var(--border);
    display: flex;
    position: relative;
    cursor: pointer;
    overflow: hidden;
}}
.filmstrip img {{
    height: {THUMB_H}px;
    width: {THUMB_W}px;
    object-fit: cover;
    flex-shrink: 0;
    margin: 2px 0;
    border-right: 1px solid #111;
    pointer-events: none;
}}

/* Annotation tracks */
.track {{
    height: 24px;
    display: flex;
    position: relative;
    border-bottom: 1px solid #1a1a2e;
}}
.track-label {{
    position: sticky;
    left: 0;
    width: 110px;
    flex-shrink: 0;
    font-size: 10px;
    color: var(--text-dim);
    text-align: right;
    padding-right: 8px;
    line-height: 24px;
    background: var(--bg-dark);
    z-index: 12;
    border-right: 1px solid var(--border);
    white-space: nowrap;
    overflow: hidden;
}}
.track-bar {{
    flex: 1;
    position: relative;
    background: var(--bg-track);
    cursor: pointer;
    min-width: 0;
}}
.event-span {{
    position: absolute;
    height: 18px;
    top: 3px;
    border-radius: 3px;
    opacity: 0.85;
    min-width: 4px;
    cursor: pointer;
    transition: opacity 0.1s, box-shadow 0.1s;
}}
.event-span:hover {{
    opacity: 1;
    z-index: 10;
    box-shadow: 0 0 8px rgba(255,255,255,0.25);
}}
.event-span.selected {{
    opacity: 1;
    outline: 2px solid #fff;
    outline-offset: -1px;
    z-index: 11;
}}

/* Playhead */
.playhead-line {{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--playhead);
    z-index: 20;
    pointer-events: none;
    box-shadow: 0 0 4px rgba(255,71,87,0.5);
}}
.playhead-handle {{
    position: absolute;
    top: -2px;
    left: -5px;
    width: 12px;
    height: 12px;
    background: var(--playhead);
    clip-path: polygon(0 0, 100% 0, 50% 100%);
}}

/* Resize handles on selected events */
.resize-handle {{
    position: absolute;
    top: 0;
    width: 8px;
    height: 100%;
    cursor: ew-resize;
    z-index: 12;
}}
.resize-handle.left {{
    left: -3px;
    border-left: 2px solid rgba(255,255,255,0.8);
}}
.resize-handle.right {{
    right: -3px;
    border-right: 2px solid rgba(255,255,255,0.8);
}}
.event-span.dragging {{
    opacity: 0.6;
    outline: 2px dashed rgba(255,255,255,0.5);
}}
.add-mode-active .track-bar {{
    cursor: crosshair;
}}
.add-mode-active .track-bar:hover {{
    background: rgba(255,255,255,0.04);
}}

/* Modal */
.modal-overlay {{
    display: flex;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.6);
    z-index: 300;
    align-items: center;
    justify-content: center;
}}
.modal {{
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    min-width: 380px;
    max-width: 480px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.8);
}}
.modal h3 {{ margin-bottom: 12px; font-size: 14px; }}
.modal label {{
    display: block; margin-bottom: 3px;
    font-size: 11px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.5px;
}}
.modal input, .modal select, .modal textarea {{
    width: 100%; padding: 6px 8px; margin-bottom: 10px;
    background: var(--bg-dark); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    font-size: 13px; font-family: inherit;
}}
.modal textarea {{ resize: vertical; min-height: 60px; }}
.modal-btns {{ display: flex; gap: 8px; margin-top: 8px; justify-content: flex-end; }}

/* Context menu */
.context-menu {{
    position: fixed;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 0;
    z-index: 250;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
    min-width: 140px;
}}
.context-menu-item {{
    padding: 6px 16px;
    cursor: pointer;
    font-size: 12px;
}}
.context-menu-item:hover {{ background: rgba(255,255,255,0.1); }}
.context-menu-item.danger {{ color: #e74c3c; }}

/* Event buttons */
.ev-btn {{
    padding: 5px 14px;
    border: none;
    border-radius: 4px;
    background: var(--playhead);
    color: #fff;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
}}
.ev-btn:hover {{ opacity: 0.85; }}
.ev-btn-dim {{
    background: #444;
}}
.ev-btn-dim:hover {{ background: #555; }}

/* Tooltip */
.tooltip {{
    display: none;
    position: fixed;
    background: var(--bg-panel);
    border: 1px solid #555;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    color: var(--text);
    z-index: 200;
    max-width: 350px;
    pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
}}
</style>
</head>
<body>

<div class="tab-bar" id="tabBar">
{tabs_html}
</div>

<div class="main" id="main">
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const EPISODES = {episodes_json};
const COLORS = {colors_json};
const LABELS = {labels_json};
const THUMBS = {{}};
{thumbs_js}
const ACTIONS = {{}};
{actions_js}
const ACTION_STEP = {max(1, GAME_FPS // 5)};  // frames between action samples
const ACTION_LABELS = ['FWD','BACK','RIGHT','LEFT','W1','W2','W3','W4','W5','W6','W7','ATK','SPD','TURN'];
const THUMB_W = {THUMB_W};
const THUMB_INTERVAL = {THUMB_INTERVAL};
const GAME_FPS = {GAME_FPS};

const TYPE_ORDER = ['weapon_pickup','health_pickup','armor_pickup','weapon_switch',
                    'shooting','combat','frag','death','respawn','other'];

let activeEpIdx = 0;
let editors = {{}};

function fmt(t) {{
    const m = Math.floor(t / 60);
    const s = t % 60;
    return String(m).padStart(2,'0') + ':' + s.toFixed(1).padStart(4,'0');
}}

function initEditor(ep, container) {{
    const vid_id = ep.vid_id;
    const duration = ep.duration;
    const tracks = ep.tracks;
    const thumbs = THUMBS[vid_id] || [];
    const nThumbs = thumbs.length;

    // Assign stable IDs to all events
    let _nextId = 1;
    const types = Object.keys(tracks).sort((a,b) =>
        (TYPE_ORDER.indexOf(a)===-1?99:TYPE_ORDER.indexOf(a)) -
        (TYPE_ORDER.indexOf(b)===-1?99:TYPE_ORDER.indexOf(b))
    );
    types.forEach(et => (tracks[et]||[]).forEach(ev => ev._id = _nextId++));

    // State
    const LABEL_W = 110;
    const basePxPerSec = (window.innerWidth - LABEL_W - 20) / duration;
    const state = {{
        zoom: basePxPerSec,
        selectedEvent: null,
        selectedEventType: null,
        editMode: null,
        dirty: false,
    }};

    function totalWidth() {{ return Math.max(duration * state.zoom, window.innerWidth - LABEL_W - 20); }}
    function timeToPx(t) {{ return LABEL_W + t * state.zoom; }}
    function pxToTime(px) {{ return Math.max(0, Math.min(duration, (px - LABEL_W) / state.zoom)); }}

    // --- DOM setup ---
    const topRow = document.createElement('div');
    topRow.className = 'top-row';
    const detailsPanel = document.createElement('div');
    detailsPanel.className = 'details-panel';
    detailsPanel.innerHTML = '<h3>Event Details</h3><div class="no-selection">Hover or click an event</div>';
    const playerPanel = document.createElement('div');
    playerPanel.className = 'player-panel';
    const video = document.createElement('video');
    video.id = vid_id;
    video.controls = true;
    video.innerHTML = '<source src="' + ep.video_filename + '" type="video/mp4">';
    playerPanel.appendChild(video);
    const actionOverlay = document.createElement('div');
    actionOverlay.className = 'action-overlay';
    playerPanel.appendChild(actionOverlay);
    topRow.appendChild(detailsPanel);
    topRow.appendChild(playerPanel);

    // Action overlay update
    const actionsArr = ACTIONS[vid_id] || [];
    const WEAPON_NAMES = ['','Fist','Pistol','Shotgun','SSG','Chaingun','Rocket','Plasma'];
    function updateActionOverlay() {{
        if (!actionsArr.length) return;
        const frame = Math.round(video.currentTime * GAME_FPS);
        const idx = Math.min(Math.floor(frame / ACTION_STEP), actionsArr.length - 1);
        if (idx < 0) return;
        const a = actionsArr[idx];
        const lines = [];
        // Movement
        const moves = [];
        if (a[0]) moves.push('FWD');
        if (a[1]) moves.push('BACK');
        if (a[2]) moves.push('RIGHT');
        if (a[3]) moves.push('LEFT');
        if (moves.length) lines.push('<span class="act-move">' + moves.join(' ') + '</span>');
        // Weapon select
        for (let w = 0; w < 7; w++) {{
            if (a[4+w]) lines.push('<span class="act-weapon">SEL ' + WEAPON_NAMES[w+1] + '</span>');
        }}
        // Attack & speed
        const combat = [];
        if (a[11]) combat.push('ATTACK');
        if (a[12]) combat.push('SPEED');
        if (combat.length) lines.push('<span class="act-active">' + combat.join(' ') + '</span>');
        // Turn
        if (Math.abs(a[13]) > 0.5) {{
            const dir = a[13] > 0 ? 'TURN R' : 'TURN L';
            lines.push('<span class="act-turn">' + dir + ' ' + Math.abs(a[13]).toFixed(1) + '</span>');
        }}
        actionOverlay.innerHTML = lines.join('\\n') || '<span style="opacity:0.3">idle</span>';
    }}

    const timelineArea = document.createElement('div');
    timelineArea.className = 'timeline-area';

    // Controls bar with editing buttons
    const controlsBar = document.createElement('div');
    controlsBar.className = 'controls-bar';
    controlsBar.innerHTML = `
        <label>Zoom</label>
        <input type="range" class="zoom-slider" min="0.5" max="30" step="0.1" value="${{basePxPerSec}}">
        <button class="ev-btn ev-btn-dim" id="addBtn_${{vid_id}}" style="font-size:11px;padding:3px 10px">+ Add</button>
        <button class="ev-btn ev-btn-dim" id="saveBtn_${{vid_id}}" style="font-size:11px;padding:3px 10px">Save</button>
        <span id="dirty_${{vid_id}}" style="display:none;color:#ff4757;font-size:11px">unsaved</span>
        <span id="verLabel_${{vid_id}}" style="font-size:11px;color:#888"></span>
        <span class="time-display">00:00.0 / ${{fmt(duration)}}</span>
    `;
    const zoomSlider = controlsBar.querySelector('.zoom-slider');
    const timeDisplay = controlsBar.querySelector('.time-display');

    const scrollContainer = document.createElement('div');
    scrollContainer.className = 'timeline-scroll';
    const content = document.createElement('div');
    content.className = 'timeline-content';
    const ruler = document.createElement('div');
    ruler.className = 'ruler';
    const filmstrip = document.createElement('div');
    filmstrip.className = 'filmstrip';
    const tracksContainer = document.createElement('div');
    const playhead = document.createElement('div');
    playhead.className = 'playhead-line';
    playhead.innerHTML = '<div class="playhead-handle"></div>';

    // --- Dirty indicator ---
    function updateDirtyIndicator() {{
        const el = controlsBar.querySelector('#dirty_' + vid_id);
        el.style.display = state.dirty ? 'inline' : 'none';
        if (typeof updateVersionLabel === 'function') updateVersionLabel();
    }}

    // --- Render ---
    function render() {{
        const tw = totalWidth();
        content.style.width = (tw + LABEL_W) + 'px';

        // Ruler ticks
        ruler.innerHTML = '';
        ruler.style.paddingLeft = LABEL_W + 'px';
        const tickInterval = state.zoom > 10 ? 5 : state.zoom > 3 ? 10 : 30;
        for (let t = 0; t <= duration; t += tickInterval) {{
            const tick = document.createElement('div');
            tick.className = 'ruler-tick';
            tick.style.left = timeToPx(t) + 'px';
            tick.textContent = fmt(t);
            ruler.appendChild(tick);
        }}

        // Filmstrip
        filmstrip.innerHTML = '';
        filmstrip.style.paddingLeft = LABEL_W + 'px';
        const thumbWidthPx = state.zoom * THUMB_INTERVAL;
        for (let i = 0; i < nThumbs; i++) {{
            const img = document.createElement('img');
            img.src = 'data:image/jpeg;base64,' + thumbs[i];
            img.style.width = Math.max(thumbWidthPx, 4) + 'px';
            img.style.height = '{THUMB_H}px';
            filmstrip.appendChild(img);
        }}

        // Tracks
        tracksContainer.innerHTML = '';
        types.forEach(etype => {{
            const track = document.createElement('div');
            track.className = 'track';
            const label = document.createElement('div');
            label.className = 'track-label';
            label.textContent = LABELS[etype] || etype;
            const bar = document.createElement('div');
            bar.className = 'track-bar';

            (tracks[etype] || []).forEach(ev => {{
                const span = document.createElement('div');
                span.className = 'event-span';
                if (state.selectedEvent && state.selectedEvent._id === ev._id) span.classList.add('selected');
                const left = ev.start * state.zoom;
                const width = Math.max((ev.end - ev.start) * state.zoom, 4);
                span.style.left = left + 'px';
                span.style.width = width + 'px';
                span.style.background = COLORS[etype] || '#666';

                // Resize handles on selected event
                if (state.selectedEvent && state.selectedEvent._id === ev._id) {{
                    const lh = document.createElement('div');
                    lh.className = 'resize-handle left';
                    lh.addEventListener('mousedown', (e) => {{ e.stopPropagation(); startResize(e, ev, 'left', span); }});
                    span.appendChild(lh);
                    const rh = document.createElement('div');
                    rh.className = 'resize-handle right';
                    rh.addEventListener('mousedown', (e) => {{ e.stopPropagation(); startResize(e, ev, 'right', span); }});
                    span.appendChild(rh);
                }}

                span.addEventListener('mouseenter', (e) => showTooltip(e, etype, ev));
                span.addEventListener('mousemove', (e) => moveTooltip(e));
                span.addEventListener('mouseleave', hideTooltip);
                span.addEventListener('click', (e) => {{
                    e.stopPropagation();
                    video.currentTime = ev.start;
                    selectEvent(etype, ev);
                }});
                span.addEventListener('contextmenu', (e) => {{
                    e.preventDefault(); e.stopPropagation();
                    showContextMenu(e.clientX, e.clientY, etype, ev);
                }});

                bar.appendChild(span);
            }});

            // Track bar click: seek or add event
            bar.addEventListener('click', (e) => {{
                if (e.target !== bar) return;
                const rect = bar.getBoundingClientRect();
                const clickTime = (e.clientX - rect.left) / state.zoom;
                if (state.editMode === 'add') {{
                    showAddEventModal(etype, clickTime);
                }} else {{
                    video.currentTime = clickTime;
                }}
            }});

            track.appendChild(label);
            track.appendChild(bar);
            tracksContainer.appendChild(track);
        }});
        updatePlayhead();
    }}

    function updatePlayhead() {{
        const t = video.currentTime || 0;
        playhead.style.left = timeToPx(t) + 'px';
        timeDisplay.textContent = fmt(t) + ' / ' + fmt(duration);
        updateActionOverlay();
    }}

    // --- Tooltip ---
    const tooltip = document.getElementById('tooltip');
    function showTooltip(e, etype, ev) {{
        const det = Object.entries(ev.details || {{}}).map(([k,v]) => k+': '+v).join(', ');
        tooltip.innerHTML = `
            <div style="font-weight:bold;color:${{COLORS[etype]||'#fff'}}">${{LABELS[etype]||etype}}</div>
            <div style="color:#888;font-size:11px">${{ev.timestamp_start}} — ${{ev.timestamp_end}}
            ${{ev.source ? '(' + ev.source + ')' : ''}}
            ${{ev.confidence ? ' conf:' + (ev.confidence*100).toFixed(0) + '%' : ''}}</div>
            <div style="margin-top:4px">${{ev.description}}</div>
            ${{det ? '<div style="margin-top:4px;color:#aaa;font-size:11px">'+det+'</div>' : ''}}`;
        tooltip.style.display = 'block';
        moveTooltip(e);
    }}
    function moveTooltip(e) {{ tooltip.style.left = (e.clientX+12)+'px'; tooltip.style.top = (e.clientY-10)+'px'; }}
    function hideTooltip() {{ tooltip.style.display = 'none'; }}

    // --- Event loop playback ---
    let loopEnd = null, loopRafId = null;
    function startEventLoop(s, e) {{
        loopEnd = e; video.currentTime = s; video.play();
        function check() {{
            if (loopEnd !== null && video.currentTime >= loopEnd) {{ video.pause(); video.currentTime = loopEnd; loopEnd = null; updatePlayhead(); return; }}
            if (!video.paused) loopRafId = requestAnimationFrame(check);
        }}
        loopRafId = requestAnimationFrame(check);
    }}
    function stopEventLoop() {{ loopEnd = null; if (loopRafId) cancelAnimationFrame(loopRafId); }}

    // --- Resize event duration ---
    function startResize(e, ev, side, spanEl) {{
        e.preventDefault();
        const startX = e.clientX;
        const origStart = ev.start, origEnd = ev.end;
        spanEl.classList.add('dragging');
        function onMove(e2) {{
            const dt = (e2.clientX - startX) / state.zoom;
            if (side === 'left') {{ ev.start = Math.max(0, Math.min(ev.end - 0.1, origStart + dt)); ev.timestamp_start = fmt(ev.start); }}
            else {{ ev.end = Math.max(ev.start + 0.1, Math.min(duration, origEnd + dt)); ev.timestamp_end = fmt(ev.end); }}
            spanEl.style.left = (ev.start * state.zoom) + 'px';
            spanEl.style.width = Math.max((ev.end - ev.start) * state.zoom, 4) + 'px';
        }}
        function onUp() {{
            spanEl.classList.remove('dragging');
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            state.dirty = true; updateDirtyIndicator();
            render();
        }}
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    }}

    // --- Select event ---
    function selectEvent(etype, ev) {{
        state.selectedEvent = ev;
        state.selectedEventType = etype;
        render(); // re-render for handles
        const det = ev.details || {{}};
        const detHtml = Object.entries(det).map(([k,v]) =>
            `<div class="detail-row"><span class="detail-label">${{k}}</span><div class="detail-value">${{v}}</div></div>`
        ).join('');
        detailsPanel.innerHTML = `
            <h3>Event Details</h3>
            <div class="detail-row"><span class="detail-label">Type</span>
                <div class="detail-value type-badge" style="background:${{COLORS[etype]||'#444'}}">${{LABELS[etype]||etype}}</div></div>
            <div class="detail-row"><span class="detail-label">Time</span>
                <div class="detail-value">${{ev.timestamp_start}} — ${{ev.timestamp_end}}</div></div>
            <div class="detail-row"><span class="detail-label">Confidence</span>
                <div class="detail-value">${{ev.confidence ? (ev.confidence*100).toFixed(0)+'%' : 'N/A'}}</div></div>
            <div class="detail-row"><span class="detail-label">Source</span>
                <div class="detail-value">${{ev.source || 'unknown'}}</div></div>
            <div class="detail-row"><span class="detail-label">Description</span>
                <div class="detail-value">${{ev.description}}</div></div>
            ${{detHtml}}
            <div style="margin-top:12px;display:flex;gap:6px;flex-wrap:wrap">
                <button class="ev-btn" id="playEventBtn">Play Event</button>
                <button class="ev-btn ev-btn-dim" id="playFromBtn">Play From Here</button>
                <button class="ev-btn" id="deleteEventBtn" style="background:#c0392b">Delete</button>
            </div>`;
        detailsPanel.querySelector('#playEventBtn').addEventListener('click', () => {{
            stopEventLoop(); startEventLoop(Math.max(0, ev.start-0.5), ev.end+0.3);
        }});
        detailsPanel.querySelector('#playFromBtn').addEventListener('click', () => {{
            stopEventLoop(); video.currentTime = ev.start; video.play();
        }});
        detailsPanel.querySelector('#deleteEventBtn').addEventListener('click', () => {{
            if (!confirm('Delete this event?')) return;
            deleteEvent(etype, ev);
        }});
    }}

    // --- Delete event ---
    function deleteEvent(etype, ev) {{
        const arr = tracks[etype];
        const idx = arr.findIndex(e => e._id === ev._id);
        if (idx !== -1) arr.splice(idx, 1);
        state.selectedEvent = null; state.selectedEventType = null;
        state.dirty = true; updateDirtyIndicator();
        detailsPanel.innerHTML = '<h3>Event Details</h3><div class="no-selection">Event deleted</div>';
        render();
    }}

    // --- Context menu ---
    function showContextMenu(x, y, etype, ev) {{
        document.querySelectorAll('.context-menu').forEach(m => m.remove());
        const menu = document.createElement('div');
        menu.className = 'context-menu';
        menu.style.left = x + 'px'; menu.style.top = y + 'px';
        const playItem = document.createElement('div');
        playItem.className = 'context-menu-item';
        playItem.textContent = 'Play Event';
        playItem.addEventListener('click', () => {{ menu.remove(); stopEventLoop(); startEventLoop(Math.max(0,ev.start-0.5), ev.end+0.3); }});
        const delItem = document.createElement('div');
        delItem.className = 'context-menu-item danger';
        delItem.textContent = 'Delete Event';
        delItem.addEventListener('click', () => {{ menu.remove(); if (confirm('Delete?')) deleteEvent(etype, ev); }});
        menu.appendChild(playItem); menu.appendChild(delItem);
        document.body.appendChild(menu);
        const close = () => {{ menu.remove(); document.removeEventListener('click', close); }};
        setTimeout(() => document.addEventListener('click', close), 0);
    }}

    // --- Add event modal ---
    function showAddEventModal(defaultType, clickTime) {{
        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        const opts = TYPE_ORDER.map(t => `<option value="${{t}}" ${{t===defaultType?'selected':''}}>${{LABELS[t]||t}}</option>`).join('');
        overlay.innerHTML = `<div class="modal">
            <h3>Add New Event</h3>
            <label>Event Type</label><select id="newType">${{opts}}</select>
            <label>Start Time (sec)</label><input type="number" id="newStart" value="${{clickTime.toFixed(1)}}" step="0.1" min="0" max="${{duration}}">
            <label>End Time (sec)</label><input type="number" id="newEnd" value="${{Math.min(duration,clickTime+2).toFixed(1)}}" step="0.1" min="0" max="${{duration}}">
            <label>Description</label><textarea id="newDesc" placeholder="What happens?"></textarea>
            <label>Details (key:value per line)</label><textarea id="newDet" placeholder="weapon: shotgun"></textarea>
            <div class="modal-btns">
                <button class="ev-btn ev-btn-dim" id="modalCancel">Cancel</button>
                <button class="ev-btn" id="modalOk">Add</button>
            </div></div>`;
        document.body.appendChild(overlay);
        overlay.querySelector('#modalCancel').addEventListener('click', () => overlay.remove());
        overlay.addEventListener('click', (e) => {{ if (e.target === overlay) overlay.remove(); }});
        overlay.querySelector('#modalOk').addEventListener('click', () => {{
            const etype = overlay.querySelector('#newType').value;
            const start = parseFloat(overlay.querySelector('#newStart').value);
            const end = parseFloat(overlay.querySelector('#newEnd').value);
            const desc = overlay.querySelector('#newDesc').value;
            const details = {{}};
            overlay.querySelector('#newDet').value.split('\\n').forEach(line => {{
                const i = line.indexOf(':');
                if (i > 0) details[line.slice(0,i).trim()] = line.slice(i+1).trim();
            }});
            const ev = {{ _id: ++_nextId, start, end, description: desc, details, confidence: 1.0,
                timestamp_start: fmt(start), timestamp_end: fmt(end), source: 'manual' }};
            if (!tracks[etype]) {{ tracks[etype] = []; if (!types.includes(etype)) types.push(etype); }}
            tracks[etype].push(ev);
            tracks[etype].sort((a,b) => a.start - b.start);
            state.dirty = true; updateDirtyIndicator();
            overlay.remove();
            state.editMode = null;
            const addBtn = controlsBar.querySelector('#addBtn_' + vid_id);
            addBtn.style.background = ''; addBtn.textContent = '+ Add';
            container.classList.remove('add-mode-active');
            render(); selectEvent(etype, ev);
        }});
    }}

    // --- Save ---
    function buildAnnotationJSON() {{
        const m = ep.ann_meta;
        const events = [];
        Object.keys(tracks).forEach(etype => {{
            (tracks[etype] || []).forEach(ev => {{
                events.push({{ event_type: etype, confidence: ev.confidence || 1.0,
                    description: ev.description || '', details: ev.details || {{}},
                    frame_start: Math.round(ev.start * GAME_FPS), frame_end: Math.round(ev.end * GAME_FPS),
                    timestamp_start: ev.timestamp_start, timestamp_end: ev.timestamp_end,
                    source: ev.source || 'manual' }});
            }});
        }});
        events.sort((a,b) => a.frame_start - b.frame_start);
        return {{ ...m, edited: true, edited_at: new Date().toISOString(), events }};
    }}
    function downloadJSON(str, filename) {{
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([str], {{type:'application/json'}}));
        a.download = filename; document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }}

    // --- Wire up controls bar buttons ---
    const addBtn = controlsBar.querySelector('#addBtn_' + vid_id);
    addBtn.addEventListener('click', () => {{
        state.editMode = state.editMode === 'add' ? null : 'add';
        addBtn.style.background = state.editMode === 'add' ? '#ff4757' : '';
        addBtn.textContent = state.editMode === 'add' ? 'Cancel' : '+ Add';
        container.classList.toggle('add-mode-active', state.editMode === 'add');
    }});
    // Version tracking
    let saveVersion = ep.next_version || 1;
    const verLabel = controlsBar.querySelector('#verLabel_' + vid_id);
    function updateVersionLabel() {{
        const cur = ep.current_version || 0;
        const label = cur === 0 ? 'original' : 'v' + cur;
        verLabel.textContent = state.dirty ? label + ' (editing)' : label;
    }}
    updateVersionLabel();

    controlsBar.querySelector('#saveBtn_' + vid_id).addEventListener('click', () => {{
        const data = buildAnnotationJSON();
        data.edit_version = saveVersion;
        data.parent_version = ep.current_version || 0;
        const filename = ep.ep_id + '_' + ep.player + '_v' + saveVersion + '.json';
        const jsonStr = JSON.stringify(data, null, 2);
        if (window.showSaveFilePicker) {{
            (async () => {{
                try {{
                    const h = await window.showSaveFilePicker({{ suggestedName: filename, types: [{{ accept: {{'application/json': ['.json']}} }}] }});
                    const w = await h.createWritable(); await w.write(jsonStr); await w.close();
                    ep.current_version = saveVersion; saveVersion++;
                    state.dirty = false; updateDirtyIndicator(); updateVersionLabel();
                }} catch(err) {{ if (err.name !== 'AbortError') {{ downloadJSON(jsonStr, filename);
                    ep.current_version = saveVersion; saveVersion++;
                    state.dirty = false; updateDirtyIndicator(); updateVersionLabel(); }} }}
            }})();
        }} else {{
            downloadJSON(jsonStr, filename);
            ep.current_version = saveVersion; saveVersion++;
            state.dirty = false; updateDirtyIndicator(); updateVersionLabel();
        }}
    }});

    // --- Scrubbing ---
    let scrubbing = false;
    function scrubFromEvent(e) {{
        const rect = scrollContainer.getBoundingClientRect();
        video.currentTime = pxToTime(e.clientX - rect.left + scrollContainer.scrollLeft);
        updatePlayhead();
    }}
    function startScrub(e) {{ scrubbing = true; video.pause(); scrubFromEvent(e); e.preventDefault(); }}
    function moveScrub(e) {{ if (scrubbing) scrubFromEvent(e); }}
    function stopScrub() {{ scrubbing = false; }}
    ruler.addEventListener('mousedown', startScrub);
    filmstrip.addEventListener('mousedown', startScrub);
    document.addEventListener('mousemove', moveScrub);
    document.addEventListener('mouseup', stopScrub);
    video.addEventListener('seeking', () => {{ if (scrubbing) stopEventLoop(); }});

    // --- Zoom ---
    zoomSlider.addEventListener('input', () => {{
        const ct = pxToTime(scrollContainer.scrollLeft + scrollContainer.clientWidth/2);
        state.zoom = parseFloat(zoomSlider.value);
        render();
        scrollContainer.scrollLeft = timeToPx(ct) - scrollContainer.clientWidth/2;
    }});
    scrollContainer.addEventListener('wheel', (e) => {{
        if (e.ctrlKey || e.metaKey) {{
            e.preventDefault();
            const rect = scrollContainer.getBoundingClientRect();
            const mx = e.clientX - rect.left + scrollContainer.scrollLeft;
            const mt = pxToTime(mx);
            state.zoom = Math.max(0.5, Math.min(30, state.zoom * (e.deltaY < 0 ? 1.15 : 0.87)));
            zoomSlider.value = state.zoom;
            render();
            scrollContainer.scrollLeft = timeToPx(mt) - (e.clientX - rect.left);
        }}
    }}, {{ passive: false }});

    // --- Playhead ---
    video.addEventListener('timeupdate', updatePlayhead);
    let rafId;
    video.addEventListener('play', () => {{
        function tick() {{
            updatePlayhead();
            const ph = timeToPx(video.currentTime);
            const sr = scrollContainer.scrollLeft + scrollContainer.clientWidth;
            if (ph > sr - 50 || ph < scrollContainer.scrollLeft + 120) scrollContainer.scrollLeft = ph - 200;
            if (!video.paused) rafId = requestAnimationFrame(tick);
        }}
        rafId = requestAnimationFrame(tick);
    }});
    video.addEventListener('pause', () => cancelAnimationFrame(rafId));

    // --- Assemble ---
    content.appendChild(ruler);
    content.appendChild(filmstrip);
    content.appendChild(tracksContainer);
    content.appendChild(playhead);
    scrollContainer.appendChild(content);
    timelineArea.appendChild(controlsBar);
    timelineArea.appendChild(scrollContainer);
    container.appendChild(topRow);
    container.appendChild(timelineArea);

    render();
    editors[vid_id] = {{ video, state, render, scrollContainer }};
}}

// --- Tab switching ---
document.addEventListener('DOMContentLoaded', () => {{
    const main = document.getElementById('main');
    const tabs = document.querySelectorAll('.tab');

    EPISODES.forEach((ep, i) => {{
        const container = document.createElement('div');
        container.className = 'episode-container' + (i === 0 ? ' active' : '');
        container.dataset.index = i;
        main.appendChild(container);
        initEditor(ep, container);
    }});

    tabs.forEach(tab => {{
        tab.addEventListener('click', () => {{
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('.episode-container').forEach(c => c.classList.remove('active'));
            const idx = parseInt(tab.dataset.index);
            document.querySelector(`.episode-container[data-index="${{idx}}"]`).classList.add('active');
        }});
    }});

    // Warn on unsaved changes
    window.addEventListener('beforeunload', (e) => {{
        const anyDirty = Object.values(editors).some(ed => ed.state && ed.state.dirty);
        if (anyDirty) {{ e.preventDefault(); e.returnValue = 'Unsaved changes'; }}
    }});
}});
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate NLE-style annotation viewer")
    parser.add_argument("--data-root", default="recordings")
    parser.add_argument("--ann-dir", default=None)
    parser.add_argument("--player", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ann_dir = args.ann_dir or os.path.join(args.data_root, "annotations")
    annotations = scan_annotations(args.data_root, ann_dir)
    if args.player:
        annotations = [a for a in annotations if a["player"] == args.player]

    if not annotations:
        print("No annotations found.")
        sys.exit(1)

    print(f"Found {len(annotations)} annotation(s)")
    for ann in annotations:
        p = ann["player"]
        is_human = ann.get("meta", {}).get(f"is_human_{p}", False)
        label = "human" if is_human else "AI"
        print(f"  {ann['episode_id'][:16]}... {p} ({label}): {len(ann.get('events', []))} events")

    output = args.output or os.path.join(ann_dir, "viewer.html")
    output_dir = os.path.dirname(output)
    html = generate_html(annotations, args.data_root, output_dir)
    with open(output, "w") as f:
        f.write(html)
    print(f"\nViewer written to {output}")
    print(f"File size: {os.path.getsize(output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
