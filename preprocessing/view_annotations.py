#!/usr/bin/env python3
"""
Generate an interactive HTML annotation viewer for Doom gameplay episodes.

Embeds the MP4 video with a timeline below showing annotation spans grouped
by event type (one track per type). Hovering shows event details, clicking
seeks the video to that moment.

Usage:
    python preprocessing/view_annotations.py --data-root recordings
    python preprocessing/view_annotations.py --data-root recordings --player p1 --episode 0
"""
import sys, os, io, json, argparse, tarfile, base64
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

GAME_FPS = 35

# Colors per event type
EVENT_COLORS = {
    "weapon_pickup": "#e74c3c",
    "health_pickup": "#2ecc71",
    "armor_pickup": "#3498db",
    "weapon_switch": "#f39c12",
    "shooting": "#e67e22",
    "combat": "#9b59b6",
    "frag": "#e74c3c",
    "death": "#c0392b",
    "respawn": "#1abc9c",
    "other": "#95a5a6",
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


def scan_annotations(data_root: str) -> list[dict]:
    """Find all annotation JSONs and their corresponding videos."""
    ann_dir = os.path.join(data_root, "annotations")
    if not os.path.isdir(ann_dir):
        return []

    results = []
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".json"):
            continue
        ann_path = os.path.join(ann_dir, fname)
        with open(ann_path) as f:
            ann = json.load(f)
        results.append(ann)
    return results


def extract_video_file(shard_path: str, member_name: str, output_dir: str) -> str:
    """Extract MP4 from tar to output_dir. Returns relative path."""
    # Use a short filename based on episode id and player
    out_name = member_name.replace("/", "_")
    out_path = os.path.join(output_dir, out_name)
    if not os.path.exists(out_path):
        with tarfile.open(shard_path, "r") as tar:
            data = tar.extractfile(tar.getmember(member_name)).read()
        with open(out_path, "wb") as f:
            f.write(data)
    return out_name


def generate_html(annotations: list[dict], data_root: str, output_dir: str) -> str:
    """Generate HTML viewer with extracted video files and annotation timelines."""

    episodes_html = []
    for ann in annotations:
        ep_id = ann["episode_id"]
        player = ann["player"]
        n_frames = ann["n_frames"]
        duration = n_frames / GAME_FPS
        events = ann.get("events", [])
        meta = ann.get("meta", {})

        # Extract video to file
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

        # Group events by type
        event_types_present = []
        seen = set()
        for ev in events:
            t = ev["event_type"]
            if t not in seen:
                event_types_present.append(t)
                seen.add(t)

        # Build track data as JSON for JS
        tracks_data = {}
        for etype in event_types_present:
            track_events = [e for e in events if e["event_type"] == etype]
            tracks_data[etype] = [{
                "start": e.get("frame_start", 0) / GAME_FPS,
                "end": e.get("frame_end", e.get("frame_start", 0)) / GAME_FPS,
                "description": e.get("description", ""),
                "details": e.get("details", {}),
                "confidence": e.get("confidence", 0),
                "timestamp_start": e.get("timestamp_start", ""),
                "timestamp_end": e.get("timestamp_end", ""),
            } for e in track_events]

        vid_id = f"vid_{ep_id[:8]}_{player}".replace("-", "")
        tracks_json = json.dumps(tracks_data)
        colors_json = json.dumps(EVENT_COLORS)
        labels_json = json.dumps(EVENT_LABELS)

        frag = meta.get(f"frag_{player}", "?")
        death = meta.get(f"death_{player}", "?")
        scenario = meta.get("scenario", "?")
        is_human = meta.get(f"is_human_{player}", False)
        player_label = f"{player.upper()} ({'human' if is_human else 'AI'})"
        passes = ann.get("passes_completed", [])

        episodes_html.append(f"""
        <div class="episode">
            <h2>{player_label} — {scenario} — {duration:.0f}s — frags: {frag}, deaths: {death}</h2>
            <p class="meta">Episode: {ep_id[:16]}... | Passes: {', '.join(passes)} | Events: {len(events)}</p>
            <div class="player-container">
                <video id="{vid_id}" width="640" height="480" controls>
                    <source src="{video_filename}" type="video/mp4">
                </video>
                <div class="timeline-container" id="timeline_{vid_id}"></div>
                <div class="tooltip" id="tooltip_{vid_id}"></div>
            </div>
            <script>
                document.addEventListener("DOMContentLoaded", function() {{
                    initTimeline("{vid_id}", {tracks_json}, {colors_json}, {labels_json}, {duration});
                }});
            </script>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Doom Gameplay Annotations</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: #0a0a1a;
        color: #e0e0e0;
        font-family: 'Segoe UI', system-ui, sans-serif;
        padding: 20px;
    }}
    h1 {{
        color: #ff6b6b;
        margin-bottom: 20px;
        font-size: 24px;
    }}
    .episode {{
        margin-bottom: 40px;
        background: #12122a;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #2a2a4a;
    }}
    .episode h2 {{
        color: #ccc;
        font-size: 16px;
        margin-bottom: 6px;
    }}
    .meta {{
        color: #666;
        font-size: 12px;
        margin-bottom: 12px;
    }}
    .player-container {{
        position: relative;
    }}
    video {{
        display: block;
        border-radius: 4px;
        margin-bottom: 8px;
        background: #000;
    }}
    .timeline-container {{
        width: 640px;
        background: #1a1a2e;
        border-radius: 4px;
        padding: 4px 0;
        position: relative;
    }}
    .track {{
        display: flex;
        align-items: center;
        height: 22px;
        margin: 1px 0;
        position: relative;
    }}
    .track-label {{
        width: 100px;
        font-size: 10px;
        color: #888;
        text-align: right;
        padding-right: 8px;
        flex-shrink: 0;
        white-space: nowrap;
        overflow: hidden;
    }}
    .track-bar {{
        flex: 1;
        height: 16px;
        position: relative;
        background: #0d0d1a;
        border-radius: 2px;
        cursor: pointer;
    }}
    .event-span {{
        position: absolute;
        height: 100%;
        border-radius: 2px;
        opacity: 0.8;
        min-width: 3px;
        transition: opacity 0.15s;
    }}
    .event-span:hover {{
        opacity: 1.0;
        z-index: 10;
        box-shadow: 0 0 6px rgba(255,255,255,0.3);
    }}
    .playhead {{
        position: absolute;
        top: 0;
        width: 1px;
        height: 100%;
        background: #fff;
        pointer-events: none;
        z-index: 20;
        opacity: 0.7;
    }}
    .tooltip {{
        display: none;
        position: absolute;
        background: #1a1a3a;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 12px;
        color: #ddd;
        z-index: 100;
        max-width: 350px;
        pointer-events: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    .tooltip .tt-type {{
        font-weight: bold;
        margin-bottom: 4px;
    }}
    .tooltip .tt-time {{
        color: #888;
        font-size: 11px;
    }}
    .tooltip .tt-desc {{
        margin-top: 4px;
    }}
    .tooltip .tt-details {{
        margin-top: 4px;
        color: #aaa;
        font-size: 11px;
    }}
</style>
</head>
<body>
<h1>Doom Gameplay Annotations</h1>
{''.join(episodes_html)}

<script>
function initTimeline(vidId, tracks, colors, labels, duration) {{
    const video = document.getElementById(vidId);
    const container = document.getElementById('timeline_' + vidId);
    const tooltip = document.getElementById('tooltip_' + vidId);

    // Sort event types by typical order
    const typeOrder = ['weapon_pickup','health_pickup','armor_pickup','weapon_switch',
                       'shooting','combat','frag','death','respawn','other'];
    const types = Object.keys(tracks).sort((a,b) =>
        (typeOrder.indexOf(a) === -1 ? 99 : typeOrder.indexOf(a)) -
        (typeOrder.indexOf(b) === -1 ? 99 : typeOrder.indexOf(b))
    );

    // Create playhead
    const playheadData = {{}};

    types.forEach(etype => {{
        const track = document.createElement('div');
        track.className = 'track';

        const label = document.createElement('div');
        label.className = 'track-label';
        label.textContent = labels[etype] || etype;
        track.appendChild(label);

        const bar = document.createElement('div');
        bar.className = 'track-bar';

        // Add playhead to each bar
        const playhead = document.createElement('div');
        playhead.className = 'playhead';
        playhead.style.left = '0%';
        bar.appendChild(playhead);
        if (!playheadData[vidId]) playheadData[vidId] = [];
        playheadData[vidId].push(playhead);

        tracks[etype].forEach(ev => {{
            const span = document.createElement('div');
            span.className = 'event-span';
            const leftPct = (ev.start / duration) * 100;
            const widthPct = Math.max(((ev.end - ev.start) / duration) * 100, 0.5);
            span.style.left = leftPct + '%';
            span.style.width = widthPct + '%';
            span.style.background = colors[etype] || '#666';

            span.addEventListener('mouseenter', (e) => {{
                const det = Object.entries(ev.details || {{}}).map(([k,v]) => k + ': ' + v).join(', ');
                tooltip.innerHTML = '<div class="tt-type" style="color:' + (colors[etype]||'#fff') + '">'
                    + (labels[etype]||etype) + '</div>'
                    + '<div class="tt-time">' + ev.timestamp_start + ' — ' + ev.timestamp_end
                    + ' (conf: ' + (ev.confidence*100).toFixed(0) + '%)</div>'
                    + '<div class="tt-desc">' + ev.description + '</div>'
                    + (det ? '<div class="tt-details">' + det + '</div>' : '');
                tooltip.style.display = 'block';
            }});
            span.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.clientX - container.getBoundingClientRect().left + 10) + 'px';
                tooltip.style.top = (e.clientY - container.getBoundingClientRect().top - 60) + 'px';
            }});
            span.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});
            span.addEventListener('click', () => {{
                video.currentTime = ev.start;
                video.play();
            }});

            bar.appendChild(span);
        }});

        // Click on empty bar area to seek
        bar.addEventListener('click', (e) => {{
            if (e.target === bar) {{
                const rect = bar.getBoundingClientRect();
                const pct = (e.clientX - rect.left) / rect.width;
                video.currentTime = pct * duration;
            }}
        }});

        track.appendChild(bar);
        container.appendChild(track);
    }});

    // Update playheads
    video.addEventListener('timeupdate', () => {{
        const pct = (video.currentTime / duration) * 100;
        (playheadData[vidId] || []).forEach(ph => {{
            ph.style.left = pct + '%';
        }});
    }});
}}
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate annotation viewer HTML")
    parser.add_argument("--data-root", default="recordings", help="Directory with shards and annotations/")
    parser.add_argument("--player", default=None, help="Filter to player (p1 or p2)")
    parser.add_argument("--output", default=None, help="Output HTML path (default: {data-root}/annotations/viewer.html)")
    args = parser.parse_args()

    annotations = scan_annotations(args.data_root)
    if args.player:
        annotations = [a for a in annotations if a["player"] == args.player]

    if not annotations:
        print("No annotations found. Run annotate_episodes.py first.")
        sys.exit(1)

    print(f"Found {len(annotations)} annotation(s)")
    for ann in annotations:
        p = ann["player"]
        is_human = ann.get("meta", {}).get(f"is_human_{p}", False)
        label = "human" if is_human else "AI"
        print(f"  {ann['episode_id'][:16]}... {p} ({label}): {len(ann.get('events', []))} events")

    output = args.output or os.path.join(args.data_root, "annotations", "viewer.html")
    output_dir = os.path.dirname(output)
    html = generate_html(annotations, args.data_root, output_dir)
    with open(output, "w") as f:
        f.write(html)
    print(f"\nViewer written to {output}")
    print(f"File size: {os.path.getsize(output) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
