#!/usr/bin/env python3
"""
Generate an interactive browser for evaluation clips.

Produces a standalone HTML that shows all eval clips grouped by category,
with video player, annotation details, and filtering.

Usage:
    python preprocessing/browse_eval_clips.py
    python preprocessing/browse_eval_clips.py --clips-dir datasets/eval_clips --port 9010
"""
import sys, os, json, argparse
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLIPS_DIR = "datasets/eval_clips"


def generate_html(clips_dir):
    index_path = os.path.join(clips_dir, "index.json")
    with open(index_path) as f:
        index = json.load(f)

    clips = index["clips"]
    categories = sorted(set(c["category"] for c in clips))

    # Build clips JSON for JS (without gemini_response to save space)
    clips_compact = []
    for c in clips:
        cc = {k: v for k, v in c.items() if k != "gemini_response"}
        # Build video path relative to the HTML file location
        cc["video_url"] = f'{c["category"]}/{c["clip_id"]}.mp4'
        clips_compact.append(cc)

    clips_json = json.dumps(clips_compact)
    categories_json = json.dumps(categories)

    cat_colors = {
        "weapon_switch": "#f39c12",
        "attack_firing": "#e74c3c",
        "attack_not_firing": "#3498db",
        "kill": "#ff4757",
        "death_respawn": "#c0392b",
    }
    colors_json = json.dumps(cat_colors)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Eval Clips Browser</title>
<style>
:root {{
    --bg: #0f0f1e;
    --bg-card: #1a1a2e;
    --bg-panel: #16213e;
    --border: #2a2a4a;
    --text: #e0e0e0;
    --text-dim: #8888aa;
    --accent: #ff4757;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
    height: 100vh;
    display: flex;
    flex-direction: column;
}}

/* Header */
.header {{
    padding: 12px 20px;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
}}
.header h1 {{ font-size: 18px; color: var(--accent); }}
.header .stats {{ color: var(--text-dim); font-size: 12px; }}

/* Filter bar */
.filters {{
    padding: 8px 20px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    flex-shrink: 0;
}}
.filter-btn {{
    padding: 4px 14px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: transparent;
    color: var(--text-dim);
    cursor: pointer;
    font-size: 12px;
    transition: all 0.15s;
}}
.filter-btn:hover {{ border-color: #555; color: var(--text); }}
.filter-btn.active {{
    border-color: var(--accent);
    color: #fff;
    background: rgba(255,71,87,0.15);
}}
.filter-count {{
    font-size: 10px;
    opacity: 0.7;
    margin-left: 4px;
}}

/* Main area */
.main {{
    flex: 1;
    display: flex;
    overflow: hidden;
}}

/* Clip list (left) */
.clip-list {{
    width: 340px;
    flex-shrink: 0;
    overflow-y: auto;
    border-right: 1px solid var(--border);
    background: var(--bg-card);
}}
.clip-item {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.1s;
}}
.clip-item:hover {{ background: rgba(255,255,255,0.03); }}
.clip-item.active {{ background: rgba(255,71,87,0.1); border-left: 3px solid var(--accent); }}
.clip-item .clip-id {{
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 2px;
}}
.clip-item .clip-cat {{
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    margin-right: 6px;
}}
.clip-item .clip-desc {{
    color: var(--text-dim);
    font-size: 11px;
    margin-top: 3px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

/* Detail panel (right) */
.detail-panel {{
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}}
.video-area {{
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    min-height: 300px;
    max-height: 480px;
    flex-shrink: 0;
}}
.video-area video {{
    max-width: 100%;
    max-height: 100%;
}}
.info-area {{
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
}}
.info-area h3 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-bottom: 10px;
}}
.info-grid {{
    display: grid;
    grid-template-columns: 120px 1fr;
    gap: 6px 12px;
    margin-bottom: 16px;
}}
.info-label {{
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
}}
.info-value {{
    font-size: 13px;
    word-break: break-word;
}}
.gt-section {{
    background: var(--bg-panel);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
}}
.gt-section h4 {{
    font-size: 12px;
    margin-bottom: 8px;
    color: var(--accent);
}}
.no-clip {{
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-dim);
    font-style: italic;
}}

/* Action overlay on video */
.video-wrapper {{
    position: relative;
    display: inline-block;
}}
</style>
</head>
<body>

<div class="header">
    <h1>Eval Clips</h1>
    <span class="stats" id="statsLabel"></span>
</div>

<div class="filters" id="filters"></div>

<div class="main">
    <div class="clip-list" id="clipList"></div>
    <div class="detail-panel" id="detailPanel">
        <div class="no-clip">Select a clip from the list</div>
    </div>
</div>

<script>
const CLIPS = {clips_json};
const CATEGORIES = {categories_json};
const CAT_COLORS = {colors_json};

let activeFilters = new Set(CATEGORIES);
let activeClipIdx = null;

// Stats
document.getElementById('statsLabel').textContent =
    CLIPS.length + ' clips across ' + CATEGORIES.length + ' categories';

// Filters
const filtersEl = document.getElementById('filters');
// "All" button
const allBtn = document.createElement('button');
allBtn.className = 'filter-btn active';
allBtn.textContent = 'All (' + CLIPS.length + ')';
allBtn.addEventListener('click', () => {{
    activeFilters = new Set(CATEGORIES);
    updateFilters();
    renderList();
}});
filtersEl.appendChild(allBtn);

CATEGORIES.forEach(cat => {{
    const count = CLIPS.filter(c => c.category === cat).length;
    const btn = document.createElement('button');
    btn.className = 'filter-btn active';
    btn.dataset.cat = cat;
    btn.innerHTML = cat.replace(/_/g, ' ') + '<span class="filter-count">(' + count + ')</span>';
    btn.style.borderColor = CAT_COLORS[cat] || '#555';
    btn.addEventListener('click', () => {{
        if (activeFilters.has(cat)) activeFilters.delete(cat);
        else activeFilters.add(cat);
        updateFilters();
        renderList();
    }});
    filtersEl.appendChild(btn);
}});

function updateFilters() {{
    filtersEl.querySelectorAll('.filter-btn').forEach(btn => {{
        const cat = btn.dataset.cat;
        if (!cat) {{ // "All" button
            btn.classList.toggle('active', activeFilters.size === CATEGORIES.length);
        }} else {{
            btn.classList.toggle('active', activeFilters.has(cat));
        }}
    }});
}}

// Clip list
const clipListEl = document.getElementById('clipList');
function renderList() {{
    clipListEl.innerHTML = '';
    CLIPS.forEach((clip, idx) => {{
        if (!activeFilters.has(clip.category)) return;
        const item = document.createElement('div');
        item.className = 'clip-item' + (idx === activeClipIdx ? ' active' : '');
        const color = CAT_COLORS[clip.category] || '#666';
        const gt = clip.ground_truth || {{}};
        let desc = '';
        if (clip.category === 'weapon_switch')
            desc = (gt.weapon_from || '?') + ' → ' + (gt.weapon_to || '?');
        else if (clip.category === 'attack_firing')
            desc = gt.weapon_type || 'unknown weapon';
        else if (clip.category === 'attack_not_firing')
            desc = (gt.weapon_type || 'unknown') + ' (idle)';
        else if (clip.category === 'kill')
            desc = gt.description || 'frag';
        else if (clip.category === 'death_respawn')
            desc = gt.description || gt.event_type || 'death/respawn';

        item.innerHTML = `
            <div class="clip-id">${{clip.clip_id}}</div>
            <span class="clip-cat" style="background:${{color}}33;color:${{color}}">${{clip.category.replace(/_/g,' ')}}</span>
            <span style="font-size:11px;color:var(--text-dim)">${{clip.duration_sec}}s</span>
            <div class="clip-desc">${{desc}}</div>
        `;
        item.addEventListener('click', () => selectClip(idx));
        clipListEl.appendChild(item);
    }});
}}

// Detail panel
const detailPanel = document.getElementById('detailPanel');
let currentVideo = null;

function selectClip(idx) {{
    activeClipIdx = idx;
    renderList();
    const clip = CLIPS[idx];
    const gt = clip.ground_truth || {{}};
    const color = CAT_COLORS[clip.category] || '#666';

    const gtHtml = Object.entries(gt).map(([k, v]) =>
        `<div class="info-label">${{k}}</div><div class="info-value">${{v}}</div>`
    ).join('');

    detailPanel.innerHTML = `
        <div class="video-area">
            <video id="clipVideo" controls autoplay loop>
                <source src="${{clip.video_url}}" type="video/mp4">
            </video>
        </div>
        <div class="info-area">
            <h3>Clip Info</h3>
            <div class="info-grid">
                <div class="info-label">Clip ID</div>
                <div class="info-value">${{clip.clip_id}}</div>
                <div class="info-label">Category</div>
                <div class="info-value" style="color:${{color}};font-weight:600">${{clip.category.replace(/_/g,' ')}}</div>
                <div class="info-label">Duration</div>
                <div class="info-value">${{clip.duration_sec}}s (${{clip.duration_frames}} frames)</div>
                <div class="info-label">Source</div>
                <div class="info-value">${{clip.source_episode.slice(0,12)}}... ${{clip.source_player}}</div>
                <div class="info-label">Frames</div>
                <div class="info-value">${{clip.frame_start}} — ${{clip.frame_end}}</div>
                <div class="info-label">Verified</div>
                <div class="info-value">${{clip.gemini_verified ? 'Yes' : 'No'}}</div>
            </div>
            <div class="gt-section">
                <h4>Ground Truth</h4>
                <div class="info-grid">${{gtHtml}}</div>
            </div>
        </div>
    `;

    // Keyboard nav
    currentVideo = document.getElementById('clipVideo');
}}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {{
    const filtered = CLIPS.map((c, i) => ({{ idx: i, cat: c.category }})).filter(x => activeFilters.has(x.cat));
    if (!filtered.length) return;
    const curPos = filtered.findIndex(x => x.idx === activeClipIdx);

    if (e.key === 'ArrowDown' || e.key === 'j') {{
        e.preventDefault();
        const next = curPos < filtered.length - 1 ? curPos + 1 : 0;
        selectClip(filtered[next].idx);
    }} else if (e.key === 'ArrowUp' || e.key === 'k') {{
        e.preventDefault();
        const prev = curPos > 0 ? curPos - 1 : filtered.length - 1;
        selectClip(filtered[prev].idx);
    }} else if (e.key === ' ' && currentVideo) {{
        e.preventDefault();
        currentVideo.paused ? currentVideo.play() : currentVideo.pause();
    }}
}});

// Init
renderList();
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Browse eval clips")
    parser.add_argument("--clips-dir", default=CLIPS_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--port", type=int, default=None, help="Start HTTP server on this port")
    args = parser.parse_args()

    output = args.output or os.path.join(args.clips_dir, "browser.html")
    html = generate_html(args.clips_dir)
    with open(output, "w") as f:
        f.write(html)
    print(f"Browser written to {output} ({os.path.getsize(output) / 1e3:.0f} KB)")

    if args.port:
        import subprocess
        print(f"Serving at http://localhost:{args.port}/browser.html")
        os.chdir(args.clips_dir)
        subprocess.run(["python", "-m", "http.server", str(args.port)])


if __name__ == "__main__":
    main()
