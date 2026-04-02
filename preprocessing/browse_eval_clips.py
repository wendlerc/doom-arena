#!/usr/bin/env python3
"""
Interactive eval clips browser with editing support.

Serves a web UI to browse, edit, delete, and re-categorize eval clips.
Uses FastAPI for the backend so edits are saved to disk immediately.

Usage:
    python preprocessing/browse_eval_clips.py
    python preprocessing/browse_eval_clips.py --clips-dir datasets/eval_clips --port 9005
"""
import sys, os, json, argparse, shutil, time
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLIPS_DIR = "datasets/eval_clips"
PORT = 9005

# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------

def make_app(clips_dir):
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    from typing import Optional

    app = FastAPI()

    index_path = os.path.join(clips_dir, "index.json")

    def load_index():
        with open(index_path) as f:
            return json.load(f)

    def save_index(data):
        # Version the index before overwriting
        backup = index_path + f".bak.{int(time.time())}"
        shutil.copy2(index_path, backup)
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_clip_annotation(clip):
        """Save individual clip JSON file."""
        clip_dir = os.path.join(clips_dir, clip["category"])
        clip_json = os.path.join(clip_dir, f'{clip["clip_id"]}.json')
        with open(clip_json, "w") as f:
            json.dump(clip, f, indent=2)

    # --- API routes ---

    @app.get("/api/clips")
    def get_clips():
        index = load_index()
        clips = []
        for c in index["clips"]:
            cc = {k: v for k, v in c.items() if k != "gemini_response"}
            cc["video_url"] = f'/files/{c["category"]}/{c["clip_id"]}.mp4'
            clips.append(cc)
        return {"clips": clips, "categories": index.get("categories", {})}

    class ClipUpdate(BaseModel):
        clip_id: str
        category: Optional[str] = None
        ground_truth: Optional[dict] = None
        gemini_verified: Optional[bool] = None
        notes: Optional[str] = None

    @app.post("/api/clips/update")
    def update_clip(update: ClipUpdate):
        index = load_index()
        clip = None
        for c in index["clips"]:
            if c["clip_id"] == update.clip_id:
                clip = c
                break
        if not clip:
            raise HTTPException(404, f"Clip {update.clip_id} not found")

        if update.category is not None and update.category != clip["category"]:
            # Move files to new category dir
            old_cat = clip["category"]
            new_cat = update.category
            os.makedirs(os.path.join(clips_dir, new_cat), exist_ok=True)
            for ext in [".mp4", "_actions.npy", ".json"]:
                old_path = os.path.join(clips_dir, old_cat, f'{clip["clip_id"]}{ext}')
                new_path = os.path.join(clips_dir, new_cat, f'{clip["clip_id"]}{ext}')
                if os.path.exists(old_path):
                    shutil.move(old_path, new_path)
            clip["category"] = new_cat

        if update.ground_truth is not None:
            clip["ground_truth"] = update.ground_truth
        if update.gemini_verified is not None:
            clip["gemini_verified"] = update.gemini_verified
        if update.notes is not None:
            clip["notes"] = update.notes
        clip["edited_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        save_clip_annotation(clip)
        save_index(index)
        return {"status": "ok", "clip_id": update.clip_id}

    @app.post("/api/clips/delete")
    def delete_clip(body: dict):
        clip_id = body.get("clip_id")
        if not clip_id:
            raise HTTPException(400, "clip_id required")

        index = load_index()
        clip = None
        for i, c in enumerate(index["clips"]):
            if c["clip_id"] == clip_id:
                clip = c
                index["clips"].pop(i)
                break
        if not clip:
            raise HTTPException(404, f"Clip {clip_id} not found")

        # Move files to _deleted/ instead of actually deleting
        deleted_dir = os.path.join(clips_dir, "_deleted")
        os.makedirs(deleted_dir, exist_ok=True)
        cat = clip["category"]
        for ext in [".mp4", "_actions.npy", ".json"]:
            src = os.path.join(clips_dir, cat, f"{clip_id}{ext}")
            if os.path.exists(src):
                shutil.move(src, os.path.join(deleted_dir, f"{clip_id}{ext}"))

        # Update category counts
        from collections import Counter
        index["categories"] = dict(Counter(c["category"] for c in index["clips"]))
        index["total_clips"] = len(index["clips"])
        save_index(index)
        return {"status": "deleted", "clip_id": clip_id}

    # --- Static files ---
    app.mount("/files", StaticFiles(directory=clips_dir), name="files")

    # --- HTML UI ---
    @app.get("/", response_class=HTMLResponse)
    def get_ui():
        return generate_html()

    return app


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html():
    cat_colors = {
        "weapon_switch": "#f39c12",
        "attack_firing": "#e74c3c",
        "attack_not_firing": "#3498db",
        "kill": "#ff4757",
        "death_respawn": "#c0392b",
        "weapon_pickup": "#27ae60",
    }
    colors_json = json.dumps(cat_colors)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Eval Clips Browser</title>
<style>
:root {{
    --bg: #0f0f1e; --bg-card: #1a1a2e; --bg-panel: #16213e;
    --border: #2a2a4a; --text: #e0e0e0; --text-dim: #8888aa; --accent: #ff4757;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
       font-size: 13px; height: 100vh; display: flex; flex-direction: column; }}
.header {{ padding: 12px 20px; background: var(--bg-card); border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
.header h1 {{ font-size: 18px; color: var(--accent); }}
.header .stats {{ color: var(--text-dim); font-size: 12px; }}
.filters {{ padding: 8px 20px; background: var(--bg-panel); border-bottom: 1px solid var(--border);
            display: flex; gap: 8px; flex-wrap: wrap; flex-shrink: 0; }}
.filter-btn {{ padding: 4px 14px; border: 1px solid var(--border); border-radius: 16px;
               background: transparent; color: var(--text-dim); cursor: pointer; font-size: 12px;
               transition: all 0.15s; }}
.filter-btn:hover {{ border-color: #555; color: var(--text); }}
.filter-btn.active {{ border-color: var(--accent); color: #fff; background: rgba(255,71,87,0.15); }}
.filter-count {{ font-size: 10px; opacity: 0.7; margin-left: 4px; }}
.main {{ flex: 1; display: flex; overflow: hidden; }}
.clip-list {{ width: 340px; flex-shrink: 0; overflow-y: auto; border-right: 1px solid var(--border);
              background: var(--bg-card); }}
.clip-item {{ padding: 10px 14px; border-bottom: 1px solid var(--border); cursor: pointer;
              transition: background 0.1s; }}
.clip-item:hover {{ background: rgba(255,255,255,0.03); }}
.clip-item.active {{ background: rgba(255,71,87,0.1); border-left: 3px solid var(--accent); }}
.clip-item .clip-id {{ font-weight: 600; font-size: 13px; margin-bottom: 2px; }}
.clip-item .clip-cat {{ display: inline-block; padding: 1px 8px; border-radius: 10px;
                        font-size: 10px; font-weight: 600; margin-right: 6px; }}
.clip-item .clip-desc {{ color: var(--text-dim); font-size: 11px; margin-top: 3px;
                         overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.clip-item.deleted {{ opacity: 0.3; text-decoration: line-through; }}
.detail-panel {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
.video-area {{ display: flex; align-items: center; justify-content: center;
               background: #000; min-height: 300px; max-height: 480px; flex-shrink: 0; }}
.video-area video {{ max-width: 100%; max-height: 100%; }}
.info-area {{ flex: 1; overflow-y: auto; padding: 16px 20px; }}
.info-area h3 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
                 color: var(--text-dim); margin-bottom: 10px; }}
.info-grid {{ display: grid; grid-template-columns: 120px 1fr; gap: 6px 12px; margin-bottom: 16px; }}
.info-label {{ font-size: 11px; color: var(--text-dim); text-transform: uppercase; }}
.info-value {{ font-size: 13px; word-break: break-word; }}
.gt-section {{ background: var(--bg-panel); border-radius: 6px; padding: 12px; margin-bottom: 12px; }}
.gt-section h4 {{ font-size: 12px; margin-bottom: 8px; color: var(--accent); }}
.no-clip {{ display: flex; align-items: center; justify-content: center;
            height: 100%; color: var(--text-dim); font-style: italic; }}
/* Buttons */
.btn {{ padding: 5px 14px; border: none; border-radius: 4px; cursor: pointer;
        font-size: 12px; font-weight: 600; transition: opacity 0.15s; color: #fff; }}
.btn:hover {{ opacity: 0.85; }}
.btn-danger {{ background: #c0392b; }}
.btn-save {{ background: #27ae60; }}
.btn-secondary {{ background: #555; }}
.btn-group {{ display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; }}
/* Editable fields */
.edit-field {{ width: 100%; padding: 5px 8px; background: var(--bg); color: var(--text);
               border: 1px solid var(--border); border-radius: 4px; font-size: 12px;
               font-family: inherit; }}
.edit-field:focus {{ border-color: var(--accent); outline: none; }}
select.edit-field {{ cursor: pointer; }}
textarea.edit-field {{ resize: vertical; min-height: 50px; }}
/* Toast */
.toast {{ position: fixed; bottom: 20px; right: 20px; padding: 10px 20px;
          background: #27ae60; color: #fff; border-radius: 6px; font-size: 13px;
          z-index: 1000; opacity: 0; transition: opacity 0.3s; pointer-events: none; }}
.toast.show {{ opacity: 1; }}
.toast.error {{ background: #c0392b; }}
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
<div class="toast" id="toast"></div>

<script>
const CAT_COLORS = {colors_json};
const ALL_CATEGORIES = ['weapon_switch','attack_firing','attack_not_firing','kill','death_respawn','weapon_pickup'];

let CLIPS = [];
let activeFilters = new Set();
let activeClipIdx = null;
let currentVideo = null;

// Toast
function showToast(msg, isError) {{
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'toast show' + (isError ? ' error' : '');
    setTimeout(() => t.className = 'toast', 2000);
}}

// API helpers
async function apiGet(url) {{
    const r = await fetch(url);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
}}
async function apiPost(url, body) {{
    const r = await fetch(url, {{
        method: 'POST', headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(body)
    }});
    if (!r.ok) throw new Error(await r.text());
    return r.json();
}}

// Load clips from API
async function loadClips() {{
    const data = await apiGet('/api/clips');
    CLIPS = data.clips;
    const cats = [...new Set(CLIPS.map(c => c.category))];
    activeFilters = new Set(cats);
    buildFilters(cats);
    renderList();
    document.getElementById('statsLabel').textContent =
        CLIPS.length + ' clips across ' + cats.length + ' categories';
}}

// Filters
function buildFilters(categories) {{
    const el = document.getElementById('filters');
    el.innerHTML = '';
    const allBtn = document.createElement('button');
    allBtn.className = 'filter-btn active';
    allBtn.textContent = 'All (' + CLIPS.length + ')';
    allBtn.addEventListener('click', () => {{
        activeFilters = new Set(categories);
        updateFilterBtns(); renderList();
    }});
    el.appendChild(allBtn);
    categories.forEach(cat => {{
        const count = CLIPS.filter(c => c.category === cat).length;
        const btn = document.createElement('button');
        btn.className = 'filter-btn active';
        btn.dataset.cat = cat;
        btn.innerHTML = cat.replace(/_/g, ' ') + '<span class="filter-count">(' + count + ')</span>';
        btn.style.borderColor = CAT_COLORS[cat] || '#555';
        btn.addEventListener('click', () => {{
            if (activeFilters.has(cat)) activeFilters.delete(cat); else activeFilters.add(cat);
            updateFilterBtns(); renderList();
        }});
        el.appendChild(btn);
    }});
}}
function updateFilterBtns() {{
    document.querySelectorAll('.filter-btn').forEach(btn => {{
        const cat = btn.dataset.cat;
        if (!cat) btn.classList.toggle('active', activeFilters.size >= ALL_CATEGORIES.length);
        else btn.classList.toggle('active', activeFilters.has(cat));
    }});
}}

// Clip list
function renderList() {{
    const el = document.getElementById('clipList');
    el.innerHTML = '';
    CLIPS.forEach((clip, idx) => {{
        if (!activeFilters.has(clip.category)) return;
        const item = document.createElement('div');
        item.className = 'clip-item' + (idx === activeClipIdx ? ' active' : '');
        const color = CAT_COLORS[clip.category] || '#666';
        const gt = clip.ground_truth || {{}};
        let desc = '';
        if (clip.category === 'weapon_switch') desc = (gt.weapon_from||'?') + ' -> ' + (gt.weapon_to||'?');
        else if (clip.category === 'attack_firing') desc = gt.weapon_type || '?';
        else if (clip.category === 'attack_not_firing') desc = (gt.weapon_type||'?') + ' (idle)';
        else if (clip.category === 'kill') desc = gt.description || 'frag';
        else if (clip.category === 'death_respawn') desc = gt.description || gt.event_type || '';
        else if (clip.category === 'weapon_pickup') desc = gt.weapon_picked_up || '?';
        if (clip.notes) desc = '[note] ' + desc;

        item.innerHTML = `
            <div class="clip-id">${{clip.clip_id}}</div>
            <span class="clip-cat" style="background:${{color}}33;color:${{color}}">${{clip.category.replace(/_/g,' ')}}</span>
            <span style="font-size:11px;color:var(--text-dim)">${{clip.duration_sec}}s</span>
            <div class="clip-desc">${{desc}}</div>`;
        item.addEventListener('click', () => selectClip(idx));
        el.appendChild(item);
    }});
}}

// Detail panel with editing
function selectClip(idx) {{
    activeClipIdx = idx;
    renderList();
    const clip = CLIPS[idx];
    const gt = clip.ground_truth || {{}};
    const color = CAT_COLORS[clip.category] || '#666';

    const catOptions = ALL_CATEGORIES.map(c =>
        `<option value="${{c}}" ${{c === clip.category ? 'selected' : ''}}>${{c.replace(/_/g,' ')}}</option>`
    ).join('');

    const gtFields = Object.entries(gt).map(([k, v]) =>
        `<div class="info-label">${{k}}</div>
         <div><input class="edit-field gt-field" data-key="${{k}}" value="${{typeof v === 'object' ? JSON.stringify(v) : v}}"></div>`
    ).join('');

    const detailPanel = document.getElementById('detailPanel');
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
                <div><select class="edit-field" id="editCategory">${{catOptions}}</select></div>
                <div class="info-label">Duration</div>
                <div class="info-value">${{clip.duration_sec}}s (${{clip.duration_frames}} frames)</div>
                <div class="info-label">Source</div>
                <div class="info-value">${{(clip.source_episode||'').slice(0,12)}}... ${{clip.source_player}}</div>
                <div class="info-label">Frames</div>
                <div class="info-value">${{clip.frame_start}} — ${{clip.frame_end}}</div>
                <div class="info-label">Verified</div>
                <div><select class="edit-field" id="editVerified">
                    <option value="true" ${{clip.gemini_verified ? 'selected' : ''}}>Yes</option>
                    <option value="false" ${{!clip.gemini_verified ? 'selected' : ''}}>No</option>
                </select></div>
                <div class="info-label">Notes</div>
                <div><textarea class="edit-field" id="editNotes" placeholder="Add notes...">${{clip.notes || ''}}</textarea></div>
            </div>
            <div class="gt-section">
                <h4>Ground Truth (editable)</h4>
                <div class="info-grid" id="gtGrid">${{gtFields}}</div>
                <button class="btn btn-secondary" id="addGtField" style="font-size:11px;padding:3px 10px;margin-top:4px">+ Add field</button>
            </div>
            <div class="btn-group">
                <button class="btn btn-save" id="saveBtn">Save Changes</button>
                <button class="btn btn-danger" id="deleteBtn">Delete Clip</button>
                <button class="btn btn-secondary" id="nextBadBtn">Mark Bad & Next</button>
            </div>
            ${{clip.edited_at ? '<div style="margin-top:8px;font-size:11px;color:var(--text-dim)">Last edited: '+clip.edited_at+'</div>' : ''}}
        </div>`;

    currentVideo = document.getElementById('clipVideo');

    // Save handler
    document.getElementById('saveBtn').addEventListener('click', async () => {{
        const newGt = {{}};
        document.querySelectorAll('.gt-field').forEach(f => {{
            let val = f.value;
            try {{ val = JSON.parse(val); }} catch(e) {{}}
            newGt[f.dataset.key] = val;
        }});
        try {{
            await apiPost('/api/clips/update', {{
                clip_id: clip.clip_id,
                category: document.getElementById('editCategory').value,
                ground_truth: newGt,
                gemini_verified: document.getElementById('editVerified').value === 'true',
                notes: document.getElementById('editNotes').value || undefined,
            }});
            clip.ground_truth = newGt;
            clip.category = document.getElementById('editCategory').value;
            clip.gemini_verified = document.getElementById('editVerified').value === 'true';
            clip.notes = document.getElementById('editNotes').value;
            clip.edited_at = new Date().toISOString();
            showToast('Saved!');
            renderList();
        }} catch(e) {{ showToast('Save failed: ' + e.message, true); }}
    }});

    // Delete handler
    document.getElementById('deleteBtn').addEventListener('click', async () => {{
        if (!confirm('Delete ' + clip.clip_id + '? (moved to _deleted/)')) return;
        try {{
            await apiPost('/api/clips/delete', {{ clip_id: clip.clip_id }});
            CLIPS.splice(idx, 1);
            activeClipIdx = null;
            showToast('Deleted');
            renderList();
            detailPanel.innerHTML = '<div class="no-clip">Clip deleted. Select another.</div>';
        }} catch(e) {{ showToast('Delete failed: ' + e.message, true); }}
    }});

    // Mark bad & next: set verified=false, add note, move to next
    document.getElementById('nextBadBtn').addEventListener('click', async () => {{
        try {{
            await apiPost('/api/clips/update', {{
                clip_id: clip.clip_id,
                gemini_verified: false,
                notes: (clip.notes || '') + ' [marked bad]',
            }});
            clip.gemini_verified = false;
            clip.notes = (clip.notes || '') + ' [marked bad]';
            showToast('Marked bad');
            // Move to next clip
            const filtered = CLIPS.map((c,i) => ({{idx:i,cat:c.category}})).filter(x => activeFilters.has(x.cat));
            const curPos = filtered.findIndex(x => x.idx === activeClipIdx);
            if (curPos < filtered.length - 1) selectClip(filtered[curPos + 1].idx);
            else renderList();
        }} catch(e) {{ showToast('Failed: ' + e.message, true); }}
    }});

    // Add GT field
    document.getElementById('addGtField').addEventListener('click', () => {{
        const grid = document.getElementById('gtGrid');
        const key = prompt('Field name:');
        if (!key) return;
        const labelDiv = document.createElement('div');
        labelDiv.className = 'info-label';
        labelDiv.textContent = key;
        const inputDiv = document.createElement('div');
        inputDiv.innerHTML = `<input class="edit-field gt-field" data-key="${{key}}" value="">`;
        grid.appendChild(labelDiv);
        grid.appendChild(inputDiv);
    }});
}}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    const filtered = CLIPS.map((c,i) => ({{idx:i,cat:c.category}})).filter(x => activeFilters.has(x.cat));
    if (!filtered.length) return;
    const curPos = filtered.findIndex(x => x.idx === activeClipIdx);

    if (e.key === 'ArrowDown' || e.key === 'j') {{
        e.preventDefault();
        selectClip(filtered[curPos < filtered.length - 1 ? curPos + 1 : 0].idx);
    }} else if (e.key === 'ArrowUp' || e.key === 'k') {{
        e.preventDefault();
        selectClip(filtered[curPos > 0 ? curPos - 1 : filtered.length - 1].idx);
    }} else if (e.key === ' ' && currentVideo) {{
        e.preventDefault();
        currentVideo.paused ? currentVideo.play() : currentVideo.pause();
    }} else if (e.key === 'x' || e.key === 'Delete') {{
        // Quick mark bad & next
        const btn = document.getElementById('nextBadBtn');
        if (btn) btn.click();
    }}
}});

// Init
loadClips();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Browse and edit eval clips")
    parser.add_argument("--clips-dir", default=CLIPS_DIR)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    import uvicorn
    app = make_app(args.clips_dir)
    print(f"Eval clips browser at http://localhost:{args.port}/")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
