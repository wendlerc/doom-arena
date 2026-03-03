#!/usr/bin/env python3
"""
Inspect PvP recordings: show both players' views side-by-side.

Generates a standalone HTML with:
  - Dataset summary (episodes, scenarios, durations, bot/policy distributions)
  - Side-by-side P1 vs P2 video frames at multiple timestamps
  - Action heatmaps for both players
  - Downloadable MP4 videos for each episode
  - Per-episode policy & bot metadata

Usage:
    python preprocessing/inspect_pvp.py
    python preprocessing/inspect_pvp.py --data-root datasets/pvp_recordings --n-episodes 5
"""
import sys, os, io, base64, argparse, random, json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAME_FPS = 35
BUTTON_NAMES = [
    "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_RIGHT", "MOVE_LEFT",
    "SELECT_WEAPON1", "SELECT_WEAPON2", "SELECT_WEAPON3", "SELECT_WEAPON4",
    "SELECT_WEAPON5", "SELECT_WEAPON6", "SELECT_WEAPON7",
    "ATTACK", "SPEED", "TURN_LEFT_RIGHT_DELTA",
]


def fig_to_base64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_pvp_episodes(data_root, max_episodes=None):
    """Load PvP episodes from WebDataset shards (reads completed shards only)."""
    import tarfile
    from pathlib import Path

    shards = sorted(Path(data_root).glob("mp-*.tar"))
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
                    key, ext = parts
                    groups.setdefault(key, {})[ext] = member.name

                for key, members in groups.items():
                    if "video_p2.mp4" not in members:
                        continue  # skip non-pvp episodes
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
                    if max_episodes and len(episodes) >= max_episodes:
                        return episodes
        except (tarfile.TarError, OSError):
            continue  # skip in-progress shards

    return episodes


def extract_file(shard_path, member_name):
    import tarfile
    with tarfile.open(shard_path, "r") as tar:
        return tar.extractfile(tar.getmember(member_name)).read()


def decode_frame(mp4_bytes, frame_idx):
    """Decode a single frame from MP4 bytes, return (H, W, 3) uint8 numpy."""
    import cv2, tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/dev/shm")
    tmp.write(mp4_bytes)
    tmp.close()
    try:
        cap = cv2.VideoCapture(tmp.name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        os.unlink(tmp.name)


def make_sidebyside(ep_info, frame_indices):
    """Create side-by-side P1/P2 frames at given indices. Returns list of base64 PNGs."""
    vid_p1 = extract_file(ep_info["shard"], ep_info["members"]["video_p1.mp4"])
    vid_p2 = extract_file(ep_info["shard"], ep_info["members"]["video_p2.mp4"])

    images = []
    for idx in frame_indices:
        f1 = decode_frame(vid_p1, idx)
        f2 = decode_frame(vid_p2, idx)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#1a1a2e")
        for ax, frame, label in [(ax1, f1, "Player 1"), (ax2, f2, "Player 2")]:
            ax.imshow(frame)
            ax.set_title(f"{label} — frame {idx}", color="#e0e0e0", fontsize=11)
            ax.axis("off")
        plt.subplots_adjust(wspace=0.05)
        images.append(fig_to_base64(fig))

    return images


def make_action_heatmaps(ep_info):
    """Create action heatmaps for both players. Returns base64 PNG."""
    acts_p1 = np.load(io.BytesIO(
        extract_file(ep_info["shard"], ep_info["members"]["actions_p1.npy"])))
    acts_p2 = np.load(io.BytesIO(
        extract_file(ep_info["shard"], ep_info["members"]["actions_p2.npy"])))

    names = ep_info["meta"].get("button_names", BUTTON_NAMES)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#1a1a2e",
                              gridspec_kw={"width_ratios": [3, 1]})

    for row, (acts, label) in enumerate([(acts_p1, "Player 1"), (acts_p2, "Player 2")]):
        ax_heat, ax_bar = axes[row]
        for ax in (ax_heat, ax_bar):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#e0e0e0", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#333")

        ax_heat.imshow(acts.T, aspect="auto", interpolation="nearest", cmap="viridis")
        ax_heat.set_yticks(range(len(names)))
        ax_heat.set_yticklabels(names, fontsize=7)
        ax_heat.set_xlabel("Frame", color="#e0e0e0", fontsize=9)
        ax_heat.set_title(f"{label} — Actions over time", color="#e0e0e0", fontsize=11)

        freq = np.mean(np.abs(acts), axis=0)
        colors = ["#00d4ff" if row == 0 else "#ff6b6b"] * len(freq)
        ax_bar.barh(range(len(names)), freq, color=colors, alpha=0.85)
        ax_bar.set_yticks(range(len(names)))
        ax_bar.set_yticklabels(names, fontsize=7)
        ax_bar.set_title("Mean |act|", color="#e0e0e0", fontsize=10)

    plt.tight_layout()
    return fig_to_base64(fig)


def policy_tag(is_random):
    """Return an HTML badge for policy type."""
    if is_random:
        return '<span class="tag tag-random">RANDOM</span>'
    return '<span class="tag tag-trained">TRAINED</span>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/pvp_recordings")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--frames-per-ep", type=int, default=5)
    parser.add_argument("--output", default="preprocessing/pvp_inspection.html")
    args = parser.parse_args()

    print(f"Loading PvP episodes from {args.data_root}...")
    all_eps = load_pvp_episodes(args.data_root)
    print(f"  Found {len(all_eps)} PvP episodes")

    if not all_eps:
        print("No PvP episodes found!")
        return

    # Summary stats
    scenarios = {}
    bot_counts = {}
    total_frames = 0
    total_frags_p1 = 0
    total_frags_p2 = 0
    n_random_p1 = 0
    n_random_p2 = 0
    for ep in all_eps:
        m = ep["meta"]
        sc = m.get("scenario", "?")
        scenarios[sc] = scenarios.get(sc, 0) + 1
        nb = m.get("n_bots", 0)
        bot_counts[nb] = bot_counts.get(nb, 0) + 1
        total_frames += m.get("n_frames", 0)
        total_frags_p1 += m.get("frag_p1", 0)
        total_frags_p2 += m.get("frag_p2", 0)
        if m.get("random_policy_p1") or m.get("random_policy"):
            n_random_p1 += 1
        if m.get("random_policy_p2") or m.get("random_policy"):
            n_random_p2 += 1

    total_hours = total_frames / GAME_FPS / 3600
    print(f"  {total_frames:,} frames ({total_hours:.1f}h)")
    print(f"  Scenarios: {scenarios}")
    print(f"  Bot counts: {bot_counts}")

    # Sample episodes to inspect
    random.seed(42)
    sample_eps = random.sample(all_eps, min(args.n_episodes, len(all_eps)))

    # Build HTML
    ep_sections = ""
    for i, ep in enumerate(sample_eps):
        m = ep["meta"]
        n_frames = m.get("n_frames", 0)
        if n_frames == 0:
            continue

        print(f"  Processing episode {i+1}/{len(sample_eps)}: {m.get('scenario')} "
              f"({n_frames} frames, {m.get('n_bots', 0)} bots)...")

        # Pick evenly spaced frames
        indices = [int(n_frames * t / (args.frames_per_ep + 1))
                   for t in range(1, args.frames_per_ep + 1)]

        frame_images = make_sidebyside(ep, indices)
        action_img = make_action_heatmaps(ep)

        # Embed MP4s as base64 for download
        vid_p1_bytes = extract_file(ep["shard"], ep["members"]["video_p1.mp4"])
        vid_p2_bytes = extract_file(ep["shard"], ep["members"]["video_p2.mp4"])
        vid_p1_b64 = base64.b64encode(vid_p1_bytes).decode()
        vid_p2_b64 = base64.b64encode(vid_p2_bytes).decode()
        ep_id = m.get("episode_id", f"ep{i+1}")[:8]

        frames_html = "\n".join(
            f'<img class="frame-img" src="data:image/png;base64,{img}" />'
            for img in frame_images
        )

        # Policy info
        rp1 = m.get("random_policy_p1", m.get("random_policy", False))
        rp2 = m.get("random_policy_p2", m.get("random_policy", False))
        n_bots = m.get("n_bots", 0)

        ep_sections += f"""
        <div class="episode-card">
            <div class="ep-header">
                <span class="ep-num">Episode {i+1}</span>
                <span class="scenario">{m.get('scenario', '?')}</span>
                <span class="bot-count">{n_bots} bots</span>
            </div>
            <div class="ep-meta-row">
                <span class="meta">{n_frames} frames ({n_frames/GAME_FPS:.1f}s)
                    | P1 frags: {m.get('frag_p1', 0):.0f}
                    | P2 frags: {m.get('frag_p2', 0):.0f}
                    | reward P1: {m.get('total_reward_p1', 0):.0f}
                    | reward P2: {m.get('total_reward_p2', 0):.0f}</span>
            </div>
            <div class="ep-meta-row">
                <span>P1: {policy_tag(rp1)}</span>
                <span>P2: {policy_tag(rp2)}</span>
                <span class="meta">ckpt: {m.get('checkpoint_p1', '?')[:40]}</span>
            </div>
            <div class="download-row">
                <a class="dl-btn" download="ep{i+1}_{ep_id}_p1.mp4"
                   href="data:video/mp4;base64,{vid_p1_b64}">Download P1 MP4</a>
                <a class="dl-btn dl-btn-p2" download="ep{i+1}_{ep_id}_p2.mp4"
                   href="data:video/mp4;base64,{vid_p2_b64}">Download P2 MP4</a>
            </div>
            <div class="frames-container">
                {frames_html}
            </div>
            <div class="actions-container">
                <img class="frame-img" src="data:image/png;base64,{action_img}" />
            </div>
        </div>
        """

    bot_dist_html = ", ".join(f"{k} bots: {v}" for k, v in sorted(bot_counts.items()))
    n_total = max(len(all_eps), 1)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PvP Data Inspection</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f0f23;
    color: #e0e0e0;
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
  }}
  h1 {{
    text-align: center;
    font-size: 2rem;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #ff6b6b 0%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .subtitle {{
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 2rem;
  }}
  .info-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .info-card {{
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #333;
  }}
  .info-card h3 {{
    color: #00d4ff;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
  }}
  .info-card .big {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
  }}
  .info-card .detail {{
    color: #888;
    font-size: 0.8rem;
    margin-top: 0.2rem;
  }}
  .episode-card {{
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    border: 1px solid #333;
  }}
  .ep-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
  }}
  .ep-meta-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.4rem;
    flex-wrap: wrap;
  }}
  .ep-num {{
    background: linear-gradient(135deg, #ff6b6b, #00d4ff);
    color: #0f0f23;
    font-weight: 700;
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
    font-size: 0.9rem;
  }}
  .scenario {{
    color: #ffa726;
    font-weight: 600;
    font-size: 1.1rem;
  }}
  .bot-count {{
    background: #2a1a4e;
    color: #c49bff;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.85rem;
  }}
  .meta {{
    color: #888;
    font-size: 0.85rem;
  }}
  .tag {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
  }}
  .tag-trained {{
    background: #1a3a2e;
    color: #4caf50;
    border: 1px solid #4caf50;
  }}
  .tag-random {{
    background: #3a1a1a;
    color: #ff6b6b;
    border: 1px solid #ff6b6b;
  }}
  .download-row {{
    display: flex;
    gap: 0.8rem;
    margin: 0.6rem 0 1rem 0;
  }}
  .dl-btn {{
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    text-decoration: none;
    background: #1a3a5e;
    color: #00d4ff;
    border: 1px solid #00d4ff;
    cursor: pointer;
    transition: background 0.2s;
  }}
  .dl-btn:hover {{
    background: #00d4ff;
    color: #0f0f23;
  }}
  .dl-btn-p2 {{
    background: #3a1a2e;
    color: #ff6b6b;
    border-color: #ff6b6b;
  }}
  .dl-btn-p2:hover {{
    background: #ff6b6b;
    color: #0f0f23;
  }}
  .frames-container {{
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }}
  .frame-img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid #2a2a4a;
  }}
  .actions-container {{
    margin-top: 0.5rem;
  }}
</style>
</head>
<body>

<h1>PvP Dual-View Data Inspection</h1>
<p class="subtitle">
  {len(all_eps)} episodes | {total_hours:.1f}h of gameplay | Both player perspectives
</p>

<div class="info-grid">
  <div class="info-card">
    <h3>Episodes</h3>
    <div class="big">{len(all_eps)}</div>
    <div class="detail">2 views per episode</div>
  </div>
  <div class="info-card">
    <h3>Total Gameplay</h3>
    <div class="big">{total_hours:.1f}h</div>
    <div class="detail">{total_frames:,} frames</div>
  </div>
  <div class="info-card">
    <h3>Avg Frags/Ep</h3>
    <div class="big">P1: {total_frags_p1/n_total:.1f} / P2: {total_frags_p2/n_total:.1f}</div>
    <div class="detail">total P1: {total_frags_p1:.0f} / P2: {total_frags_p2:.0f}</div>
  </div>
  <div class="info-card">
    <h3>Scenarios</h3>
    <div class="big">{len(scenarios)}</div>
    <div class="detail">{', '.join(f'{k}: {v}' for k,v in scenarios.items())}</div>
  </div>
  <div class="info-card">
    <h3>Bot Distribution</h3>
    <div class="big">{len(bot_counts)} configs</div>
    <div class="detail">{bot_dist_html}</div>
  </div>
  <div class="info-card">
    <h3>Random Policy</h3>
    <div class="big">P1: {n_random_p1}/{n_total} | P2: {n_random_p2}/{n_total}</div>
    <div class="detail">{100*n_random_p1/n_total:.0f}% / {100*n_random_p2/n_total:.0f}% random</div>
  </div>
</div>

{ep_sections}

</body>
</html>"""

    with open(args.output, "w") as f:
        f.write(html)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
