#!/usr/bin/env python3
"""
Inspect PvP latent dataset: validate encode-decode quality, frame-action
alignment, policy behavior, and dataset completeness.

Generates a standalone HTML report with:
  - Dataset summary (episodes, hours, bot/policy distributions)
  - Side-by-side P1 vs P2 decoded frames at multiple timestamps
  - Original vs decoded comparison with PSNR metrics
  - Action heatmaps for both players
  - Per-episode integrity checks (frame counts, alignment)

Usage:
    python preprocessing/inspect_latents.py
    python preprocessing/inspect_latents.py --latent-dir datasets/pvp_latents --video-dir datasets/pvp_recordings --n-episodes 5
"""
import sys, os, io, base64, argparse, random, json, tarfile
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
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
MODEL_ID = "mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers"


def fig_to_base64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_latent_episodes(latent_dir):
    """Load all episodes from latent WebDataset shards."""
    from pathlib import Path
    shards = sorted(Path(latent_dir).glob("latent-*.tar"))
    episodes = []

    for shard_path in shards:
        try:
            with tarfile.open(str(shard_path), "r") as tar:
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
            print(f"  Warning: skipping {shard_path.name}: {e}")
            continue

    return episodes


def extract_npy(shard_path, member_name):
    """Extract and load a numpy array from a tar shard."""
    with tarfile.open(shard_path, "r") as tar:
        return np.load(io.BytesIO(tar.extractfile(tar.getmember(member_name)).read()))


def decode_video_frame(video_dir, episode_id, player, frame_idx):
    """Try to find and decode a frame from the original video recordings."""
    import cv2, tempfile
    from pathlib import Path

    # Search through recording shards for this episode
    for shard_path in sorted(Path(video_dir).glob("mp-*.tar")):
        try:
            with tarfile.open(str(shard_path), "r") as tar:
                for member in tar.getmembers():
                    if episode_id in member.name and f"video_{player}.mp4" in member.name:
                        mp4_bytes = tar.extractfile(member).read()
                        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/dev/shm")
                        tmp.write(mp4_bytes)
                        tmp.close()
                        try:
                            cap = cv2.VideoCapture(tmp.name)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            cap.release()
                            if ret:
                                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        finally:
                            os.unlink(tmp.name)
                        return None
        except (tarfile.TarError, OSError):
            continue
    return None


def decode_latents(dc_ae, latents, batch_size=16):
    """Decode latent array back to uint8 RGB frames.

    Args:
        latents: (N, 32, 15, 20) float16 numpy array
    Returns:
        (N, H, W, 3) uint8 numpy array
    """
    n = latents.shape[0]
    frames = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(latents[start:end]).cuda().half()
        with torch.no_grad():
            recon = dc_ae.decode(batch).sample
        recon_uint8 = recon.mul(0.5).add(0.5).clamp_(0, 1).mul_(255).byte()
        # (B, 3, H, W) -> (B, H, W, 3)
        frames.append(recon_uint8.permute(0, 2, 3, 1).cpu().numpy())
        del batch, recon, recon_uint8
    return np.concatenate(frames, axis=0)


def compute_psnr(original, reconstructed):
    """Compute PSNR between two uint8 images."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def make_decoded_sidebyside(dc_ae, ep, frame_indices):
    """Decode latent frames for P1 and P2 at given indices, return side-by-side base64 PNGs."""
    lat_p1 = extract_npy(ep["shard"], ep["members"]["latents_p1.npy"])
    has_p2 = "latents_p2.npy" in ep["members"]
    lat_p2 = extract_npy(ep["shard"], ep["members"]["latents_p2.npy"]) if has_p2 else None

    images = []
    for idx in frame_indices:
        if idx >= len(lat_p1):
            continue
        # Decode single frames
        f1 = decode_latents(dc_ae, lat_p1[idx:idx+1])[0]

        if has_p2 and idx < len(lat_p2):
            f2 = decode_latents(dc_ae, lat_p2[idx:idx+1])[0]
        else:
            f2 = np.zeros_like(f1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#1a1a2e")
        for ax, frame, label in [(ax1, f1, "Player 1"), (ax2, f2, "Player 2")]:
            ax.imshow(frame)
            ax.set_title(f"{label} — frame {idx}", color="#e0e0e0", fontsize=11)
            ax.axis("off")
        plt.subplots_adjust(wspace=0.05)
        images.append(fig_to_base64(fig))

    return images


def make_recon_comparison(dc_ae, ep, video_dir, frame_indices):
    """Compare original video frames with latent-decoded frames. Returns base64 PNG and PSNRs."""
    episode_id = ep["meta"].get("episode_id", "")
    lat_p1 = extract_npy(ep["shard"], ep["members"]["latents_p1.npy"])

    psnrs = []
    images = []

    for idx in frame_indices:
        if idx >= len(lat_p1):
            continue
        decoded = decode_latents(dc_ae, lat_p1[idx:idx+1])[0]
        original = decode_video_frame(video_dir, episode_id, "p1", idx)

        if original is None:
            continue

        psnr = compute_psnr(original, decoded)
        psnrs.append(psnr)

        diff = np.abs(original.astype(np.int16) - decoded.astype(np.int16)).astype(np.uint8)
        diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="#1a1a2e")
        titles = [f"Original (frame {idx})", f"Decoded (PSNR: {psnr:.1f} dB)", "Difference (5x)"]
        for ax, img, title in zip(axes, [original, decoded, diff_amplified], titles):
            ax.imshow(img)
            ax.set_title(title, color="#e0e0e0", fontsize=10)
            ax.axis("off")
        plt.subplots_adjust(wspace=0.05)
        images.append(fig_to_base64(fig))

    return images, psnrs


def make_action_heatmaps(ep):
    """Create action heatmaps for both players. Returns base64 PNG."""
    acts_p1 = extract_npy(ep["shard"], ep["members"]["actions_p1.npy"])
    has_p2 = "actions_p2.npy" in ep["members"]
    acts_p2 = extract_npy(ep["shard"], ep["members"]["actions_p2.npy"]) if has_p2 else None

    names = ep["meta"].get("button_names", BUTTON_NAMES)
    n_rows = 2 if has_p2 else 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), facecolor="#1a1a2e",
                              gridspec_kw={"width_ratios": [3, 1]})
    if n_rows == 1:
        axes = [axes]

    datasets = [(acts_p1, "Player 1")]
    if has_p2:
        datasets.append((acts_p2, "Player 2"))

    for row, (acts, label) in enumerate(datasets):
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
    if is_random:
        return '<span class="tag tag-random">RANDOM</span>'
    return '<span class="tag tag-trained">TRAINED</span>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", default="datasets/pvp_latents")
    parser.add_argument("--video-dir", default="datasets/pvp_recordings",
                        help="Original video recordings for decode comparison")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--frames-per-ep", type=int, default=5)
    parser.add_argument("--n-recon-compare", type=int, default=3,
                        help="Number of episodes to do original-vs-decoded comparison")
    parser.add_argument("--output", default="preprocessing/latent_inspection.html")
    parser.add_argument("--no-decode", action="store_true",
                        help="Skip decoding latents (only show metadata and actions)")
    args = parser.parse_args()

    print(f"Loading latent episodes from {args.latent_dir}...")
    all_eps = load_latent_episodes(args.latent_dir)
    print(f"  Found {len(all_eps)} episodes")

    if not all_eps:
        print("No episodes found!")
        return

    # Summary stats
    scenarios = {}
    bot_counts = {}
    total_frames = 0
    n_pvp = 0
    n_random_p1 = 0
    n_random_p2 = 0
    integrity_issues = []

    for i, ep in enumerate(all_eps):
        m = ep["meta"]
        sc = m.get("scenario", "?")
        scenarios[sc] = scenarios.get(sc, 0) + 1
        nb = m.get("n_bots", 0)
        bot_counts[nb] = bot_counts.get(nb, 0) + 1
        total_frames += m.get("n_latent_frames", m.get("n_frames", 0))

        if m.get("is_pvp"):
            n_pvp += 1
        if m.get("random_policy_p1"):
            n_random_p1 += 1
        if m.get("random_policy_p2"):
            n_random_p2 += 1

        # Quick integrity check: verify member presence
        has_p1 = "latents_p1.npy" in ep["members"]
        has_p2 = "latents_p2.npy" in ep["members"]
        has_act_p1 = "actions_p1.npy" in ep["members"]
        has_act_p2 = "actions_p2.npy" in ep["members"]

        if m.get("is_pvp") and not has_p2:
            integrity_issues.append(f"ep {i}: PvP but missing latents_p2")
        if not has_p1:
            integrity_issues.append(f"ep {i}: missing latents_p1")
        if not has_act_p1:
            integrity_issues.append(f"ep {i}: missing actions_p1")

    total_hours = total_frames / GAME_FPS / 3600
    n_total = max(len(all_eps), 1)
    print(f"  {total_frames:,} frames ({total_hours:.1f}h)")
    print(f"  PvP episodes: {n_pvp}/{n_total}")
    print(f"  Integrity issues: {len(integrity_issues)}")

    # Load DC-AE for decoding (unless --no-decode)
    dc_ae = None
    if not args.no_decode:
        from diffusers import AutoencoderDC
        print(f"Loading DC-AE decoder: {MODEL_ID} (fp16)...")
        dc_ae = AutoencoderDC.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        dc_ae = dc_ae.to("cuda").eval()
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Sample episodes
    random.seed(42)
    sample_eps = random.sample(all_eps, min(args.n_episodes, len(all_eps)))

    # Detailed frame-action alignment check on sampled episodes
    alignment_table = []
    for ep in sample_eps:
        try:
            lat_p1 = extract_npy(ep["shard"], ep["members"]["latents_p1.npy"])
            act_p1 = extract_npy(ep["shard"], ep["members"]["actions_p1.npy"])
            n_lat = lat_p1.shape[0]
            n_act = act_p1.shape[0]
            aligned = n_lat == n_act

            row = {
                "scenario": ep["meta"].get("scenario", "?"),
                "n_latents_p1": n_lat,
                "n_actions_p1": n_act,
                "aligned_p1": aligned,
                "latent_shape": str(lat_p1.shape),
            }

            if "latents_p2.npy" in ep["members"]:
                lat_p2 = extract_npy(ep["shard"], ep["members"]["latents_p2.npy"])
                act_p2 = extract_npy(ep["shard"], ep["members"]["actions_p2.npy"])
                row["n_latents_p2"] = lat_p2.shape[0]
                row["n_actions_p2"] = act_p2.shape[0]
                row["aligned_p2"] = lat_p2.shape[0] == act_p2.shape[0]
                row["p1_p2_match"] = n_lat == lat_p2.shape[0]

            alignment_table.append(row)
        except Exception as e:
            alignment_table.append({"scenario": "ERROR", "error": str(e)})

    # Build episode sections
    ep_sections = ""
    all_psnrs = []

    for i, ep in enumerate(sample_eps):
        m = ep["meta"]
        n_frames = m.get("n_latent_frames", m.get("n_frames", 0))
        if n_frames == 0:
            continue

        print(f"  Processing episode {i+1}/{len(sample_eps)}: {m.get('scenario')} "
              f"({n_frames} frames, {m.get('n_bots', 0)} bots)...")

        indices = [int(n_frames * t / (args.frames_per_ep + 1))
                   for t in range(1, args.frames_per_ep + 1)]

        # Decoded side-by-side frames
        frames_html = ""
        if dc_ae is not None:
            frame_images = make_decoded_sidebyside(dc_ae, ep, indices)
            frames_html = "\n".join(
                f'<img class="frame-img" src="data:image/png;base64,{img}" />'
                for img in frame_images
            )

        # Original vs decoded comparison (first N episodes only)
        recon_html = ""
        if dc_ae is not None and i < args.n_recon_compare:
            print(f"    Comparing with original video...")
            recon_images, psnrs = make_recon_comparison(dc_ae, ep, args.video_dir, indices[:3])
            all_psnrs.extend(psnrs)
            if recon_images:
                recon_html = '<div class="recon-section"><h3 style="color:#ffa726;margin:0.8rem 0 0.4rem">Original vs Decoded Comparison</h3>\n'
                recon_html += "\n".join(
                    f'<img class="frame-img" src="data:image/png;base64,{img}" />'
                    for img in recon_images
                )
                recon_html += "</div>"

        # Action heatmaps
        action_img = make_action_heatmaps(ep)

        # Policy info
        rp1 = m.get("random_policy_p1", m.get("random_policy", False))
        rp2 = m.get("random_policy_p2", m.get("random_policy", False))
        n_bots = m.get("n_bots", 0)

        # Alignment info
        ainfo = alignment_table[i] if i < len(alignment_table) else {}
        align_ok = ainfo.get("aligned_p1", False) and ainfo.get("p1_p2_match", True)
        align_badge = '<span class="tag tag-trained">ALIGNED</span>' if align_ok else '<span class="tag tag-random">MISALIGNED</span>'

        ep_sections += f"""
        <div class="episode-card">
            <div class="ep-header">
                <span class="ep-num">Episode {i+1}</span>
                <span class="scenario">{m.get('scenario', '?')}</span>
                <span class="bot-count">{n_bots} bots</span>
                {align_badge}
            </div>
            <div class="ep-meta-row">
                <span class="meta">{n_frames} frames ({n_frames/GAME_FPS:.1f}s)
                    | P1 frags: {m.get('frag_p1', 0):.0f}
                    | P2 frags: {m.get('frag_p2', 0):.0f}
                    | latent: {ainfo.get('latent_shape', '?')}</span>
            </div>
            <div class="ep-meta-row">
                <span>P1: {policy_tag(rp1)}</span>
                <span>P2: {policy_tag(rp2)}</span>
                <span class="meta">P1 latents={ainfo.get('n_latents_p1','?')} acts={ainfo.get('n_actions_p1','?')}
                    | P2 latents={ainfo.get('n_latents_p2','?')} acts={ainfo.get('n_actions_p2','?')}</span>
            </div>
            <div class="frames-container">
                {frames_html}
            </div>
            {recon_html}
            <div class="actions-container">
                <img class="frame-img" src="data:image/png;base64,{action_img}" />
            </div>
        </div>
        """

    # Summary PSNR stats
    psnr_summary = ""
    if all_psnrs:
        psnr_arr = np.array(all_psnrs)
        psnr_summary = f"""
        <div class="info-card">
            <h3>Decode Quality (PSNR)</h3>
            <div class="big">{np.mean(psnr_arr):.1f} dB</div>
            <div class="detail">min {np.min(psnr_arr):.1f} / max {np.max(psnr_arr):.1f} / std {np.std(psnr_arr):.1f}</div>
        </div>"""

    # Integrity section
    integrity_html = ""
    if integrity_issues:
        issues_list = "\n".join(f"<li>{iss}</li>" for iss in integrity_issues[:20])
        integrity_html = f"""
        <div class="episode-card" style="border-color:#ff6b6b">
            <h3 style="color:#ff6b6b">Integrity Issues ({len(integrity_issues)})</h3>
            <ul style="color:#e0e0e0;font-size:0.85rem">{issues_list}</ul>
        </div>"""

    # Alignment summary table
    align_rows = ""
    for j, row in enumerate(alignment_table):
        if "error" in row:
            align_rows += f'<tr><td>{j+1}</td><td colspan="6" style="color:#ff6b6b">{row["error"]}</td></tr>'
            continue
        ok1 = "ok" if row.get("aligned_p1") else "MISMATCH"
        ok2 = "ok" if row.get("aligned_p2") else "MISMATCH" if "aligned_p2" in row else "-"
        match = "ok" if row.get("p1_p2_match") else "MISMATCH" if "p1_p2_match" in row else "-"
        align_rows += f"""<tr>
            <td>{j+1}</td><td>{row['scenario']}</td>
            <td>{row.get('n_latents_p1','?')}</td><td>{row.get('n_actions_p1','?')}</td>
            <td style="color:{'#4caf50' if ok1=='ok' else '#ff6b6b'}">{ok1}</td>
            <td style="color:{'#4caf50' if ok2=='ok' else '#ff6b6b'}">{ok2}</td>
            <td style="color:{'#4caf50' if match=='ok' else '#ff6b6b'}">{match}</td>
        </tr>"""

    bot_dist_html = ", ".join(f"{k} bots: {v}" for k, v in sorted(bot_counts.items()))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PvP Latent Dataset Inspection</title>
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
  .subtitle {{ text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 2rem; }}
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
  .info-card .big {{ font-size: 1.6rem; font-weight: 700; color: #fff; }}
  .info-card .detail {{ color: #888; font-size: 0.8rem; margin-top: 0.2rem; }}
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
  .scenario {{ color: #ffa726; font-weight: 600; font-size: 1.1rem; }}
  .bot-count {{
    background: #2a1a4e;
    color: #c49bff;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.85rem;
  }}
  .meta {{ color: #888; font-size: 0.85rem; }}
  .tag {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
  }}
  .tag-trained {{ background: #1a3a2e; color: #4caf50; border: 1px solid #4caf50; }}
  .tag-random {{ background: #3a1a1a; color: #ff6b6b; border: 1px solid #ff6b6b; }}
  .frames-container {{
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin: 0.5rem 0 1rem 0;
  }}
  .frame-img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid #2a2a4a;
  }}
  .actions-container {{ margin-top: 0.5rem; }}
  .recon-section {{ margin: 0.5rem 0; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 1rem 0;
  }}
  th, td {{
    padding: 0.4rem 0.8rem;
    border: 1px solid #333;
    text-align: center;
  }}
  th {{ background: #16213e; color: #00d4ff; }}
  td {{ background: #1a1a2e; }}
</style>
</head>
<body>

<h1>PvP Latent Dataset Inspection</h1>
<p class="subtitle">
  {len(all_eps)} episodes | {total_hours:.1f}h of gameplay | Latent shape: (N, 32, 15, 20) float16
</p>

<div class="info-grid">
  <div class="info-card">
    <h3>Episodes</h3>
    <div class="big">{len(all_eps)}</div>
    <div class="detail">{n_pvp} PvP (2 players)</div>
  </div>
  <div class="info-card">
    <h3>Total Gameplay</h3>
    <div class="big">{total_hours:.1f}h</div>
    <div class="detail">{total_frames:,} frames</div>
  </div>
  <div class="info-card">
    <h3>Bot Distribution</h3>
    <div class="big">{len(bot_counts)} configs</div>
    <div class="detail">{bot_dist_html}</div>
  </div>
  <div class="info-card">
    <h3>Random Policy</h3>
    <div class="big">P1: {n_random_p1} | P2: {n_random_p2}</div>
    <div class="detail">{100*n_random_p1/n_total:.0f}% / {100*n_random_p2/n_total:.0f}% random</div>
  </div>
  <div class="info-card">
    <h3>Integrity</h3>
    <div class="big" style="color:{'#4caf50' if not integrity_issues else '#ff6b6b'}">{'PASS' if not integrity_issues else f'{len(integrity_issues)} issues'}</div>
    <div class="detail">All episodes have required files</div>
  </div>
  {psnr_summary}
</div>

<div class="episode-card">
    <h3 style="color:#00d4ff;margin-bottom:0.8rem">Frame-Action Alignment Check</h3>
    <table>
        <tr><th>#</th><th>Scenario</th><th>Latents P1</th><th>Actions P1</th><th>P1 Align</th><th>P2 Align</th><th>P1=P2</th></tr>
        {align_rows}
    </table>
</div>

{integrity_html}

{ep_sections}

</body>
</html>"""

    with open(args.output, "w") as f:
        f.write(html)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
