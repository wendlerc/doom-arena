#!/usr/bin/env python3
"""
Validate DC-AE-Lite encode-decode quality on Doom gameplay frames.

Samples diverse frames from the dataset, runs encode→decode through the
autoencoder, and generates a standalone HTML report with:
  - Side-by-side original vs reconstructed images
  - Per-frame PSNR and SSIM metrics
  - Summary statistics and compression ratio info

Usage:
    python preprocessing/validate_ae.py
"""
import sys, os, time, base64, io, subprocess
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = "datasets/mp_recordings"
MODEL_ID = "mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers"
N_SAMPLES = 20
BATCH_SIZE = 4  # frames per encode batch (low to keep VRAM safe)
OUTPUT_HTML = "preprocessing/ae_validation_report.html"


# ── Helpers ──────────────────────────────────────────────────────────────────

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def frame_to_base64(frame_hwc_uint8: np.ndarray) -> str:
    """Convert (H, W, 3) uint8 numpy array to base64 PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    ax.imshow(frame_hwc_uint8)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def compute_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def compute_ssim(orig: np.ndarray, recon: np.ndarray) -> float:
    """Simplified SSIM on full image (luminance channel)."""
    from skimage.metrics import structural_similarity
    # Convert to grayscale for SSIM
    orig_gray = np.mean(orig.astype(np.float64), axis=2)
    recon_gray = np.mean(recon.astype(np.float64), axis=2)
    return structural_similarity(orig_gray, recon_gray, data_range=255.0)


def system_info():
    gpu_name = "N/A"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"], text=True,
        ).strip()
        parts = out.split(", ")
        gpu_name = f"{parts[0]} ({int(parts[1])/1024:.0f} GB)"
    except Exception:
        pass
    return gpu_name


# ── Sample diverse frames ────────────────────────────────────────────────────

def sample_frames(ds, n_samples: int) -> list[dict]:
    """Sample diverse frames from across the dataset."""
    import random
    random.seed(42)

    # Pick episodes spread across scenarios and temporal positions
    n_eps = len(ds)
    ep_indices = sorted(random.sample(range(n_eps), min(n_samples, n_eps)))

    samples = []
    for ep_idx in ep_indices:
        ep = ds[ep_idx]
        n_frames = ep.n_frames
        # Pick a frame at a random position within the episode
        frame_idx = random.randint(0, max(0, n_frames - 1))
        samples.append({
            "ep_idx": ep_idx,
            "frame_idx": frame_idx,
            "scenario": ep.meta.get("scenario", "?"),
            "episode_id": ep.meta.get("episode_id", "?")[:12],
        })
        # Free video cache
        del ep
    return samples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from diffusers import AutoencoderDC
    from doom_arena.fast_loader import DoomDataset

    print(f"Loading dataset from {DATA_ROOT}...")
    ds = DoomDataset(DATA_ROOT, verbose=False)
    print(f"  {len(ds)} episodes")

    print(f"Loading DC-AE model: {MODEL_ID}...")
    t0 = time.perf_counter()
    dc_ae = AutoencoderDC.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    dc_ae = dc_ae.to("cuda").eval()
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Count model parameters
    n_params = sum(p.numel() for p in dc_ae.parameters()) / 1e6

    print(f"Sampling {N_SAMPLES} frames...")
    frame_specs = sample_frames(ds, N_SAMPLES)

    results = []
    for i, spec in enumerate(frame_specs):
        ep = ds[spec["ep_idx"]]
        video = ep.video_uint8  # (N, 3, H, W) uint8 GPU

        frame_chw = video[spec["frame_idx"]]  # (3, H, W) uint8 GPU

        # Normalize to [-1, 1]
        x = frame_chw.unsqueeze(0).float().div_(255.0).mul_(2.0).sub_(1.0)

        with torch.no_grad():
            latent = dc_ae.encode(x).latent
            recon = dc_ae.decode(latent).sample

        # Denormalize to uint8
        recon_uint8 = recon.mul(0.5).add(0.5).clamp_(0, 1).mul_(255).byte()

        # Move to CPU numpy as (H, W, 3)
        orig_hwc = frame_chw.permute(1, 2, 0).cpu().numpy()
        recon_hwc = recon_uint8[0].permute(1, 2, 0).cpu().numpy()

        psnr = compute_psnr(orig_hwc, recon_hwc)
        ssim = compute_ssim(orig_hwc, recon_hwc)

        # Create side-by-side image
        orig_b64 = frame_to_base64(orig_hwc)
        recon_b64 = frame_to_base64(recon_hwc)

        # Difference image (amplified 5x for visibility)
        diff = np.abs(orig_hwc.astype(np.int16) - recon_hwc.astype(np.int16))
        diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
        diff_b64 = frame_to_base64(diff_amplified)

        results.append({
            **spec,
            "psnr": psnr,
            "ssim": ssim,
            "orig_b64": orig_b64,
            "recon_b64": recon_b64,
            "diff_b64": diff_b64,
            "latent_min": latent.min().item(),
            "latent_max": latent.max().item(),
            "latent_mean": latent.mean().item(),
            "latent_std": latent.std().item(),
        })

        print(f"  [{i+1}/{N_SAMPLES}] ep={spec['ep_idx']} frame={spec['frame_idx']} "
              f"PSNR={psnr:.1f}dB SSIM={ssim:.3f} ({spec['scenario']})")

        # Free VRAM
        del video, ep
        torch.cuda.empty_cache()

    # ── Summary stats ──
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]

    print(f"\nSummary:")
    print(f"  PSNR: {np.mean(psnrs):.1f} ± {np.std(psnrs):.1f} dB "
          f"(min={np.min(psnrs):.1f}, max={np.max(psnrs):.1f})")
    print(f"  SSIM: {np.mean(ssims):.3f} ± {np.std(ssims):.3f} "
          f"(min={np.min(ssims):.3f}, max={np.max(ssims):.3f})")

    # ── Distribution chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                    facecolor="#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#333")

    ax1.hist(psnrs, bins=10, color="#00d4ff", edgecolor="#1a1a2e", alpha=0.85)
    ax1.set_xlabel("PSNR (dB)", color="#e0e0e0")
    ax1.set_ylabel("Count", color="#e0e0e0")
    ax1.set_title("PSNR Distribution", color="#e0e0e0", fontsize=13)
    ax1.axvline(np.mean(psnrs), color="#ff6b6b", linestyle="--", label=f"Mean: {np.mean(psnrs):.1f}")
    ax1.legend(facecolor="#16213e", edgecolor="#333", labelcolor="#e0e0e0")

    ax2.hist(ssims, bins=10, color="#00d4ff", edgecolor="#1a1a2e", alpha=0.85)
    ax2.set_xlabel("SSIM", color="#e0e0e0")
    ax2.set_ylabel("Count", color="#e0e0e0")
    ax2.set_title("SSIM Distribution", color="#e0e0e0", fontsize=13)
    ax2.axvline(np.mean(ssims), color="#ff6b6b", linestyle="--", label=f"Mean: {np.mean(ssims):.3f}")
    ax2.legend(facecolor="#16213e", edgecolor="#333", labelcolor="#e0e0e0")

    dist_chart_b64 = fig_to_base64(fig)

    # ── Latent stats chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                    facecolor="#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#333")

    means = [r["latent_mean"] for r in results]
    stds = [r["latent_std"] for r in results]
    ax1.bar(range(len(means)), means, color="#00d4ff", alpha=0.85)
    ax1.set_xlabel("Sample", color="#e0e0e0")
    ax1.set_ylabel("Mean", color="#e0e0e0")
    ax1.set_title("Latent Mean per Frame", color="#e0e0e0", fontsize=13)

    ax2.bar(range(len(stds)), stds, color="#ffa726", alpha=0.85)
    ax2.set_xlabel("Sample", color="#e0e0e0")
    ax2.set_ylabel("Std", color="#e0e0e0")
    ax2.set_title("Latent Std per Frame", color="#e0e0e0", fontsize=13)

    latent_chart_b64 = fig_to_base64(fig)

    # ── Generate HTML ──
    gpu_name = system_info()

    # Build per-frame rows
    frame_rows = ""
    for i, r in enumerate(results):
        frame_rows += f"""
        <div class="frame-card">
            <div class="frame-header">
                <span class="frame-num">#{i+1}</span>
                <span class="scenario">{r['scenario']}</span>
                <span class="meta">ep={r['ep_idx']} frame={r['frame_idx']}</span>
            </div>
            <div class="images-row">
                <div class="img-col">
                    <div class="img-label">Original</div>
                    <img src="data:image/png;base64,{r['orig_b64']}" />
                </div>
                <div class="img-col">
                    <div class="img-label">Reconstructed</div>
                    <img src="data:image/png;base64,{r['recon_b64']}" />
                </div>
                <div class="img-col">
                    <div class="img-label">Difference (5x)</div>
                    <img src="data:image/png;base64,{r['diff_b64']}" />
                </div>
            </div>
            <div class="metrics-row">
                <span class="metric">PSNR: <b>{r['psnr']:.1f} dB</b></span>
                <span class="metric">SSIM: <b>{r['ssim']:.3f}</b></span>
                <span class="metric">Latent: [{r['latent_min']:.1f}, {r['latent_max']:.1f}]</span>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>DC-AE Validation Report</title>
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
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%);
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
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
  }}
  .info-card .big {{
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
  }}
  .info-card .detail {{
    color: #888;
    font-size: 0.85rem;
    margin-top: 0.3rem;
  }}
  .section {{
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    border: 1px solid #333;
  }}
  .section h2 {{
    color: #00d4ff;
    font-size: 1.3rem;
    margin-bottom: 1rem;
  }}
  .chart-img {{
    display: block;
    margin: 0 auto;
    max-width: 100%;
    border-radius: 8px;
  }}
  .frame-card {{
    background: #16213e;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border: 1px solid #2a2a4a;
  }}
  .frame-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
  }}
  .frame-num {{
    background: #00d4ff;
    color: #0f0f23;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.85rem;
  }}
  .scenario {{
    color: #ffa726;
    font-weight: 600;
  }}
  .meta {{
    color: #888;
    font-size: 0.85rem;
  }}
  .images-row {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.8rem;
    margin-bottom: 0.8rem;
  }}
  .img-col {{
    text-align: center;
  }}
  .img-label {{
    color: #aaa;
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .img-col img {{
    width: 100%;
    border-radius: 6px;
    border: 1px solid #333;
  }}
  .metrics-row {{
    display: flex;
    gap: 2rem;
    justify-content: center;
  }}
  .metric {{
    color: #ccc;
    font-size: 0.9rem;
  }}
  .metric b {{
    color: #00d4ff;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid #2a2a4a;
  }}
  th {{
    color: #00d4ff;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
  }}
  tr:hover {{ background: #16213e; }}
</style>
</head>
<body>

<h1>DC-AE Encode-Decode Validation</h1>
<p class="subtitle">
  Model: dc-ae-lite-f32c32-sana-1.1 | {N_SAMPLES} frames |
  GPU: {gpu_name}
</p>

<div class="info-grid">
  <div class="info-card">
    <h3>Mean PSNR</h3>
    <div class="big">{np.mean(psnrs):.1f} dB</div>
    <div class="detail">min {np.min(psnrs):.1f} / max {np.max(psnrs):.1f}</div>
  </div>
  <div class="info-card">
    <h3>Mean SSIM</h3>
    <div class="big">{np.mean(ssims):.3f}</div>
    <div class="detail">min {np.min(ssims):.3f} / max {np.max(ssims):.3f}</div>
  </div>
  <div class="info-card">
    <h3>Spatial Compression</h3>
    <div class="big">32x</div>
    <div class="detail">480x640 &rarr; 15x20 latent (32 ch)</div>
  </div>
  <div class="info-card">
    <h3>Data Compression</h3>
    <div class="big">48x</div>
    <div class="detail">921,600 &rarr; 19,200 values (float16)</div>
  </div>
  <div class="info-card">
    <h3>Model Size</h3>
    <div class="big">{n_params:.0f}M</div>
    <div class="detail">VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB peak</div>
  </div>
  <div class="info-card">
    <h3>Latent Shape</h3>
    <div class="big">(32, 15, 20)</div>
    <div class="detail">Per-frame: 19.2 KB at float16</div>
  </div>
</div>

<div class="section">
  <h2>Quality Distribution</h2>
  <img class="chart-img" src="data:image/png;base64,{dist_chart_b64}" />
</div>

<div class="section">
  <h2>Latent Statistics</h2>
  <img class="chart-img" src="data:image/png;base64,{latent_chart_b64}" />
</div>

<div class="section">
  <h2>Per-Frame Results</h2>
  <table>
    <tr>
      <th>#</th><th>Scenario</th><th>Episode</th><th>Frame</th>
      <th>PSNR (dB)</th><th>SSIM</th><th>Latent Range</th>
    </tr>
    {"".join(
        f'<tr><td>{i+1}</td><td>{r["scenario"]}</td><td>{r["ep_idx"]}</td>'
        f'<td>{r["frame_idx"]}</td><td>{r["psnr"]:.1f}</td><td>{r["ssim"]:.3f}</td>'
        f'<td>[{r["latent_min"]:.1f}, {r["latent_max"]:.1f}]</td></tr>'
        for i, r in enumerate(results)
    )}
  </table>
</div>

<div class="section">
  <h2>Side-by-Side Comparisons</h2>
  {frame_rows}
</div>

</body>
</html>"""

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"\nReport saved to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
