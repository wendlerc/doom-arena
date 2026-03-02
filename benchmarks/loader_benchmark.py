#!/usr/bin/env python3
"""
Benchmark: fast_loader (NVDEC GPU) vs loader (cv2 CPU).

Runs decode, batch, and seek benchmarks, then generates a standalone HTML report
with embedded matplotlib charts.

Usage:
    python benchmarks/loader_benchmark.py
"""
import sys, os, time, base64, io, platform, subprocess
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DATA_ROOT = "datasets/mp_recordings"

# ── Helpers ──────────────────────────────────────────────────────────────────

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


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

    try:
        cpu_name = subprocess.check_output(
            ["lscpu"], text=True,
        )
        for line in cpu_name.splitlines():
            if "Model name" in line:
                cpu_name = line.split(":")[1].strip()
                break
    except Exception:
        cpu_name = platform.processor() or "Unknown"

    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    return {
        "gpu": gpu_name,
        "cpu": cpu_name,
        "cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch": torch.__version__,
        "python": platform.python_version(),
    }


# ── Benchmarks ───────────────────────────────────────────────────────────────

def bench_episode_decode(n_episodes=10):
    """Full episode decode: GPU (NVDEC) vs CPU (cv2)."""
    from doom_arena.fast_loader import DoomDataset as FastDS
    from doom_arena.loader import DoomDataset as SimpleDS

    fast_ds = FastDS(DATA_ROOT, verbose=False)
    simple_ds = SimpleDS(DATA_ROOT, verbose=False)

    indices = list(range(min(n_episodes, len(fast_ds))))

    gpu_results = []  # (n_frames, seconds)
    cpu_results = []

    for i in indices:
        # GPU
        ep = fast_ds[i]
        t0 = time.perf_counter()
        v = ep.video_uint8
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        nf = v.shape[0]
        gpu_results.append((nf, dt))
        del v, ep
        torch.cuda.empty_cache()

        # CPU
        ep = simple_ds[i]
        t0 = time.perf_counter()
        v = ep.video
        dt = time.perf_counter() - t0
        cpu_results.append((nf, dt))
        del v, ep

    return gpu_results, cpu_results


def bench_train_loader(n_batches=200):
    """DoomTrainLoader warm batch throughput."""
    from doom_arena.fast_loader import DoomTrainLoader

    loader = DoomTrainLoader(
        DATA_ROOT, clip_len=16, stride=8, batch_size=32,
        device="cuda", max_cache=4, verbose=False,
    )

    times = []
    vram_samples = []
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        vram_samples.append(torch.cuda.memory_allocated())
        t0 = time.perf_counter()
        if i >= n_batches:
            break

    t_arr = np.array(times)
    median = np.median(t_arr)
    cold_mask = t_arr > 5 * median
    warm = t_arr[~cold_mask]
    cold = t_arr[cold_mask]

    return {
        "all_times": t_arr,
        "warm_times": warm,
        "cold_times": cold,
        "n_cold": len(cold),
        "n_warm": len(warm),
        "warm_mean_ms": np.mean(warm) * 1000,
        "warm_p50_ms": np.median(warm) * 1000,
        "warm_p99_ms": np.percentile(warm, 99) * 1000,
        "warm_fps": len(warm) * 32 * 16 / np.sum(warm),
        "vram_mb": np.mean(vram_samples) / 1e6,
        "vram_peak_mb": torch.cuda.max_memory_allocated() / 1e6,
        "batch_size": 32,
        "clip_len": 16,
    }


def bench_single_frame():
    """Single frame seek: GPU vs CPU."""
    from doom_arena.fast_loader import DoomDataset as FastDS
    from doom_arena.loader import DoomDataset as SimpleDS

    fast_ds = FastDS(DATA_ROOT, verbose=False)
    simple_ds = SimpleDS(DATA_ROOT, verbose=False)

    frame_indices = [0, 100, 500, 2000, 5000]
    gpu_times = []
    cpu_times = []

    for fi in frame_indices:
        # GPU
        ep = fast_ds[0]
        t0 = time.perf_counter()
        f = ep.get_frame(fi)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gpu_times.append(dt)
        del f, ep
        torch.cuda.empty_cache()

        # CPU
        ep = simple_ds[0]
        t0 = time.perf_counter()
        f = ep.get_frame(fi)
        dt = time.perf_counter() - t0
        cpu_times.append(dt)
        del f, ep

    return frame_indices, gpu_times, cpu_times


# ── Charts ───────────────────────────────────────────────────────────────────

COLORS = {
    "gpu": "#00d2ff",
    "cpu": "#ff6b6b",
    "accent": "#ffd93d",
    "bg": "#1a1a2e",
    "card": "#16213e",
    "text": "#e0e0e0",
    "grid": "#2a2a4a",
}


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, color=COLORS["text"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=COLORS["text"], fontsize=11)
    ax.set_ylabel(ylabel, color=COLORS["text"], fontsize=11)
    ax.tick_params(colors=COLORS["text"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, alpha=0.2, color=COLORS["grid"])


def chart_episode_decode(gpu_results, cpu_results):
    n = len(gpu_results)
    gpu_fps = [nf / dt for nf, dt in gpu_results]
    cpu_fps = [nf / dt for nf, dt in cpu_results]
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    # Bar chart: fps per episode
    w = 0.35
    ax1.bar(x - w/2, gpu_fps, w, label="NVDEC (GPU)", color=COLORS["gpu"], alpha=0.85)
    ax1.bar(x + w/2, cpu_fps, w, label="cv2 (CPU)", color=COLORS["cpu"], alpha=0.85)
    ax1.legend(fontsize=10, facecolor=COLORS["card"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["text"])
    style_ax(ax1, "Per-Episode Decode Speed", "Episode Index", "Decode FPS")
    ax1.set_xticks(x)

    # Summary comparison
    avg_gpu = sum(nf for nf, _ in gpu_results) / sum(dt for _, dt in gpu_results)
    avg_cpu = sum(nf for nf, _ in cpu_results) / sum(dt for _, dt in cpu_results)
    speedup = avg_gpu / avg_cpu

    bars = ax2.barh(
        ["cv2\n(CPU)", "NVDEC\n(GPU)"],
        [avg_cpu, avg_gpu],
        color=[COLORS["cpu"], COLORS["gpu"]],
        height=0.5, alpha=0.85,
    )
    for bar, val in zip(bars, [avg_cpu, avg_gpu]):
        ax2.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f} fps", va="center", color=COLORS["text"], fontsize=12,
                 fontweight="bold")
    style_ax(ax2, f"Average Decode Speed ({speedup:.1f}x)", "Frames per Second", "")
    ax2.set_xlim(0, max(avg_gpu, avg_cpu) * 1.3)

    plt.tight_layout()
    return fig_to_base64(fig), avg_gpu, avg_cpu, speedup


def chart_train_loader(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    # Histogram of warm batch times
    ax1.hist(results["warm_times"] * 1000, bins=40, color=COLORS["gpu"], alpha=0.8,
             edgecolor=COLORS["bg"])
    ax1.axvline(results["warm_p50_ms"], color=COLORS["accent"], linestyle="--",
                linewidth=2, label=f"p50 = {results['warm_p50_ms']:.0f}ms")
    ax1.axvline(results["warm_p99_ms"], color=COLORS["cpu"], linestyle="--",
                linewidth=2, label=f"p99 = {results['warm_p99_ms']:.0f}ms")
    ax1.legend(fontsize=10, facecolor=COLORS["card"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["text"])
    style_ax(ax1, "Warm Batch Latency Distribution", "Latency (ms)", "Count")

    # Timeline of batch times
    ax2.scatter(range(len(results["all_times"])), results["all_times"] * 1000,
                s=8, alpha=0.6, color=COLORS["gpu"])
    cold_idx = np.where(results["all_times"] > np.median(results["all_times"]) * 5)[0]
    if len(cold_idx):
        ax2.scatter(cold_idx, results["all_times"][cold_idx] * 1000,
                    s=30, color=COLORS["cpu"], zorder=5, label="Cache miss (cold)")
        ax2.legend(fontsize=10, facecolor=COLORS["card"], edgecolor=COLORS["grid"],
                   labelcolor=COLORS["text"])
    style_ax(ax2, "Batch Latency Timeline", "Batch Index", "Latency (ms)")

    plt.tight_layout()
    return fig_to_base64(fig)


def chart_single_frame(frame_indices, gpu_times, cpu_times):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    x = np.arange(len(frame_indices))
    w = 0.35
    ax.bar(x - w/2, [t * 1000 for t in gpu_times], w,
           label="NVDEC (GPU)", color=COLORS["gpu"], alpha=0.85)
    ax.bar(x + w/2, [t * 1000 for t in cpu_times], w,
           label="cv2 (CPU)", color=COLORS["cpu"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(fi) for fi in frame_indices])
    ax.legend(fontsize=10, facecolor=COLORS["card"], edgecolor=COLORS["grid"],
              labelcolor=COLORS["text"])
    style_ax(ax, "Single Frame Seek Time (uncached)", "Frame Index", "Time (ms)")

    plt.tight_layout()
    return fig_to_base64(fig)


def chart_architecture():
    """Visual architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Draw pipeline boxes
    boxes = [
        (0.5, 4.5, "WebDataset\nTar Shards", COLORS["grid"]),
        (2.5, 4.5, "Tar Extract\n(bytes)", COLORS["grid"]),
        (4.5, 4.5, "NVDEC\nHW Decode", COLORS["gpu"]),
        (6.5, 4.5, "CPU RAM\nLRU Cache", COLORS["accent"]),
        (8.5, 4.5, "Pinned\nTransfer", COLORS["gpu"]),

        (0.5, 1.5, "WebDataset\nTar Shards", COLORS["grid"]),
        (2.5, 1.5, "Tar Extract\n(bytes)", COLORS["grid"]),
        (4.5, 1.5, "cv2\nCPU Decode", COLORS["cpu"]),
        (6.5, 1.5, "NumPy\nArray", COLORS["cpu"]),
        (8.5, 1.5, "Manual\nCopy", COLORS["cpu"]),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x - 0.7, y - 0.5), 1.4, 1.0,
                              facecolor=color, alpha=0.2, edgecolor=color,
                              linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", color=COLORS["text"],
                fontsize=9, fontweight="bold", zorder=3)

    # Arrows
    for row_y in [4.5, 1.5]:
        for x_start in [1.2, 3.2, 5.2, 7.2]:
            ax.annotate("", xy=(x_start + 0.6, row_y), xytext=(x_start, row_y),
                        arrowprops=dict(arrowstyle="->", color=COLORS["text"],
                                       lw=1.5), zorder=4)

    # Labels
    ax.text(5, 5.7, "fast_loader.py  (NVDEC GPU Pipeline)",
            ha="center", color=COLORS["gpu"], fontsize=13, fontweight="bold")
    ax.text(5, 2.7, "loader.py  (cv2 CPU Pipeline)",
            ha="center", color=COLORS["cpu"], fontsize=13, fontweight="bold")

    # Final output
    for y, label, color in [(4.5, "GPU Tensor\n(B,T,3,H,W)", COLORS["gpu"]),
                             (1.5, "NumPy Array\n(N,H,W,3)", COLORS["cpu"])]:
        ax.text(9.8, y, label, ha="center", va="center",
                color=color, fontsize=9, fontweight="bold")

    plt.tight_layout()
    return fig_to_base64(fig)


# ── HTML Report ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Doom Arena — Data Loader Benchmark</title>
<style>
  :root {{
    --bg: #0f0f23;
    --card: #1a1a2e;
    --border: #2a2a4a;
    --text: #e0e0e0;
    --gpu: #00d2ff;
    --cpu: #ff6b6b;
    --accent: #ffd93d;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 2.2rem;
    background: linear-gradient(135deg, var(--gpu), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }}
  h2 {{
    font-size: 1.4rem;
    color: var(--gpu);
    margin: 2rem 0 1rem;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.4rem;
  }}
  .subtitle {{ color: #888; font-size: 0.95rem; margin-bottom: 2rem; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .stat {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
  }}
  .stat .value {{
    font-size: 2rem;
    font-weight: 700;
    display: block;
    margin-bottom: 0.2rem;
  }}
  .stat .label {{ font-size: 0.85rem; color: #888; }}
  .gpu {{ color: var(--gpu); }}
  .cpu {{ color: var(--cpu); }}
  .accent {{ color: var(--accent); }}
  img {{ width: 100%; border-radius: 8px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
  }}
  th, td {{
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{ color: var(--gpu); font-weight: 600; }}
  .tag {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
  }}
  .tag-gpu {{ background: rgba(0,210,255,0.15); color: var(--gpu); }}
  .tag-cpu {{ background: rgba(255,107,107,0.15); color: var(--cpu); }}
  .sysinfo {{ font-size: 0.85rem; color: #888; }}
  .sysinfo span {{ color: var(--text); }}
</style>
</head>
<body>

<h1>Doom Arena Data Loader Benchmark</h1>
<p class="subtitle">fast_loader.py (NVDEC GPU) vs loader.py (cv2 CPU) &mdash; {date}</p>

<div class="stats-grid">
  <div class="stat">
    <span class="value gpu">{warm_fps:.0f}</span>
    <span class="label">Training FPS (warm)</span>
  </div>
  <div class="stat">
    <span class="value gpu">{warm_p50:.0f}ms</span>
    <span class="label">Batch Latency (p50)</span>
  </div>
  <div class="stat">
    <span class="value accent">{decode_speedup:.1f}x</span>
    <span class="label">Decode Speedup</span>
  </div>
  <div class="stat">
    <span class="value">{vram_peak:.0f} MB</span>
    <span class="label">Peak VRAM</span>
  </div>
</div>

<h2>Architecture</h2>
<div class="card">
  <img src="data:image/png;base64,{arch_chart}" alt="Architecture comparison">
</div>

<h2>Episode Decode Speed</h2>
<div class="card">
  <img src="data:image/png;base64,{decode_chart}" alt="Decode benchmark">
  <p style="margin-top:0.8rem; font-size:0.9rem; color:#888;">
    Measured over {n_decode_eps} episodes. GPU decode uses NVIDIA NVDEC hardware via PyNvVideoCodec.
    CPU decode uses OpenCV VideoCapture. Both include tar extraction and temp file overhead.
  </p>
</div>

<h2>Training Batch Throughput</h2>
<div class="card">
  <img src="data:image/png;base64,{train_chart}" alt="Training throughput">
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Warm throughput</td><td class="gpu">{warm_fps:.0f} frames/s</td></tr>
    <tr><td>Batch latency (p50 / p99)</td><td>{warm_p50:.0f}ms / {warm_p99:.0f}ms</td></tr>
    <tr><td>Cold batches (cache miss)</td><td class="cpu">{n_cold} of {n_total} ({cold_pct:.1f}%)</td></tr>
    <tr><td>Batch size / Clip length</td><td>32 clips &times; 16 frames = 512 frames/batch</td></tr>
    <tr><td>VRAM usage (batch)</td><td>{vram_batch:.0f} MB</td></tr>
    <tr><td>VRAM peak</td><td>{vram_peak:.0f} MB</td></tr>
  </table>
</div>

<h2>Single Frame Seek</h2>
<div class="card">
  <img src="data:image/png;base64,{seek_chart}" alt="Single frame seek">
  <p style="margin-top:0.8rem; font-size:0.9rem; color:#888;">
    Time to extract one frame from a fresh (uncached) episode. Includes tar extraction +
    temp file + decoder init + seek. Once video is cached, frame access is &lt;0.1ms.
  </p>
</div>

<h2>Feature Comparison</h2>
<div class="card">
  <table>
    <tr>
      <th>Feature</th>
      <th><span class="tag tag-gpu">fast_loader</span></th>
      <th><span class="tag tag-cpu">loader</span></th>
    </tr>
    <tr><td>Video decode backend</td><td>NVDEC hardware (PyNvVideoCodec)</td><td>cv2 (CPU)</td></tr>
    <tr><td>Output format</td><td>torch.Tensor (GPU/CPU)</td><td>numpy.ndarray (CPU)</td></tr>
    <tr><td>Video layout</td><td>(N, C, H, W) &mdash; channels-first</td><td>(N, H, W, C) &mdash; channels-last</td></tr>
    <tr><td>Training pipeline</td><td>DoomTrainLoader (clip batching, shuffle)</td><td>Manual iteration</td></tr>
    <tr><td>Pinned memory</td><td>Pre-allocated pinned buffers</td><td>N/A</td></tr>
    <tr><td>Caching</td><td>LRU CPU RAM cache + grouped iteration</td><td>Per-episode lazy load</td></tr>
    <tr><td>Temp files</td><td>/dev/shm (RAM disk)</td><td>/tmp (disk)</td></tr>
    <tr><td>CPU fallback</td><td>Automatic (cv2)</td><td>N/A (cv2 only)</td></tr>
    <tr><td>Visualization</td><td>show_frame, play, plot_actions, plot_rewards</td><td>Same</td></tr>
    <tr><td>Multi-core</td><td>Single-threaded (NVDEC offloads CPU)</td><td>Single-threaded</td></tr>
  </table>
</div>

<h2>System</h2>
<div class="card sysinfo">
  <p><strong>GPU:</strong> <span>{sys_gpu}</span></p>
  <p><strong>CPU:</strong> <span>{sys_cpu}</span> ({sys_cores} cores)</p>
  <p><strong>RAM:</strong> <span>{sys_ram:.0f} GB</span></p>
  <p><strong>CUDA:</strong> <span>{sys_cuda}</span> &nbsp;|&nbsp;
     <strong>PyTorch:</strong> <span>{sys_pytorch}</span> &nbsp;|&nbsp;
     <strong>Python:</strong> <span>{sys_python}</span></p>
</div>

</body>
</html>
"""


def main():
    print("=" * 60)
    print("  Doom Arena Data Loader Benchmark")
    print("=" * 60)

    sysinfo = system_info()
    print(f"\nSystem: {sysinfo['gpu']} | {sysinfo['cpu']} ({sysinfo['cores']} cores)")

    # 1. Episode decode
    print("\n[1/4] Episode decode benchmark (10 episodes)...")
    gpu_res, cpu_res = bench_episode_decode(10)
    avg_gpu = sum(nf for nf, _ in gpu_res) / sum(dt for _, dt in gpu_res)
    avg_cpu = sum(nf for nf, _ in cpu_res) / sum(dt for _, dt in cpu_res)
    print(f"  GPU: {avg_gpu:.0f} fps | CPU: {avg_cpu:.0f} fps | Speedup: {avg_gpu/avg_cpu:.1f}x")

    # 2. Training loader
    print("\n[2/4] Training batch benchmark (200 batches)...")
    train_res = bench_train_loader(200)
    print(f"  Warm: {train_res['warm_fps']:.0f} fps ({train_res['warm_p50_ms']:.0f}ms p50)")

    # 3. Single frame seek
    print("\n[3/4] Single frame seek benchmark...")
    seek_indices, seek_gpu, seek_cpu = bench_single_frame()
    print(f"  GPU avg: {np.mean(seek_gpu)*1000:.0f}ms | CPU avg: {np.mean(seek_cpu)*1000:.0f}ms")

    # 4. Generate charts
    print("\n[4/4] Generating report...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    decode_b64, d_gpu, d_cpu, d_speedup = chart_episode_decode(gpu_res, cpu_res)
    train_b64 = chart_train_loader(train_res)
    seek_b64 = chart_single_frame(seek_indices, seek_gpu, seek_cpu)
    arch_b64 = chart_architecture()

    from datetime import datetime
    html = HTML_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        warm_fps=train_res["warm_fps"],
        warm_p50=train_res["warm_p50_ms"],
        warm_p99=train_res["warm_p99_ms"],
        decode_speedup=d_speedup,
        vram_peak=train_res["vram_peak_mb"],
        vram_batch=train_res["vram_mb"],
        n_cold=train_res["n_cold"],
        n_total=train_res["n_warm"] + train_res["n_cold"],
        cold_pct=train_res["n_cold"] / (train_res["n_warm"] + train_res["n_cold"]) * 100,
        n_decode_eps=len(gpu_res),
        arch_chart=arch_b64,
        decode_chart=decode_b64,
        train_chart=train_b64,
        seek_chart=seek_b64,
        sys_gpu=sysinfo["gpu"],
        sys_cpu=sysinfo["cpu"],
        sys_cores=sysinfo["cores"],
        sys_ram=sysinfo["ram_gb"],
        sys_cuda=sysinfo["cuda"],
        sys_pytorch=sysinfo["pytorch"],
        sys_python=sysinfo["python"],
    )

    out_path = "benchmarks/report.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nReport saved to {out_path}")
    print(f"Open in browser: file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
