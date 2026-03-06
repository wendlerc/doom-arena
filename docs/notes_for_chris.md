# Notes for Chris

## Dataset Size Comparison: MP4 Recordings vs Latent Shards

Both measured including both player perspectives (2 video streams per PvP episode).

| Format | Total Size | MB per minute of video | Details |
|--------|-----------|----------------------|---------|
| **MP4 recordings** | 902 GB | **~46 MB/min** | Raw mp4 video + actions + rewards in WebDataset tars |
| **Latent shards** | 528 GB (65% done, ~812 GB projected) | **~40 MB/min** | DC-AE-Lite f32c32 latents, float16, shape (N,32,15,20) |

So latents are only **~13% smaller** per minute of video than the raw MP4s. The latents
are uncompressed float16 numpy arrays (32 channels x 15 x 20 spatial), while the MP4s
benefit from video compression (H.264). The latent representation trades spatial resolution
for channel depth, but without video-codec compression, the savings are modest.

---

## Compression Benchmark on Latent Shards

Tested on 5 episodes (10 perspectives, 41.7 min of video) from `latent-000001.tar`.

| Method | MB/min | % of raw | Lossless? | Notes |
|--------|--------|----------|-----------|-------|
| **Raw (baseline)** | 40.3 | 100% | yes | Uncompressed float16 numpy |
| **zstd** | 37.1 | 92.1% | yes | Generic compression, ~8% savings |
| **delta + zstd** | 36.6 | 90.8% | ~yes | Frame diffs + zstd. fp16 roundtrip err ~0.17 |
| **delta + int8 + zstd** | 12.7 | 31.5% | NO | 69% savings but huge MSE (~100k). Unusable. |
| **delta + int8 + lz4** | 19.0 | 47.1% | NO | Fast but same lossy problem |

**Projected savings on full ~812 GB dataset:**
- zstd alone: saves ~64 GB (to ~748 GB)
- delta+zstd: saves ~74 GB (to ~738 GB)
- delta+int8+zstd: saves ~557 GB (to ~255 GB) — but destroys signal

**Takeaway:** Lossless compression barely helps (~9% savings) because float16 latent
values don't have much redundancy that generic compressors can exploit. The big savings
(69%) require lossy int8 quantization, but the naive per-array min/max quantization
produces unacceptably large errors. A smarter quantization scheme (per-channel, learned
codebook, or fp8) might bridge the gap.

---

## Storage Cost Equations: MP4 vs Latent VAE

### Variables

| Symbol | Meaning | Our value |
|--------|---------|-----------|
| H, W | Video resolution (pixels) | 480, 640 |
| F | Frame rate (fps) | 35 |
| T | Duration (seconds) | — |
| s | VAE spatial compression factor | 32 |
| c | Latent channels | 32 |
| p | Bytes per value (fp16=2, fp8=1) | 2 |
| t | Temporal compression factor (1 = none) | 1 |
| b | MP4 effective bits per pixel | ~0.57 (measured) |

### MP4 Storage

H.264 exploits spatial AND temporal redundancy. Empirically:

```
S_mp4 = b × H × W × F × T / 8     (bytes)
```

The key is that `b` (effective bits/pixel) stays roughly constant or even *decreases*
as resolution grows, because more pixels = more spatial redundancy to exploit.

From our data: b = 46 MB/min × 8 / (480 × 640 × 35 × 60) ≈ 0.57 bits/pixel.

### Latent VAE Storage (current: spatial-only compression)

Each frame is independently encoded. No temporal compression, no entropy coding:

```
S_latent = c × (H/s) × (W/s) × p × F × T     (bytes)
```

This is fully deterministic — double the resolution → 4x the storage.

### Latent VAE + Temporal Compression (hypothetical 3D VAE)

A temporal compression factor `t` reduces the number of stored latent frames:

```
S_latent_temporal = c × (H/s) × (W/s) × p × (F/t) × T     (bytes)
```

Or equivalently: `S_latent_temporal = S_latent / t`

### Storage ratio: Latent vs MP4

```
                    c × (H/s) × (W/s) × p × F
R = S_latent / S_mp4 = ─────────────────────────────
                          b × H × W × F / 8

                        c × p × 8
                    = ─────────────
                        b × s²
```

Note: **F and T cancel out** (ratio is independent of framerate and duration),
and **H, W cancel out** (ratio is independent of resolution!). This means:

> The storage ratio between latent and MP4 depends ONLY on the VAE design
> (channels c, spatial factor s, precision p) and the MP4 quality (b).

With temporal compression factor t:

```
R_temporal = c × p × 8 / (b × s² × t)
```

### Plugging in numbers

**Our current setup** (DC-AE f32c32, fp16, no temporal compression):

```
R = 32 × 2 × 8 / (0.57 × 32²) = 512 / 583.7 = 0.88
```

Latent is 88% the size of MP4. Matches our measurement (~40 vs ~46 MB/min).

**Hypothetical scenarios:**

| Scenario | s | c | p | t | R (latent/MP4) | MB/min | Savings vs MP4 |
|----------|---|---|---|---|----------------|--------|----------------|
| **Current** (DC-AE f32c32, fp16) | 32 | 32 | 2 | 1 | 0.88 | 40.3 | 12% smaller |
| Current + fp8 | 32 | 32 | 1 | 1 | 0.44 | 20.2 | 56% smaller |
| Current + temporal 4x | 32 | 32 | 2 | 4 | 0.22 | 10.1 | 78% smaller |
| Current + temporal 8x | 32 | 32 | 2 | 8 | 0.11 | 5.0 | 89% smaller |
| Current + fp8 + temporal 4x | 32 | 32 | 1 | 4 | 0.11 | 5.0 | 89% smaller |
| Current + fp8 + temporal 8x | 32 | 32 | 1 | 8 | 0.055 | 2.5 | 95% smaller |
| Cosmos-style (s=8, c=16, temporal 8x) | 8 | 16 | 2 | 8 | 0.55 | 25.2 | 45% smaller |
| Higher-res 1080p (1080×1920)* | 32 | 32 | 2 | 1 | 0.88 | 40.3 | 12% smaller |

*Resolution doesn't affect the ratio! The latent and MP4 scale the same in this model.
In practice, MP4 gets relatively better at higher res (b decreases), so the ratio would
actually favor MP4 more.

**Key insight:** Temporal compression is the biggest lever we have. A 3D VAE with 4-8x
temporal compression would cut our dataset from ~812 GB to ~100-200 GB while keeping
the same latent channel structure. fp8 alone halves it. Combined, we could get under 100 GB.

**But note on Cosmos-style:** Lower spatial compression (s=8 vs s=32) means much larger
spatial dims (60×80 vs 15×20), which actually makes latents *bigger* per frame despite
fewer channels. The only reason Cosmos works is the aggressive temporal compression (8x).

---

## Storage Costs by Resolution

All at 35 fps. MP4 bits-per-pixel (b) estimated for game content — decreases at higher
resolution because H.264 exploits more spatial redundancy. 480p value is measured from
our dataset; others are estimates.

### MB per minute of video

| Resolution | H×W | Latent dims | b (bpp) | MP4 | Latent (fp16) | Ratio | t=4 | t=8 | t=8+fp8 |
|------------|-----|-------------|---------|-----|---------------|-------|-----|-----|---------|
| 240p | 240×426 | 8×14 | 0.80 | 21.5 | 15.1 | 0.70 | 3.8 | 1.9 | 0.9 |
| 360p | 360×640 | 12×20 | 0.68 | 41.1 | 32.3 | 0.78 | 8.1 | 4.0 | 2.0 |
| **480p** | **480×640** | **15×20** | **0.57** | **46.0** | **40.3** | **0.88** | **10.1** | **5.0** | **2.5** |
| 720p | 720×1280 | 23×40 | 0.40 | 96.8 | 123.6 | 1.28 | 30.9 | 15.5 | 7.7 |
| 1080p | 1080×1920 | 34×60 | 0.30 | 163.3 | 274.2 | 1.68 | 68.5 | 34.3 | 17.1 |
| 1440p | 1440×2560 | 45×80 | 0.22 | 212.9 | 483.8 | 2.27 | 121.0 | 60.5 | 30.2 |
| 4K | 2160×3840 | 68×120 | 0.17 | 370.1 | 1096.7 | 2.96 | 274.2 | 137.1 | 68.5 |

### Crossover point: when do latents beat MP4?

| Resolution | Temporal compression needed |
|------------|---------------------------|
| 240p-480p | Already smaller (ratio < 1) |
| 720p | t >= 1.3x |
| 1080p | t >= 1.7x |
| 1440p | t >= 2.3x |
| 4K | t >= 3.0x |

### Our full PvP dataset at each resolution (~340 hours)

| Resolution | MP4 | Latent (fp16) | Latent (t=8, fp8) |
|------------|-----|---------------|-------------------|
| 240p | 0.44 TB | 0.31 TB | 19 GB |
| 480p | 0.94 TB | 0.82 TB | 51 GB |
| 720p | 1.97 TB | 2.52 TB | 158 GB |
| 1080p | 3.33 TB | 5.59 TB | 350 GB |
| 4K | 7.55 TB | 22.37 TB | 1.4 TB |

**Key takeaway:** At our current 480p, raw latents barely beat MP4. But at 720p+,
latents WITHOUT temporal compression are actually LARGER than MP4. The crossover
happens because latent size scales as (H/s)×(W/s) = H×W/s² (quadratic in resolution),
while MP4's bits-per-pixel decreases at higher res. Temporal compression (3D VAE)
becomes essential at higher resolutions — even a modest t=2 would keep latents
competitive at 1080p.

---

## Latent Loader Throughput

Measured with `clip_len=16`, `batch_size=8`, `num_workers=4`.

| Version | Storage | Frames/s | Clips/s | Notes |
|---------|---------|----------|---------|-------|
| 1 clip per episode | NFS | 6 | 0.4 | Read 200MB episode, use 16 frames |
| 1 clip per episode | Local NVMe | 85 | 5.3 | Same waste, just faster I/O |
| **All clips per episode** | **NFS** | **2,000** | **124** | Amortize shard read across all frames |
| **All clips per episode** | **Local NVMe** | **20,000** | **1,250** | CPU-bound (numpy decode) |

**Key insight:** The original loader read an entire ~200MB episode (~10,000 frames) but
only extracted 1 random clip of 16 frames — a 600x I/O waste. By yielding all ~500
non-overlapping clips per episode, we amortize the shard read and throughput jumps
from 6 fps to 2,000 fps on NFS (330x) or 20,000 fps on local NVMe (3,300x).

### Realistic training params: `clip_len=70`, `batch_size=64`, `num_workers=4`

| Storage | Frames/s | Batches/s | Seconds/batch |
|---------|----------|-----------|---------------|
| NFS | 3,356 | 0.75 | 1.33s |
| **Local NVMe** | **22,371** | **4.99** | **0.20s** |

Local NVMe gives a batch every 200ms — will saturate any GPU. NFS at 1.33s/batch
is borderline depending on model size.

### Setup for training

Dataset is ~812 GB projected. Local NVMe has 1.4 TB free (on osaka).

```bash
rsync -av datasets/pvp_latents/ /tmp/pvp_latents/
```

Then point the loader at `/tmp/pvp_latents`.

wandb runs: https://wandb.ai/chrisxx/doom-arena

---

## How to read `ps aux` output

Here's an annotated real example from the encoding job:

```
wendler  2700113  109  18.2  67238428  24031484  ?  Rl  Mar03  2632:23  python -u encode_dataset.py ...
```

Columns left to right:

| Column | Example | Meaning |
|--------|---------|---------|
| **USER** | `wendler` | Who owns the process |
| **PID** | `2700113` | Process ID (use this to `kill` it) |
| **%CPU** | `109` | CPU usage (>100% = multiple cores) |
| **%MEM** | `18.2` | Percentage of total system RAM |
| **VSZ** | `67238428` | Virtual memory in KB (~64 GB here, includes mapped but unused memory) |
| **RSS** | `24031484` | Actual physical RAM in KB (~23 GB here -- this is the real memory usage) |
| **TTY** | `?` | Terminal. `?` = no terminal (background/daemon process) |
| **STAT** | `Rl` | Process state (see below) |
| **START** | `Mar03` | When the process started |
| **TIME** | `2632:23` | Total CPU time consumed (2632 minutes = ~44 hours) |
| **COMMAND** | `python ...` | The actual command line |

### STAT codes (most common)

| Code | Meaning |
|------|---------|
| `R` | Running (actively using CPU right now) |
| `S` | Sleeping (waiting for I/O, timer, signal, etc.) |
| `D` | Uninterruptible sleep (usually disk I/O -- can't be killed) |
| `T` | Stopped (e.g. Ctrl+Z) |
| `Z` | Zombie (finished but parent hasn't collected exit status) |
| `l` | Multi-threaded |
| `+` | Foreground process (attached to terminal) |
| `s` | Session leader |

### The flags: `ps aux`

- **a** = show processes from all users
- **u** = user-oriented format (the nice columns above)
- **x** = include processes not attached to a terminal (background jobs, daemons)

### Useful variations

```bash
ps aux | grep something     # find specific processes
ps aux --sort=-%mem          # sort by memory usage (biggest first)
ps aux --sort=-%cpu          # sort by CPU usage
ps -ef                       # alternative format (shows parent PID, useful for process trees)
ps auxf                      # show process tree (forest view)
```
