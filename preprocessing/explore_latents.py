# %% [markdown]
# # Explore Latent Video Dataset
# Load encoded DC-AE latent episodes, inspect shapes/stats, decode back to frames.

# %%
import sys, os, io, json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
import webdataset as wds
import matplotlib.pyplot as plt
from pathlib import Path

LATENT_DIR = "datasets/pvp_latents"

# %% [markdown]
# ## 1. Index shards and stream episodes

# %%
shards = sorted(Path(LATENT_DIR).glob("latent-*.tar"))
shard_urls = [str(s) for s in shards]
print(f"Found {len(shards)} shards, {sum(s.stat().st_size for s in shards) / 1e9:.1f} GB total")

# %% [markdown]
# ## 2. Build a WebDataset pipeline

# %%
def make_dataset(shard_urls, shuffle=False, clip_len=0, player="p1"):
    """Build a WebDataset that yields parsed episode dicts.

    If clip_len > 0, only loads the arrays for `player` and slices a random
    clip *during* npy deserialization to avoid loading the full ~190MB array.
    """
    ds = wds.WebDataset(shard_urls, shardshuffle=shuffle, handler=wds.warn_and_continue)
    if shuffle:
        ds = ds.shuffle(100)

    def decode_sample(sample):
        out = {"__key__": sample["__key__"]}
        for k, v in sample.items():
            if k == "meta.json":
                out["meta"] = json.loads(v)
            elif k.endswith(".npy"):
                name = k.replace(".npy", "")
                # When clip sampling, skip arrays we don't need
                if clip_len > 0 and player not in name:
                    continue
                out[name] = np.load(io.BytesIO(v))
        return out

    ds = ds.map(decode_sample)

    if clip_len > 0:
        def _clip(sample):
            lat = sample[f"latents_{player}"]
            n = lat.shape[0]
            start = np.random.randint(0, max(1, n - clip_len))
            end = start + clip_len
            return {
                "latents": lat[start:end],
                "actions": sample[f"actions_{player}"][start:end],
                "rewards": sample[f"rewards_{player}"][start:end],
            }
        ds = ds.map(_clip)

    return ds


ds = make_dataset(shard_urls)

# Load first episode to inspect
sample = next(iter(ds))
print(f"Episode: {sample['__key__']}")
print(f"Scenario: {sample['meta'].get('scenario', '?')}")
print(f"PvP: {sample['meta'].get('is_pvp', False)}")
print()
for k, v in sample.items():
    if isinstance(v, np.ndarray):
        print(f"  {k:20s}  shape={str(v.shape):20s}  dtype={v.dtype}  range=[{v.min():.3f}, {v.max():.3f}]")

# %% [markdown]
# ## 3. Load a batch of latent clips (for training)
# Note: each episode's latent array is ~190MB over NFS, so first batches take
# a few seconds per sample. With `clip_len>0`, we skip loading the other
# player's arrays which halves the I/O.

# %%
def numpy_collate(batch):
    return {
        "latents": torch.from_numpy(np.stack([s["latents"] for s in batch])),
        "actions": torch.from_numpy(np.stack([s["actions"] for s in batch])),
        "rewards": torch.from_numpy(np.stack([s["rewards"] for s in batch])),
    }


CLIP_LEN = 16
BATCH_SIZE = 8

train_ds = make_dataset(shard_urls, shuffle=True, clip_len=CLIP_LEN, player="p1")
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, collate_fn=numpy_collate, num_workers=2,
)

batch = next(iter(train_loader))
for k, v in batch.items():
    print(f"  {k:10s}  {str(v.shape):25s}  {v.dtype}")

# %% [markdown]
# ## 4. Latent statistics

# %%
lat = batch["latents"].float()
print(f"Latent stats (batch of clips):")
print(f"  mean={lat.mean():.4f}  std={lat.std():.4f}")
print(f"  min={lat.min():.4f}  max={lat.max():.4f}")

# Per-channel stats
channel_mean = lat.mean(dim=(0, 1, 3, 4))  # (32,)
channel_std = lat.std(dim=(0, 1, 3, 4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(32), channel_mean.numpy())
axes[0].set_title("Per-channel mean")
axes[0].set_xlabel("Channel")
axes[1].bar(range(32), channel_std.numpy())
axes[1].set_title("Per-channel std")
axes[1].set_xlabel("Channel")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Action distribution

# %%
act = batch["actions"]  # (B, T, 14)
button_names = sample["meta"].get("button_names", [f"btn_{i}" for i in range(14)])

act_flat = act.reshape(-1, 14).numpy()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(14), act_flat.mean(axis=0))
ax.set_xticks(range(14))
ax.set_xticklabels(button_names, rotation=45, ha="right")
ax.set_title("Action frequency (batch)")
ax.set_ylabel("Mean activation")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Load DC-AE and decode latents back to frames

# %%
from diffusers import AutoencoderDC

MODEL_ID = "mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading DC-AE on {device}...")
dc_ae = AutoencoderDC.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
print(f"  VRAM: {torch.cuda.memory_allocated(device)/1e9:.2f} GB" if device == "cuda" else "  (CPU)")

# %%
def decode_latents(dc_ae, latents, device="cuda", batch_size=16):
    """Decode latent array back to uint8 RGB frames.

    Args:
        latents: (N, 32, 15, 20) float16 numpy array or torch tensor
    Returns:
        (N, H, W, 3) uint8 numpy array
    """
    if isinstance(latents, np.ndarray):
        latents = torch.from_numpy(latents)
    latents = latents.to(device).half()

    frames = []
    for start in range(0, len(latents), batch_size):
        b = latents[start:start + batch_size]
        with torch.no_grad():
            decoded = dc_ae.decode(b).sample  # (B, 3, H, W)
        rgb = decoded.mul(0.5).add(0.5).clamp_(0, 1).mul_(255).byte()
        frames.append(rgb.permute(0, 2, 3, 1).cpu().numpy())  # (B, H, W, 3)
    return np.concatenate(frames, axis=0)


# Decode the first clip from our batch
sample_latents = batch["latents"][0].numpy()  # (T, 32, 15, 20)
decoded_frames = decode_latents(dc_ae, sample_latents)
print(f"Decoded: {decoded_frames.shape} {decoded_frames.dtype}")

# %% [markdown]
# ## 7. Visualize decoded frames

# %%
n_show = min(8, decoded_frames.shape[0])
fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
for i in range(n_show):
    axes[i].imshow(decoded_frames[i])
    axes[i].set_title(f"t={i}")
    axes[i].axis("off")
fig.suptitle("Decoded latent clip", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Side-by-side P1 vs P2 (PvP episode)

# %%
# Find a PvP episode
pvp_ds = make_dataset(shard_urls).select(lambda s: s["meta"].get("is_pvp", False))
try:
    pvp_sample = next(iter(pvp_ds))
    t = pvp_sample["latents_p1"].shape[0] // 2  # mid-game frame

    frames_p1 = decode_latents(dc_ae, pvp_sample["latents_p1"][t:t+1])
    frames_p2 = decode_latents(dc_ae, pvp_sample["latents_p2"][t:t+1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(frames_p1[0])
    axes[0].set_title("Player 1")
    axes[0].axis("off")
    axes[1].imshow(frames_p2[0])
    axes[1].set_title("Player 2")
    axes[1].axis("off")
    meta = pvp_sample["meta"]
    fig.suptitle(f"PvP: {meta.get('scenario','?')} | "
                 f"P1 frags={meta.get('frag_p1','?')} P2 frags={meta.get('frag_p2','?')}")
    plt.tight_layout()
    plt.show()
except StopIteration:
    print("No PvP episodes found in current shards")
