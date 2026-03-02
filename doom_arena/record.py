"""
WebDataset multiplayer recording pipeline.

Records two AI players playing Doom deathmatch at 640x480, storing frames as
H.264 video, actions (button presses + turn delta), rewards, and audio in
WebDataset tar shards. Supports multiple maps, bot counts, and checkpoint sampling.

Usage:
    doom-record --experiment seed0 --total-hours 100 --num-workers 8 --device cuda
"""
from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

import vizdoom as vzd

# --- Constants ---
BASE_PORT = 5400
DECISION_INTERVAL = 2  # model decides every 2 tics (matching training env_frameskip=2)
GAME_FPS = 35
NUM_WEAPONS = 8

# Scenarios with weights
# Note: bots are disabled in 2-player multiplayer to avoid ViZDoom sync deadlocks.
# Two AI players fight each other directly in deathmatch mode.
SCENARIOS = {
    "dwango5_3min": {"wad": "dwango5.wad", "map": "map01", "bots": 0, "timelimit": 3.0, "weight": 0.40},
    "dwango5_5min": {"wad": "dwango5.wad", "map": "map01", "bots": 0, "timelimit": 5.0, "weight": 0.45},
    "ssl2_duel": {"wad": "ssl2.wad", "map": "map01", "bots": 0, "timelimit": 3.0, "weight": 0.15},
}

# Buttons matching training (dwango5_dm_continuous_weap.cfg)
TRAINING_BUTTONS = [
    vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
    vzd.Button.MOVE_RIGHT, vzd.Button.MOVE_LEFT,
    vzd.Button.SELECT_WEAPON1, vzd.Button.SELECT_WEAPON2,
    vzd.Button.SELECT_WEAPON3, vzd.Button.SELECT_WEAPON4,
    vzd.Button.SELECT_WEAPON5, vzd.Button.SELECT_WEAPON6,
    vzd.Button.SELECT_WEAPON7,
    vzd.Button.ATTACK, vzd.Button.SPEED,
    vzd.Button.TURN_LEFT_RIGHT_DELTA,
]

BUTTON_NAMES = [str(b).split(".")[-1] for b in TRAINING_BUTTONS]


def _sf_scenarios_dir() -> str:
    """Get the SF scenarios directory where dwango5.wad, ssl2.wad etc. live."""
    import sf_examples.vizdoom.doom.scenarios as sc_mod
    # Namespace package: __file__ is None, but __path__ works
    return list(sc_mod.__path__)[0]


def _find_wad(wad_name: str) -> str:
    """Find a WAD file, searching SF scenarios dir then vzd.scenarios_path."""
    for base in [_sf_scenarios_dir(), vzd.scenarios_path]:
        path = os.path.join(base, wad_name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"WAD not found: {wad_name}")


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# --- Game Setup ---

def create_game(
    wad_path: str,
    doom_map: str,
    port: int,
    is_host: bool,
    timelimit: float,
    num_bots: int = 0,
    enable_audio: bool = True,
) -> vzd.DoomGame:
    """Create a DoomGame instance configured for multiplayer recording at 640x480."""
    game = vzd.DoomGame()

    game.set_doom_scenario_path(wad_path)
    game.set_doom_map(doom_map)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    for button in TRAINING_BUTTONS:
        game.add_available_button(button)

    for var in [
        vzd.GameVariable.SELECTED_WEAPON, vzd.GameVariable.SELECTED_WEAPON_AMMO,
        vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR, vzd.GameVariable.USER2,
        vzd.GameVariable.ATTACK_READY, vzd.GameVariable.PLAYER_COUNT,
        vzd.GameVariable.FRAGCOUNT, vzd.GameVariable.DEATHCOUNT,
        vzd.GameVariable.HITCOUNT, vzd.GameVariable.DAMAGECOUNT,
    ]:
        game.add_available_game_variable(var)
    for i in range(10):
        game.add_available_game_variable(getattr(vzd.GameVariable, f"WEAPON{i}"))
    for i in range(10):
        game.add_available_game_variable(getattr(vzd.GameVariable, f"AMMO{i}"))

    if enable_audio:
        try:
            game.set_sound_enabled(True)
            game.set_audio_buffer_enabled(True)
            game.set_audio_sampling_rate(vzd.SamplingRate.SR_44100)
            game.set_audio_buffer_size(4)
        except Exception:
            pass

    if is_host:
        # -deathmatch: proper deathmatch mode (matching SF training config).
        # ASYNC_PLAYER mode avoids multiplayer sync deadlocks at 640x480.
        game.add_game_args(
            f"-host 2 -port {port} -deathmatch "
            f"+timelimit {timelimit:.1f} "
            "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 "
            "+sv_spawnfarthest 1 +sv_nocrouch 1 +sv_nojump 1 "
            "+sv_nofreelook 1 +sv_noexit 1 "
            "+viz_respawn_delay 0 +viz_nocheat 1 "
            "+viz_connect_timeout 60"
        )
        game.add_game_args("+name P1 +colorset 0")
    else:
        game.add_game_args(f"-join 127.0.0.1:{port} +viz_connect_timeout 30")
        game.add_game_args("+name P2 +colorset 3")

    game.set_episode_timeout(int(timelimit * 60 * game.get_ticrate()))

    return game


# --- Observation Preprocessing ---

def extract_measurements(game: vzd.DoomGame) -> np.ndarray:
    """Extract measurements vector matching DoomAdditionalInput wrapper (23 values)."""
    measurements = np.zeros(7 + NUM_WEAPONS + NUM_WEAPONS, dtype=np.float32)
    get = game.get_game_variable

    selected_weapon = round(max(0, get(vzd.GameVariable.SELECTED_WEAPON)))
    sel_ammo = min(max(0.0, get(vzd.GameVariable.SELECTED_WEAPON_AMMO)) / 15.0, 5.0)
    health = max(0.0, get(vzd.GameVariable.HEALTH)) / 30.0
    armor = get(vzd.GameVariable.ARMOR) / 30.0
    kills = get(vzd.GameVariable.USER2) / 10.0
    attack_ready = get(vzd.GameVariable.ATTACK_READY)
    num_players = get(vzd.GameVariable.PLAYER_COUNT) / 5.0

    i = 0
    for val in [selected_weapon, sel_ammo, health, armor, kills, attack_ready, num_players]:
        measurements[i] = float(val)
        i += 1
    for w in range(NUM_WEAPONS):
        measurements[i] = float(max(0.0, get(getattr(vzd.GameVariable, f"WEAPON{w}"))))
        i += 1
    for w in range(NUM_WEAPONS):
        measurements[i] = min(max(0.0, get(getattr(vzd.GameVariable, f"AMMO{w}"))) / 15.0, 5.0)
        i += 1

    return measurements


def preprocess_for_model(screen_buffer: np.ndarray, measurements: np.ndarray, device: torch.device) -> dict:
    """Convert raw 640x480 CHW screen buffer to model input format (128x72 CHW).

    Two-step resize to match training pipeline:
      Training: ViZDoom renders at 160x120 → ResizeWrapper scales to 128x72
      Here: 640x480 → 160x120 (approximate native) → 128x72
    """
    hwc = np.transpose(screen_buffer, (1, 2, 0))
    # Step 1: downscale to training render resolution
    small = cv2.resize(hwc, (160, 120), interpolation=cv2.INTER_AREA)
    # Step 2: resize to model input (matches ResizeWrapper which uses INTER_NEAREST)
    resized = cv2.resize(small, (128, 72), interpolation=cv2.INTER_NEAREST)
    chw = np.transpose(resized, (2, 0, 1))

    obs_t = torch.from_numpy(chw).float().unsqueeze(0).to(device)
    meas_t = torch.from_numpy(measurements).float().unsqueeze(0).to(device)
    return {"obs": obs_t, "measurements": meas_t}


# --- Action Conversion ---

def _build_action_spaces():
    """Build the action sub-spaces matching doom_action_space_full_discretized()."""
    import gymnasium as gym
    from sample_factory.algo.utils.spaces.discretized import Discretized
    return [
        gym.spaces.Discrete(3),
        gym.spaces.Discrete(3),
        gym.spaces.Discrete(8),
        gym.spaces.Discrete(2),
        gym.spaces.Discrete(2),
        Discretized(21, min_action=-12.5, max_action=12.5),
    ]


_ACTION_SPACES = None


def get_action_spaces():
    global _ACTION_SPACES
    if _ACTION_SPACES is None:
        _ACTION_SPACES = _build_action_spaces()
    return _ACTION_SPACES


def convert_action(actions: np.ndarray) -> list:
    """Convert model output (per sub-space indices) to binary button list for ViZDoom.

    Returns: list of 14 floats (13 binary + 1 continuous turn delta).
    """
    import gymnasium as gym
    from sample_factory.algo.utils.spaces.discretized import Discretized

    spaces = get_action_spaces()
    if actions.ndim == 2:
        actions = actions[0]

    flat = []
    for i, space in enumerate(spaces):
        act_val = int(actions[i])
        if isinstance(space, Discretized):
            flat.append(space.to_continuous(act_val))
        elif isinstance(space, gym.spaces.Discrete):
            one_hot = [0] * (space.n - 1)
            if act_val > 0:
                one_hot[act_val - 1] = 1
            flat.extend(one_hot)
    return flat


# --- Checkpoint Sampling ---

def discover_checkpoints(experiment: str, train_dir: str) -> list:
    """Find all checkpoints, return list of (path, reward)."""
    ckpt_dir = Path(train_dir) / experiment / "checkpoint_p0"
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for f in ckpt_dir.glob("*.pth"):
        match = re.search(r"reward_([-\d.]+?)\.pth", f.name)
        reward = float(match.group(1)) if match else 0.0
        checkpoints.append((str(f), reward))

    return sorted(checkpoints, key=lambda x: x[1])


def sample_checkpoint(checkpoints: list, temperature: float = 0.5) -> str:
    """Sample a checkpoint weighted by reward using softmax."""
    if not checkpoints:
        raise ValueError("No checkpoints found")
    if len(checkpoints) == 1:
        return checkpoints[0][0]

    rewards = np.array([r for _, r in checkpoints], dtype=np.float64)
    logits = rewards / max(temperature, 0.01)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    idx = np.random.choice(len(checkpoints), p=weights)
    return checkpoints[idx][0]


# --- Model Loading ---

def load_model(experiment: str, train_dir: str, checkpoint_path: str, device: str = "cuda"):
    """Load a Sample Factory model for inference. Returns (actor_critic, rnn_size, device)."""
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict
    from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

    register_vizdoom_components()

    cfg_path = Path(train_dir) / experiment / "cfg.json"
    if not cfg_path.exists():
        cfg_path = Path(train_dir) / experiment / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg = AttrDict(cfg)
    cfg.train_dir = train_dir
    cfg.experiment = experiment
    cfg.no_render = True
    cfg.skip_measurements_head = True

    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)
    env = make_env_func_batched(cfg, env_config=env_config, render_mode="rgb_array")

    dev = torch.device(device)
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic = actor_critic.to(dev)

    checkpoint_dict = Learner.load_checkpoint([checkpoint_path], dev)
    actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)
    actor_critic.eval()

    rnn_size = get_rnn_size(cfg)
    env.close()

    return actor_critic, rnn_size, dev


# --- Video Encoding ---

def encode_video(frames: list, fps: int = GAME_FPS) -> bytes:
    """Encode a list of (H, W, 3) uint8 RGB frames to mp4 bytes."""
    if not frames:
        return b""
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


# --- Episode Recording ---

def record_episode(
    actor_critic,
    rnn_size: int,
    device: torch.device,
    scenario_name: str,
    scenario: dict,
    port: int,
    checkpoint_name: str = "",
    enable_audio: bool = True,
) -> dict | None:
    """Record a single multiplayer episode. Returns episode data dict or None on failure."""

    try:
        wad_path = _find_wad(scenario["wad"])
    except FileNotFoundError as e:
        print(f"[record] {e}")
        return None

    doom_map = scenario["map"]
    timelimit = scenario["timelimit"]
    num_bots = scenario["bots"]

    host_game = create_game(wad_path, doom_map, port, True, timelimit, num_bots, enable_audio)
    join_game = create_game(wad_path, doom_map, port, False, timelimit, 0, enable_audio)

    # Init via threads (host blocks waiting for join to connect)
    errors = [None, None]

    def init_host():
        try:
            host_game.init()
        except Exception as e:
            errors[0] = e

    def init_join():
        try:
            time.sleep(1.5)
            join_game.init()
        except Exception as e:
            errors[1] = e

    t_host = threading.Thread(target=init_host)
    t_join = threading.Thread(target=init_join)
    t_host.start()
    t_join.start()
    t_host.join(timeout=45)
    t_join.join(timeout=45)

    if errors[0] or errors[1]:
        try:
            host_game.close()
        except Exception:
            pass
        try:
            join_game.close()
        except Exception:
            pass
        print(f"[record] Init failed: host={errors[0]}, join={errors[1]}")
        return None

    # Add bots after init
    if num_bots > 0:
        for _ in range(num_bots):
            host_game.send_game_command("addbot")

    # RNN states
    rnn_p1 = torch.zeros(1, rnn_size, device=device)
    rnn_p2 = torch.zeros(1, rnn_size, device=device)

    # Current actions
    action_p1 = [0.0] * len(TRAINING_BUTTONS)
    action_p2 = [0.0] * len(TRAINING_BUTTONS)

    frames_p1, frames_p2 = [], []
    actions_p1, actions_p2 = [], []
    rewards_p1, rewards_p2 = [], []
    audio_chunks_p1, audio_chunks_p2 = [], []

    tic = 0
    t0 = time.perf_counter()
    has_audio = False
    frag_p1 = frag_p2 = death_p1 = death_p2 = 0.0

    def _advance_both(act_h, act_j):
        """Advance both games 1 tic using make_action (ASYNC_PLAYER mode)."""
        host_game.make_action(act_h)
        join_game.make_action(act_j)

    zero_action = [0.0] * len(TRAINING_BUTTONS)

    while not host_game.is_episode_finished() and not join_game.is_episode_finished():
        state_p1 = host_game.get_state()
        state_p2 = join_game.get_state()

        if state_p1 is None or state_p2 is None:
            # Player dead/respawning — advance both with zero action
            _advance_both(zero_action, zero_action)
            if host_game.is_player_dead():
                host_game.respawn_player()
            if join_game.is_player_dead():
                join_game.respawn_player()
            tic += 1
            continue

        # Record frames (CHW → HWC)
        frame_p1 = np.transpose(state_p1.screen_buffer, (1, 2, 0))
        frame_p2 = np.transpose(state_p2.screen_buffer, (1, 2, 0))
        frames_p1.append(frame_p1)
        frames_p2.append(frame_p2)

        # Decision step (also capture audio here — buffer covers DECISION_INTERVAL tics)
        if tic % DECISION_INTERVAL == 0:
            if enable_audio:
                try:
                    ab1 = state_p1.audio_buffer
                    ab2 = state_p2.audio_buffer
                    if ab1 is not None:
                        audio_chunks_p1.append(ab1.copy())
                        has_audio = True
                    if ab2 is not None:
                        audio_chunks_p2.append(ab2.copy())
                except AttributeError:
                    pass

            with torch.no_grad():
                meas_p1 = extract_measurements(host_game)
                obs_p1 = preprocess_for_model(state_p1.screen_buffer, meas_p1, device)
                norm_p1 = actor_critic.normalize_obs(obs_p1)
                res_p1 = actor_critic(norm_p1, rnn_p1)
                rnn_p1 = res_p1["new_rnn_states"]
                action_p1 = convert_action(res_p1["actions"].cpu().numpy())

                meas_p2 = extract_measurements(join_game)
                obs_p2 = preprocess_for_model(state_p2.screen_buffer, meas_p2, device)
                norm_p2 = actor_critic.normalize_obs(obs_p2)
                res_p2 = actor_critic(norm_p2, rnn_p2)
                rnn_p2 = res_p2["new_rnn_states"]
                action_p2 = convert_action(res_p2["actions"].cpu().numpy())

        actions_p1.append(np.array(action_p1, dtype=np.float32))
        actions_p2.append(np.array(action_p2, dtype=np.float32))

        # Track stats during episode (may not be readable after episode end)
        if tic % DECISION_INTERVAL == 0:
            frag_p1 = host_game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
            frag_p2 = join_game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
            death_p1 = host_game.get_game_variable(vzd.GameVariable.DEATHCOUNT)
            death_p2 = join_game.get_game_variable(vzd.GameVariable.DEATHCOUNT)

        # Advance both games 1 tic
        action_list_p1 = [float(a) for a in action_p1]
        action_list_p2 = [float(a) for a in action_p2]
        _advance_both(action_list_p1, action_list_p2)

        rewards_p1.append(float(host_game.get_last_reward()))
        rewards_p2.append(float(join_game.get_last_reward()))

        # Immediate respawn (matching SF's _process_game_step)
        if host_game.is_player_dead():
            host_game.respawn_player()
        if join_game.is_player_dead():
            join_game.respawn_player()

        tic += 1

    duration = time.perf_counter() - t0
    game_tics = tic

    host_game.close()
    join_game.close()

    if not frames_p1:
        return None

    # Encode videos
    video_p1 = encode_video(frames_p1, GAME_FPS)
    video_p2 = encode_video(frames_p2, GAME_FPS)

    result = {
        "video_p1": video_p1,
        "video_p2": video_p2,
        "actions_p1": np.stack(actions_p1),
        "actions_p2": np.stack(actions_p2),
        "rewards_p1": np.array(rewards_p1, dtype=np.float32),
        "rewards_p2": np.array(rewards_p2, dtype=np.float32),
        "n_frames": len(frames_p1),
        "game_tics": game_tics,
        "duration_s": duration,
        "scenario": scenario_name,
        "map": doom_map,
        "n_bots": num_bots,
        "timelimit": timelimit,
        "checkpoint": checkpoint_name,
        "frag_p1": frag_p1,
        "frag_p2": frag_p2,
        "death_p1": death_p1,
        "death_p2": death_p2,
        "total_reward_p1": float(np.sum(rewards_p1)),
        "total_reward_p2": float(np.sum(rewards_p2)),
    }

    if has_audio and audio_chunks_p1:
        result["audio_p1"] = np.concatenate(audio_chunks_p1, axis=0)
    if has_audio and audio_chunks_p2:
        result["audio_p2"] = np.concatenate(audio_chunks_p2, axis=0)

    return result


# --- Scenario Sampling ---

def sample_scenario() -> tuple:
    """Sample a scenario weighted by configured weights. Returns (name, config)."""
    names = list(SCENARIOS.keys())
    weights = [SCENARIOS[n]["weight"] for n in names]
    name = random.choices(names, weights=weights, k=1)[0]
    return name, SCENARIOS[name]


# --- Worker ---

def record_worker(
    worker_id: int,
    experiments: list,
    train_dir: str,
    checkpoint_temp: float,
    device_str: str,
    enable_audio: bool,
    output_dir: str,
    shard_size_mb: int,
    target_secs: float,
    progress_queue,
    wandb_project: str | None = None,
):
    """Worker function for parallel recording. Runs in spawned subprocess."""
    import webdataset as wds

    port = BASE_PORT + worker_id * 10

    # Discover checkpoints for all experiments
    all_checkpoints = {}
    for exp in experiments:
        ckpts = discover_checkpoints(exp, train_dir)
        if ckpts:
            all_checkpoints[exp] = ckpts

    if not all_checkpoints:
        progress_queue.put(("error", worker_id, "No checkpoints found"))
        return

    # Load initial model (will reload on checkpoint change)
    current_exp = random.choice(list(all_checkpoints.keys()))
    current_ckpt_path = sample_checkpoint(all_checkpoints[current_exp], checkpoint_temp)
    actor_critic, rnn_size, dev = load_model(current_exp, train_dir, current_ckpt_path, device_str)
    current_ckpt_name = Path(current_ckpt_path).name

    # wandb for this worker
    wandb_run = None
    if wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                group="recording",
                name=f"record_worker_{worker_id}",
                config={
                    "worker_id": worker_id,
                    "experiments": experiments,
                    "device": device_str,
                    "target_hours": target_secs / 3600,
                },
                reinit=True,
            )
        except Exception as e:
            print(f"[worker {worker_id}] wandb init failed: {e}")

    shard_pat = os.path.join(output_dir, f"mp-{worker_id:04d}-%06d.tar")
    accumulated_secs = 0.0
    ep_count = 0

    with wds.ShardWriter(shard_pat, maxsize=shard_size_mb * 1024 * 1024) as writer:
        while accumulated_secs < target_secs:
            # Sample checkpoint (may change per episode for diversity)
            exp = random.choice(list(all_checkpoints.keys()))
            ckpt_path = sample_checkpoint(all_checkpoints[exp], checkpoint_temp)
            ckpt_name = Path(ckpt_path).name

            if ckpt_path != current_ckpt_path:
                try:
                    from sample_factory.algo.learning.learner import Learner
                    checkpoint_dict = Learner.load_checkpoint([ckpt_path], dev)
                    actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)
                    current_ckpt_path = ckpt_path
                    current_ckpt_name = ckpt_name
                except Exception as e:
                    print(f"[worker {worker_id}] checkpoint reload failed: {e}")

            scenario_name, scenario = sample_scenario()

            try:
                ep_data = record_episode(
                    actor_critic, rnn_size, dev,
                    scenario_name, scenario, port,
                    checkpoint_name=current_ckpt_name,
                    enable_audio=enable_audio,
                )
            except Exception as e:
                progress_queue.put(("error", worker_id, f"{scenario_name}: {e}"))
                time.sleep(2)
                continue

            if ep_data is None:
                progress_queue.put(("skip", worker_id, scenario_name))
                time.sleep(1)
                continue

            # Write to shard
            ep_id = str(uuid.uuid4())
            meta = {
                "episode_id": ep_id,
                "scenario": ep_data["scenario"],
                "map": ep_data["map"],
                "n_bots": ep_data["n_bots"],
                "timelimit_min": ep_data["timelimit"],
                "checkpoint_p1": ep_data["checkpoint"],
                "checkpoint_p2": ep_data["checkpoint"],
                "button_names": BUTTON_NAMES,
                "decision_interval": DECISION_INTERVAL,
                "fps": GAME_FPS,
                "resolution": "640x480",
                "n_frames": ep_data["n_frames"],
                "total_reward_p1": ep_data["total_reward_p1"],
                "total_reward_p2": ep_data["total_reward_p2"],
                "frag_p1": ep_data["frag_p1"],
                "frag_p2": ep_data["frag_p2"],
                "death_p1": ep_data["death_p1"],
                "death_p2": ep_data["death_p2"],
                "game_tics": ep_data["game_tics"],
                "duration_s": ep_data["duration_s"],
                "timestamp": time.time(),
                "worker_id": worker_id,
            }

            sample = {
                "__key__": f"ep_{ep_id}",
                "video_p1.mp4": ep_data["video_p1"],
                "video_p2.mp4": ep_data["video_p2"],
                "actions_p1.npy": _npy_bytes(ep_data["actions_p1"]),
                "actions_p2.npy": _npy_bytes(ep_data["actions_p2"]),
                "rewards_p1.npy": _npy_bytes(ep_data["rewards_p1"]),
                "rewards_p2.npy": _npy_bytes(ep_data["rewards_p2"]),
                "meta.json": json.dumps(meta).encode(),
            }
            if "audio_p1" in ep_data:
                sample["audio_p1.npy"] = _npy_bytes(ep_data["audio_p1"])
            if "audio_p2" in ep_data:
                sample["audio_p2.npy"] = _npy_bytes(ep_data["audio_p2"])

            writer.write(sample)

            game_secs = ep_data["game_tics"] / GAME_FPS
            accumulated_secs += game_secs
            ep_count += 1
            tics_per_sec = ep_data["game_tics"] / max(ep_data["duration_s"], 0.01)

            progress_queue.put((
                "done", worker_id, scenario_name, game_secs,
                ep_data["frag_p1"], ep_data["frag_p2"],
                ep_data["n_frames"], tics_per_sec,
            ))

            if wandb_run:
                try:
                    import wandb
                    wandb_run.log({
                        "episode": ep_count,
                        "game_hours": accumulated_secs / 3600,
                        "reward_p1": ep_data["total_reward_p1"],
                        "reward_p2": ep_data["total_reward_p2"],
                        "frags_p1": ep_data["frag_p1"],
                        "frags_p2": ep_data["frag_p2"],
                        "n_frames": ep_data["n_frames"],
                        "tics_per_sec": tics_per_sec,
                        "scenario": scenario_name,
                        "n_bots": ep_data["n_bots"],
                        "video_mb": (len(ep_data["video_p1"]) + len(ep_data["video_p2"])) / 1e6,
                    })
                except Exception:
                    pass

    progress_queue.put(("worker_done", worker_id, accumulated_secs, ep_count))
    if wandb_run:
        wandb_run.finish()


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Record multiplayer Doom games as WebDataset")
    parser.add_argument("--experiment", default="00_bots_128_fs2_narrow_see_0",
                        help="Experiment name(s), comma-separated for variety")
    parser.add_argument("--train-dir", default="./sf_train_dir", help="SF train directory")
    parser.add_argument("--total-hours", type=float, default=100.0, help="Target gameplay hours")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel game instances")
    parser.add_argument("--output", default="datasets/mp_recordings", help="Output directory")
    parser.add_argument("--shard-size", type=int, default=512, help="Max shard size in MB")
    parser.add_argument("--checkpoint-temp", type=float, default=0.5,
                        help="Softmax temperature for checkpoint sampling")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--enable-audio", type=bool, default=True, help="Enable audio capture")
    parser.add_argument("--wandb-project", default="doom-deathmatch", help="wandb project (empty to disable)")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiment.split(",")]
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    total_secs = args.total_hours * 3600.0
    secs_per_worker = total_secs / args.num_workers

    print(f"[doom-record] Target: {args.total_hours:.1f}h, {args.num_workers} workers")
    print(f"  Experiments: {experiments}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint temperature: {args.checkpoint_temp}")
    print(f"  Shard size: {args.shard_size} MB")

    ctx = mp.get_context("spawn")
    with mp.Manager() as manager:
        progress_queue = manager.Queue()

        procs = []
        for wid in range(args.num_workers):
            p = ctx.Process(
                target=record_worker,
                name=f"record-worker-{wid}",
                args=(
                    wid, experiments, args.train_dir,
                    args.checkpoint_temp, args.device,
                    args.enable_audio, output_dir,
                    args.shard_size, secs_per_worker,
                    progress_queue,
                    args.wandb_project if args.wandb_project else None,
                ),
            )
            p.start()
            procs.append(p)

        # Progress monitoring
        accumulated_secs = 0.0
        total_episodes = 0
        workers_done = 0

        try:
            from tqdm import tqdm
            pbar = tqdm(total=args.total_hours, unit="h", desc="Recording", ncols=100)
        except ImportError:
            pbar = None

        while workers_done < args.num_workers:
            try:
                msg = progress_queue.get(timeout=10.0)
            except Exception:
                alive = sum(1 for p in procs if p.is_alive())
                if alive == 0:
                    break
                continue

            kind = msg[0]
            if kind == "done":
                _, wid, sc, game_secs, f1, f2, nf, tps = msg
                accumulated_secs += game_secs
                total_episodes += 1
                hours = accumulated_secs / 3600.0
                if pbar:
                    pbar.n = min(hours, args.total_hours)
                    pbar.set_postfix(ep=total_episodes, sc=sc, tps=f"{tps:.0f}")
                    pbar.refresh()
                else:
                    print(f"  [{hours:.2f}h] ep={total_episodes} {sc} "
                          f"frags={f1:.0f}/{f2:.0f} frames={nf} {tps:.0f} tics/s")
            elif kind == "worker_done":
                _, wid, wsecs, weps = msg
                workers_done += 1
                print(f"  Worker {wid} done: {wsecs/3600:.2f}h, {weps} episodes")
            elif kind == "error":
                _, wid, err = msg
                print(f"  [worker {wid}] ERROR: {err}")
            elif kind == "skip":
                _, wid, sc = msg
                print(f"  [worker {wid}] skipped {sc}")

        if pbar:
            pbar.close()

        for p in procs:
            p.join(timeout=30)

    shards = sorted(Path(output_dir).glob("mp-*.tar"))
    print(f"\n[doom-record] Done. {accumulated_secs/3600:.2f}h collected, "
          f"{total_episodes} episodes, {len(shards)} shards in '{output_dir}'")


if __name__ == "__main__":
    main()
