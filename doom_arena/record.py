"""
Demo-based WebDataset recording pipeline for Doom gameplay.

Approach: play at native 160x120 (matching training resolution exactly),
record ViZDoom .lmp demo files, then replay at 640x480 for high-res video.
This gives correct policy behavior AND high-quality video output.

Supports two modes:
  - pvp: 2 AI players fighting each other via ViZDoom multiplayer
  - bots: 1 AI player + N bots (more reliable, no sync issues)

Usage:
    doom-record --experiment seed0 --total-hours 100 --num-workers 8 --device cpu
"""
from __future__ import annotations

import argparse
import io
import json
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
REPLAY_RESOLUTION = (640, 480)

# Scenarios with weights
SCENARIOS = {
    "dwango5_3min": {"wad": "dwango5.wad", "map": "map01", "bots": 4, "timelimit": 3.0, "weight": 0.40},
    "dwango5_5min": {"wad": "dwango5.wad", "map": "map01", "bots": 4, "timelimit": 5.0, "weight": 0.45},
    "ssl2_duel": {"wad": "ssl2.wad", "map": "map01", "bots": 2, "timelimit": 3.0, "weight": 0.15},
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

GAME_VARIABLES = [
    vzd.GameVariable.SELECTED_WEAPON, vzd.GameVariable.SELECTED_WEAPON_AMMO,
    vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR, vzd.GameVariable.USER2,
    vzd.GameVariable.ATTACK_READY, vzd.GameVariable.PLAYER_COUNT,
    vzd.GameVariable.FRAGCOUNT, vzd.GameVariable.DEATHCOUNT,
    vzd.GameVariable.HITCOUNT, vzd.GameVariable.DAMAGECOUNT,
]

DEATHMATCH_ARGS = (
    "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 "
    "+sv_spawnfarthest 1 +sv_nocrouch 1 +sv_nojump 1 "
    "+sv_nofreelook 1 +sv_noexit 1 "
    "+viz_respawn_delay 0 +viz_nocheat 1"
)


def _sf_scenarios_dir() -> str:
    """Get the SF scenarios directory where dwango5.wad, ssl2.wad etc. live."""
    import sf_examples.vizdoom.doom.scenarios as sc_mod
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


def _add_game_variables(game: vzd.DoomGame):
    """Add all game variables needed for measurements and stats."""
    for var in GAME_VARIABLES:
        game.add_available_game_variable(var)
    for i in range(10):
        game.add_available_game_variable(getattr(vzd.GameVariable, f"WEAPON{i}"))
    for i in range(10):
        game.add_available_game_variable(getattr(vzd.GameVariable, f"AMMO{i}"))


# --- Game Setup ---

def create_play_game(
    wad_path: str,
    doom_map: str,
    timelimit: float,
    port: int | None = None,
    is_host: bool = True,
    num_bots: int = 0,
) -> vzd.DoomGame:
    """Create game at 160x120 for policy inference + demo recording.

    Uses PLAYER mode for single-player (bots), ASYNC_PLAYER for multiplayer
    (avoids sync deadlocks; at 160x120 processing is fast enough to not skip).
    """
    game = vzd.DoomGame()
    game.set_doom_scenario_path(wad_path)
    game.set_doom_map(doom_map)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_window_visible(False)

    if port is not None:
        # Multiplayer needs ASYNC_PLAYER to avoid deadlocks
        game.set_mode(vzd.Mode.ASYNC_PLAYER)
    else:
        game.set_mode(vzd.Mode.PLAYER)

    for button in TRAINING_BUTTONS:
        game.add_available_button(button)
    _add_game_variables(game)

    if port is not None:
        # Multiplayer (pvp mode)
        if is_host:
            game.add_game_args(
                f"-host 2 -port {port} -deathmatch "
                f"+timelimit {timelimit:.1f} {DEATHMATCH_ARGS} "
                "+viz_connect_timeout 60"
            )
            game.add_game_args("+name P1 +colorset 0")
        else:
            game.add_game_args(f"-join 127.0.0.1:{port} +viz_connect_timeout 30")
            game.add_game_args("+name P2 +colorset 3")
    else:
        # Single player with bots
        game.add_game_args(
            f"-deathmatch +timelimit {timelimit:.1f} {DEATHMATCH_ARGS}"
        )
        game.add_game_args("+name Player +colorset 0")

    game.set_episode_timeout(int(timelimit * 60 * game.get_ticrate()))
    return game


def create_replay_game(wad_path: str, doom_map: str) -> vzd.DoomGame:
    """Create game at 640x480 SPECTATOR mode for replaying demos as high-res video."""
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
    game.set_mode(vzd.Mode.SPECTATOR)

    for button in TRAINING_BUTTONS:
        game.add_available_button(button)
    _add_game_variables(game)

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
    """Convert native 160x120 CHW screen buffer to model input format (128x72 CHW).

    Single-step INTER_NEAREST resize — pixel-identical to training's ResizeWrapper.
    """
    hwc = np.transpose(screen_buffer, (1, 2, 0))
    resized = cv2.resize(hwc, (128, 72), interpolation=cv2.INTER_NEAREST)
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


RANDOM_POLICY_PROB = 0.20  # per-player probability of using random actions (avoids policy bias)

# Bot count options for PvP mode (weighted toward more bots for chaotic games)
PVP_BOT_COUNTS = [0, 2, 2, 4, 4, 4, 6, 6]


def sample_random_action() -> list:
    """Sample a uniformly random action from the action space."""
    import gymnasium as gym
    from sample_factory.algo.utils.spaces.discretized import Discretized

    spaces = get_action_spaces()
    flat = []
    for space in spaces:
        act_val = np.random.randint(space.n) if hasattr(space, "n") else 0
        if isinstance(space, Discretized):
            flat.append(space.to_continuous(act_val))
        elif isinstance(space, gym.spaces.Discrete):
            one_hot = [0] * (space.n - 1)
            if act_val > 0:
                one_hot[act_val - 1] = 1
            flat.extend(one_hot)
    return flat


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


# --- Demo Replay ---

def replay_demo(wad_path: str, doom_map: str, demo_path: str) -> list:
    """Replay a .lmp demo at 640x480 and return list of RGB frames."""
    game = create_replay_game(wad_path, doom_map)
    game.init()
    game.replay_episode(demo_path)

    frames = []
    while not game.is_episode_finished():
        state = game.get_state()
        if state is not None:
            frame = np.transpose(state.screen_buffer, (1, 2, 0))
            frames.append(frame)
        game.advance_action()

    game.close()
    return frames


# --- Single-Player Episode (bots mode) ---

def _play_single_player(
    actor_critic,
    rnn_size: int,
    device: torch.device,
    wad_path: str,
    doom_map: str,
    timelimit: float,
    num_bots: int,
    demo_path: str,
    use_random_policy: bool = False,
) -> dict | None:
    """Play one episode at 160x120 with bots, recording a demo. Returns play data."""
    game = create_play_game(wad_path, doom_map, timelimit, num_bots=num_bots)
    game.init()
    for _ in range(num_bots):
        game.send_game_command("addbot")

    game.new_episode(demo_path)

    rnn = torch.zeros(1, rnn_size, device=device)
    action = [0.0] * len(TRAINING_BUTTONS)
    zero_action = [0.0] * len(TRAINING_BUTTONS)
    all_actions = []
    all_rewards = []

    tic = 0
    t0 = time.perf_counter()

    while not game.is_episode_finished():
        state = game.get_state()

        if state is None:
            # Death tic — still record action/reward for frame alignment
            all_actions.append(np.array(zero_action, dtype=np.float32))
            game.make_action(zero_action)
            all_rewards.append(float(game.get_last_reward()))
            if game.is_player_dead():
                game.respawn_player()
            tic += 1
            continue

        if tic % DECISION_INTERVAL == 0:
            if use_random_policy:
                action = sample_random_action()
            else:
                with torch.no_grad():
                    meas = extract_measurements(game)
                    obs = preprocess_for_model(state.screen_buffer, meas, device)
                    norm = actor_critic.normalize_obs(obs)
                    result = actor_critic(norm, rnn)
                    rnn = result["new_rnn_states"]
                    action = convert_action(result["actions"].cpu().numpy())

        all_actions.append(np.array(action, dtype=np.float32))
        game.make_action([float(a) for a in action])
        all_rewards.append(float(game.get_last_reward()))

        if game.is_player_dead():
            game.respawn_player()
        tic += 1

    duration = time.perf_counter() - t0

    frag = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
    death = game.get_game_variable(vzd.GameVariable.DEATHCOUNT)
    hits = game.get_game_variable(vzd.GameVariable.HITCOUNT)
    dmg = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
    game.close()

    if not all_actions:
        return None

    return {
        "actions": np.stack(all_actions),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "game_tics": tic,
        "duration_s": duration,
        "frag": float(frag),
        "death": float(death),
        "hits": float(hits),
        "damage": float(dmg),
    }


# --- Multiplayer Episode (pvp mode) ---

def _play_multiplayer(
    actor_critic,
    rnn_size: int,
    device: torch.device,
    wad_path: str,
    doom_map: str,
    timelimit: float,
    port: int,
    num_bots: int = 0,
    use_random_p1: bool = False,
    use_random_p2: bool = False,
) -> dict | None:
    """Play one 2-player episode at 160x120, recording frames and actions.

    Uses ASYNC_PLAYER mode with threaded make_action to avoid sync deadlocks.
    No demo recording (not supported in ASYNC_PLAYER multiplayer) — instead
    captures 160x120 frames directly and upscales to 640x480 for video.
    Optionally adds bots alongside the 2 AI players for more chaotic games.
    Each player independently may use random or trained policy.
    """
    host_game = create_play_game(wad_path, doom_map, timelimit, port=port, is_host=True)
    join_game = create_play_game(wad_path, doom_map, timelimit, port=port, is_host=False)

    # Init via threads (host blocks waiting for join)
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
        for g in [host_game, join_game]:
            try:
                g.close()
            except Exception:
                pass
        print(f"[record] Init failed: host={errors[0]}, join={errors[1]}")
        return None

    # Add bots (host only, after init but before new_episode)
    for _ in range(num_bots):
        host_game.send_game_command("addbot")

    # Start episodes via threads (new_episode() syncs between players)
    ep_errors = [None, None]
    def start_host_ep():
        try:
            host_game.new_episode()
        except Exception as e:
            ep_errors[0] = e
    def start_join_ep():
        try:
            join_game.new_episode()
        except Exception as e:
            ep_errors[1] = e

    te_host = threading.Thread(target=start_host_ep)
    te_join = threading.Thread(target=start_join_ep)
    te_host.start()
    te_join.start()
    te_host.join(timeout=30)
    te_join.join(timeout=30)

    if ep_errors[0] or ep_errors[1]:
        for g in [host_game, join_game]:
            try:
                g.close()
            except Exception:
                pass
        print(f"[record] new_episode failed: host={ep_errors[0]}, join={ep_errors[1]}")
        return None

    rnn_p1 = torch.zeros(1, rnn_size, device=device)
    rnn_p2 = torch.zeros(1, rnn_size, device=device)
    action_p1 = [0.0] * len(TRAINING_BUTTONS)
    action_p2 = [0.0] * len(TRAINING_BUTTONS)
    zero_action = [0.0] * len(TRAINING_BUTTONS)

    frames_p1, frames_p2 = [], []
    actions_p1, actions_p2 = [], []
    rewards_p1, rewards_p2 = [], []

    tic = 0
    t0 = time.perf_counter()
    last_progress = time.perf_counter()
    frag_p1 = frag_p2 = death_p1 = death_p2 = 0.0

    def _advance_both(act_h, act_j):
        """Advance both games via threads to avoid multiplayer sync deadlocks."""
        t1 = threading.Thread(target=host_game.make_action, args=(act_h,))
        t2 = threading.Thread(target=join_game.make_action, args=(act_j,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

    while not host_game.is_episode_finished() and not join_game.is_episode_finished():
        state_p1 = host_game.get_state()
        state_p2 = join_game.get_state()

        if state_p1 is None or state_p2 is None:
            # Death tic — record zero actions for frame alignment
            actions_p1.append(np.array(zero_action, dtype=np.float32))
            actions_p2.append(np.array(zero_action, dtype=np.float32))
            _advance_both(zero_action, zero_action)
            rewards_p1.append(float(host_game.get_last_reward()))
            rewards_p2.append(float(join_game.get_last_reward()))
            if host_game.is_player_dead():
                host_game.respawn_player()
            if join_game.is_player_dead():
                join_game.respawn_player()
            tic += 1
            last_progress = time.perf_counter()
            continue

        # Record frames (160x120 → upscale to 640x480 later)
        frame_p1 = np.transpose(state_p1.screen_buffer, (1, 2, 0))
        frame_p2 = np.transpose(state_p2.screen_buffer, (1, 2, 0))
        frames_p1.append(frame_p1)
        frames_p2.append(frame_p2)

        if tic % DECISION_INTERVAL == 0:
            # P1 action
            if use_random_p1:
                action_p1 = sample_random_action()
            else:
                with torch.no_grad():
                    meas_p1 = extract_measurements(host_game)
                    obs_p1 = preprocess_for_model(state_p1.screen_buffer, meas_p1, device)
                    norm_p1 = actor_critic.normalize_obs(obs_p1)
                    res_p1 = actor_critic(norm_p1, rnn_p1)
                    rnn_p1 = res_p1["new_rnn_states"]
                    action_p1 = convert_action(res_p1["actions"].cpu().numpy())

            # P2 action
            if use_random_p2:
                action_p2 = sample_random_action()
            else:
                with torch.no_grad():
                    meas_p2 = extract_measurements(join_game)
                    obs_p2 = preprocess_for_model(state_p2.screen_buffer, meas_p2, device)
                    norm_p2 = actor_critic.normalize_obs(obs_p2)
                    res_p2 = actor_critic(norm_p2, rnn_p2)
                    rnn_p2 = res_p2["new_rnn_states"]
                    action_p2 = convert_action(res_p2["actions"].cpu().numpy())

        actions_p1.append(np.array(action_p1, dtype=np.float32))
        actions_p2.append(np.array(action_p2, dtype=np.float32))

        if tic % DECISION_INTERVAL == 0:
            frag_p1 = host_game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
            frag_p2 = join_game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
            death_p1 = host_game.get_game_variable(vzd.GameVariable.DEATHCOUNT)
            death_p2 = join_game.get_game_variable(vzd.GameVariable.DEATHCOUNT)

        act_list_p1 = [float(a) for a in action_p1]
        act_list_p2 = [float(a) for a in action_p2]
        _advance_both(act_list_p1, act_list_p2)
        rewards_p1.append(float(host_game.get_last_reward()))
        rewards_p2.append(float(join_game.get_last_reward()))

        if host_game.is_player_dead():
            host_game.respawn_player()
        if join_game.is_player_dead():
            join_game.respawn_player()

        tic += 1
        last_progress = time.perf_counter()

        # Deadlock detection
        if time.perf_counter() - last_progress > 30:
            print(f"[record] Deadlock detected at tic {tic}, aborting")
            break

    duration = time.perf_counter() - t0

    host_game.close()
    join_game.close()

    if not frames_p1:
        return None

    # Upscale 160x120 frames to 640x480 for video
    w, h = REPLAY_RESOLUTION
    upscaled_p1 = [cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC) for f in frames_p1]
    upscaled_p2 = [cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC) for f in frames_p2]

    n_frames = len(upscaled_p1)
    return {
        "frames_p1": upscaled_p1,
        "frames_p2": upscaled_p2,
        "actions_p1": np.stack(actions_p1[:n_frames]),
        "actions_p2": np.stack(actions_p2[:n_frames]),
        "rewards_p1": np.array(rewards_p1[:n_frames], dtype=np.float32),
        "rewards_p2": np.array(rewards_p2[:n_frames], dtype=np.float32),
        "n_frames": n_frames,
        "game_tics": tic,
        "duration_s": duration,
        "frag_p1": float(frag_p1),
        "frag_p2": float(frag_p2),
        "death_p1": float(death_p1),
        "death_p2": float(death_p2),
    }


# --- Episode Recording (main entry point) ---

def record_episode(
    actor_critic,
    rnn_size: int,
    device: torch.device,
    scenario_name: str,
    scenario: dict,
    port: int,
    checkpoint_name: str = "",
    mode: str = "bots",
) -> dict | None:
    """Record a single episode.

    Bots mode: play at 160x120, save .lmp demo, replay at 640x480 for high-res video
    PvP mode: play at 160x120 with 2 AI players (threaded), upscale frames to 640x480

    Args:
        mode: "bots" (1 player + bots, demo-based) or "pvp" (2 AI players, direct capture)
    """
    try:
        wad_path = _find_wad(scenario["wad"])
    except FileNotFoundError as e:
        print(f"[record] {e}")
        return None

    doom_map = scenario["map"]
    timelimit = scenario["timelimit"]
    num_bots = scenario["bots"]

    if mode == "pvp":
        # --- PvP: randomize bot count and per-player random policy ---
        num_bots = random.choice(PVP_BOT_COUNTS)
        use_random_p1 = random.random() < RANDOM_POLICY_PROB
        use_random_p2 = random.random() < RANDOM_POLICY_PROB

        play_data = _play_multiplayer(
            actor_critic, rnn_size, device,
            wad_path, doom_map, timelimit, port,
            num_bots=num_bots,
            use_random_p1=use_random_p1,
            use_random_p2=use_random_p2,
        )
        if play_data is None:
            return None

        n_frames = play_data["n_frames"]
        video_p1 = encode_video(play_data["frames_p1"], GAME_FPS)
        video_p2 = encode_video(play_data["frames_p2"], GAME_FPS)

        return {
            "video_p1": video_p1,
            "video_p2": video_p2,
            "demo_p1": None,
            "demo_p2": None,
            "actions_p1": play_data["actions_p1"],
            "actions_p2": play_data["actions_p2"],
            "rewards_p1": play_data["rewards_p1"],
            "rewards_p2": play_data["rewards_p2"],
            "n_frames": n_frames,
            "game_tics": play_data["game_tics"],
            "duration_s": play_data["duration_s"],
            "scenario": scenario_name,
            "map": doom_map,
            "n_bots": num_bots,
            "timelimit": timelimit,
            "checkpoint": checkpoint_name,
            "mode": mode,
            "frag_p1": play_data["frag_p1"],
            "frag_p2": play_data["frag_p2"],
            "death_p1": play_data["death_p1"],
            "death_p2": play_data["death_p2"],
            "total_reward_p1": float(np.sum(play_data["rewards_p1"])),
            "total_reward_p2": float(np.sum(play_data["rewards_p2"])),
            "random_policy_p1": use_random_p1,
            "random_policy_p2": use_random_p2,
        }

    else:
        # --- Bots: demo-based (play at 160x120, replay at 640x480) ---
        use_random_policy = random.random() < RANDOM_POLICY_PROB
        tmp_dir = tempfile.mkdtemp(prefix="doom_demo_")
        demo_path = os.path.join(tmp_dir, "p1.lmp")

        try:
            play_data = _play_single_player(
                actor_critic, rnn_size, device,
                wad_path, doom_map, timelimit, num_bots,
                demo_path,
                use_random_policy=use_random_policy,
            )
            if play_data is None:
                return None

            # Replay at 640x480
            frames = replay_demo(wad_path, doom_map, demo_path)
            demo_bytes = Path(demo_path).read_bytes()

            if not frames:
                return None

            # Align: trim to match action count
            n_actions = len(play_data["actions"])
            if len(frames) > n_actions:
                frames = frames[:n_actions]
            elif len(frames) < n_actions:
                play_data["actions"] = play_data["actions"][:len(frames)]
                play_data["rewards"] = play_data["rewards"][:len(frames)]

            n_frames = len(frames)
            video_p1 = encode_video(frames, GAME_FPS)

            return {
                "video_p1": video_p1,
                "video_p2": b"",
                "demo_p1": demo_bytes,
                "demo_p2": None,
                "actions_p1": play_data["actions"][:n_frames],
                "actions_p2": np.zeros((0, len(TRAINING_BUTTONS)), dtype=np.float32),
                "rewards_p1": play_data["rewards"][:n_frames],
                "rewards_p2": np.array([], dtype=np.float32),
                "n_frames": n_frames,
                "game_tics": play_data["game_tics"],
                "duration_s": play_data["duration_s"],
                "scenario": scenario_name,
                "map": doom_map,
                "n_bots": num_bots,
                "timelimit": timelimit,
                "checkpoint": checkpoint_name,
                "mode": mode,
                "frag_p1": play_data["frag"],
                "frag_p2": 0.0,
                "death_p1": play_data["death"],
                "death_p2": 0.0,
                "total_reward_p1": float(np.sum(play_data["rewards"][:n_frames])),
                "total_reward_p2": 0.0,
                "random_policy": use_random_policy,
            }

        finally:
            try:
                os.unlink(demo_path)
            except OSError:
                pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass


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
    game_mode: str,
    output_dir: str,
    shard_size_mb: int,
    target_secs: float,
    progress_queue,
    wandb_project: str | None = None,
    base_port: int = BASE_PORT,
):
    """Worker function for parallel recording. Runs in spawned subprocess."""
    # Limit PyTorch to 1 CPU thread per worker to avoid contention
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    import webdataset as wds

    port = base_port + worker_id * 10

    # Discover checkpoints for all experiments
    all_checkpoints = {}
    for exp in experiments:
        ckpts = discover_checkpoints(exp, train_dir)
        if ckpts:
            all_checkpoints[exp] = ckpts

    if not all_checkpoints:
        progress_queue.put(("error", worker_id, "No checkpoints found"))
        return

    # Load initial model
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
                    "mode": game_mode,
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
                    mode=game_mode,
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
                "play_resolution": "160x120",
                "video_resolution": "640x480",
                "mode": ep_data["mode"],
                "n_frames": ep_data["n_frames"],
                "total_reward_p1": ep_data["total_reward_p1"],
                "total_reward_p2": ep_data["total_reward_p2"],
                "frag_p1": ep_data["frag_p1"],
                "frag_p2": ep_data["frag_p2"],
                "death_p1": ep_data["death_p1"],
                "death_p2": ep_data["death_p2"],
                "game_tics": ep_data["game_tics"],
                "duration_s": ep_data["duration_s"],
                "random_policy_p1": ep_data.get("random_policy_p1", ep_data.get("random_policy", False)),
                "random_policy_p2": ep_data.get("random_policy_p2", ep_data.get("random_policy", False)),
                "timestamp": time.time(),
                "worker_id": worker_id,
            }

            sample = {
                "__key__": f"ep_{ep_id}",
                "video_p1.mp4": ep_data["video_p1"],
                "actions_p1.npy": _npy_bytes(ep_data["actions_p1"]),
                "rewards_p1.npy": _npy_bytes(ep_data["rewards_p1"]),
                "meta.json": json.dumps(meta).encode(),
            }
            if ep_data.get("demo_p1"):
                sample["demo_p1.lmp"] = ep_data["demo_p1"]
            if ep_data["video_p2"]:
                sample["video_p2.mp4"] = ep_data["video_p2"]
            if ep_data["actions_p2"].size > 0:
                sample["actions_p2.npy"] = _npy_bytes(ep_data["actions_p2"])
                sample["rewards_p2.npy"] = _npy_bytes(ep_data["rewards_p2"])
            if ep_data.get("demo_p2"):
                sample["demo_p2.lmp"] = ep_data["demo_p2"]

            writer.write(sample)

            game_secs = ep_data["game_tics"] / GAME_FPS
            accumulated_secs += game_secs
            ep_count += 1
            tics_per_sec = ep_data["game_tics"] / max(ep_data["duration_s"], 0.01)

            progress_queue.put((
                "done", worker_id, scenario_name, game_secs,
                ep_data["frag_p1"], ep_data["frag_p2"],
                ep_data["n_frames"], tics_per_sec,
                ep_data["n_bots"],
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
                        "video_mb": len(ep_data["video_p1"]) / 1e6,
                    })
                except Exception:
                    pass

    progress_queue.put(("worker_done", worker_id, accumulated_secs, ep_count))
    if wandb_run:
        wandb_run.finish()


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Record Doom games as WebDataset (demo-based)")
    parser.add_argument("--experiment", default="00_bots_128_fs2_narrow_see_0",
                        help="Experiment name(s), comma-separated for variety")
    parser.add_argument("--train-dir", default="./sf_train_dir", help="SF train directory")
    parser.add_argument("--total-hours", type=float, default=100.0, help="Target gameplay hours")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel game instances")
    parser.add_argument("--output", default="datasets/mp_recordings", help="Output directory")
    parser.add_argument("--shard-size", type=int, default=512, help="Max shard size in MB")
    parser.add_argument("--checkpoint-temp", type=float, default=0.5,
                        help="Softmax temperature for checkpoint sampling")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--mode", choices=["bots", "pvp"], default="bots",
                        help="Game mode: bots (1 player + bots) or pvp (2 AI players)")
    parser.add_argument("--wandb-project", default="", help="wandb project (empty to disable)")
    parser.add_argument("--base-port", type=int, default=BASE_PORT,
                        help="Base port for multiplayer (offset by worker_id*10)")
    parser.add_argument("--worker-id-offset", type=int, default=0,
                        help="Offset added to worker IDs (for multi-node runs)")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiment.split(",")]
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    total_secs = args.total_hours * 3600.0
    secs_per_worker = total_secs / args.num_workers

    print(f"[doom-record] Target: {args.total_hours:.1f}h, {args.num_workers} workers, mode={args.mode}")
    print(f"  Experiments: {experiments}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Play: 160x120 | Video: 640x480 | Demo-based recording")

    ctx = mp.get_context("spawn")
    with mp.Manager() as manager:
        progress_queue = manager.Queue()

        procs = []
        for local_wid in range(args.num_workers):
            wid = local_wid + args.worker_id_offset
            p = ctx.Process(
                target=record_worker,
                name=f"record-worker-{wid}",
                args=(
                    wid, experiments, args.train_dir,
                    args.checkpoint_temp, args.device,
                    args.mode, output_dir,
                    args.shard_size, secs_per_worker,
                    progress_queue,
                    args.wandb_project if args.wandb_project else None,
                    args.base_port,
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
                _, wid, sc, game_secs, f1, f2, nf, tps, nb = msg
                accumulated_secs += game_secs
                total_episodes += 1
                hours = accumulated_secs / 3600.0
                if pbar:
                    pbar.n = min(hours, args.total_hours)
                    pbar.set_postfix(ep=total_episodes, sc=sc, bots=nb, tps=f"{tps:.0f}")
                    pbar.refresh()
                else:
                    print(f"  [{hours:.2f}h] ep={total_episodes} {sc} "
                          f"frags={f1:.0f}/{f2:.0f} frames={nf} bots={nb} {tps:.0f} tics/s")
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
