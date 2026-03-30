"""
Record human-vs-AI gameplay as WebDataset shards.

The AI runs as a multiplayer host (ASYNC_PLAYER, 160x120, hidden window)
in a background thread. The human joins in SPECTATOR mode (640x480, visible
window) with keyboard/mouse in the main thread. Both players' frames,
actions, and rewards are recorded independently and aligned afterward.

Usage:
    doom-record-human --experiment my_run --num-bots 4 --timelimit 5
    doom-record-human --experiment my_run --episodes 3 --output datasets/human_recordings
    doom-record-human-arena --experiment my_run   # you + best checkpoint + 7 bots, 5 min → datasets/recordings
    uv run python -m doom_arena.record_human arena --experiment my_run   # if console scripts break
"""
from __future__ import annotations

import argparse
import json
import os
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

from doom_arena.record import (
    TRAINING_BUTTONS,
    BUTTON_NAMES,
    DEATHMATCH_ARGS,
    DECISION_INTERVAL,
    GAME_FPS,
    REPLAY_RESOLUTION,
    _add_game_variables,
    _find_wad,
    _npy_bytes,
    convert_action,
    create_play_game,
    encode_video,
    extract_measurements,
    load_model,
    preprocess_for_model,
)
from sf_examples.vizdoom.doom.multiplayer.doom_multiagent import (
    DEFAULT_UDP_PORT,
    find_available_port,
)

_HUMAN_SHARD_RE = re.compile(r"^human-(\d{6})\.tar$")


def _next_human_shard_start(output_dir: str) -> int:
    """First shard index that does not collide with existing human-######.tar files."""
    out = Path(output_dir)
    max_idx = -1
    for p in out.glob("human-*.tar"):
        m = _HUMAN_SHARD_RE.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def create_human_game(
    wad_path: str,
    doom_map: str,
    timelimit: float,
    port: int,
) -> vzd.DoomGame:
    """Create the human player's game: SPECTATOR mode, 640x480, visible window."""
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
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    for button in TRAINING_BUTTONS:
        game.add_available_button(button)
    _add_game_variables(game)

    game.add_game_args(
        f"-join 127.0.0.1:{port} "
        f"+viz_connect_timeout 30 "
        f"+name Human +colorset 3"
    )

    game.set_episode_timeout(int(timelimit * 60 * game.get_ticrate()))
    return game


def _configure_human_controls(game: vzd.DoomGame):
    """Set up WASD + Shift + 1-7 controls and low mouse sensitivity."""
    binds = [
        'bind w +forward',
        'bind s +back',
        'bind a +moveleft',
        'bind d +moveright',
        'bind shift +speed',
        'bind 1 "slot 1"',
        'bind 2 "slot 2"',
        'bind 3 "slot 3"',
        'bind 4 "slot 4"',
        'bind 5 "slot 5"',
        'bind 6 "slot 6"',
        'bind 7 "slot 7"',
    ]
    for cmd in binds:
        game.send_game_command(cmd)
    game.send_game_command("set mouse_sensitivity 0.1")


def _ai_thread_fn(
    actor_critic,
    rnn_size: int,
    device: torch.device,
    ai_game: vzd.DoomGame,
    results: dict,
):
    """Run the AI agent loop in a background thread, recording frames/actions/rewards."""
    try:
        rnn = torch.zeros(1, rnn_size, device=device)
        ai_action = [0.0] * len(TRAINING_BUTTONS)
        zero_action = [0.0] * len(TRAINING_BUTTONS)

        frames = []
        actions = []
        rewards = []
        tic = 0

        while not ai_game.is_episode_finished():
            state = ai_game.get_state()

            if state is None:
                # Death tic
                actions.append(np.array(zero_action, dtype=np.float32))
                ai_game.make_action(zero_action)
                rewards.append(float(ai_game.get_last_reward()))
                if ai_game.is_player_dead():
                    ai_game.respawn_player()
                tic += 1
                continue

            # Capture frame (160x120)
            frame = np.transpose(state.screen_buffer, (1, 2, 0))
            frames.append(frame)

            # Decide action
            if tic % DECISION_INTERVAL == 0:
                with torch.no_grad():
                    meas = extract_measurements(ai_game)
                    obs = preprocess_for_model(state.screen_buffer, meas, device)
                    norm = actor_critic.normalize_obs(obs)
                    result = actor_critic(norm, rnn)
                    rnn = result["new_rnn_states"]
                    ai_action = convert_action(result["actions"].cpu().numpy())

            actions.append(np.array(ai_action, dtype=np.float32))
            ai_game.make_action([float(a) for a in ai_action])
            rewards.append(float(ai_game.get_last_reward()))

            if ai_game.is_player_dead():
                ai_game.respawn_player()
            tic += 1

        results["frames"] = frames
        results["actions"] = actions
        results["rewards"] = rewards
        results["tic"] = tic
        results["frag"] = float(ai_game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
        results["death"] = float(ai_game.get_game_variable(vzd.GameVariable.DEATHCOUNT))

    except Exception as e:
        results["error"] = str(e)
        import traceback
        traceback.print_exc()


def play_and_record(
    actor_critic,
    rnn_size: int,
    device: torch.device,
    wad_path: str,
    doom_map: str,
    timelimit: float,
    port: int,
    num_bots: int,
) -> dict | None:
    """Play one human-vs-AI episode, recording both perspectives.

    AI runs in a background thread; human runs in the main thread.
    The ViZDoom multiplayer engine handles synchronization internally.

    Returns episode data dict or None on failure.
    """
    # Create both games
    ai_game = create_play_game(
        wad_path, doom_map, timelimit, port=port, is_host=True, num_bots=0,
    )
    human_game = create_human_game(wad_path, doom_map, timelimit, port)

    # Init via threads (host blocks waiting for join)
    errors = [None, None]

    def init_ai():
        try:
            ai_game.init()
        except Exception as e:
            errors[0] = e

    def init_human():
        try:
            time.sleep(2)
            human_game.init()
        except Exception as e:
            errors[1] = e

    t_ai = threading.Thread(target=init_ai)
    t_human = threading.Thread(target=init_human)
    t_ai.start()
    t_human.start()
    t_ai.join(timeout=45)
    t_human.join(timeout=45)

    if errors[0] or errors[1]:
        for g in [ai_game, human_game]:
            try:
                g.close()
            except Exception:
                pass
        print(f"[record-human] Init failed: ai={errors[0]}, human={errors[1]}")
        return None

    # Configure human controls (must be after init)
    _configure_human_controls(human_game)

    # Start episodes via threads (new_episode resets the map; bots must be added after)
    ep_errors = [None, None]

    def start_ai_ep():
        try:
            ai_game.new_episode()
        except Exception as e:
            ep_errors[0] = e

    def start_human_ep():
        try:
            human_game.new_episode()
        except Exception as e:
            ep_errors[1] = e

    te_ai = threading.Thread(target=start_ai_ep)
    te_human = threading.Thread(target=start_human_ep)
    te_ai.start()
    te_human.start()
    te_ai.join(timeout=30)
    te_human.join(timeout=30)

    if ep_errors[0] or ep_errors[1]:
        for g in [ai_game, human_game]:
            try:
                g.close()
            except Exception:
                pass
        print(f"[record-human] new_episode failed: ai={ep_errors[0]}, human={ep_errors[1]}")
        return None

    # Match Sample Factory VizdoomEnvMultiplayer.reset(): add bots only after the episode has started.
    if num_bots > 0:
        ai_game.send_game_command("removebots")
        for _ in range(num_bots):
            ai_game.send_game_command("addbot")

    print("[record-human] Game started! Use keyboard/mouse in the game window.")
    print("[record-human] Press Ctrl+C to end early and save the recording.\n")

    # --- Run AI in background thread ---
    ai_results = {}
    ai_thread = threading.Thread(
        target=_ai_thread_fn,
        args=(actor_critic, rnn_size, device, ai_game, ai_results),
        daemon=True,
    )
    ai_thread.start()

    # --- Run human in main thread ---
    zero_action = [0.0] * len(TRAINING_BUTTONS)
    human_actions = []
    human_rewards = []

    # Write human frames to temp video on-the-fly (avoids multi-GB RAM)
    w, h = REPLAY_RESOLUTION
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video_path = tmp_video.name
    tmp_video.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    human_writer = cv2.VideoWriter(tmp_video_path, fourcc, GAME_FPS, (w, h))
    human_frame_count = 0

    t0 = time.perf_counter()

    try:
        while not human_game.is_episode_finished():
            human_game.advance_action()

            state = human_game.get_state()
            if state is not None and state.screen_buffer is not None:
                frame = np.transpose(state.screen_buffer, (1, 2, 0))
                human_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                human_frame_count += 1

            # Record human's action
            human_act = human_game.get_last_action()
            human_actions.append(np.array(human_act, dtype=np.float32))
            human_rewards.append(float(human_game.get_last_reward()))

            if human_game.is_player_dead():
                human_game.respawn_player()
    except KeyboardInterrupt:
        print("\n[record-human] Ctrl+C received — ending match and saving recording...")

    duration = time.perf_counter() - t0
    frag_human = frag_ai = death_human = death_ai = 0.0
    try:
        frag_human = float(human_game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
        death_human = float(human_game.get_game_variable(vzd.GameVariable.DEATHCOUNT))
    except Exception:
        pass

    human_writer.release()
    try:
        human_game.close()
    except Exception:
        pass

    # Wait for AI thread to finish
    ai_thread.join(timeout=15)
    try:
        ai_game.close()
    except Exception:
        pass

    if "error" in ai_results:
        print(f"[record-human] AI error: {ai_results['error']}")
        os.unlink(tmp_video_path)
        return None

    if not ai_results.get("frames"):
        print("[record-human] No AI frames captured.")
        os.unlink(tmp_video_path)
        return None

    # Read human video bytes
    with open(tmp_video_path, "rb") as f:
        human_video_bytes = f.read()
    os.unlink(tmp_video_path)

    # Upscale AI frames to 640x480 and encode
    ai_frames_up = [cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
                     for f in ai_results["frames"]]
    ai_video_bytes = encode_video(ai_frames_up, GAME_FPS)

    n_human = len(human_actions)
    n_ai = len(ai_results["actions"])
    n_frames = min(n_human, n_ai, human_frame_count, len(ai_results["frames"]))

    frag_ai = ai_results.get("frag", 0.0)
    death_ai = ai_results.get("death", 0.0)

    # Print results
    print(f"\n{'=' * 50}")
    print("  MATCH RESULTS")
    print(f"{'=' * 50}")
    print(f"  Human:  {int(frag_human)} frags, {int(death_human)} deaths")
    print(f"  AI:     {int(frag_ai)} frags, {int(death_ai)} deaths")
    print(f"  Frames: {n_frames} (human={n_human}, ai={n_ai}), Duration: {duration:.1f}s")
    print(f"{'=' * 50}")

    return {
        "human_video": human_video_bytes,
        "ai_video": ai_video_bytes,
        "human_actions": np.stack(human_actions[:n_frames]),
        "ai_actions": np.stack(ai_results["actions"][:n_frames]),
        "human_rewards": np.array(human_rewards[:n_frames], dtype=np.float32),
        "ai_rewards": np.array(ai_results["rewards"][:n_frames], dtype=np.float32),
        "n_frames": n_frames,
        "game_tics": max(n_human, n_ai),
        "duration_s": duration,
        "frag_human": frag_human,
        "frag_ai": frag_ai,
        "death_human": death_human,
        "death_ai": death_ai,
    }


def _build_record_human_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--experiment", default="00_bots_128_fs2_narrow_see_0",
                   help="AI model experiment name")
    p.add_argument("--train-dir", default="./sf_train_dir")
    p.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    p.add_argument("--num-bots", type=int, default=4, help="Number of built-in DM bots (addbot)")
    p.add_argument("--timelimit", type=float, default=5.0, help="Match duration in minutes")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="datasets/human_recordings", help="Output directory")
    p.add_argument("--port", type=int, default=None, help="UDP port (auto-detected if not set)")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    p.add_argument("--wad", default="dwango5.wad", help="WAD file name")
    p.add_argument("--map", default="map01", help="Map name")
    return p


def run_record_human(args: argparse.Namespace) -> None:
    import webdataset as wds
    from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

    register_vizdoom_components()

    wad_path = _find_wad(args.wad)
    port = args.port or find_available_port(DEFAULT_UDP_PORT)

    from doom_arena.record import select_checkpoint_path

    ckpt_path = select_checkpoint_path(args.experiment, args.train_dir, args.checkpoint)
    ckpt_path = str(Path(ckpt_path).resolve())

    actor_critic, rnn_size, dev = load_model(
        args.experiment, args.train_dir, ckpt_path, args.device,
    )
    ckpt_name = Path(ckpt_path).name
    m_rw = re.search(r"reward_([-\d.]+)\.pth$", ckpt_name)
    reward_tag = float(m_rw.group(1)) if m_rw else None
    rw_line = f"score tag in filename: {reward_tag}" if reward_tag is not None else "no reward_* tag in filename"

    print(
        "\n"
        + "=" * 72
        + "\n"
        "  HUMAN RECORDING — who is controlled by what\n"
        "  • You (human): spectator client, keyboard/mouse.\n"
        "  • Exactly ONE neural policy: multiplayer host (this checkpoint).\n"
        "  • --num-bots are ZDoom engine bots (addbot), NOT extra checkpoints.\n"
        "    They will not feel like “different policies” you trained.\n"
        "=" * 72
        + f"\n  experiment:      {args.experiment}\n"
        f"  selection mode:    {args.checkpoint}\n"
        f"  checkpoint file:   {ckpt_name}\n"
        f"  {rw_line}\n"
        f"  full path:         {ckpt_path}\n"
        f"  device:            {args.device}\n"
        + "=" * 72
        + "\n"
    )

    os.makedirs(args.output, exist_ok=True)
    shard_pat = os.path.join(args.output, "human-%06d.tar")
    start_shard = _next_human_shard_start(args.output)
    if start_shard > 0:
        print(f"[record-human] Resuming shard numbering at human-{start_shard:06d}.tar")

    with wds.ShardWriter(
        shard_pat, maxsize=512 * 1024 * 1024, start_shard=start_shard
    ) as writer:
        for ep_idx in range(args.episodes):
            print(f"\n--- Episode {ep_idx + 1}/{args.episodes} ---")
            print(f"Playing on {args.wad} {args.map} with {args.num_bots} bots, "
                  f"{args.timelimit:.1f} min")

            ep_data = play_and_record(
                actor_critic, rnn_size, dev,
                wad_path, args.map, args.timelimit,
                port, args.num_bots,
            )

            if ep_data is None:
                print("[record-human] Episode failed, skipping.")
                continue

            ep_id = str(uuid.uuid4())
            meta = {
                "episode_id": ep_id,
                "mode": "human",
                "is_human_p1": True,
                "learned_agent_count": 1,
                "engine_bots": args.num_bots,
                "checkpoint_reward_in_filename": reward_tag,
                "scenario": f"{args.wad.replace('.wad', '')}_{args.timelimit:.0f}min",
                "map": args.map,
                "n_bots": args.num_bots,
                "timelimit_min": args.timelimit,
                "checkpoint_p2": ckpt_name,
                "button_names": BUTTON_NAMES,
                "decision_interval": DECISION_INTERVAL,
                "fps": GAME_FPS,
                "play_resolution": "160x120",
                "video_resolution": "640x480",
                "n_frames": ep_data["n_frames"],
                "total_reward_p1": float(np.sum(ep_data["human_rewards"])),
                "total_reward_p2": float(np.sum(ep_data["ai_rewards"])),
                "frag_p1": ep_data["frag_human"],
                "frag_p2": ep_data["frag_ai"],
                "death_p1": ep_data["death_human"],
                "death_p2": ep_data["death_ai"],
                "game_tics": ep_data["game_tics"],
                "duration_s": ep_data["duration_s"],
                "random_policy_p1": False,
                "random_policy_p2": False,
                "timestamp": time.time(),
            }

            sample = {
                "__key__": f"ep_{ep_id}",
                "video_p1.mp4": ep_data["human_video"],
                "video_p2.mp4": ep_data["ai_video"],
                "actions_p1.npy": _npy_bytes(ep_data["human_actions"]),
                "actions_p2.npy": _npy_bytes(ep_data["ai_actions"]),
                "rewards_p1.npy": _npy_bytes(ep_data["human_rewards"]),
                "rewards_p2.npy": _npy_bytes(ep_data["ai_rewards"]),
                "meta.json": json.dumps(meta).encode(),
            }
            writer.write(sample)
            print(f"[record-human] Episode {ep_idx + 1} saved ({ep_data['n_frames']} frames)")

    shards = sorted(Path(args.output).glob("human-*.tar"))
    print(f"\n[record-human] Done. {args.episodes} episodes, {len(shards)} shards in '{args.output}'")


def main():
    parser = _build_record_human_parser("Record human-vs-AI gameplay as WebDataset shards")
    args = parser.parse_args()
    run_record_human(args)


def main_arena():
    """Human + best-reward checkpoint AI + 7 DM bots, 5 min, WebDataset under datasets/recordings."""
    parser = _build_record_human_parser(
        "Record one 5-minute match: you vs best checkpoint AI with 7 built-in bots → datasets/recordings"
    )
    parser.set_defaults(
        timelimit=5.0,
        num_bots=7,
        episodes=1,
        output="datasets/recordings",
        checkpoint="best",
    )
    args = parser.parse_args()
    print(
        "[record-human-arena] Human (you) + trained AI host + 7 engine bots | "
        f"{args.timelimit:g} min → {args.output}/human-*.tar"
    )
    run_record_human(args)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "arena":
        sys.argv.pop(1)
        main_arena()
    else:
        main()
