#!/usr/bin/env python3
"""
Play against a trained AI agent in ViZDoom deathmatch.

The AI runs as the multiplayer host in a background thread.
The human joins via a visible ViZDoom window in SPECTATOR mode
(you play using the game window directly with keyboard/mouse).

Usage:
    doom-play --experiment my_run --num-bots 4 --timelimit 5
    doom-play --experiment 00_bots_128_fs2_narrow_see_0 --record results/vs_ai.mp4
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import vizdoom as vzd

from doom_arena.agent import SFAgent, _ensure_registered, load_cfg, get_reward, is_done, extract_frame

from sf_examples.vizdoom.doom.doom_utils import doom_env_by_name, make_doom_env_impl
from sf_examples.vizdoom.doom.multiplayer.doom_multiagent import find_available_port, DEFAULT_UDP_PORT


def _ai_thread(
    agent: SFAgent,
    port: int,
    num_bots: int,
    timelimit: float,
    results: dict,
):
    """Run the AI agent as multiplayer host."""
    try:
        doom_spec = doom_env_by_name("doom_deathmatch_bots")

        # max_num_players = AI + human (bots don't count)
        env = make_doom_env_impl(
            doom_spec,
            cfg=agent.cfg,
            player_id=0,
            num_agents=1,
            max_num_players=2,
            num_bots=num_bots,
            render_mode="rgb_array",
        )
        env.unwrapped.timelimit = timelimit
        env.unwrapped.init_info = {"port": port}

        print(f"[AI] Starting as host on port {port} (waiting for human to join)...")
        obs, info = env.reset()
        agent.reset_rnn()
        print(f"[AI] Game started! Fighting with {num_bots} bots.")

        ep_reward = 0.0
        step = 0

        while True:
            action = agent.act(obs)
            obs, rew, terminated, truncated, infos = env.step(action)
            ep_reward += get_reward(rew)
            step += 1

            if is_done(terminated, truncated):
                break

        env.close()

        results["ai_reward"] = ep_reward
        results["ai_steps"] = step
        if infos:
            inf = infos[0] if isinstance(infos, list) else infos
            if isinstance(inf, dict):
                results["ai_info"] = {k: float(v) for k, v in inf.items() if isinstance(v, (int, float))}

        print(f"[AI] Episode finished: reward={ep_reward:.1f}, steps={step}")

    except Exception as e:
        results["ai_error"] = str(e)
        print(f"[AI] Error: {e}")
        import traceback
        traceback.print_exc()


def _find_cfg_path():
    """Find the dwango5 deathmatch config file from the SF package."""
    import sf_examples.vizdoom.doom.scenarios as scenarios_pkg
    scenarios_dir = os.path.dirname(scenarios_pkg.__file__) if hasattr(scenarios_pkg, '__file__') else None
    if scenarios_dir is None:
        # Fallback: search relative to sf_examples
        import sf_examples.vizdoom.doom as doom_pkg
        scenarios_dir = os.path.join(os.path.dirname(doom_pkg.__file__), "scenarios")
    return os.path.join(scenarios_dir, "dwango5_dm_continuous_weap.cfg")


def main():
    ap = argparse.ArgumentParser(description="Play against a trained AI in deathmatch")
    ap.add_argument("--experiment", default="00_bots_128_fs2_narrow_see_0")
    ap.add_argument("--train-dir", default="./sf_train_dir")
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ap.add_argument("--num-bots", type=int, default=4, help="Number of built-in bots")
    ap.add_argument("--timelimit", type=float, default=5.0, help="Match duration in minutes")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--record", default=None, help="Save video of human's view to this path")
    ap.add_argument("--port", type=int, default=None, help="UDP port (auto-detected if not set)")
    args = ap.parse_args()

    _ensure_registered()

    # Load AI agent
    agent = SFAgent(args.experiment, args.train_dir, args.checkpoint, args.device)

    # Find available port
    port = args.port or find_available_port(DEFAULT_UDP_PORT)

    # Start AI in background thread
    ai_results = {}
    ai_thread = threading.Thread(
        target=_ai_thread,
        args=(agent, port, args.num_bots, args.timelimit, ai_results),
        daemon=True,
    )
    ai_thread.start()

    # Give the AI time to start hosting
    time.sleep(3)

    # Set up human player
    cfg_path = _find_cfg_path()
    human_game = vzd.DoomGame()
    human_game.load_config(cfg_path)
    human_game.set_window_visible(True)
    human_game.set_mode(vzd.Mode.SPECTATOR)
    human_game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    human_game.set_screen_format(vzd.ScreenFormat.RGB24)
    human_game.set_render_hud(True)
    human_game.set_render_crosshair(True)
    human_game.set_render_weapon(True)

    human_game.add_game_args(
        f"-join 127.0.0.1:{port} "
        f"+viz_connect_timeout 10 "
        f"+name Human +colorset 3"
    )
    human_game.set_episode_timeout(int(args.timelimit * 60 * 35))  # 35 tics/sec

    print(f"\n[Human] Joining game on port {port}...")
    print("[Human] Use keyboard/mouse in the game window to play!")
    print("[Human] Close the window or press ESC to quit.\n")

    human_game.init()

    # Human play loop
    frames = []
    while not human_game.is_episode_finished():
        human_game.advance_action()

        if human_game.is_player_dead():
            human_game.respawn_player()

        # Capture frames for recording
        if args.record:
            state = human_game.get_state()
            if state is not None and state.screen_buffer is not None:
                frame = state.screen_buffer  # (H, W, C) in RGB24
                if frame.ndim == 3:
                    frames.append(frame.copy())

    # Get human stats
    human_frags = human_game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
    human_deaths = human_game.get_game_variable(vzd.GameVariable.DEATHCOUNT)
    human_game.close()

    # Wait for AI thread to finish
    ai_thread.join(timeout=10)

    # Print results
    print("\n" + "=" * 50)
    print("  MATCH RESULTS")
    print("=" * 50)
    print(f"  Human:  {int(human_frags)} frags, {int(human_deaths)} deaths")
    if "ai_info" in ai_results:
        ai_info = ai_results["ai_info"]
        ai_frags = ai_info.get("FRAGCOUNT", 0)
        ai_deaths = ai_info.get("DEATHCOUNT", 0)
        print(f"  AI:     {int(ai_frags)} frags, {int(ai_deaths)} deaths")
    elif "ai_reward" in ai_results:
        print(f"  AI:     reward={ai_results['ai_reward']:.1f}")
    if "ai_error" in ai_results:
        print(f"  AI Error: {ai_results['ai_error']}")
    print("=" * 50)

    # Save recording
    if args.record and frames:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 35, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"\nSaved recording: {args.record} ({len(frames)} frames)")

    agent.close()


if __name__ == "__main__":
    main()
