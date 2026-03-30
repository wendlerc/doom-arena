#!/usr/bin/env python3
"""
Headless check: 2-player ViZDoom deathmatch + addbot N.

PLAYER_COUNT is only *human* net players (typically 2 here), not bots — do not use it
to count bots. We log host HITCOUNT/DAMAGECOUNT/FRAGCOUNT to see if combat is happening.

Mirrors record_human ordering (new_episode first, then removebots/addbot on host).

Usage (from repo root, package on PYTHONPATH):
  PYTHONPATH=. uv run python scripts/diagnose_mp_bots.py
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import vizdoom as vzd

from doom_arena.record import (
    DEATHMATCH_ARGS,
    TRAINING_BUTTONS,
    _add_game_variables,
    _find_wad,
)
from sf_examples.vizdoom.doom.multiplayer.doom_multiagent import (
    DEFAULT_UDP_PORT,
    find_available_port,
)


def _make_game(
    wad_path: str,
    doom_map: str,
    timelimit: float,
    port: int,
    *,
    is_host: bool,
    host_slots: int,
) -> vzd.DoomGame:
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
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    for button in TRAINING_BUTTONS:
        game.add_available_button(button)
    _add_game_variables(game)
    if is_host:
        game.add_game_args(
            f"-host {host_slots} -port {port} -deathmatch "
            f"+timelimit {timelimit:.1f} {DEATHMATCH_ARGS} "
            "+viz_connect_timeout 60"
        )
        game.add_game_args("+name P1 +colorset 0")
    else:
        game.add_game_args(f"-join 127.0.0.1:{port} +viz_connect_timeout 30")
        game.add_game_args("+name P2 +colorset 3")
    game.set_episode_timeout(int(timelimit * 60 * game.get_ticrate()))
    return game


def _advance_both(g1, g2, a1, a2):
    t1 = threading.Thread(target=g1.make_action, args=(a1,))
    t2 = threading.Thread(target=g2.make_action, args=(a2,))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)


def run_probe(host_slots: int, num_bots: int, max_tics: int) -> None:
    from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

    register_vizdoom_components()

    wad_path = _find_wad("dwango5.wad")
    port = find_available_port(DEFAULT_UDP_PORT)
    timelimit = 5.0
    host = _make_game(wad_path, "map01", timelimit, port, is_host=True, host_slots=host_slots)
    join = _make_game(wad_path, "map01", timelimit, port, is_host=False, host_slots=host_slots)

    errs = [None, None]

    def ih():
        try:
            host.init()
        except Exception as e:
            errs[0] = e

    def ij():
        try:
            time.sleep(2.0)
            join.init()
        except Exception as e:
            errs[1] = e

    th = threading.Thread(target=ih)
    tj = threading.Thread(target=ij)
    th.start()
    tj.start()
    th.join(timeout=60)
    tj.join(timeout=60)
    if errs[0] or errs[1]:
        print(f"[diagnose] init failed host={errs[0]} join={errs[1]}")
        return

    ep = [None, None]

    def eh():
        try:
            host.new_episode()
        except Exception as e:
            ep[0] = e

    def ej():
        try:
            join.new_episode()
        except Exception as e:
            ep[1] = e

    te = threading.Thread(target=eh)
    tj2 = threading.Thread(target=ej)
    te.start()
    tj2.start()
    te.join(timeout=45)
    tj2.join(timeout=45)
    if ep[0] or ep[1]:
        print(f"[diagnose] new_episode failed host={ep[0]} join={ep[1]}")
        host.close()
        join.close()
        return

    host.send_game_command("removebots")
    for i in range(num_bots):
        host.send_game_command("addbot")
    print(
        f"[diagnose] -host {host_slots}, addbot x{num_bots}. "
        "PLAYER_COUNT=human connectors only; watch hits/damage/frags for bot activity."
    )

    z = [0.0] * len(TRAINING_BUTTONS)
    zf = [float(x) for x in z]
    rng = np.random.default_rng(0)
    for tic in range(max_tics):
        # Mild random aggression so DM isn't perfectly idle
        a1 = list(zf)
        a2 = list(zf)
        if rng.random() < 0.4:
            a1[0] = 1.0
            a1[11] = 1.0
            a1[-1] = float(rng.uniform(-5, 5))
        if rng.random() < 0.4:
            a2[0] = 1.0
            a2[11] = 1.0
            a2[-1] = float(rng.uniform(-5, 5))
        _advance_both(host, join, a1, a2)
        if host.is_player_dead():
            host.respawn_player()
        if join.is_player_dead():
            join.respawn_player()
        if tic % 35 == 0 or tic < 5:
            pc = host.get_game_variable(vzd.GameVariable.PLAYER_COUNT)
            hit = host.get_game_variable(vzd.GameVariable.HITCOUNT)
            dmg = host.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
            fr = host.get_game_variable(vzd.GameVariable.FRAGCOUNT)
            print(
                f"  tic={tic:4d} PLAYER_COUNT={pc} "
                f"host_hits={hit} host_dmg={dmg} host_frags={fr}"
            )
        if host.is_episode_finished() or join.is_episode_finished():
            print(f"  episode finished early at tic={tic}")
            break

    host.close()
    join.close()
    print("[diagnose] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-slots", type=int, default=2, help="-host N (default 2 like record_human)")
    ap.add_argument("--num-bots", type=int, default=7)
    ap.add_argument("--max-tics", type=int, default=400, help="~11s at 35 tics/s")
    args = ap.parse_args()

    print(f"=== diagnose_mp_bots -host {args.host_slots} ===")
    run_probe(host_slots=args.host_slots, num_bots=args.num_bots, max_tics=args.max_tics)


if __name__ == "__main__":
    main()
