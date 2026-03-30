"""Test: does -deathmatch work with frameskip=4 (matching SF training)?"""
import threading, time
import numpy as np
import vizdoom as vzd

sf_scenarios = __import__("sf_examples.vizdoom.doom.scenarios", fromlist=[""])
WAD = list(sf_scenarios.__path__)[0] + "/dwango5.wad"

def make_game(is_host, port=5450):
    game = vzd.DoomGame()
    game.set_doom_scenario_path(WAD)
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.CRCGCB)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    for b in [vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
              vzd.Button.MOVE_RIGHT, vzd.Button.MOVE_LEFT,
              vzd.Button.SELECT_WEAPON1, vzd.Button.SELECT_WEAPON2,
              vzd.Button.SELECT_WEAPON3, vzd.Button.SELECT_WEAPON4,
              vzd.Button.SELECT_WEAPON5, vzd.Button.SELECT_WEAPON6,
              vzd.Button.SELECT_WEAPON7,
              vzd.Button.ATTACK, vzd.Button.SPEED, vzd.Button.TURN_LEFT_RIGHT_DELTA]:
        game.add_available_button(b)
    for v in [vzd.GameVariable.FRAGCOUNT, vzd.GameVariable.DEATHCOUNT,
              vzd.GameVariable.HEALTH, vzd.GameVariable.KILLCOUNT]:
        game.add_available_game_variable(v)
    if is_host:
        game.add_game_args(
            f"-host 2 -port {port} -deathmatch "
            f"+timelimit 1.0 "
            "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 "
            "+sv_spawnfarthest 1 +sv_nocrouch 1 +sv_nojump 1 "
            "+sv_nofreelook 1 +sv_noexit 1 "
            "+viz_respawn_delay 10 +viz_connect_timeout 30"
        )
        game.add_game_args("+name P1 +colorset 0")
    else:
        game.add_game_args(f"-join 127.0.0.1 -port {port} +viz_connect_timeout 30")
        game.add_game_args("+name P2 +colorset 3")
    game.set_episode_timeout(int(1.0 * 60 * 35))
    return game

host = make_game(True)
join = make_game(False)

def init_h():
    host.init()
    for _ in range(4):
        host.send_game_command("addbot")
def init_j():
    time.sleep(1.5)
    join.init()

th = threading.Thread(target=init_h)
tj = threading.Thread(target=init_j)
th.start(); tj.start()
th.join(30); tj.join(30)
print("Connected!", flush=True)

tic = 0
frameskip = 4
zero_action = [0.0] * 14
frag_h = frag_j = death_h = death_j = 0.0

t0 = time.perf_counter()
while not host.is_episode_finished() and not join.is_episode_finished():
    s1 = host.get_state()
    s2 = join.get_state()

    if s1 is None or s2 is None:
        r = [0.0]
        def sh(): r[0] = host.make_action(zero_action, 1)
        def sj(): join.make_action(zero_action, 1)
        t1 = threading.Thread(target=sh)
        t2 = threading.Thread(target=sj)
        t1.start(); t2.start()
        t1.join(); t2.join()
        if host.is_player_dead(): host.respawn_player()
        if join.is_player_dead(): join.respawn_player()
        tic += 1
        continue

    # Random actions (attack + move + turn)
    action = zero_action.copy()
    action[0] = 1  # forward
    action[11] = 1  # attack
    action[13] = np.random.uniform(-5, 5)  # turn

    r1 = [0.0]; r2 = [0.0]
    def sh(): r1[0] = host.make_action(action, frameskip)
    def sj(): r2[0] = join.make_action(action, frameskip)
    t1 = threading.Thread(target=sh)
    t2 = threading.Thread(target=sj)
    t1.start(); t2.start()
    t1.join(); t2.join()

    frag_h = host.get_game_variable(vzd.GameVariable.FRAGCOUNT)
    death_h = host.get_game_variable(vzd.GameVariable.DEATHCOUNT)
    frag_j = join.get_game_variable(vzd.GameVariable.FRAGCOUNT)
    death_j = join.get_game_variable(vzd.GameVariable.DEATHCOUNT)

    if host.is_player_dead(): host.respawn_player()
    if join.is_player_dead(): join.respawn_player()

    tic += frameskip

    if tic % 200 == 0:
        print(f"tic={tic}: H frags={frag_h} deaths={death_h} | J frags={frag_j} deaths={death_j}", flush=True)

dt = time.perf_counter() - t0
print(f"\nDone: {tic} tics in {dt:.1f}s ({tic/dt:.0f} tics/s)", flush=True)
print(f"Final: H frags={frag_h} deaths={death_h} | J frags={frag_j} deaths={death_j}", flush=True)
host.close(); join.close()
