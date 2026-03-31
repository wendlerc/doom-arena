"""ViZDoom game lifecycle, game loop, and per-player I/O for the web server.

Manages:
  - Creating ViZDoom game instances (host + joiners)
  - Threaded initialization and episode start
  - Main game loop: get state, compute actions, advance all players concurrently
  - JPEG frame encoding and queuing for WebSocket streaming
  - Recording buffers for post-game saving
"""
from __future__ import annotations

import json
import logging
import queue
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

import vizdoom as vzd

from doom_arena.record import (
    BUTTON_NAMES,
    DEATHMATCH_ARGS,
    DECISION_INTERVAL,
    GAME_FPS,
    GAME_VARIABLES,
    TRAINING_BUTTONS,
    _add_game_variables,
    _find_wad,
    convert_action,
    encode_video,
    extract_measurements,
    load_model,
    preprocess_for_model,
    select_checkpoint_path,
)

from doom_arena.web.input_mapping import (
    ZERO_ACTION,
    build_frame_message,
    parse_input_message,
)

logger = logging.getLogger("doom_arena.web")

# JPEG encode quality for streaming frames to browser
JPEG_QUALITY = 70


@dataclass
class PlayerIO:
    """Per-player communication channel between game loop and WebSocket."""

    slot_id: int
    player_type: str  # "human", "ai", "host_dummy"
    name: str = ""

    # Browser ↔ game loop queues
    input_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=4))
    frame_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))

    # Connection tracking
    connected: threading.Event = field(default_factory=threading.Event)

    # Recording buffers (per episode, reset between episodes)
    frames: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    # AI model state
    model: object = None  # actor_critic
    rnn_state: object = None  # torch tensor
    rnn_size: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # ViZDoom game instance
    game: vzd.DoomGame | None = None

    # Stats
    frags: float = 0.0
    deaths: float = 0.0

    def reset_recording(self):
        self.frames = []
        self.actions = []
        self.rewards = []
        self.frags = 0.0
        self.deaths = 0.0

    def reset_rnn(self):
        if self.rnn_size > 0:
            self.rnn_state = torch.zeros(1, self.rnn_size, device=self.device)


def _create_web_game(
    wad_path: str,
    doom_map: str,
    timelimit: float,
    port: int,
    is_host: bool,
    num_players: int,
    resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_640X480,
    player_name: str = "Player",
    colorset: int = 0,
) -> vzd.DoomGame:
    """Create a ViZDoom game instance for web multiplayer.

    Always uses ASYNC_PLAYER mode and hidden window.
    """
    game = vzd.DoomGame()
    game.set_doom_scenario_path(wad_path)
    game.set_doom_map(doom_map)
    game.set_screen_resolution(resolution)
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
            f"-host {num_players} -port {port} -deathmatch "
            f"+timelimit {timelimit:.1f} {DEATHMATCH_ARGS} "
            "+viz_connect_timeout 120"
        )
    else:
        game.add_game_args(
            f"-join 127.0.0.1:{port} +viz_connect_timeout 60"
        )

    game.add_game_args(f"+name {player_name} +colorset {colorset}")
    game.set_episode_timeout(int(timelimit * 60 * game.get_ticrate()))

    return game


def _find_available_port(start: int = 5800) -> int:
    """Find an available UDP port starting from `start`."""
    import socket
    port = start
    while port < start + 200:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise RuntimeError(f"No available UDP port found in range {start}-{start+200}")


def _encode_frame_jpeg(screen_buffer: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a ViZDoom CRCGCB screen buffer to JPEG bytes.

    Input: (3, H, W) uint8 array (CHW RGB from ViZDoom).
    Output: JPEG bytes.
    """
    # CHW → HWC, RGB → BGR for cv2
    hwc = np.transpose(screen_buffer, (1, 2, 0))
    bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


class GameRunner:
    """Manages the lifecycle of a ViZDoom multiplayer game session.

    Creates game instances, runs the game loop in a dedicated thread,
    handles AI inference, human input, frame streaming, and recording.
    """

    def __init__(
        self,
        wad: str,
        doom_map: str,
        timelimit: float,
        num_bots: int,
        num_episodes: int,
        players: list[dict],  # [{type, name, slot_id, experiment?, checkpoint?, device?}]
        on_episode_end: callable | None = None,
        on_game_finished: callable | None = None,
    ):
        self.wad = wad
        self.doom_map = doom_map
        self.timelimit = timelimit
        self.num_bots = num_bots
        self.num_episodes = num_episodes
        self.on_episode_end = on_episode_end
        self.on_game_finished = on_game_finished

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._port: int = 0

        # Build PlayerIO list
        self.player_ios: list[PlayerIO] = []
        for p in players:
            pio = PlayerIO(
                slot_id=p["slot_id"],
                player_type=p["type"],
                name=p.get("name", f"Player_{p['slot_id']}"),
            )
            self.player_ios.append(pio)

        # Separate host dummy if no AI player exists to be host
        self._host_dummy: PlayerIO | None = None
        self._ai_configs = {p["slot_id"]: p for p in players if p["type"] == "ai"}

        # All episode recordings
        self.episode_recordings: list[dict] = []

        # Current episode state (broadcast to clients)
        self.current_episode = 0
        self.state = "init"  # init, playing, between_episodes, finished, error
        self.error_message = ""

    @property
    def port(self) -> int:
        return self._port

    def get_player_io(self, slot_id: int) -> PlayerIO | None:
        for pio in self.player_ios:
            if pio.slot_id == slot_id:
                return pio
        return None

    def start(self):
        """Start the game runner thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="game-runner")
        self._thread.start()

    def stop(self):
        """Signal the game runner to stop."""
        self._stop_event.set()

    def wait(self, timeout: float | None = None):
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self):
        """Main game runner: init games, run episodes, cleanup."""
        try:
            self._port = _find_available_port()
            wad_path = _find_wad(self.wad)

            # Load AI models
            self._load_ai_models()

            # Determine networked player count (all humans + all AIs + host_dummy if needed)
            # We always create a dedicated host instance
            humans = [p for p in self.player_ios if p.player_type == "human"]
            ais = [p for p in self.player_ios if p.player_type == "ai"]

            # Total networked players = 1 (host_dummy) + humans + ais
            # Host dummy is always player slot, takes zero actions
            num_networked = 1 + len(humans) + len(ais)

            # Create host dummy
            self._host_dummy = PlayerIO(
                slot_id=-1,
                player_type="host_dummy",
                name="Server",
            )

            logger.info(
                f"Starting game: {self.wad} {self.doom_map}, "
                f"{len(humans)} humans, {len(ais)} AI, {self.num_bots} bots, "
                f"port {self._port}, {num_networked} networked players"
            )

            # Create all game instances
            self._host_dummy.game = _create_web_game(
                wad_path, self.doom_map, self.timelimit,
                self._port, is_host=True, num_players=num_networked,
                resolution=vzd.ScreenResolution.RES_160X120,
                player_name="Server", colorset=7,
            )

            joiners: list[PlayerIO] = []
            colorset = 0
            for pio in self.player_ios:
                res = (
                    vzd.ScreenResolution.RES_640X480
                    if pio.player_type == "human"
                    else vzd.ScreenResolution.RES_160X120
                )
                pio.game = _create_web_game(
                    wad_path, self.doom_map, self.timelimit,
                    self._port, is_host=False, num_players=num_networked,
                    resolution=res,
                    player_name=pio.name[:15].replace(" ", "_"),
                    colorset=colorset,
                )
                colorset = (colorset + 1) % 8
                joiners.append(pio)

            # Init all games via threads
            if not self._init_games(joiners):
                self.state = "error"
                self.error_message = "Failed to initialize ViZDoom games"
                return

            # Run episodes
            for ep in range(self.num_episodes):
                if self._stop_event.is_set():
                    break

                self.current_episode = ep
                self.state = "playing"
                logger.info(f"Starting episode {ep + 1}/{self.num_episodes}")

                # Start episode
                if not self._start_episode(joiners):
                    self.state = "error"
                    self.error_message = f"Failed to start episode {ep + 1}"
                    break

                # Add bots
                if self.num_bots > 0:
                    self._host_dummy.game.send_game_command("removebots")
                    for _ in range(self.num_bots):
                        self._host_dummy.game.send_game_command("addbot")

                # Reset recording buffers
                for pio in self.player_ios:
                    pio.reset_recording()
                    pio.reset_rnn()

                # Run game loop
                ep_data = self._game_loop(joiners)

                if ep_data:
                    self.episode_recordings.append(ep_data)

                # Broadcast scores
                if self.on_episode_end:
                    self.on_episode_end(ep, ep_data)

                # Between episodes pause
                if ep < self.num_episodes - 1 and not self._stop_event.is_set():
                    self.state = "between_episodes"
                    scores = self._build_scores()
                    for pio in self.player_ios:
                        if pio.player_type == "human":
                            try:
                                msg = json.dumps({
                                    "type": "scores",
                                    "episode": ep + 1,
                                    "total_episodes": self.num_episodes,
                                    "scores": scores,
                                    "next_in": 5,
                                })
                                pio.frame_queue.put(("text", msg), timeout=2)
                            except queue.Full:
                                pass
                    time.sleep(5)

            self.state = "finished"
            # Send final scores
            scores = self._build_scores()
            for pio in self.player_ios:
                if pio.player_type == "human":
                    try:
                        msg = json.dumps({
                            "type": "game_over",
                            "scores": scores,
                            "episodes_played": len(self.episode_recordings),
                        })
                        pio.frame_queue.put(("text", msg), timeout=2)
                    except queue.Full:
                        pass

            if self.on_game_finished:
                self.on_game_finished(self.episode_recordings)

        except Exception as e:
            logger.exception(f"GameRunner error: {e}")
            self.state = "error"
            self.error_message = str(e)
        finally:
            self._cleanup()

    def _load_ai_models(self):
        """Load AI models for all AI players."""
        train_dir = "sf_train_dir"
        for pio in self.player_ios:
            if pio.player_type != "ai":
                continue
            cfg = self._ai_configs[pio.slot_id]
            experiment = cfg.get("experiment", "")
            checkpoint_mode = cfg.get("checkpoint", "best")
            device = cfg.get("device", "cpu")
            if not experiment:
                logger.warning(f"AI player {pio.name} has no experiment, will use random policy")
                continue
            try:
                ckpt_path = select_checkpoint_path(experiment, train_dir, checkpoint_mode)
                model, rnn_size, dev = load_model(experiment, train_dir, ckpt_path, device)
                pio.model = model
                pio.rnn_size = rnn_size
                pio.device = dev
                pio.rnn_state = torch.zeros(1, rnn_size, device=dev)
                logger.info(f"Loaded AI model for {pio.name}: {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to load model for {pio.name}: {e}")

    def _init_games(self, joiners: list[PlayerIO]) -> bool:
        """Initialize all game instances via threads. Returns True on success."""
        errors = {}

        def init_game(pio, delay=0.0):
            try:
                if delay > 0:
                    time.sleep(delay)
                pio.game.init()
            except Exception as e:
                errors[pio.slot_id] = e

        threads = []
        # Host first (no delay)
        t = threading.Thread(target=init_game, args=(self._host_dummy, 0.0))
        threads.append(t)
        t.start()

        # Joiners with staggered delay
        for i, pio in enumerate(joiners):
            delay = 2.0 + i * 0.5  # stagger to avoid race conditions
            t = threading.Thread(target=init_game, args=(pio, delay))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60)

        if errors:
            logger.error(f"Game init errors: {errors}")
            return False

        return True

    def _start_episode(self, joiners: list[PlayerIO]) -> bool:
        """Start a new episode on all games. Returns True on success."""
        errors = {}

        def start_ep(pio):
            try:
                pio.game.new_episode()
            except Exception as e:
                errors[pio.slot_id] = e

        threads = []
        for pio in [self._host_dummy] + joiners:
            t = threading.Thread(target=start_ep, args=(pio,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        if errors:
            logger.error(f"Episode start errors: {errors}")
            return False

        return True

    def _game_loop(self, joiners: list[PlayerIO]) -> dict | None:
        """Run one episode's game loop. Returns episode data dict."""
        tic = 0
        t0 = time.perf_counter()
        host_game = self._host_dummy.game
        zero = list(ZERO_ACTION)

        while not self._stop_event.is_set():
            # Check if episode is finished
            if host_game.is_episode_finished():
                break
            any_joiner_done = any(
                pio.game.is_episode_finished()
                for pio in joiners
                if pio.game is not None
            )
            if any_joiner_done:
                break

            # Get states and compute actions for each player
            all_games = [self._host_dummy] + joiners
            actions_per_player: list[list[float]] = []

            for pio in all_games:
                state = pio.game.get_state()

                if pio.player_type == "host_dummy":
                    actions_per_player.append(zero)
                    if pio.game.is_player_dead():
                        pio.game.respawn_player()
                    continue

                if state is None:
                    # Dead — zero action
                    pio.actions.append(np.array(zero, dtype=np.float32))
                    pio.rewards.append(float(pio.game.get_last_reward()))
                    actions_per_player.append(zero)
                    if pio.game.is_player_dead():
                        pio.game.respawn_player()
                    continue

                # Human player: get input from browser, send frame
                if pio.player_type == "human":
                    action = self._handle_human_tic(pio, state, tic)
                    actions_per_player.append(action)

                # AI player: model inference
                elif pio.player_type == "ai":
                    action = self._handle_ai_tic(pio, state, tic)
                    actions_per_player.append(action)

                # Record action and reward
                pio.actions.append(np.array(action, dtype=np.float32))

                # Update stats periodically
                if tic % DECISION_INTERVAL == 0:
                    pio.frags = pio.game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
                    pio.deaths = pio.game.get_game_variable(vzd.GameVariable.DEATHCOUNT)

            # Advance all games concurrently
            self._advance_all(all_games, actions_per_player)

            # Collect rewards after advance
            for pio in joiners:
                if pio.game is not None and len(pio.rewards) < len(pio.actions):
                    pio.rewards.append(float(pio.game.get_last_reward()))

                # Respawn dead players
                if pio.game.is_player_dead():
                    pio.game.respawn_player()

            tic += 1

        duration = time.perf_counter() - t0

        # Build episode data
        if not any(len(pio.frames) > 0 for pio in joiners):
            return None

        ep_data = {
            "episode_id": uuid.uuid4().hex[:16],
            "tic": tic,
            "duration_s": duration,
            "players": {},
        }
        for pio in joiners:
            n_frames = len(pio.frames)
            if n_frames == 0:
                continue
            n_actions = min(len(pio.actions), n_frames)
            n_rewards = min(len(pio.rewards), n_frames)
            ep_data["players"][pio.slot_id] = {
                "name": pio.name,
                "type": pio.player_type,
                "frames": pio.frames[:n_frames],
                "actions": np.stack(pio.actions[:n_actions]) if pio.actions else np.zeros((0, 14), dtype=np.float32),
                "rewards": np.array(pio.rewards[:n_rewards], dtype=np.float32) if pio.rewards else np.zeros(0, dtype=np.float32),
                "frags": float(pio.frags),
                "deaths": float(pio.deaths),
                "n_frames": n_frames,
            }

        return ep_data

    def _handle_human_tic(self, pio: PlayerIO, state, tic: int) -> list[float]:
        """Process one tic for a human player: get input, encode frame."""
        # Get latest input (drain queue, keep last)
        action = list(ZERO_ACTION)
        if pio.connected.is_set():
            while True:
                try:
                    msg_data = pio.input_queue.get_nowait()
                    action = parse_input_message(msg_data)
                except queue.Empty:
                    break

        # Encode frame and queue for streaming
        screen = state.screen_buffer
        jpeg_data = _encode_frame_jpeg(screen)

        health = int(max(0, pio.game.get_game_variable(vzd.GameVariable.HEALTH)))
        frags = int(pio.game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
        deaths = int(pio.game.get_game_variable(vzd.GameVariable.DEATHCOUNT))

        frame_msg = build_frame_message(tic, health, frags, deaths, jpeg_data)

        # Drop oldest frame if queue is full
        try:
            pio.frame_queue.put_nowait(("binary", frame_msg))
        except queue.Full:
            try:
                pio.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                pio.frame_queue.put_nowait(("binary", frame_msg))
            except queue.Full:
                pass

        # Record frame (HWC RGB for video encoding)
        hwc = np.transpose(screen, (1, 2, 0))
        pio.frames.append(hwc)

        return action

    def _handle_ai_tic(self, pio: PlayerIO, state, tic: int) -> list[float]:
        """Process one tic for an AI player: model inference."""
        # Record frame every tic (for video)
        hwc = np.transpose(state.screen_buffer, (1, 2, 0))
        pio.frames.append(hwc)

        # Only run inference every DECISION_INTERVAL tics
        if tic % DECISION_INTERVAL != 0 and pio.actions:
            return pio.actions[-1].tolist()

        if pio.model is None:
            from doom_arena.record import sample_random_action
            return sample_random_action()

        with torch.no_grad():
            meas = extract_measurements(pio.game)
            obs = preprocess_for_model(state.screen_buffer, meas, pio.device)
            norm = pio.model.normalize_obs(obs)
            result = pio.model(norm, pio.rnn_state)
            pio.rnn_state = result["new_rnn_states"]
            return convert_action(result["actions"].cpu().numpy())

    def _advance_all(self, all_players: list[PlayerIO], actions: list[list[float]]):
        """Advance all games concurrently via threads."""
        threads = []
        for pio, action in zip(all_players, actions):
            if pio.game is None:
                continue
            act = [float(a) for a in action]
            t = threading.Thread(target=pio.game.make_action, args=(act,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

    def _build_scores(self) -> list[dict]:
        """Build scores list from current player stats."""
        scores = []
        for pio in self.player_ios:
            scores.append({
                "slot_id": pio.slot_id,
                "name": pio.name,
                "type": pio.player_type,
                "frags": float(pio.frags),
                "deaths": float(pio.deaths),
            })
        return sorted(scores, key=lambda s: -s["frags"])

    def _cleanup(self):
        """Close all game instances."""
        all_pios = list(self.player_ios)
        if self._host_dummy:
            all_pios.append(self._host_dummy)
        for pio in all_pios:
            if pio.game is not None:
                try:
                    pio.game.close()
                except Exception:
                    pass
                pio.game = None

    def handle_disconnect(self, slot_id: int):
        """Handle a human player disconnecting: replace with a bot."""
        pio = self.get_player_io(slot_id)
        if pio:
            pio.connected.clear()
            # Add a bot to replace the disconnected player
            if self._host_dummy and self._host_dummy.game:
                try:
                    self._host_dummy.game.send_game_command("addbot")
                    logger.info(f"Player {pio.name} disconnected, replaced with bot")
                except Exception as e:
                    logger.warning(f"Failed to add replacement bot: {e}")
