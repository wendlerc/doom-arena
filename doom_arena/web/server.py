"""FastAPI web server for Doom Arena game lobby and recording.

Provides:
  - REST API for session management (create, list, start)
  - WebSocket endpoints for lobby (player list, chat) and gameplay (frames, input)
  - Static file serving for the frontend
  - Xvfb auto-detection for headless servers

Usage:
    doom-web --host 0.0.0.0 --port 8666
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from doom_arena.web.session import SessionManager, SessionState
from doom_arena.web.game_runner import GameRunner
from doom_arena.web.recorder import save_session_recording

logger = logging.getLogger("doom_arena.web")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Doom Arena", docs_url=None, redoc_url=None)

# Global state
session_manager = SessionManager()
game_runners: dict[str, GameRunner] = {}


# --- Static Pages ---

@app.get("/")
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/lobby/{session_id}")
async def lobby_page(session_id: str):
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return HTMLResponse((STATIC_DIR / "lobby.html").read_text())


@app.get("/game/{session_id}")
async def game_page(session_id: str):
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return HTMLResponse((STATIC_DIR / "game.html").read_text())


# --- REST API ---

@app.post("/api/sessions")
async def create_session(config: dict):
    session = await session_manager.create_session(config)
    return {
        "session_id": session.session_id,
        "admin_token": session.admin_token,
        "join_url": f"/lobby/{session.session_id}",
        "admin_url": f"/lobby/{session.session_id}?admin={session.admin_token}",
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.to_dict()


@app.get("/api/sessions")
async def list_sessions():
    return await session_manager.list_sessions()


@app.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str, admin_token: str = Query(...)):
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.admin_token != admin_token:
        raise HTTPException(403, "Invalid admin token")
    if session.state != SessionState.LOBBY:
        raise HTTPException(400, f"Cannot start: session is {session.state.value}")
    if not session.connected_humans():
        raise HTTPException(400, "No connected human players")

    await session_manager.set_state(session_id, SessionState.STARTING)

    # Build player list for GameRunner
    players = []
    for p in session.players:
        pd = {
            "slot_id": p.slot_id,
            "type": p.player_type,
            "name": p.name,
        }
        if p.player_type == "ai":
            pd["experiment"] = p.experiment
            pd["checkpoint"] = p.checkpoint
            pd["device"] = p.device
        players.append(pd)

    # Create and start game runner — capture event loop for thread-safe callback
    loop = asyncio.get_running_loop()

    def on_game_finished(episode_recordings):
        loop.call_soon_threadsafe(
            asyncio.ensure_future,
            _on_game_finished(session_id, episode_recordings),
        )

    runner = GameRunner(
        wad=session.wad,
        doom_map=session.doom_map,
        timelimit=session.timelimit,
        num_bots=session.num_bots,
        num_episodes=session.num_episodes,
        players=players,
        on_game_finished=on_game_finished,
    )
    game_runners[session_id] = runner
    runner.start()

    await session_manager.set_state(session_id, SessionState.PLAYING)

    # Notify all lobby WebSockets
    await _broadcast_lobby(session_id, {
        "type": "game_started",
        "game_url": f"/game/{session_id}",
    })

    return {"status": "started"}


async def _on_game_finished(session_id: str, episode_recordings: list[dict]):
    """Called when a game finishes — save recordings and update session state."""
    session = await session_manager.get_session(session_id)
    if not session:
        return

    # Save recording
    shard_path = save_session_recording(
        session_id=session_id,
        session_config=session.config_dict(),
        episode_recordings=episode_recordings,
    )

    session.results = {
        "shard_path": shard_path,
        "episodes_played": len(episode_recordings),
    }
    await session_manager.set_state(session_id, SessionState.FINISHED)

    # Cleanup runner
    runner = game_runners.pop(session_id, None)
    if runner:
        runner.stop()

    logger.info(f"Session {session_id} finished. Recording: {shard_path}")


# --- Lobby WebSocket ---

# Track lobby connections: session_id -> list of (ws, slot_id)
_lobby_connections: dict[str, list[tuple[WebSocket, int]]] = {}


async def _broadcast_lobby(session_id: str, msg: dict):
    """Broadcast a message to all lobby WebSocket connections for a session."""
    conns = _lobby_connections.get(session_id, [])
    data = json.dumps(msg)
    for ws, _ in conns:
        try:
            await ws.send_text(data)
        except Exception:
            pass


@app.websocket("/ws/lobby/{session_id}")
async def lobby_websocket(websocket: WebSocket, session_id: str):
    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    slot_id = -1

    if session_id not in _lobby_connections:
        _lobby_connections[session_id] = []

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg["type"] == "join":
                name = msg.get("name", "").strip()
                player = await session_manager.add_player(session_id, name)
                if player is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Cannot join: lobby full or game started",
                    }))
                    continue

                slot_id = player.slot_id
                player.connected = True
                _lobby_connections[session_id].append((websocket, slot_id))

                await websocket.send_text(json.dumps({
                    "type": "joined",
                    "slot_id": slot_id,
                    "join_token": player.join_token,
                    "name": player.name,
                }))

                # Broadcast updated player list
                session = await session_manager.get_session(session_id)
                await _broadcast_lobby(session_id, {
                    "type": "player_list",
                    "players": [p.to_dict() for p in session.players],
                })

            elif msg["type"] == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Lobby WS error: {e}")
    finally:
        # Remove from lobby connections
        if session_id in _lobby_connections:
            _lobby_connections[session_id] = [
                (ws, sid) for ws, sid in _lobby_connections[session_id]
                if ws is not websocket
            ]
        # Remove player from session if still in lobby
        if slot_id >= 0:
            session = await session_manager.get_session(session_id)
            if session and session.state == SessionState.LOBBY:
                await session_manager.remove_player(session_id, slot_id)
                session = await session_manager.get_session(session_id)
                if session:
                    await _broadcast_lobby(session_id, {
                        "type": "player_list",
                        "players": [p.to_dict() for p in session.players],
                    })


# --- Game WebSocket ---

@app.websocket("/ws/game/{session_id}/{slot_id}")
async def game_websocket(
    websocket: WebSocket,
    session_id: str,
    slot_id: int,
    token: str = Query(""),
):
    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    # Verify player token
    player = None
    for p in session.players:
        if p.slot_id == slot_id and p.join_token == token:
            player = p
            break

    if not player:
        await websocket.close(code=4003, reason="Invalid player or token")
        return

    runner = game_runners.get(session_id)
    if not runner:
        await websocket.close(code=4005, reason="Game not started")
        return

    pio = runner.get_player_io(slot_id)
    if not pio:
        await websocket.close(code=4006, reason="Player not in game")
        return

    await websocket.accept()
    pio.connected.set()

    # Run send and receive concurrently
    send_task = asyncio.create_task(_game_send_frames(websocket, pio))
    recv_task = asyncio.create_task(_game_recv_input(websocket, pio))

    try:
        done, pending = await asyncio.wait(
            [send_task, recv_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except Exception as e:
        logger.warning(f"Game WS error: {e}")
    finally:
        pio.connected.clear()
        runner.handle_disconnect(slot_id)
        send_task.cancel()
        recv_task.cancel()


async def _game_send_frames(websocket: WebSocket, pio):
    """Send frames from the game loop to the browser."""
    while True:
        try:
            # Poll frame queue with async sleep to avoid blocking
            try:
                msg_type, data = pio.frame_queue.get_nowait()
            except Exception:
                await asyncio.sleep(0.005)
                continue

            if msg_type == "binary":
                await websocket.send_bytes(data)
            elif msg_type == "text":
                await websocket.send_text(data)

        except WebSocketDisconnect:
            break
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Frame send error: {e}")
            break


async def _game_recv_input(websocket: WebSocket, pio):
    """Receive input from the browser and queue for the game loop."""
    while True:
        try:
            data = await websocket.receive_bytes()
            # Drop oldest if queue is full
            try:
                pio.input_queue.put_nowait(data)
            except Exception:
                try:
                    pio.input_queue.get_nowait()
                except Exception:
                    pass
                try:
                    pio.input_queue.put_nowait(data)
                except Exception:
                    pass

        except WebSocketDisconnect:
            break
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Input recv error: {e}")
            break


# --- Xvfb Auto-Detection ---

def _ensure_display():
    """Check for $DISPLAY and launch Xvfb if missing."""
    if os.environ.get("DISPLAY"):
        return

    logger.info("No $DISPLAY found, launching Xvfb...")
    try:
        # Find a free display number
        for display_num in range(99, 200):
            lock_file = f"/tmp/.X{display_num}-lock"
            if not os.path.exists(lock_file):
                break
        else:
            raise RuntimeError("No free X display number found")

        display = f":{display_num}"
        proc = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1280x960x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give Xvfb a moment to start
        import time
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError(f"Xvfb exited immediately (code {proc.returncode})")

        os.environ["DISPLAY"] = display
        logger.info(f"Xvfb started on {display} (PID {proc.pid})")
    except FileNotFoundError:
        logger.warning(
            "Xvfb not installed. Install with: apt install xvfb\n"
            "Or run with: xvfb-run doom-web"
        )
    except Exception as e:
        logger.warning(f"Failed to start Xvfb: {e}")


# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Doom Arena Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8666, help="HTTP port")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    _ensure_display()

    # Mount static files (CSS, JS)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
