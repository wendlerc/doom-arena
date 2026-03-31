"""In-memory session and lobby management for Doom Arena web server."""
from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


class SessionState(enum.Enum):
    LOBBY = "lobby"
    STARTING = "starting"
    PLAYING = "playing"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class PlayerSlot:
    slot_id: int
    player_type: str  # "human" or "ai"
    name: str = ""
    join_token: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    connected: bool = False
    ws: Any = None  # WebSocket reference (not serialized)
    # AI-specific
    experiment: str = ""
    checkpoint: str = "best"
    device: str = "cpu"

    def to_dict(self) -> dict:
        d = {
            "slot_id": self.slot_id,
            "name": self.name,
            "type": self.player_type,
            "connected": self.connected,
        }
        if self.player_type == "ai":
            d["experiment"] = self.experiment
        return d


@dataclass
class GameSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    admin_token: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    state: SessionState = SessionState.LOBBY
    created_at: float = field(default_factory=time.time)

    # Configuration
    wad: str = "dwango5.wad"
    doom_map: str = "map01"
    timelimit: float = 5.0
    num_bots: int = 4
    num_episodes: int = 1
    max_human_players: int = 4

    # Runtime
    players: list[PlayerSlot] = field(default_factory=list)
    udp_port: int = 0
    error_message: str = ""
    results: dict = field(default_factory=dict)

    def human_players(self) -> list[PlayerSlot]:
        return [p for p in self.players if p.player_type == "human"]

    def ai_players(self) -> list[PlayerSlot]:
        return [p for p in self.players if p.player_type == "ai"]

    def connected_humans(self) -> list[PlayerSlot]:
        return [p for p in self.players if p.player_type == "human" and p.connected]

    def next_slot_id(self) -> int:
        used = {p.slot_id for p in self.players}
        i = 0
        while i in used:
            i += 1
        return i

    def config_dict(self) -> dict:
        return {
            "wad": self.wad,
            "doom_map": self.doom_map,
            "timelimit": self.timelimit,
            "num_bots": self.num_bots,
            "num_episodes": self.num_episodes,
            "max_human_players": self.max_human_players,
        }

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "config": self.config_dict(),
            "players": [p.to_dict() for p in self.players],
            "results": self.results if self.state == SessionState.FINISHED else None,
        }


class SessionManager:
    """Thread-safe in-memory session store."""

    def __init__(self):
        self._sessions: dict[str, GameSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, config: dict) -> GameSession:
        async with self._lock:
            session = GameSession(
                wad=config.get("wad", "dwango5.wad"),
                doom_map=config.get("doom_map", "map01"),
                timelimit=float(config.get("timelimit", 5.0)),
                num_bots=int(config.get("num_bots", 4)),
                num_episodes=int(config.get("num_episodes", 1)),
                max_human_players=int(config.get("max_human_players", 4)),
            )
            # Add AI players from config
            for ai_cfg in config.get("ai_players", []):
                slot = PlayerSlot(
                    slot_id=session.next_slot_id(),
                    player_type="ai",
                    name=ai_cfg.get("name", f"AI_{session.next_slot_id()}"),
                    experiment=ai_cfg.get("experiment", ""),
                    checkpoint=ai_cfg.get("checkpoint", "best"),
                    device=ai_cfg.get("device", "cpu"),
                    connected=True,  # AI is always "connected"
                )
                session.players.append(slot)
            self._sessions[session.session_id] = session
            return session

    async def get_session(self, session_id: str) -> Optional[GameSession]:
        return self._sessions.get(session_id)

    async def add_player(self, session_id: str, name: str) -> Optional[PlayerSlot]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.state != SessionState.LOBBY:
                return None
            humans = session.human_players()
            if len(humans) >= session.max_human_players:
                return None
            slot = PlayerSlot(
                slot_id=session.next_slot_id(),
                player_type="human",
                name=name or f"Player_{session.next_slot_id() + 1}",
            )
            session.players.append(slot)
            return slot

    async def remove_player(self, session_id: str, slot_id: int) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.players = [p for p in session.players if p.slot_id != slot_id]

    async def set_state(self, session_id: str, state: SessionState) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.state = state

    async def list_sessions(self) -> list[dict]:
        return [s.to_dict() for s in self._sessions.values()]

    async def cleanup_stale(self, max_age_hours: float = 2.0) -> int:
        cutoff = time.time() - max_age_hours * 3600
        async with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.created_at < cutoff]
            for sid in stale:
                del self._sessions[sid]
            return len(stale)
