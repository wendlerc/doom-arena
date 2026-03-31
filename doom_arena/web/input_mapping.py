"""Browser input → ViZDoom action vector conversion.

Binary protocol:
  byte 0:     0x02 (input message type)
  bytes 1-2:  uint16 key_state bitmask
  bytes 3-6:  float32 mouse_dx (accumulated movementX pixels)

Key bitmask layout:
  bit 0:  W (MOVE_FORWARD)
  bit 1:  S (MOVE_BACKWARD)
  bit 2:  D (MOVE_RIGHT)
  bit 3:  A (MOVE_LEFT)
  bit 4:  1 (SELECT_WEAPON1)
  bit 5:  2 (SELECT_WEAPON2)
  bit 6:  3 (SELECT_WEAPON3)
  bit 7:  4 (SELECT_WEAPON4)
  bit 8:  5 (SELECT_WEAPON5)
  bit 9:  6 (SELECT_WEAPON6)
  bit 10: 7 (SELECT_WEAPON7)
  bit 11: Mouse left click (ATTACK)
  bit 12: Shift (SPEED / run)

Output: 14 floats matching TRAINING_BUTTONS order:
  [MOVE_FWD, MOVE_BWD, MOVE_RIGHT, MOVE_LEFT,
   WEAP1..WEAP7, ATTACK, SPEED, TURN_DELTA]
"""
from __future__ import annotations

import struct

# Scales browser movementX pixels to ViZDoom turn delta.
# ViZDoom turn range is roughly [-12.5, 12.5] degrees per tic.
# Typical mouse movementX is -30..+30 px per frame at normal sensitivity.
MOUSE_SENSITIVITY = 0.4

INPUT_MSG_TYPE = 0x02
INPUT_MSG_SIZE = 7  # 1 + 2 + 4

ZERO_ACTION = [0.0] * 14


def parse_input_message(data: bytes) -> list[float]:
    """Parse a binary input message into a 14-float ViZDoom action vector.

    Returns ZERO_ACTION for malformed messages.
    """
    if len(data) < INPUT_MSG_SIZE or data[0] != INPUT_MSG_TYPE:
        return list(ZERO_ACTION)

    key_state = struct.unpack_from("<H", data, 1)[0]
    mouse_dx = struct.unpack_from("<f", data, 3)[0]

    action = [0.0] * 14

    # Movement (bits 0-3 → indices 0-3)
    action[0] = 1.0 if key_state & (1 << 0) else 0.0  # W → MOVE_FORWARD
    action[1] = 1.0 if key_state & (1 << 1) else 0.0  # S → MOVE_BACKWARD
    action[2] = 1.0 if key_state & (1 << 2) else 0.0  # D → MOVE_RIGHT
    action[3] = 1.0 if key_state & (1 << 3) else 0.0  # A → MOVE_LEFT

    # Weapons (bits 4-10 → indices 4-10)
    for i in range(7):
        action[4 + i] = 1.0 if key_state & (1 << (4 + i)) else 0.0

    # Attack (bit 11 → index 11)
    action[11] = 1.0 if key_state & (1 << 11) else 0.0

    # Speed/run (bit 12 → index 12)
    action[12] = 1.0 if key_state & (1 << 12) else 0.0

    # Turn delta (continuous, from mouse movement)
    turn = mouse_dx * MOUSE_SENSITIVITY
    turn = max(-12.5, min(12.5, turn))
    action[13] = turn

    return action


def build_frame_message(
    tic: int,
    health: int,
    frags: int,
    deaths: int,
    jpeg_data: bytes,
) -> bytes:
    """Build a binary frame message to send to the browser.

    Format:
      byte 0:     0x01 (frame message type)
      bytes 1-4:  uint32 tic number
      bytes 5-6:  int16 health
      bytes 7-8:  int16 frags
      bytes 9-10: int16 deaths
      bytes 11+:  JPEG data
    """
    header = struct.pack("<BIhhh", 0x01, tic, health, frags, deaths)
    return header + jpeg_data


FRAME_HEADER_SIZE = 11  # 1 + 4 + 2 + 2 + 2


def build_score_message(episode: int, total_episodes: int, scores: list[dict]) -> bytes:
    """Build a JSON-encoded score message.

    Sent as text WebSocket message with type prefix.
    """
    import json
    msg = {
        "type": "scores",
        "episode": episode,
        "total_episodes": total_episodes,
        "scores": scores,
    }
    return json.dumps(msg).encode()
