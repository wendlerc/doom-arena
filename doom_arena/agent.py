"""Shared model loading and inference utilities for Sample Factory ViZDoom agents."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components


def _ensure_registered():
    """Register vizdoom components (idempotent)."""
    register_vizdoom_components()


def load_cfg(experiment: str, train_dir: str = "./sf_train_dir") -> AttrDict:
    """Load a Sample Factory config from an experiment directory."""
    cfg_path = Path(train_dir) / experiment / "cfg.json"
    if not cfg_path.exists():
        cfg_path = Path(train_dir) / experiment / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config found in {Path(train_dir) / experiment}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg = AttrDict(cfg)
    cfg.train_dir = train_dir
    cfg.experiment = experiment
    cfg.no_render = True
    cfg.skip_measurements_head = True
    return cfg


@dataclass
class EpisodeStats:
    """Stats from a single episode."""
    reward: float = 0.0
    steps: int = 0
    frames: list = field(default_factory=list)
    game_vars: dict = field(default_factory=dict)


class SFAgent:
    """Wraps a Sample Factory checkpoint for inference."""

    def __init__(
        self,
        experiment: str,
        train_dir: str = "./sf_train_dir",
        checkpoint: str = "best",
        device: str = "cpu",
    ):
        _ensure_registered()

        self.cfg = load_cfg(experiment, train_dir)
        self.device = torch.device(device)

        # Create environment (needed for obs/action space)
        env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)
        self.env = make_env_func_batched(
            self.cfg, env_config=env_config, render_mode="rgb_array"
        )

        # Create and load model
        self.actor_critic = create_actor_critic(
            self.cfg, self.env.observation_space, self.env.action_space
        )
        self.actor_critic = self.actor_critic.to(self.device)

        name_prefix = "best" if checkpoint == "best" else "checkpoint"
        # Try requested, then fall back
        for prefix in [f"{name_prefix}_*", "best_*", "checkpoint_*"]:
            checkpoints = Learner.get_checkpoints(
                Learner.checkpoint_dir(self.cfg, 0), prefix
            )
            if checkpoints:
                break

        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in {Learner.checkpoint_dir(self.cfg, 0)}"
            )

        checkpoint_dict = Learner.load_checkpoint(checkpoints, self.device)
        self.actor_critic.load_state_dict(checkpoint_dict["model"], strict=False)
        self.actor_critic.eval()

        self.checkpoint_name = Path(checkpoints[-1]).name
        self.rnn_size = get_rnn_size(self.cfg)
        self.num_agents = self.env.num_agents

        self._rnn_states = None
        self.reset_rnn()

        print(f"Loaded {experiment} [{self.checkpoint_name}] on {device}")

    def reset_rnn(self):
        """Reset RNN hidden states."""
        self._rnn_states = torch.zeros(
            self.num_agents, self.rnn_size, device=self.device
        )

    def obs_to_torch(self, obs: dict) -> dict:
        """Convert observation dict to torch tensors."""
        obs_torch = {}
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                obs_torch[key] = val.float().to(self.device)
            else:
                t = torch.from_numpy(np.array(val)).float()
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                obs_torch[key] = t.to(self.device)
        return obs_torch

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        """Given raw obs dict, return actions and update RNN states."""
        obs_torch = self.obs_to_torch(obs)
        normalized = self.actor_critic.normalize_obs(obs_torch)
        result = self.actor_critic(normalized, self._rnn_states)
        self._rnn_states = result["new_rnn_states"]
        return result["actions"].cpu().numpy()

    def close(self):
        """Close the environment."""
        self.env.close()


def is_done(terminated, truncated) -> bool:
    """Check if episode is done, handling both scalar and array returns."""
    if isinstance(terminated, (list, np.ndarray)):
        return bool(terminated[0]) or bool(truncated[0])
    return bool(terminated) or bool(truncated)


def get_reward(rew) -> float:
    """Extract scalar reward from potentially array reward."""
    if isinstance(rew, (list, np.ndarray)):
        return float(rew[0]) if len(rew) > 0 else float(rew)
    return float(rew)


def extract_frame(obs: dict) -> np.ndarray | None:
    """Extract an RGB frame (H, W, C) from obs dict."""
    if "obs" not in obs:
        return None
    raw = obs["obs"]
    if isinstance(raw, torch.Tensor):
        raw = raw[0].cpu().numpy()  # [C, H, W]
        raw = np.transpose(raw, (1, 2, 0))  # [H, W, C]
    return raw.copy()


def run_episodes(
    agent: SFAgent,
    num_episodes: int,
    collect_frames: bool = False,
) -> list[EpisodeStats]:
    """Run N episodes with the agent, return list of EpisodeStats."""
    results = []

    for ep in range(num_episodes):
        obs, info = agent.env.reset()
        agent.reset_rnn()
        stats = EpisodeStats()

        while True:
            action = agent.act(obs)
            obs, rew, terminated, truncated, infos = agent.env.step(action)
            stats.reward += get_reward(rew)
            stats.steps += 1

            if collect_frames:
                frame = extract_frame(obs)
                if frame is not None:
                    stats.frames.append(frame)

            if is_done(terminated, truncated):
                break

        # Extract game variables from final info
        if infos:
            inf = infos[0] if isinstance(infos, list) else infos
            if isinstance(inf, dict):
                for key in [
                    "FRAGCOUNT", "DEATHCOUNT", "HITCOUNT",
                    "DAMAGECOUNT", "HEALTH", "ARMOR",
                ]:
                    if key in inf:
                        stats.game_vars[key.lower()] = float(inf[key])

        results.append(stats)

    return results
