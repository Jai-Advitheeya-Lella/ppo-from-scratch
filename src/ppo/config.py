"""Hyperparameter configuration and per-environment presets."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PPOConfig:
    """All PPO hyperparameters. Defaults follow CleanRL's ``ppo.py`` (Atari-free, classic control)."""

    env_id: str = "CartPole-v1"
    seed: int = 1
    total_timesteps: int = 500_000
    num_envs: int = 4
    num_steps: int = 128
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    # Implementation details that the ablation toggles.
    norm_adv: bool = True
    clip_vloss: bool = True
    norm_obs: bool = False
    norm_reward: bool = False
    orthogonal_init: bool = True
    bootstrap_truncated: bool = True
    # Network.
    hidden_sizes: tuple[int, ...] = field(default=(64, 64))
    activation: str = "tanh"
    # Bookkeeping.
    threads: int = 1
    eval_episodes: int = 20

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PPOConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if "hidden_sizes" in kwargs:
            kwargs["hidden_sizes"] = tuple(kwargs["hidden_sizes"])
        return cls(**kwargs)


_MUJOCO = dict(
    total_timesteps=1_000_000,
    num_envs=1,
    num_steps=2048,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=10,
    ent_coef=0.0,
    norm_obs=True,
    norm_reward=True,
)

#: Per-environment presets. Classic control follows CleanRL ``ppo.py``; LunarLander
#: follows the rl-baselines3-zoo tuned values plus observation/reward normalisation
#: (without reward scaling, value clipping at 0.2 in raw return units throttles the
#: critic; see README); MuJoCo follows CleanRL
#: ``ppo_continuous_action.py`` (itself the original OpenAI baselines ``mujoco`` preset).
PRESETS: dict[str, dict[str, Any]] = {
    "CartPole-v1": dict(total_timesteps=500_000),
    "LunarLander-v3": dict(
        total_timesteps=1_000_000,
        num_envs=16,
        num_steps=1024,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.98,
        num_minibatches=256,
        update_epochs=4,
        ent_coef=0.01,
        norm_obs=True,
        norm_reward=True,
    ),
    "HalfCheetah-v5": dict(_MUJOCO),
    "Hopper-v5": dict(_MUJOCO),
    "Walker2d-v5": dict(_MUJOCO),
}


def make_config(env_id: str, **overrides: Any) -> PPOConfig:
    """Build a config from the preset for ``env_id`` with keyword overrides applied last."""
    params: dict[str, Any] = {"env_id": env_id, **PRESETS.get(env_id, {})}
    params.update({k: v for k, v in overrides.items() if v is not None})
    return PPOConfig.from_dict(params)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Expose every config field as an optional CLI flag (``None`` = use preset)."""
    for f in dataclasses.fields(PPOConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "env_id":
            parser.add_argument(flag, type=str, default="CartPole-v1")
        elif f.type in ("bool", bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif f.name == "hidden_sizes":
            parser.add_argument(flag, type=int, nargs="+", default=None)
        elif f.name == "target_kl":
            parser.add_argument(flag, type=float, default=None)
        else:
            caster = {"int": int, "float": float, "str": str}[str(f.type)]
            parser.add_argument(flag, type=caster, default=None)
