"""Environment construction helpers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv


def make_vec_env(env_id: str, num_envs: int, render_mode: str | None = None) -> SyncVectorEnv:
    """Synchronous vector env that resets finished sub-environments in the same step.

    With ``AutoresetMode.SAME_STEP`` the observation returned for a finished
    sub-environment is already the first observation of the next episode, and the
    true final observation is exposed in ``info["final_obs"]``. This keeps every
    buffer slot a real transition and lets the trainer bootstrap through time-limit
    truncations.
    """

    def thunk() -> gym.Env:
        return gym.make(env_id, render_mode=render_mode)

    return SyncVectorEnv([thunk for _ in range(num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)


def space_info(env: gym.Env | SyncVectorEnv) -> tuple[int, int, bool, np.ndarray | None, np.ndarray | None]:
    """Return ``(obs_dim, action_dim, continuous, low, high)`` for a (vector) env."""
    obs_space = env.single_observation_space if isinstance(env, SyncVectorEnv) else env.observation_space
    act_space = env.single_action_space if isinstance(env, SyncVectorEnv) else env.action_space
    obs_dim = int(np.prod(obs_space.shape))
    if isinstance(act_space, gym.spaces.Discrete):
        return obs_dim, int(act_space.n), False, None, None
    if isinstance(act_space, gym.spaces.Box):
        return obs_dim, int(np.prod(act_space.shape)), True, act_space.low, act_space.high
    raise NotImplementedError(f"Unsupported action space {act_space}")
