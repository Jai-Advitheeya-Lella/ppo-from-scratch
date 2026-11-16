"""Generalized Advantage Estimation (Schulman et al., 2016)."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    r"""Compute GAE(gamma, lambda) advantages and lambda-returns for a rollout.

    Uses the backward recursion

    .. math::
        \delta_t = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t), \qquad
        \hat A_t = \delta_t + \gamma \lambda (1 - d_t) \hat A_{t+1}.

    Args:
        rewards: ``(T, N)`` rewards ``r_t`` received after acting in ``s_t``.
            Time-limit truncations should already have ``gamma * V(s_final)``
            folded in (see :mod:`ppo.trainer`).
        values: ``(T, N)`` critic values ``V(s_t)``.
        dones: ``(T, N)`` 1.0 if the episode ended at transition ``t``
            (terminated or truncated), in which case ``s_{t+1}`` in the buffer is
            the first state of a new episode and must not be bootstrapped from.
        last_value: ``(N,)`` value of the state following the final step.
        gamma: Discount factor.
        gae_lambda: GAE trace-decay parameter.

    Returns:
        ``(advantages, returns)`` each of shape ``(T, N)`` where
        ``returns = advantages + values`` are the regression targets for the critic.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(num_steps)):
        next_value = last_value if t == num_steps - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values
