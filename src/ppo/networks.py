"""Actor-critic networks with orthogonal initialisation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ppo.distributions import Categorical, DiagGaussian

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}


def layer_init(layer: nn.Linear, gain: float, orthogonal: bool = True) -> nn.Linear:
    """Orthogonal weight init with a given gain and zero bias (Huang et al. 2022, detail #2).

    When ``orthogonal`` is False the PyTorch default (Kaiming-uniform) init is kept,
    which is what the ablation compares against.
    """
    if orthogonal:
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


def mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    activation: str,
    out_gain: float,
    orthogonal: bool,
) -> nn.Sequential:
    """MLP with hidden gains sqrt(2) and a custom gain on the output layer."""
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [layer_init(nn.Linear(prev, h), math.sqrt(2), orthogonal), act()]
        prev = h
    layers.append(layer_init(nn.Linear(prev, out_dim), out_gain, orthogonal))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate policy and value MLPs (no shared trunk).

    Discrete action spaces use a categorical policy over logits. Continuous
    spaces use a diagonal Gaussian whose mean is produced by the network and
    whose log standard deviation is a state-independent learned vector
    initialised to zero (Huang et al. 2022, continuous details #1-#2).

    Args:
        obs_dim: Flattened observation dimension.
        action_dim: Number of discrete actions, or the action-vector dimension.
        continuous: Whether the action space is continuous.
        hidden_sizes: Hidden layer widths shared by actor and critic.
        activation: ``"tanh"`` or ``"relu"``.
        orthogonal_init: Use orthogonal init (policy head gain 0.01, value head gain 1).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        continuous: bool,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        orthogonal_init: bool = True,
    ) -> None:
        super().__init__()
        self.continuous = continuous
        self.critic = mlp(obs_dim, hidden_sizes, 1, activation, 1.0, orthogonal_init)
        self.actor = mlp(obs_dim, hidden_sizes, action_dim, activation, 0.01, orthogonal_init)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def value(self, obs: Tensor) -> Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: Tensor) -> Categorical | DiagGaussian:
        out = self.actor(obs)
        if self.continuous:
            return DiagGaussian(out, self.log_std)
        return Categorical(out)

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Sample actions for a batch of observations.

        Returns:
            ``(actions, log_probs, values)``.
        """
        dist = self.distribution(obs)
        actions = dist.mode() if deterministic else dist.sample()
        return actions, dist.log_prob(actions), self.value(obs)

    def evaluate(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Log-probs, entropies and values of stored actions under the current parameters."""
        dist = self.distribution(obs)
        return dist.log_prob(actions), dist.entropy(), self.value(obs)
