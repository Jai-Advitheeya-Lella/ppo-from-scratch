"""Action distributions implemented directly with tensor ops.

Only the operations PPO needs are provided: sampling, log-probabilities,
entropy and the deterministic mode. They are checked against
``torch.distributions`` in ``tests/test_distributions.py``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_LOG_2PI = math.log(2.0 * math.pi)


class Categorical:
    """Categorical distribution over ``K`` actions parameterised by logits.

    Args:
        logits: Unnormalised log-probabilities with shape ``(..., K)``.
    """

    def __init__(self, logits: Tensor) -> None:
        # Normalise once: log p_k = z_k - logsumexp(z).
        self.log_probs_all = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    @property
    def probs(self) -> Tensor:
        return self.log_probs_all.exp()

    def sample(self) -> Tensor:
        """Draw a sample with the Gumbel-max trick: argmax_k (log p_k + G_k)."""
        u = torch.rand_like(self.log_probs_all).clamp_(1e-12, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(u))
        return torch.argmax(self.log_probs_all + gumbel, dim=-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        """Log-probability of integer ``actions`` with shape ``(...)``."""
        return self.log_probs_all.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> Tensor:
        """Shannon entropy H = -sum_k p_k log p_k."""
        return -(self.probs * self.log_probs_all).sum(dim=-1)

    def mode(self) -> Tensor:
        return torch.argmax(self.log_probs_all, dim=-1)


class DiagGaussian:
    """Multivariate normal with diagonal covariance.

    Log-probabilities and entropies are summed over the last (action) axis, so
    the distribution behaves like a single joint distribution over the action
    vector.

    Args:
        mean: Means with shape ``(..., D)``.
        log_std: Log standard deviations, broadcastable to ``mean``.
    """

    def __init__(self, mean: Tensor, log_std: Tensor) -> None:
        self.mean = mean
        self.log_std = log_std.expand_as(mean)
        self.std = self.log_std.exp()

    def sample(self) -> Tensor:
        """Reparameterised sample mean + std * eps, eps ~ N(0, I)."""
        return self.mean + self.std * torch.randn_like(self.mean)

    def log_prob(self, actions: Tensor) -> Tensor:
        """log N(a; mu, diag(sigma^2)) = -1/2 sum_d [((a-mu)/sigma)^2 + 2 log sigma + log 2pi]."""
        z = (actions - self.mean) / self.std
        return -0.5 * (z.pow(2) + 2.0 * self.log_std + _LOG_2PI).sum(dim=-1)

    def entropy(self) -> Tensor:
        """H = sum_d (1/2 + 1/2 log 2pi + log sigma_d)."""
        return (0.5 + 0.5 * _LOG_2PI + self.log_std).sum(dim=-1)

    def mode(self) -> Tensor:
        return self.mean
