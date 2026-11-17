"""Running statistics for observation normalization and reward scaling."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Streaming mean/variance using the parallel update of Chan et al. (1979).

    Args:
        shape: Shape of a single sample.
        epsilon: Pseudo-count for the initial (zero-mean, unit-variance) prior,
            which avoids division by zero before any data has been seen.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        """Merge a batch of samples with shape ``(B, *shape)`` into the statistics."""
        batch = np.asarray(batch, dtype=np.float64)
        b_mean = batch.mean(axis=0)
        b_var = batch.var(axis=0)
        b_count = batch.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        new_mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def state_dict(self) -> dict[str, np.ndarray | float]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict[str, np.ndarray | float]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])


class ObsNormalizer:
    """Standardises observations with running statistics and clips the result.

    Args:
        shape: Observation shape.
        clip: Normalised observations are clipped to ``[-clip, clip]``.
        epsilon: Added to the variance for numerical stability.
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.rms.update(obs)
        normed = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


class RewardScaler:
    r"""Divides rewards by the running std of the discounted return.

    This is the "reward scaling" of Engstrom et al. (2020) one of the continuous-control
    details in Huang et al. (2022): maintain per-environment
    discounted returns :math:`G \leftarrow \gamma G + r`, track their variance,
    and return :math:`r / \sqrt{\mathrm{Var}[G] + \epsilon}` (clipped). The mean is
    *not* subtracted, so the sign of each reward is preserved.

    Args:
        num_envs: Number of parallel environments.
        gamma: Discount used for the return accumulator.
        clip: Scaled rewards are clipped to ``[-clip, clip]``.
        epsilon: Added to the variance for numerical stability.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.rms = RunningMeanStd(())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        scaled = rewards / np.sqrt(self.rms.var + self.epsilon)
        self.returns[dones.astype(bool)] = 0.0
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)
