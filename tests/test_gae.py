import numpy as np
import pytest
import torch

from ppo.gae import compute_gae


def reference_gae(rewards, values, dones, last_value, gamma, lam):
    """Direct definition: A_t = sum_l (gamma*lam)^l delta_{t+l}, truncated at episode ends."""
    T, N = rewards.shape
    next_values = np.concatenate([values[1:], last_value[None]], axis=0)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    adv = np.zeros_like(rewards)
    for n in range(N):
        for t in range(T):
            total, coef = 0.0, 1.0
            for k in range(t, T):
                total += coef * deltas[k, n]
                if dones[k, n]:
                    break
                coef *= gamma * lam
            adv[t, n] = total
    return adv


def random_rollout(seed: int, T: int = 37, N: int = 5):
    rng = np.random.default_rng(seed)
    rewards = rng.normal(size=(T, N))
    values = rng.normal(size=(T, N))
    dones = (rng.random((T, N)) < 0.1).astype(np.float64)
    last_value = rng.normal(size=N)
    return rewards, values, dones, last_value


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("gamma,lam", [(0.99, 0.95), (0.9, 0.0), (0.999, 1.0), (0.5, 0.7)])
def test_gae_matches_reference(seed: int, gamma: float, lam: float) -> None:
    r, v, d, lv = random_rollout(seed)
    adv, ret = compute_gae(*map(torch.from_numpy, (r, v, d, lv)), gamma, lam)
    expected = reference_gae(r, v, d, lv, gamma, lam)
    np.testing.assert_allclose(adv.numpy(), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ret.numpy(), expected + v, rtol=1e-10, atol=1e-10)


def test_lambda_zero_is_td_error() -> None:
    r, v, d, lv = random_rollout(0)
    adv, _ = compute_gae(*map(torch.from_numpy, (r, v, d, lv)), 0.99, 0.0)
    nv = np.concatenate([v[1:], lv[None]])
    np.testing.assert_allclose(adv.numpy(), r + 0.99 * nv * (1 - d) - v, atol=1e-12)


def test_lambda_one_is_monte_carlo_return_minus_value() -> None:
    gamma = 0.97
    r, v, d, lv = random_rollout(1)
    _, ret = compute_gae(*map(torch.from_numpy, (r, v, d, lv)), gamma, 1.0)
    T, N = r.shape
    mc = np.zeros_like(r)
    for n in range(N):
        g = lv[n]
        for t in reversed(range(T)):
            g = r[t, n] + gamma * g * (1 - d[t, n])
            mc[t, n] = g
    np.testing.assert_allclose(ret.numpy(), mc, atol=1e-10)


def test_done_blocks_bootstrap_across_episodes() -> None:
    # Two-step rollout where the episode ends at t=0: A_0 must ignore V(s_1) and everything after.
    rewards = torch.tensor([[1.0], [5.0]])
    values = torch.tensor([[0.5], [100.0]])
    dones = torch.tensor([[1.0], [0.0]])
    adv, _ = compute_gae(rewards, values, dones, torch.tensor([1000.0]), 0.99, 0.95)
    assert adv[0, 0].item() == pytest.approx(1.0 - 0.5)
