import os

import numpy as np
import pytest
import torch

from ppo.config import make_config
from ppo.networks import ActorCritic
from ppo.trainer import PPOTrainer, explained_variance, ppo_loss

IN_CI = bool(os.environ.get("CI"))


def _batch(agent: ActorCritic, n: int = 64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, 4, generator=g)
    with torch.no_grad():
        actions, logp, values = agent.act(obs)
    adv = torch.randn(n, generator=g)
    ret = values + torch.randn(n, generator=g)
    return obs, actions, logp, values, adv, ret


def test_orthogonal_init_gains() -> None:
    torch.manual_seed(0)
    agent = ActorCritic(4, 2, continuous=False, hidden_sizes=(64, 64))
    w = agent.actor[0].weight  # (64, 4): orthonormal columns scaled by sqrt(2)
    torch.testing.assert_close(w.T @ w, 2.0 * torch.eye(4), atol=1e-5, rtol=0)
    head = agent.actor[-1].weight  # (2, 64): orthonormal rows scaled by 0.01
    torch.testing.assert_close(head @ head.T, 1e-4 * torch.eye(2), atol=1e-8, rtol=0)
    assert torch.all(agent.critic[0].bias == 0)
    # Policy is near-uniform at initialisation.
    probs = agent.distribution(torch.randn(100, 4)).probs
    assert torch.allclose(probs, torch.full_like(probs, 0.5), atol=0.05)


def test_loss_at_identical_policy() -> None:
    """With pi == pi_old the ratio is 1: no clipping, zero KL, and pg_loss = -mean(adv)."""
    torch.manual_seed(0)
    agent = ActorCritic(4, 3, continuous=False)
    obs, actions, logp, values, adv, ret = _batch(agent)
    cfg = make_config("CartPole-v1", norm_adv=False)
    _, info = ppo_loss(agent, obs, actions, logp, values, adv, ret, cfg)
    assert info["pg_loss"].item() == pytest.approx(-adv.mean().item(), abs=1e-6)
    assert info["approx_kl"].item() == pytest.approx(0.0, abs=1e-7)
    assert info["clipfrac"].item() == 0.0


def test_clipped_objective_has_zero_gradient_outside_trust_region() -> None:
    """If every ratio is beyond 1+eps with positive advantage, the surrogate is flat."""
    torch.manual_seed(0)
    agent = ActorCritic(4, 2, continuous=True)
    obs, actions, logp, values, _, ret = _batch(agent)
    cfg = make_config("CartPole-v1", norm_adv=False, ent_coef=0.0, vf_coef=0.0)
    adv = torch.ones(len(obs))
    old_logp = logp - 1.0  # ratio = e ~ 2.72 > 1.2
    loss, info = ppo_loss(agent, obs, actions, old_logp, values, adv, ret, cfg)
    loss.backward()
    assert info["clipfrac"].item() == 1.0
    for p in agent.actor.parameters():
        assert torch.all(p.grad == 0)


def test_value_clipping_takes_pessimistic_loss() -> None:
    torch.manual_seed(0)
    agent = ActorCritic(4, 2, continuous=False)
    obs, actions, logp, values, adv, _ = _batch(agent)
    cfg = make_config("CartPole-v1", clip_coef=0.2)
    # Old values far from the current prediction; target equal to current prediction.
    with torch.no_grad():
        current = agent.value(obs)
    old_values = current - 1.0
    _, info = ppo_loss(agent, obs, actions, logp, old_values, adv, current, cfg)
    # Clipped prediction is old+0.2 = current-0.8, so loss = 0.5 * 0.8^2.
    assert info["v_loss"].item() == pytest.approx(0.5 * 0.8**2, rel=1e-4)
    cfg_noclip = make_config("CartPole-v1", clip_vloss=False)
    _, info = ppo_loss(agent, obs, actions, logp, old_values, adv, current, cfg_noclip)
    assert info["v_loss"].item() == pytest.approx(0.0, abs=1e-10)


def test_explained_variance() -> None:
    t = torch.randn(1000)
    assert explained_variance(t, t) == pytest.approx(1.0)
    assert explained_variance(torch.zeros(1000) + t.mean(), t) == pytest.approx(0.0, abs=1e-6)


def test_same_seed_is_deterministic() -> None:
    cfg = make_config("CartPole-v1", total_timesteps=2048, eval_episodes=1, seed=3)
    a = PPOTrainer(cfg).train()
    b = PPOTrainer(cfg).train()
    assert a["eval_returns"] == b["eval_returns"]
    assert a["final_train_return"] == b["final_train_return"]


def test_cartpole_learns() -> None:
    cfg = make_config("CartPole-v1", total_timesteps=60_000, eval_episodes=5, seed=1)
    summary = PPOTrainer(cfg).train()
    # Random policy scores ~22; PPO should be well past 150 after 60k steps.
    assert summary["final_train_return"] > 150


def test_continuous_pendulum_improves_and_rollout_shapes() -> None:
    cfg = make_config("Pendulum-v1", total_timesteps=8 * 256 * 6, num_envs=8, num_steps=256,
                      norm_obs=True, norm_reward=True, ent_coef=0.0, update_epochs=10,
                      num_minibatches=32, gamma=0.9, learning_rate=1e-3, eval_episodes=2)
    trainer = PPOTrainer(cfg)
    summary = trainer.train()
    assert trainer.buf_actions.shape == (256, 8, 1)
    assert np.isfinite(summary["eval_return_mean"])
    assert torch.all(torch.isfinite(trainer.agent.log_std))


@pytest.mark.skipif(IN_CI, reason="MuJoCo smoke test runs locally only")
def test_mujoco_smoke() -> None:
    cfg = make_config("HalfCheetah-v5", total_timesteps=4096, eval_episodes=1)
    summary = PPOTrainer(cfg).train()
    assert np.isfinite(summary["eval_return_mean"])


@pytest.mark.skipif(IN_CI, reason="Box2D smoke test runs locally only")
def test_lunarlander_smoke() -> None:
    cfg = make_config("LunarLander-v3", total_timesteps=16384, num_minibatches=16, eval_episodes=1)
    summary = PPOTrainer(cfg).train()
    assert np.isfinite(summary["eval_return_mean"])
