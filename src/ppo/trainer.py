"""PPO training loop (Schulman et al., 2017) with the details of Huang et al. (2022)."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ppo.config import PPOConfig
from ppo.envs import make_vec_env, space_info
from ppo.gae import compute_gae
from ppo.networks import ActorCritic
from ppo.normalization import ObsNormalizer, RewardScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class UpdateStats:
    """Diagnostics averaged over all minibatches of one PPO update."""

    pg_loss: float
    v_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    explained_var: float


def explained_variance(pred: Tensor, target: Tensor) -> float:
    """1 - Var[target - pred] / Var[target]; 1 is a perfect critic, <= 0 is no better than a constant."""
    var_t = target.var()
    return float("nan") if var_t == 0 else float(1.0 - (target - pred).var() / var_t)


def ppo_loss(
    agent: ActorCritic,
    obs: Tensor,
    actions: Tensor,
    old_logp: Tensor,
    old_values: Tensor,
    advantages: Tensor,
    returns: Tensor,
    cfg: PPOConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    r"""Clipped surrogate objective plus value loss and entropy bonus for one minibatch.

    .. math::
        L = -\mathbb{E}[\min(\rho_t \hat A_t, \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t)]
            + c_v L^V - c_e \mathbb{E}[H[\pi_\theta(\cdot|s_t)]]

    where :math:`\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)`.
    """
    new_logp, entropy, new_values = agent.evaluate(obs, actions)
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()

    if cfg.norm_adv:  # per-minibatch advantage normalisation (detail #7)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss = torch.max(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
    ).mean()

    if cfg.clip_vloss:  # value clipping around the old prediction (detail #9)
        v_unclipped = (new_values - returns).pow(2)
        v_clipped = old_values + torch.clamp(new_values - old_values, -cfg.clip_coef, cfg.clip_coef)
        v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - returns).pow(2)).mean()
    else:
        v_loss = 0.5 * (new_values - returns).pow(2).mean()

    ent = entropy.mean()
    loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

    with torch.no_grad():
        # Low-variance unbiased KL estimator k3 = (rho - 1) - log rho (Schulman, 2020).
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
    info = {"pg_loss": pg_loss.detach(), "v_loss": v_loss.detach(), "entropy": ent.detach(),
            "approx_kl": approx_kl, "clipfrac": clipfrac}
    return loss, info


class PPOTrainer:
    """Collects rollouts from a vector env and optimises the PPO objective.

    Args:
        cfg: Hyperparameters.
        run_dir: Output directory for CSV logs, the final checkpoint and eval results.
            ``None`` disables all file output (used by tests).
    """

    def __init__(self, cfg: PPOConfig, run_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        torch.set_num_threads(cfg.threads)
        set_seed(cfg.seed)

        self.envs = make_vec_env(cfg.env_id, cfg.num_envs)
        self.envs.action_space.seed(cfg.seed)
        obs_dim, act_dim, self.continuous, self.act_low, self.act_high = space_info(self.envs)
        self.obs_dim, self.act_dim = obs_dim, act_dim

        self.agent = ActorCritic(obs_dim, act_dim, self.continuous, cfg.hidden_sizes,
                                 cfg.activation, cfg.orthogonal_init)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.obs_norm = ObsNormalizer((obs_dim,)) if cfg.norm_obs else None
        self.reward_scaler = RewardScaler(cfg.num_envs, cfg.gamma) if cfg.norm_reward else None

        T, N = cfg.num_steps, cfg.num_envs
        act_shape = (act_dim,) if self.continuous else ()
        self.buf_obs = torch.zeros((T, N, obs_dim))
        self.buf_actions = torch.zeros((T, N, *act_shape))
        self.buf_logp = torch.zeros((T, N))
        self.buf_rewards = torch.zeros((T, N))
        self.buf_dones = torch.zeros((T, N))
        self.buf_values = torch.zeros((T, N))

        self.global_step = 0
        self.episodes: list[tuple[int, float, int]] = []  # (global_step, return, length)

    # ------------------------------------------------------------------ helpers
    def _normalize_obs(self, obs: np.ndarray, update: bool) -> np.ndarray:
        if self.obs_norm is None:
            return obs.astype(np.float32)
        return self.obs_norm(obs, update=update)

    def _env_action(self, actions: Tensor) -> np.ndarray:
        a = actions.numpy()
        if self.continuous:  # sample is unclipped for log-probs; clip only what the env sees (detail #4)
            a = np.clip(a, self.act_low, self.act_high)
        return a

    # ----------------------------------------------------------------- rollout
    def collect_rollout(self, obs: np.ndarray, ep_ret: np.ndarray, ep_len: np.ndarray) -> np.ndarray:
        """Fill the rollout buffer with ``num_steps`` transitions per environment."""
        cfg = self.cfg
        for t in range(cfg.num_steps):
            obs_t = torch.from_numpy(obs)
            actions, logp, values = self.agent.act(obs_t)
            next_obs_raw, reward, terminated, truncated, info = self.envs.step(self._env_action(actions))
            self.global_step += cfg.num_envs
            done = np.logical_or(terminated, truncated)

            ep_ret += reward
            ep_len += 1
            for i in np.flatnonzero(done):
                self.episodes.append((self.global_step, float(ep_ret[i]), int(ep_len[i])))
                ep_ret[i], ep_len[i] = 0.0, 0

            next_obs = self._normalize_obs(next_obs_raw, update=True)
            r = self.reward_scaler(reward, done) if self.reward_scaler else reward.astype(np.float32)
            r = torch.as_tensor(r, dtype=torch.float32).clone()

            # Bootstrap through time limits: a truncated episode did not really end,
            # so add gamma * V(s_final) to the last reward (Pardo et al., 2018).
            trunc_only = np.flatnonzero(truncated & ~terminated)
            if cfg.bootstrap_truncated and trunc_only.size:
                final = np.stack([info["final_obs"][i] for i in trunc_only])
                final_t = torch.from_numpy(self._normalize_obs(final, update=False))
                with torch.no_grad():
                    r[trunc_only] += cfg.gamma * self.agent.value(final_t)

            self.buf_obs[t] = obs_t
            self.buf_actions[t] = actions
            self.buf_logp[t] = logp
            self.buf_values[t] = values
            self.buf_rewards[t] = r
            self.buf_dones[t] = torch.from_numpy(done.astype(np.float32))
            obs = next_obs
        return obs

    # ------------------------------------------------------------------ update
    def update(self, last_obs: np.ndarray) -> UpdateStats:
        cfg = self.cfg
        with torch.no_grad():
            last_value = self.agent.value(torch.from_numpy(last_obs))
        advantages, returns = compute_gae(self.buf_rewards, self.buf_values, self.buf_dones,
                                          last_value, cfg.gamma, cfg.gae_lambda)

        b_obs = self.buf_obs.reshape(-1, self.obs_dim)
        b_actions = self.buf_actions.reshape((-1, self.act_dim) if self.continuous else (-1,))
        b_logp, b_adv = self.buf_logp.reshape(-1), advantages.reshape(-1)
        b_ret, b_val = returns.reshape(-1), self.buf_values.reshape(-1)

        sums: dict[str, float] = {}
        n_mb = 0
        params = list(self.agent.parameters())
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(cfg.batch_size)
            stop = False
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]
                loss, info = ppo_loss(self.agent, b_obs[mb], b_actions[mb], b_logp[mb], b_val[mb],
                                      b_adv[mb], b_ret[mb], cfg)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)  # detail #11
                self.optimizer.step()
                for k, v in info.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                n_mb += 1
                if cfg.target_kl is not None and float(info["approx_kl"]) > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break

        means = {k: v / n_mb for k, v in sums.items()}
        return UpdateStats(explained_var=explained_variance(b_val, b_ret), **means)

    # -------------------------------------------------------------------- main
    def train(self) -> dict[str, float]:
        cfg = self.cfg
        num_updates = cfg.total_timesteps // cfg.batch_size
        obs_raw, _ = self.envs.reset(seed=cfg.seed)
        obs = self._normalize_obs(obs_raw, update=True)
        ep_ret = np.zeros(cfg.num_envs, dtype=np.float64)
        ep_len = np.zeros(cfg.num_envs, dtype=np.int64)

        update_rows: list[dict[str, float]] = []
        t0 = time.time()
        for update in range(1, num_updates + 1):
            if cfg.anneal_lr:  # linear decay to zero (detail #4)
                frac = 1.0 - (update - 1.0) / num_updates
                self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate
            obs = self.collect_rollout(obs, ep_ret, ep_len)
            stats = self.update(obs)
            sps = self.global_step / (time.time() - t0)
            recent = [r for _, r, _ in self.episodes[-20:]]
            update_rows.append({"global_step": self.global_step, "lr": self.optimizer.param_groups[0]["lr"],
                                "sps": sps, **stats.__dict__})
            if update % max(1, num_updates // 20) == 0 or update == num_updates:
                mean_r = np.mean(recent) if recent else float("nan")
                print(f"[{cfg.env_id} seed={cfg.seed}] step={self.global_step:>8d} "
                      f"return(last20)={mean_r:8.1f} kl={stats.approx_kl:.4f} "
                      f"clipfrac={stats.clipfrac:.3f} ev={stats.explained_var:.2f} sps={sps:.0f}",
                      flush=True)
        train_time = time.time() - t0
        self.envs.close()

        eval_returns = evaluate(self.agent, cfg.env_id, self.obs_norm, cfg.eval_episodes,
                                seed=cfg.seed + 10_000)
        summary = {
            "env_id": cfg.env_id, "seed": cfg.seed, "total_timesteps": self.global_step,
            "train_time_s": train_time,
            "final_train_return": float(np.mean([r for _, r, _ in self.episodes[-20:]])),
            "eval_return_mean": float(np.mean(eval_returns)),
            "eval_return_std": float(np.std(eval_returns)),
            "eval_returns": eval_returns,
        }
        if self.run_dir is not None:
            self.save(update_rows, summary)
        return summary

    def save(self, update_rows: list[dict[str, float]], summary: dict) -> None:
        assert self.run_dir is not None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "episodes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["global_step", "return", "length"])
            w.writerows(self.episodes)
        with open(self.run_dir / "updates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(update_rows[0].keys()))
            w.writeheader()
            w.writerows(update_rows)
        (self.run_dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        torch.save({"model": self.agent.state_dict(), "config": self.cfg.to_dict(),
                    "obs_rms": self.obs_norm.rms.state_dict() if self.obs_norm else None},
                   self.run_dir / "model.pt")
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def evaluate(
    agent: ActorCritic,
    env_id: str,
    obs_norm: ObsNormalizer | None,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> list[float]:
    """Run ``episodes`` full episodes with frozen normalisation statistics."""
    import gymnasium as gym

    env = gym.make(env_id)
    _, _, continuous, low, high = space_info(env)
    returns: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            x = obs_norm(obs[None], update=False) if obs_norm else obs[None].astype(np.float32)
            action, _, _ = agent.act(torch.from_numpy(x), deterministic=deterministic)
            a = action[0].numpy()
            if continuous:
                a = np.clip(a, low, high)
            obs, r, term, trunc, _ = env.step(a if continuous else int(a))
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return returns


def load_agent(path: Path) -> tuple[ActorCritic, ObsNormalizer | None, PPOConfig]:
    """Load a checkpoint written by :meth:`PPOTrainer.save`."""
    import gymnasium as gym

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PPOConfig.from_dict(ckpt["config"])
    env = gym.make(cfg.env_id)
    obs_dim, act_dim, continuous, _, _ = space_info(env)
    env.close()
    agent = ActorCritic(obs_dim, act_dim, continuous, cfg.hidden_sizes, cfg.activation)
    agent.load_state_dict(ckpt["model"])
    obs_norm = None
    if ckpt["obs_rms"] is not None:
        obs_norm = ObsNormalizer((obs_dim,))
        obs_norm.rms.load_state_dict(ckpt["obs_rms"])
    return agent, obs_norm, cfg
