"""Reference baseline: Stable-Baselines3 PPO with hyperparameters matched to our preset.

Writes the same ``episodes.csv`` / ``summary.json`` files as ``ppo.train`` so the two
implementations can be plotted on the same axes.

    python scripts/sb3_baseline.py --env-id HalfCheetah-v5 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from ppo.config import make_config


class EpisodeLogger(BaseCallback):
    """Records (global_step, return, length) of every finished training episode."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[int, float, int]] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.rows.append((self.num_timesteps, float(info["episode"]["r"]), int(info["episode"]["l"])))
        return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = make_config(args.env_id, seed=args.seed, total_timesteps=args.total_timesteps)
    run_dir = args.run_dir or Path("runs") / "sb3" / cfg.env_id / "default" / f"seed{cfg.seed}"
    torch.set_num_threads(cfg.threads)

    venv = make_vec_env(cfg.env_id, n_envs=cfg.num_envs, seed=cfg.seed)
    if cfg.norm_obs or cfg.norm_reward:
        venv = VecNormalize(venv, norm_obs=cfg.norm_obs, norm_reward=cfg.norm_reward, gamma=cfg.gamma,
                            clip_obs=10.0, clip_reward=10.0)

    lr = cfg.learning_rate
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=(lambda frac: frac * lr) if cfg.anneal_lr else lr,
        n_steps=cfg.num_steps, batch_size=cfg.minibatch_size, n_epochs=cfg.update_epochs,
        gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_coef,
        clip_range_vf=cfg.clip_coef if cfg.clip_vloss else None,
        normalize_advantage=cfg.norm_adv, ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm, target_kl=cfg.target_kl,
        policy_kwargs=dict(net_arch=dict(pi=list(cfg.hidden_sizes), vf=list(cfg.hidden_sizes)),
                           activation_fn=torch.nn.Tanh, ortho_init=cfg.orthogonal_init, log_std_init=0.0),
        seed=cfg.seed, device="cpu", verbose=0,
    )
    logger = EpisodeLogger()
    t0 = time.time()
    model.learn(total_timesteps=cfg.total_timesteps, callback=logger)
    train_time = time.time() - t0

    # Deterministic evaluation with frozen normalisation statistics.
    eval_env = make_vec_env(cfg.env_id, n_envs=1, seed=cfg.seed + 10_000)
    if isinstance(venv, VecNormalize):
        eval_env = VecNormalize(eval_env, norm_obs=cfg.norm_obs, norm_reward=False, training=False,
                                clip_obs=10.0)
        eval_env.obs_rms = venv.obs_rms
    returns: list[float] = []
    obs = eval_env.reset()
    while len(returns) < cfg.eval_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = eval_env.step(action)
        if done[0]:
            returns.append(float(info[0]["episode"]["r"]))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_step", "return", "length"])
        w.writerows(logger.rows)
    summary = {
        "env_id": cfg.env_id, "seed": cfg.seed, "total_timesteps": int(model.num_timesteps),
        "train_time_s": train_time,
        "final_train_return": float(np.mean([r for _, r, _ in logger.rows[-20:]])),
        "eval_return_mean": float(np.mean(returns)), "eval_return_std": float(np.std(returns)),
        "eval_returns": returns,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[sb3 {cfg.env_id} seed={cfg.seed}] eval {summary['eval_return_mean']:.1f} "
          f"+/- {summary['eval_return_std']:.1f} ({train_time:.0f}s)")


if __name__ == "__main__":
    main()
