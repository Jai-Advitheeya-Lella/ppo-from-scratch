"""Aggregate ``runs/`` into learning-curve figures and markdown tables.

    python scripts/make_figures.py

Outputs ``assets/learning_curves.png``, ``assets/ablation_<env>.png`` and
``assets/results.md`` (the tables pasted into the README).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ASSETS = ROOT / "assets"
SEEDS = [1, 2, 3, 4, 5]
# Categorical slots 1 and 2 of the validated reference palette; ink colours for text.
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ENV_STEPS = {"CartPole-v1": 500_000, "LunarLander-v3": 1_000_000, "HalfCheetah-v5": 1_000_000,
             "Hopper-v5": 1_000_000}
ABLATIONS = {
    "no_norm_obs": "no obs. normalisation",
    "no_norm_reward": "no reward scaling",
    "no_norm_adv": "no advantage normalisation",
    "no_clip_vloss": "no value-loss clipping",
    "no_ortho_init": "default (Kaiming) init",
    "no_bootstrap_truncated": "no time-limit bootstrap",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 100,
})


def run_dir(impl: str, env: str, variant: str, seed: int) -> Path:
    return RUNS / impl / env / variant / f"seed{seed}"


def curve(path: Path, grid: np.ndarray, window: int = 20) -> np.ndarray:
    """Mean return of the last ``window`` training episodes finished at or before each grid step."""
    data = np.loadtxt(path / "episodes.csv", delimiter=",", skiprows=1)
    steps, rets = data[:, 0], data[:, 1]
    cs = np.concatenate([[0.0], np.cumsum(rets)])
    hi = np.searchsorted(steps, grid, side="right")
    lo = np.maximum(hi - window, 0)
    out = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    out[hi == 0] = np.nan
    return out


def curves(impl: str, env: str, variant: str, grid: np.ndarray) -> np.ndarray:
    return np.stack([curve(run_dir(impl, env, variant, s), grid) for s in SEEDS])


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile-bootstrap CI of the mean over axis 0 (seeds)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], size=(n_boot, x.shape[0]))
    means = np.nanmean(x[idx], axis=1)
    return np.nanpercentile(means, 2.5, axis=0), np.nanpercentile(means, 97.5, axis=0)


def plot_band(ax: plt.Axes, grid: np.ndarray, y: np.ndarray, color: str, label: str, ls: str = "-") -> None:
    lo, hi = bootstrap_ci(y)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, np.nanmean(y, axis=0), color=color, lw=2, ls=ls, label=label)


def summaries(impl: str, env: str, variant: str) -> dict[str, np.ndarray]:
    rows = [json.loads((run_dir(impl, env, variant, s) / "summary.json").read_text()) for s in SEEDS]
    return {"eval": np.array([r["eval_return_mean"] for r in rows]),
            "time": np.array([r["train_time_s"] for r in rows])}


def perm_test(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation test on the difference of means (C(10,5)=252 relabellings)."""
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    n = len(a)
    diffs = [abs(pooled[list(i)].mean() - np.delete(pooled, list(i)).mean())
             for i in itertools.combinations(range(len(pooled)), n)]
    return float(np.mean(np.array(diffs) >= obs - 1e-9))


def fmt(x: np.ndarray) -> str:
    return f"{x.mean():.0f} ± {x.std(ddof=1):.0f}"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    md: list[str] = []

    # ---------------------------------------------------------------- main curves
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), constrained_layout=True)
    md += ["| Environment | Steps | Ours (final eval return) | SB3 PPO (final eval return) | Ours train time / run |",
           "|---|---|---|---|---|"]
    for ax, env in zip(axes.flat, ENV_STEPS):
        grid = np.linspace(ENV_STEPS[env] / 100, ENV_STEPS[env], 100)
        plot_band(ax, grid, curves("ppo", env, "default", grid), BLUE, "this repo (PyTorch)")
        plot_band(ax, grid, curves("sb3", env, "default", grid), ORANGE, "Stable-Baselines3 2.9.2", ls="--")
        ax.set_title(env)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("training return (last 20 episodes)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"))
        ours, ref = summaries("ppo", env, "default"), summaries("sb3", env, "default")
        md.append(f"| {env} | {ENV_STEPS[env] // 1000}k | {fmt(ours['eval'])} | {fmt(ref['eval'])} | "
                  f"{ours['time'].mean() / 60:.1f} min |")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("PPO learning curves, 5 seeds (line: mean, band: 95% bootstrap CI of the mean)", fontsize=11)
    fig.savefig(ASSETS / "learning_curves.png", dpi=150)
    plt.close(fig)
    md.append("")
    md.append("Per-seed final evaluation returns (20 deterministic episodes each):")
    md.append("")
    for env in ENV_STEPS:
        o, r = summaries("ppo", env, "default")["eval"], summaries("sb3", env, "default")["eval"]
        md.append(f"- {env}: ours {np.round(o).astype(int).tolist()}; SB3 {np.round(r).astype(int).tolist()}")

    # ------------------------------------------------------------------ ablations
    for env in ("LunarLander-v3", "HalfCheetah-v5"):
        grid = np.linspace(ENV_STEPS[env] / 100, ENV_STEPS[env], 100)
        base = curves("ppo", env, "default", grid)
        base_eval = summaries("ppo", env, "default")["eval"]
        base_auc = np.nanmean(base, axis=1)
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True, constrained_layout=True)
        md += ["", f"**{env}** (5 seeds each; final deterministic-evaluation return, mean ± std across seeds; "
               "*p* is an exact two-sided permutation test against the default; \"Train AUC\" is the mean "
               "training return over the whole run, a measure of sample efficiency):", "",
               "| Variant | Eval return | Δ vs default | p | Train AUC | p (AUC) | Per-seed eval |",
               "|---|---|---|---|---|---|---|",
               f"| default | {fmt(base_eval)} | – | – | {fmt(base_auc)} | – | "
               f"{np.round(base_eval).astype(int).tolist()} |"]
        for ax, (name, label) in zip(axes.flat, ABLATIONS.items()):
            plot_band(ax, grid, base, BLUE, "default")
            var = curves("ppo", env, name, grid)
            plot_band(ax, grid, var, ORANGE, label)
            auc = np.nanmean(var, axis=1)
            ax.set_title(label, fontsize=10)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"))
            ev = summaries("ppo", env, name)["eval"]
            md.append(f"| {label} | {fmt(ev)} | {ev.mean() - base_eval.mean():+.0f} | "
                      f"{perm_test(base_eval, ev):.3f} | {fmt(auc)} | {perm_test(base_auc, auc):.3f} | "
                      f"{np.round(ev).astype(int).tolist()} |")
        inter = "no_norm_reward_no_clip_vloss"
        if env == "LunarLander-v3" and (RUNS / "ppo" / env / inter).exists():
            ev = summaries("ppo", env, inter)["eval"]
            nr = summaries("ppo", env, "no_norm_reward")["eval"]
            auc = np.nanmean(curves("ppo", env, inter, grid), axis=1)
            md.append(f"| no reward scaling *and* no value-loss clipping | {fmt(ev)} | {ev.mean() - base_eval.mean():+.0f} | "
                      f"{perm_test(base_eval, ev):.3f} (vs. no reward scaling alone: {perm_test(nr, ev):.3f}) | "
                      f"{fmt(auc)} | {perm_test(base_auc, auc):.3f} | {np.round(ev).astype(int).tolist()} |")
        for ax in axes[-1]:
            ax.set_xlabel("environment steps")
        for ax in axes[:, 0]:
            ax.set_ylabel("training return")
        handles = [plt.Line2D([], [], color=BLUE, lw=2, label="default"),
                   plt.Line2D([], [], color=ORANGE, lw=2, label="one detail removed")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2)
        fig.suptitle(f"{env}: removing one implementation detail at a time (mean and 95% CI over 5 seeds)",
                     fontsize=11)
        fig.savefig(ASSETS / f"ablation_{env.split('-')[0].lower()}.png", dpi=150)
        plt.close(fig)

    (ASSETS / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
