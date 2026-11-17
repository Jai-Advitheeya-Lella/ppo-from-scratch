"""Run the full experiment grid (main results, ablations, SB3 baseline) with resumption.

Each run writes to ``runs/<impl>/<env>/<variant>/seed<k>/``; runs whose ``summary.json``
already exists are skipped, so the script can be interrupted and restarted.

    python scripts/run_experiments.py --groups main sb3 ablation interaction --workers 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1, 2, 3, 4, 5]

#: Ablations remove one implementation detail at a time from the full configuration.
ABLATIONS: dict[str, list[str]] = {
    "no_norm_obs": ["--norm-obs", "false"],
    "no_norm_reward": ["--norm-reward", "false"],
    "no_norm_adv": ["--norm-adv", "false"],
    "no_clip_vloss": ["--clip-vloss", "false"],
    "no_ortho_init": ["--orthogonal-init", "false"],
    "no_bootstrap_truncated": ["--bootstrap-truncated", "false"],
}
#: Extra LunarLander run testing whether value-loss clipping is what hurts when reward scaling is off.
INTERACTION = {"no_norm_reward_no_clip_vloss": ["--norm-reward", "false", "--clip-vloss", "false"]}
MAIN_ENVS = ["HalfCheetah-v5", "LunarLander-v3", "CartPole-v1", "Hopper-v5"]
ABLATION_ENVS = ["LunarLander-v3", "HalfCheetah-v5"]


def jobs_for(group: str) -> list[tuple[Path, list[str]]]:
    py = sys.executable
    out: list[tuple[Path, list[str]]] = []
    if group == "main":
        for env in MAIN_ENVS:
            for s in SEEDS:
                d = ROOT / "runs" / "ppo" / env / "default" / f"seed{s}"
                out.append((d, [py, "-m", "ppo.train", "--env-id", env, "--seed", str(s), "--run-dir", str(d)]))
    elif group == "sb3":
        for env in MAIN_ENVS:
            for s in SEEDS:
                d = ROOT / "runs" / "sb3" / env / "default" / f"seed{s}"
                out.append((d, [py, str(ROOT / "scripts" / "sb3_baseline.py"), "--env-id", env,
                                "--seed", str(s), "--run-dir", str(d)]))
    elif group == "ablation":
        for env in ABLATION_ENVS:
            for name, flags in ABLATIONS.items():
                for s in SEEDS:
                    d = ROOT / "runs" / "ppo" / env / name / f"seed{s}"
                    out.append((d, [py, "-m", "ppo.train", "--env-id", env, "--seed", str(s),
                                    "--run-dir", str(d), *flags]))
    elif group == "interaction":
        for name, flags in INTERACTION.items():
            for s in SEEDS:
                d = ROOT / "runs" / "ppo" / "LunarLander-v3" / name / f"seed{s}"
                out.append((d, [py, "-m", "ppo.train", "--env-id", "LunarLander-v3", "--seed", str(s),
                                "--run-dir", str(d), *flags]))
    else:
        raise ValueError(group)
    return out


def run(job: tuple[Path, list[str]]) -> str:
    run_dir, cmd = job
    if (run_dir / "summary.json").exists():
        return f"skip {run_dir.relative_to(ROOT)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "egl"}
    with open(run_dir / "train.log", "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
    msg = f"{status} {run_dir.relative_to(ROOT)}"
    print(msg, flush=True)
    return msg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups", nargs="+", default=["main", "sb3", "ablation"])
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    jobs = [j for g in args.groups for j in jobs_for(g)]
    print(f"{len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for _ in pool.map(run, jobs):
            pass


if __name__ == "__main__":
    main()
