"""Command-line entry point: ``python -m ppo.train --env-id LunarLander-v3 --seed 1``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppo.config import add_config_args, make_config
from ppo.trainer import PPOTrainer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Output directory (default: runs/ppo/<env>/default/seed<k>)")
    args = vars(parser.parse_args(argv))
    run_dir = args.pop("run_dir")
    env_id = args.pop("env_id")
    overrides = {k: (tuple(v) if k == "hidden_sizes" and v else v) for k, v in args.items()}
    cfg = make_config(env_id, **overrides)
    if run_dir is None:
        run_dir = Path("runs") / "ppo" / cfg.env_id / "default" / f"seed{cfg.seed}"
    print(json.dumps(cfg.to_dict()), flush=True)
    summary = PPOTrainer(cfg, run_dir).train()
    print(f"eval return: {summary['eval_return_mean']:.1f} +/- {summary['eval_return_std']:.1f} "
          f"({summary['train_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
