"""Render GIFs of trained agents (deterministic policy) from ``runs/ppo/<env>/default/seed<k>/model.pt``.

    MUJOCO_GL=egl python scripts/make_gifs.py --seed 1

Writes one GIF per environment plus a side-by-side ``hero.gif`` to ``assets/``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ppo.envs import space_info  # noqa: E402
from ppo.trainer import load_agent  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: env id -> (frame stride, max frames, output size (w, h), title)
LAYOUT = {
    "LunarLander-v3": (2, 160, (270, 180), "LunarLander-v3"),
    "HalfCheetah-v5": (1, 160, (270, 180), "HalfCheetah-v5"),
    "Hopper-v5": (6, 160, (270, 180), "Hopper-v5"),
}


def rollout_frames(env_id: str, ckpt: Path, stride: int, max_frames: int, seed: int) -> tuple[list[np.ndarray], float]:
    """Run one deterministic episode and return the (strided) RGB frames and the return."""
    agent, obs_norm, _ = load_agent(ckpt)
    if "v5" in env_id:  # MuJoCo: keep the camera locked on the torso so the agent stays in view
        cam = {"HalfCheetah-v5": {"distance": 4.5, "lookat": np.array([0.0, 0.0, 0.2]), "elevation": -10.0,
                                  "azimuth": 90.0},
               "Hopper-v5": {"distance": 4.0, "lookat": np.array([0.0, 0.0, 0.9]), "elevation": -10.0,
                             "azimuth": 90.0}}[env_id]
        env = gym.make(env_id, render_mode="rgb_array", width=480, height=320, default_camera_config=cam)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
    _, _, continuous, low, high = space_info(env)
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total, done, t = 0.0, False, 0
    while not done:  # run the episode to its end so the reported return is the full-episode return
        if t % stride == 0 and len(frames) < max_frames:
            viewer = getattr(env.unwrapped, "mujoco_renderer", None) and env.unwrapped.mujoco_renderer.viewer
            if viewer is not None:  # follow the agent along x (free camera, so move its look-at point)
                viewer.cam.lookat[0] = env.unwrapped.data.qpos[0]
            frames.append(env.render())
        x = obs_norm(obs[None], update=False) if obs_norm else obs[None].astype(np.float32)
        action, _, _ = agent.act(torch.from_numpy(x), deterministic=True)
        a = action[0].numpy()
        a = np.clip(a, low, high) if continuous else int(a)
        obs, r, term, trunc, _ = env.step(a)
        total += float(r)
        done = term or trunc
        t += 1
    env.close()
    return frames, total


def fit(frame: np.ndarray, size: tuple[int, int], title: str, ret: float) -> np.ndarray:
    """Resize to ``size`` (letterboxed) and stamp a caption."""
    img = Image.fromarray(frame)
    img.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (252, 252, 251))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    text = f"{title}  |  return {ret:.0f}"
    box = draw.textbbox((6, 4), text, font=font)
    draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 3), fill=(252, 252, 251))
    draw.text((6, 4), text, fill=(11, 11, 11), font=font)
    return np.asarray(canvas)


def save_gif(frames: list[np.ndarray], path: Path, fps: int, colors: int = 64) -> None:
    """Write a looping GIF with one shared palette (smaller than per-frame palettes)."""
    imgs = [Image.fromarray(f) for f in frames]
    pal = imgs[len(imgs) // 2].quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1, help="training seed whose checkpoint is rendered")
    p.add_argument("--eval-seed", type=int, default=0, help="environment seed of the rendered episode")
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()
    torch.set_num_threads(1)

    panels: dict[str, list[np.ndarray]] = {}
    for env_id, (stride, max_frames, size, title) in LAYOUT.items():
        ckpt = ROOT / "runs" / "ppo" / env_id / "default" / f"seed{args.seed}" / "model.pt"
        frames, ret = rollout_frames(env_id, ckpt, stride, max_frames, args.eval_seed)
        print(f"{env_id}: {len(frames)} frames, return {ret:.1f}")
        panels[env_id] = [fit(f, size, title, ret) for f in frames]
        out = ROOT / "assets" / f"{env_id.split('-')[0].lower()}.gif"
        save_gif(panels[env_id], out, args.fps)
        print(f"  wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    n = max(len(v) for v in panels.values())
    hero = []
    for i in range(n):  # shorter clips freeze on their last frame
        hero.append(np.concatenate([v[min(i, len(v) - 1)] for v in panels.values()], axis=1))
    out = ROOT / "assets" / "hero.gif"
    save_gif(hero, out, args.fps)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
