"""Generate a synthetic DVS dataset of 8-way heading clips.

Each clip is a (T, 2, H, W) binary tensor: channel 0 = ON spikes (brightness up),
channel 1 = OFF spikes (brightness down). Labels are integers 0..7 for the
cardinal + diagonal compass headings in ``heading_net.config.HEADING_LABELS``.

Scenes are procedural random shapes + noise translated along the heading vector
over ``num_timesteps`` frames. Consecutive-frame differences are thresholded to
produce polarity spikes. Crude versus a real DVS simulator, but preserves the
direction-of-motion signal a heading classifier actually needs.

Usage:
    uv run python scripts/gen_heading_dataset.py \
        --train 2000 --val 400 --out data/synth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `src/` importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heading_net.config import DataConfig, HEADING_VECTORS, NUM_CLASSES


def _random_scene(rng: np.random.Generator, cfg: DataConfig) -> np.ndarray:
    """Draw random filled rectangles/ellipses on a noise background."""
    h, w = cfg.height, cfg.width
    # Oversize canvas so shapes can translate off-screen without introducing
    # a wrap-around edge inside the field of view
    pad = max(h, w) // 2
    canvas = rng.normal(0.5, cfg.noise_std, size=(h + 2 * pad, w + 2 * pad)).astype(np.float32)

    n_shapes = rng.integers(cfg.num_shapes_min, cfg.num_shapes_max + 1)
    for _ in range(n_shapes):
        kind = rng.integers(0, 2)
        cy = rng.integers(0, canvas.shape[0])
        cx = rng.integers(0, canvas.shape[1])
        size_y = rng.integers(8, 40)
        size_x = rng.integers(8, 40)
        intensity = rng.uniform(0.0, 1.0)

        y0, y1 = max(0, cy - size_y), min(canvas.shape[0], cy + size_y)
        x0, x1 = max(0, cx - size_x), min(canvas.shape[1], cx + size_x)
        if kind == 0:
            canvas[y0:y1, x0:x1] = intensity
        else:
            ys = np.arange(y0, y1) - cy
            xs = np.arange(x0, x1) - cx
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            mask = (yy / max(size_y, 1)) ** 2 + (xx / max(size_x, 1)) ** 2 <= 1.0
            canvas[y0:y1, x0:x1] = np.where(mask, intensity, canvas[y0:y1, x0:x1])

    return np.clip(canvas, 0.0, 1.0), pad


def _translate_view(scene: np.ndarray, pad: int, cy: float, cx: float,
                    height: int, width: int) -> np.ndarray:
    """Grab a HxW window from the padded scene at (cy, cx) using nearest-neighbor."""
    y0 = int(round(pad + cy))
    x0 = int(round(pad + cx))
    y0 = np.clip(y0, 0, scene.shape[0] - height)
    x0 = np.clip(x0, 0, scene.shape[1] - width)
    return scene[y0:y0 + height, x0:x0 + width]


def _make_clip(rng: np.random.Generator, cfg: DataConfig, label: int) -> np.ndarray:
    scene, pad = _random_scene(rng, cfg)
    dy, dx = HEADING_VECTORS[label]
    speed = rng.uniform(cfg.pixels_per_step_min, cfg.pixels_per_step_max)

    start_cy = rng.uniform(-pad / 4, pad / 4)
    start_cx = rng.uniform(-pad / 4, pad / 4)

    frames = np.empty((cfg.num_timesteps + 1, cfg.height, cfg.width), dtype=np.float32)
    for t in range(cfg.num_timesteps + 1):
        cy = start_cy + dy * speed * t
        cx = start_cx + dx * speed * t
        frames[t] = _translate_view(scene, pad, cy, cx, cfg.height, cfg.width)

    # Per-frame brightness delta → polarity spikes
    deltas = np.diff(frames, axis=0)
    spikes = np.zeros((cfg.num_timesteps, 2, cfg.height, cfg.width), dtype=np.uint8)
    spikes[:, 0] = (deltas > cfg.diff_threshold).astype(np.uint8)
    spikes[:, 1] = (deltas < -cfg.diff_threshold).astype(np.uint8)
    return spikes


def generate(split_dir: Path, n: int, cfg: DataConfig, seed: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, NUM_CLASSES, size=n)
    for i, label in enumerate(labels):
        clip = _make_clip(rng, cfg, int(label))
        np.savez_compressed(split_dir / f"{i:06d}.npz", spikes=clip, label=int(label))
        if (i + 1) % 200 == 0:
            print(f"  {split_dir.name}: {i + 1}/{n}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("data/synth"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = DataConfig()
    print(f"Config: {cfg}")
    print(f"Generating {args.train} train / {args.val} val clips into {args.out}")

    generate(args.out / "train", args.train, cfg, seed=args.seed)
    generate(args.out / "val", args.val, cfg, seed=args.seed + 1)

    # Preview one clip per class as a sanity PNG
    preview_dir = args.out / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 999)
    for label in range(NUM_CLASSES):
        clip = _make_clip(rng, cfg, label)
        accum = clip.sum(axis=0)  # (2, H, W)
        img = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        img[..., 0] = np.clip(accum[0] * 32, 0, 255)  # ON → red
        img[..., 2] = np.clip(accum[1] * 32, 0, 255)  # OFF → blue
        from PIL import Image
        Image.fromarray(img).save(preview_dir / f"class{label}.png")
    print(f"Wrote per-class preview PNGs to {preview_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
