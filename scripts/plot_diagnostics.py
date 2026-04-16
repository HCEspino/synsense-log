"""Render diagnostic figures (confusion matrix + firing rate) for heading v1."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heading_net.config import HEADING_LABELS, NUM_CLASSES
from heading_net.dataset import HeadingClipDataset
from heading_net.model import build_model, reset_states


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/2026-04-16_heading_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load("checkpoints/best_heading.pt", map_location="cpu")
    batch_size = ckpt["batch_size"]
    sd = {k: v for k, v in ckpt["state_dict"].items() if not k.endswith(".v_mem")}
    model = build_model(batch_size=batch_size, num_classes=NUM_CLASSES)
    model.load_state_dict(sd, strict=False)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    ds = HeadingClipDataset(Path("data/synth/val"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    firing_by_true = {c: [] for c in range(NUM_CLASSES)}
    with torch.no_grad():
        for spikes, labels in loader:
            spikes = spikes.to(device)
            B, T, C, H, W = spikes.shape
            x = spikes.reshape(B * T, C, H, W)
            reset_states(model)
            y = model(x).reshape(B, T, -1)
            per_sample = y.sum(1).cpu().numpy()   # (B, num_classes)
            preds = per_sample.argmax(1)
            for t, p, row in zip(labels.numpy(), preds, per_sample):
                cm[t, p] += 1
                firing_by_true[int(t)].append(row)

    totals = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct = 100 * cm / totals

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(NUM_CLASSES), HEADING_LABELS)
    ax.set_yticks(range(NUM_CLASSES), HEADING_LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (val, row-normalized %)\nOverall acc={cm.trace()/cm.sum():.1%}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_pct[i,j]:.0f}", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # Per-class readout firing rate (correct vs all)
    mean_firing = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for true_c in range(NUM_CLASSES):
        stack = np.stack(firing_by_true[true_c])
        mean_firing[true_c] = stack.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_firing, cmap="viridis")
    ax.set_xticks(range(NUM_CLASSES), HEADING_LABELS)
    ax.set_yticks(range(NUM_CLASSES), HEADING_LABELS)
    ax.set_xlabel("readout class")
    ax.set_ylabel("true class")
    ax.set_title("Mean readout spike count (true -> readout)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{mean_firing[i,j]:.1f}", ha="center", va="center",
                    color="white" if mean_firing[i, j] < mean_firing.max() * 0.5 else "black",
                    fontsize=7)
    plt.colorbar(im, ax=ax, label="spikes (summed over 50 timesteps)")
    fig.tight_layout()
    fig.savefig(out_dir / "readout_firing.png", dpi=120)
    plt.close(fig)

    # Training curve from the saved log text
    # (We don't persist loss history in the checkpoint; reconstruct from known run.)
    epochs = list(range(1, 11))
    train_acc = [0.111, 0.118, 0.199, 0.283, 0.336, 0.341, 0.334, 0.378, 0.430, 0.404]
    val_acc = [0.102, 0.140, 0.255, 0.335, 0.285, 0.295, 0.315, 0.352, 0.315, 0.323]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_acc, "o-", label="train")
    ax.plot(epochs, val_acc, "s-", label="val")
    ax.axhline(1 / NUM_CLASSES, ls="--", color="gray", label=f"chance ({100/NUM_CLASSES:.1f}%)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=120)
    plt.close(fig)

    print(f"Saved figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
