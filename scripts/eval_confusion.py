"""Compute per-class confusion matrix + output-layer firing rate on val set."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heading_net.config import HEADING_LABELS, NUM_CLASSES
from heading_net.dataset import HeadingClipDataset
from heading_net.model import build_model, reset_states


def main() -> int:
    ckpt = torch.load("checkpoints/best_heading.pt", map_location="cpu")
    batch_size = ckpt["batch_size"]
    model = build_model(batch_size=batch_size, num_classes=NUM_CLASSES)
    sd = {k: v for k, v in ckpt["state_dict"].items() if not k.endswith(".v_mem")}
    model.load_state_dict(sd, strict=False)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    ds = HeadingClipDataset(Path("data/synth/val"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    firing = []
    with torch.no_grad():
        for spikes, labels in loader:
            spikes = spikes.to(device)
            B, T, C, H, W = spikes.shape
            x = spikes.reshape(B * T, C, H, W)
            reset_states(model)
            y = model(x).reshape(B, T, -1)
            firing.append(y.sum(1).cpu().numpy())   # spikes per class per sample
            preds = y.sum(1).argmax(1).cpu().numpy()
            for t, p in zip(labels.numpy(), preds):
                cm[t, p] += 1

    firing = np.concatenate(firing, axis=0)  # (N, C)
    totals = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct = 100 * cm / totals

    print("Confusion matrix (row=true, col=pred, % of row):")
    print("      " + "  ".join(f"{c:>5}" for c in HEADING_LABELS))
    for i, row in enumerate(cm_pct):
        print(f"{HEADING_LABELS[i]:<5} " + "  ".join(f"{v:5.1f}" for v in row))

    print()
    print("Per-class readout firing rate (spikes summed over T, averaged over N):")
    for c in range(NUM_CLASSES):
        mean_when_true = firing[:, c].mean()
        print(f"  {HEADING_LABELS[c]}: mean={mean_when_true:.2f}")

    print(f"\nOverall acc: {cm.trace() / cm.sum():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
