"""Train the heading-classifier SNN on the synthetic dataset.

Usage:
    uv run python scripts/train_heading.py --data data/synth --epochs 15 --out checkpoints
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heading_net.config import NUM_CLASSES, HEADING_LABELS
from heading_net.dataset import HeadingClipDataset
from heading_net.model import build_model, reset_states


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def flatten_bt(spikes: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """(B, T, C, H, W) -> (B*T, C, H, W), return batch_size and T."""
    B, T, C, H, W = spikes.shape
    return spikes.reshape(B * T, C, H, W), B, T


def forward_classify(model: nn.Sequential, spikes: torch.Tensor) -> torch.Tensor:
    """Run the SNN, integrate output spikes over time -> (B, num_classes) logits."""
    x, B, T = flatten_bt(spikes)
    reset_states(model)
    y = model(x)                     # (B*T, num_classes)
    y = y.reshape(B, T, -1).sum(1)   # (B, num_classes)
    return y


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    loss_fn = nn.CrossEntropyLoss()
    for spikes, labels in loader:
        spikes = spikes.to(device)
        labels = labels.to(device)
        logits = forward_classify(model, spikes)
        loss = loss_fn(logits, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * spikes.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += spikes.size(0)
    return total_loss / total_n, total_correct / total_n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/synth"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default=None, help="cpu / mps / cuda (auto if unset)")
    parser.add_argument("--quick", action="store_true", help="1 epoch + 32-sample subset (smoke test)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Device: {device}")

    train_ds = HeadingClipDataset(args.data / "train")
    val_ds = HeadingClipDataset(args.data / "val")

    if args.quick:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(32, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(16, len(val_ds)))))
        args.epochs = 1

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=True)
    print(f"Train clips: {len(train_ds)}   Val clips: {len(val_ds)}")

    model = build_model(batch_size=args.batch, num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    best_val = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, device, None)
        dt = time.time() - t0
        print(
            f"epoch {epoch+1:2d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}  "
            f"({dt:.1f}s)"
        )
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "state_dict": model.state_dict(),
                "batch_size": args.batch,
                "num_classes": NUM_CLASSES,
                "val_acc": val_acc,
                "labels": list(HEADING_LABELS),
            }
            torch.save(ckpt, args.out / "best_heading.pt")
            print(f"  saved checkpoint (val_acc={val_acc:.3f})")

    print(f"Best val acc: {best_val:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
