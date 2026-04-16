"""Torch Dataset wrapping the synthetic heading clips produced by gen_heading_dataset.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HeadingClipDataset(Dataset):
    """Yields (spikes, label) with spikes shape (T, 2, H, W) as float32."""

    def __init__(self, split_dir: Path) -> None:
        self.paths = sorted(Path(split_dir).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz clips found under {split_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        z = np.load(self.paths[idx])
        spikes = torch.from_numpy(z["spikes"].astype(np.float32))
        label = int(z["label"])
        return spikes, label
