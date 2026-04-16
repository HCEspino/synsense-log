"""Speck2f-deployable heading-classifier SNN definition.

The architecture is chosen to satisfy sinabs-dynapcnn backend constraints:
- Conv2d layers with bias=False
- Stride 1 or 2, kernel 3x3 (plus final 1x1 if used)
- IAFSqueeze neurons (standard sinabs pattern for dynapcnn deployment)
- No BatchNorm, no Dropout in the backbone
"""

from __future__ import annotations

import torch
import torch.nn as nn
import sinabs.layers as sl


def build_model(batch_size: int, num_classes: int = 8) -> nn.Sequential:
    """Create the training/eval SNN. ``batch_size`` is frozen into IAFSqueeze."""
    return nn.Sequential(
        # 128 -> 64
        nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1, bias=False),
        sl.IAFSqueeze(batch_size=batch_size),
        # 64 -> 32
        nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
        sl.IAFSqueeze(batch_size=batch_size),
        # 32 -> 16
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
        sl.IAFSqueeze(batch_size=batch_size),
        # 16 -> 8
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        sl.IAFSqueeze(batch_size=batch_size),
        # 8 -> 4
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
        sl.IAFSqueeze(batch_size=batch_size),
        # Flatten + readout
        nn.Flatten(),
        nn.Linear(4 * 4 * 32, num_classes, bias=False),
        sl.IAFSqueeze(batch_size=batch_size),
    )


def reset_states(model: nn.Sequential) -> None:
    """Zero IAF membrane potentials between samples."""
    for m in model.modules():
        if isinstance(m, sl.IAFSqueeze):
            m.reset_states()
