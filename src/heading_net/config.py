"""Shared constants for the 8-way heading classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass


NUM_CLASSES = 8

HEADING_LABELS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

HEADING_VECTORS = tuple(
    (math.cos(math.radians(90 - 45 * i)), -math.sin(math.radians(90 - 45 * i)))
    for i in range(NUM_CLASSES)
)


@dataclass(frozen=True)
class DataConfig:
    height: int = 128
    width: int = 128
    num_timesteps: int = 50         # 50 * 5ms = 250ms inference window
    pixels_per_step_min: float = 0.8
    pixels_per_step_max: float = 2.0
    diff_threshold: float = 0.05    # normalized intensity change to emit a spike
    num_shapes_min: int = 4
    num_shapes_max: int = 12
    noise_std: float = 0.05
