"""Live DVS viewer for the Synsense Speck2f devkit.

Streams raw events from the on-board 128x128 DVS sensor and renders them as
a decaying intensity map. Close the matplotlib window (or Ctrl-C) to exit.

Run with sudo on macOS (libusb needs to detach the kernel driver):
    sudo .venv/bin/python scripts/view_dvs_stream.py
"""

from __future__ import annotations

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import samna
from samna.speck2f.configuration import SpeckConfiguration


DVS_H, DVS_W = 128, 128
FRAME_DT = 0.05   # seconds between frame updates (~20 FPS)
DECAY = 0.6       # per-frame intensity decay so old events fade out
VMAX = 8.0        # saturation for the colormap


def find_speck2f():
    for dev in samna.device.get_unopened_devices():
        if "speck2f" in repr(dev).lower():
            return dev
    return None


def build_config() -> SpeckConfiguration:
    cfg = SpeckConfiguration()
    cfg.dvs_layer.monitor_enable = True
    return cfg


def accumulate(frame: np.ndarray, events) -> int:
    frame *= DECAY
    n = 0
    for e in events:
        x = getattr(e, "x", None)
        y = getattr(e, "y", None)
        if x is None or y is None:
            continue
        if 0 <= x < DVS_W and 0 <= y < DVS_H:
            frame[y, x] += 1.0
            n += 1
    return n


def main() -> int:
    target = find_speck2f()
    if target is None:
        print("No Speck2fDevKit detected. Plug in the board and retry.")
        return 1

    dk = samna.device.open_device(target)
    model = dk.get_model()
    model.apply_configuration(build_config())

    source = dk.get_model_source_node()
    sink = samna.BasicSinkNode_speck2f_event_output_event()
    graph = samna.graph.EventFilterGraph()
    graph.sequential([source, sink])
    graph.start()

    frame = np.zeros((DVS_H, DVS_W), dtype=np.float32)

    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(frame, cmap="gray", vmin=0.0, vmax=VMAX, interpolation="nearest")
    title = ax.set_title("Speck2f DVS stream  —  close window to exit")
    ax.set_axis_off()
    fig.tight_layout()

    print("Streaming. Wave your hand in front of the sensor. Close the window to stop.")
    last_rate_t = time.time()
    rate_count = 0
    try:
        while plt.fignum_exists(fig.number):
            events = sink.get_events()
            if events:
                rate_count += accumulate(frame, events)
            img.set_data(frame)

            now = time.time()
            if now - last_rate_t >= 1.0:
                rate = rate_count / (now - last_rate_t)
                title.set_text(f"Speck2f DVS  —  {rate:,.0f} ev/s")
                last_rate_t = now
                rate_count = 0

            fig.canvas.draw_idle()
            plt.pause(FRAME_DT)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        graph.stop()
        samna.device.close_device(dk)
        plt.ioff()

    return 0


if __name__ == "__main__":
    sys.exit(main())
