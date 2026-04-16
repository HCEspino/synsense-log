"""Deploy the trained heading classifier to Speck2f and show live predictions.

Loads the best checkpoint, wraps it in DynapcnnNetwork, uploads to the chip with
DVS input enabled, and streams output spikes back to the host. A matplotlib
window shows the live DVS accumulator on the left and the per-class spike count
bar chart on the right.

Run with sudo on macOS:
    sudo .venv/bin/python scripts/deploy_and_run.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heading_net.config import HEADING_LABELS, HEADING_VECTORS, NUM_CLASSES
from heading_net.model import build_model

import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork


DVS_H, DVS_W = 128, 128
FRAME_DT = 0.05
DVS_DECAY = 0.6
CLASS_WINDOW_SEC = 0.5   # rolling window for class spike counts


def load_model(ckpt_path: Path) -> torch.nn.Sequential:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(batch_size=1, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: val_acc={ckpt.get('val_acc', 'n/a')}")
    return model


def event_class(e) -> int | None:
    """Extract the class index from a Speck2f output event. The readout is a
    Linear -> IAF projecting to NUM_CLASSES units, which after chip conversion
    typically surfaces as either the .feature or .x attribute."""
    for attr in ("feature", "channel", "x"):
        v = getattr(e, attr, None)
        if v is not None and 0 <= v < NUM_CLASSES:
            return int(v)
    return None


def accumulate_dvs(frame: np.ndarray, events) -> None:
    frame *= DVS_DECAY
    for e in events:
        x = getattr(e, "x", None)
        y = getattr(e, "y", None)
        if x is None or y is None:
            continue
        if 0 <= x < DVS_W and 0 <= y < DVS_H:
            frame[y, x] += 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/best_heading.pt"))
    args = parser.parse_args()

    if not args.ckpt.exists():
        print(f"Checkpoint not found: {args.ckpt}. Train first with scripts/train_heading.py")
        return 1

    model = load_model(args.ckpt)
    dcnn = DynapcnnNetwork(
        snn=model,
        input_shape=(2, DVS_H, DVS_W),
        dvs_input=True,
        discretize=True,
    )
    print("DynapcnnNetwork ready.")
    print(dcnn.memory_summary())

    # Upload to chip with DVS input and final-layer output monitoring
    dcnn.to(device="speck2fdevkit:0", monitor_layers=["dvs", -1])
    print("Uploaded to Speck2fDevKit. Streaming...")

    # Set up samna graph: device source -> two sinks (DVS raw + output layer)
    source = dcnn.samna_device.get_model_source_node()
    dvs_sink = samna.BasicSinkNode_speck2f_event_output_event()
    out_sink = samna.BasicSinkNode_speck2f_event_output_event()
    graph = samna.graph.EventFilterGraph()
    # Both sinks read the same stream; class events and DVS events are
    # differentiated by type downstream.
    graph.sequential([source, dvs_sink])
    graph.sequential([source, out_sink])
    graph.start()

    dvs_frame = np.zeros((DVS_H, DVS_W), dtype=np.float32)
    class_events = deque()  # (timestamp, class_idx)

    plt.ion()
    fig, (ax_dvs, ax_bar) = plt.subplots(1, 2, figsize=(10, 5))
    dvs_img = ax_dvs.imshow(dvs_frame, cmap="gray", vmin=0, vmax=8)
    ax_dvs.set_title("DVS stream")
    ax_dvs.set_axis_off()

    bars = ax_bar.bar(HEADING_LABELS, [0] * NUM_CLASSES, color="steelblue")
    ax_bar.set_ylim(0, 50)
    ax_bar.set_title("Class spike counts (0.5s window)")
    ax_bar.set_ylabel("spikes")
    prediction_text = ax_bar.text(0.5, 0.95, "", transform=ax_bar.transAxes,
                                  ha="center", va="top", fontsize=14, weight="bold")
    fig.tight_layout()

    print("Wave / rotate the board to trigger motion. Close the window to exit.")
    try:
        while plt.fignum_exists(fig.number):
            # DVS accumulation
            dvs_events = dvs_sink.get_events()
            if dvs_events:
                accumulate_dvs(dvs_frame, dvs_events)
            dvs_img.set_data(dvs_frame)

            # Output-layer class events
            now = time.time()
            for e in out_sink.get_events():
                cls = event_class(e)
                if cls is not None:
                    class_events.append((now, cls))

            # Drop events outside the rolling window
            cutoff = now - CLASS_WINDOW_SEC
            while class_events and class_events[0][0] < cutoff:
                class_events.popleft()

            counts = np.zeros(NUM_CLASSES, dtype=int)
            for _, cls in class_events:
                counts[cls] += 1
            for bar, c in zip(bars, counts):
                bar.set_height(c)
            ymax = max(10, int(counts.max() * 1.2) + 1)
            ax_bar.set_ylim(0, ymax)

            if counts.sum() >= 5:
                pred = int(counts.argmax())
                prediction_text.set_text(f"heading: {HEADING_LABELS[pred]}")
            else:
                prediction_text.set_text("(waiting for events)")

            fig.canvas.draw_idle()
            plt.pause(FRAME_DT)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        graph.stop()
        plt.ioff()

    return 0


if __name__ == "__main__":
    sys.exit(main())
