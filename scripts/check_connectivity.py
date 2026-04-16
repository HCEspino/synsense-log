"""Quick connectivity check for the Synsense Speck2f devkit.

Lists unopened USB devices visible to samna, attempts to open the first
Speck2fDevKit it finds, and prints basic identifiers. Exits non-zero if no
Speck2f devkit is detected so the script is CI-friendly.
"""

from __future__ import annotations

import sys

import samna


TARGET_KEYWORD = "speck2f"


def list_devices() -> list:
    devices = samna.device.get_unopened_devices()
    print(f"samna sees {len(devices)} unopened device(s):")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev}")
    return devices


def find_speck2f(devices: list):
    for dev in devices:
        if TARGET_KEYWORD in repr(dev).lower():
            return dev
    return None


def main() -> int:
    devices = list_devices()
    target = find_speck2f(devices)

    if target is None:
        print(f"\nNo Speck2f devkit detected. Plug in the board via USB and retry.")
        return 1

    print(f"\nOpening {target} ...")
    dk = samna.device.open_device(target)
    print(f"Opened: {type(dk).__name__}")

    model = dk.get_model()
    print(f"Model: {type(model).__name__}")
    print(f"Configuration type: {type(model.get_configuration()).__name__}")

    samna.device.close_device(dk)
    print("Closed device cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
