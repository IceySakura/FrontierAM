"""Public synthetic examples and interface check.

Run: python public-dev/check.py path/to/solution.py
Import examples() from this file to obtain (raw, reference) pairs.
Reference planes and printed errors follow the order 0°, 45°, 90°, 135°.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

import numpy as np


def examples():
    """Yield two deterministic, independently generated uint8 image pairs."""
    rng = np.random.default_rng(173)
    for height, width in [(48, 64), (1024, 1024)]:
        y, x = np.ogrid[:height, :width]
        edge = x + y // 2 > width // 2
        intensity = (100 + 52 * edge + 11 * np.cos(y / 7) + 8 * np.sin(x / 3)
                     + rng.integers(-2, 3, size=(height, width)))
        q = 17 * np.cos(x / 5) + 6 * edge
        u = 11 * np.sin(y / 9) - 8 * edge
        planes = np.stack((intensity + q, intensity + u, intensity - q, intensity - u))
        reference = np.rint(planes).astype(np.uint8)
        raw = np.empty((height, width), dtype=np.uint8)
        # Repeating 2×2 sensor tile: [90°, 45°] / [135°, 0°].
        for row, col, channel in [(0, 0, 2), (0, 1, 1), (1, 0, 3), (1, 1, 0)]:
            raw[row::2, col::2] = reference[channel, row::2, col::2]
        raw.flags.writeable = False
        yield raw, reference


def channel_rmse(output, reference):
    """Check the return contract, then score all pixels in each channel."""
    if not isinstance(output, np.ndarray) or output.shape != reference.shape:
        raise ValueError(f"expected an ndarray of shape {reference.shape}")
    if output.dtype.kind != "f" or output.dtype.itemsize not in (4, 8):
        raise ValueError("expected float32 or float64 output")
    if not np.isfinite(output).all():
        raise ValueError("output contains non-finite values")

    difference = np.subtract(output, reference, dtype=np.float64)
    # Scale before squaring so even very large finite outputs can be scored.
    scale = np.max(np.abs(difference), axis=(1, 2))
    difference /= np.where(scale > 0, scale, 1)[:, None, None]
    return scale * np.sqrt(np.mean(np.square(difference), axis=(1, 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    args = parser.parse_args()
    path = args.solution.resolve(strict=True)
    os.environ["SEED"] = "1"
    sys.path.insert(0, str(path.parent))
    reconstruct = runpy.run_path(str(path))["reconstruct"]

    scene_errors = []
    for raw, reference in examples():
        errors = channel_rmse(reconstruct(raw), reference)
        scene_error = (errors / 4).sum()
        scene_errors.append(scene_error)
        print(f"{raw.shape[0]}x{raw.shape[1]} RMSE (0,45,90,135):",
              *(f"{error:.6f}" for error in errors), f"mean={scene_error:.6f}")
    print(f"public mean RMSE: {(np.array(scene_errors) / len(scene_errors)).sum():.6f}")


if __name__ == "__main__":
    main()
