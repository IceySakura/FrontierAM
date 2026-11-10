"""Full-frame reconstruction for the simulated [[90, 45], [135, 0]] polarimeter."""
from pathlib import Path
import sys

import numpy as np

PROBE = Path(__file__).resolve().parents[1] / "probe"
sys.path.insert(0, str(PROBE))
from methods.liu import recon
from metrics import bilinear


PAPER_ORDER = [2, 1, 0, 3]
RIM = 5  # LEPD diagonal stage (2), axial stage (2), then LEIC correction (1).
PAD = 6  # At least RIM and even to preserve the 2x2 mosaic phase.


def reconstruct(raw, method="bilinear-edge"):
    """Return (0°,45°,90°,135°) float planes; raw is in the sensor's 0..255 units."""
    if method not in ("bilinear-edge", "reflect", "original", "bilinear"):
        raise ValueError(f"unknown method: {method}")
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or min(raw.shape) < 2 or not np.isfinite(raw).all():
        raise ValueError("expected a finite 2-D mosaic with at least two rows and columns")
    if method == "bilinear" or (method == "bilinear-edge" and min(raw.shape) <= 2 * RIM):
        return bilinear(raw)[PAPER_ORDER]
    span = float(np.ptp(raw))
    gain = 255 / span if span else 1.0
    if method == "reflect":
        extended = np.pad(raw, PAD, mode="reflect")
        extended_planes = recon(extended * gain, "LEIC") / gain
        return extended_planes[:, PAD:-PAD, PAD:-PAD][PAPER_ORDER]

    planes = recon(raw * gain, "LEIC") / gain
    if method == "bilinear-edge":
        # Give each strip a neighbor beyond the seam. Start at even coordinates
        # so the local origin has the same filter phase, including for odd sizes.
        h, w = raw.shape
        bottom = (h - RIM - 1) // 2 * 2
        right = (w - RIM - 1) // 2 * 2
        planes[:, :RIM, :] = bilinear(raw[:RIM + 1, :])[:, :RIM, :]
        planes[:, -RIM:, :] = bilinear(raw[bottom:, :])[:, -RIM:, :]
        planes[:, :, :RIM] = bilinear(raw[:, :RIM + 1])[:, :, :RIM]
        planes[:, :, -RIM:] = bilinear(raw[:, right:])[:, :, -RIM:]
    return planes[PAPER_ORDER]
