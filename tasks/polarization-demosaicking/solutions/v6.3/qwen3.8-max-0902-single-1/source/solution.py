"""Polarization DoFP demosaicking: one mosaic frame -> four intensity images.

Sensor tile (rows/cols 0-indexed, period 2 in both directions):

        even col   odd col
    even row  90deg   45deg
    odd  row 135deg    0deg

Method (see README.md): work in the Stokes domain instead of interpolating each
angle on its own.  With g = S0/2, s1 = S1/2, s2 = S2/2 the four images are

    I0 = g + s1   I90 = g - s1   I45 = g + s2   I135 = g - s2

and every raw sample is  v(p) = g(p) + sgn(p) * s(p)  with sgn(p) = (-1)^(col+1)
and s = s1 when (row+col) is even, s = s2 when it is odd.  g carries all of the
scene structure while s1/s2 are small (degree of polarisation is only a few
percent) and smooth, so only s1 and s2 have to be interpolated; g is then
recovered exactly at the sampled pixel from the raw reading itself.

Only NumPy is used.  Everything is float32, separable and out-of-place free so a
1024x1024 frame finishes in ~80 ms of compute.
"""

import numpy as np

__all__ = ["reconstruct"]

# ---------------------------------------------------------------- utilities --


def _pad(a, p):
    """Symmetric padding that also works for axes shorter than the pad width."""
    if p <= 0:
        return a
    mode = "reflect" if min(a.shape) > p else "edge"
    return np.pad(a, p, mode=mode)


def _diag_sum(a, d):
    """Sum of the four diagonal neighbours at offset (+-d, +-d)."""
    ap = _pad(a, d)
    return (ap[0:-2 * d, 0:-2 * d] + ap[0:-2 * d, 2 * d:]
            + ap[2 * d:, 0:-2 * d] + ap[2 * d:, 2 * d:])


def _box(a):
    """One 3x3 box pass (normalised)."""
    ap = _pad(a, 1)
    s = (ap[0:-2, 0:-2] + ap[0:-2, 1:-1] + ap[0:-2, 2:]
         + ap[1:-1, 0:-2] + ap[1:-1, 1:-1] + ap[1:-1, 2:]
         + ap[2:, 0:-2] + ap[2:, 1:-1] + ap[2:, 2:])
    return s * (1.0 / 9.0)


def _axial_sum(a):
    """Sum of the four axis-aligned neighbours."""
    ap = _pad(a, 1)
    return ap[0:-2, 1:-1] + ap[2:, 1:-1] + ap[1:-1, 0:-2] + ap[1:-1, 2:]


# ------------------------------------------------------------------- core ----

# Number of 3x3 box passes applied to the interpolated Stokes components.
_NSMOOTH = 3


def _parity(H, W):
    """sgn = (-1)^(col+1) as (1,W); q0 = True where s1 is the observed term."""
    sgn = np.empty((1, W), np.float32)
    sgn[0, 0::2] = -1.0
    sgn[0, 1::2] = 1.0
    q0 = np.zeros((H, W), np.bool_)
    q0[0::2, 0::2] = True
    q0[1::2, 1::2] = True
    return sgn, q0


def _fill_checker(x, valid, den):
    """Keep samples where `valid`, fill the rest from the 4 axial neighbours."""
    return np.where(valid, x, _axial_sum(np.where(valid, x, 0.0)) / den)


def reconstruct(raw):
    """Estimate the 0/45/90/135 degree images from one DoFP exposure.

    Parameters
    ----------
    raw : read-only 2-D uint8 array (H, W), readings in 0..255.

    Returns
    -------
    (4, H, W) float32 array in the order 0deg, 45deg, 90deg, 135deg.
    """
    v = np.asarray(raw, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError("raw must be 2-D")
    H, W = v.shape

    sgn, q0 = _parity(H, W)

    # --- Stokes samples -----------------------------------------------------
    # v(p) - diag_mean(v)(p) isolates 2*sgn(p)*s(p) up to a residual that is
    # proportional to the curvature of g.  The kernel below is chosen so that
    # constant, linear *and* quadratic profiles of g cancel exactly, which
    # leaves only third-order terms of g as interference:
    #     s_hat = sgn * (3/8 v - 1/8 * diag1_sum(v) + 1/32 * diag2_sum(v))
    s_hat = sgn * (v * 0.375 - _diag_sum(v, 1) * 0.125
                   + _diag_sum(v, 2) * (1.0 / 32.0))

    # --- separate + smooth s1 and s2 ---------------------------------------
    # s1 is sampled on the (row+col) even checkerboard, s2 on the odd one; the
    # axial neighbours of a hole always belong to the right sub-lattice.
    mq = q0.astype(np.float32)
    den1 = np.maximum(_axial_sum(mq), 1e-6)
    den2 = np.maximum(_axial_sum(1.0 - mq), 1e-6)

    s1 = _fill_checker(s_hat, q0, den1)
    s2 = _fill_checker(s_hat, ~q0, den2)
    for _ in range(_NSMOOTH):
        s1 = _box(s1)
        s2 = _box(s2)

    # --- recombine ----------------------------------------------------------
    # g keeps the raw reading's full sharpness: at the sampled pixel the
    # corresponding output reproduces v exactly.
    st = np.where(q0, s1, s2)
    g = v - sgn * st

    out = np.empty((4, H, W), np.float32)
    np.add(g, s1, out=out[0])        # 0 deg
    np.add(g, s2, out=out[1])        # 45 deg
    np.subtract(g, s1, out=out[2])   # 90 deg
    np.subtract(g, s2, out=out[3])   # 135 deg

    np.clip(out, 0.0, 255.0, out=out)
    # Guarantees the finiteness contract even for pathological input.
    return np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0,
                         copy=False).astype(np.float32, copy=False)
