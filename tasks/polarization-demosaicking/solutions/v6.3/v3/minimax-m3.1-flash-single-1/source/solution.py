"""Polarization demosaicking: recover the 0/45/90/135 degree intensity planes
from a single polarizer mosaic.

The polarizer wheel repeats with a 2x2 period, so every pixel carries exactly
one of the four polarization measurements:

        even col   odd col
even row    90 deg    45 deg
odd row    135 deg     0 deg

Method
------
Write the plane that owns parity ``j`` as ``mu + d_j``, where ``mu`` is the
common (grayscale) intensity content of the scene and ``d_j`` the smooth
per-angle polarization deviation; then ``sum_j d_j = 0``.  The mosaic pixel
observed at a position of parity ``j`` reads ``mu + d_j``, hence

    plane_k = mosaic + d_k - d_j                          (*)

Two independent estimates of the left hand side are combined.

``S_k`` - spline interpolation of the samples that belong to plane ``k``.
         It carries the plane's own polarization deviation but only the
         lower half of the spatial spectrum, so the top octave of the scene
         texture is missing at the three quarters of the positions where the
         plane is not directly observed.

``M_k`` - a wide low-pass of the same lattice.  ``M_k`` tracks ``mu + d_k``,
         so ``M_k - M_j`` is a smooth estimate of the polarization deviation
         difference ``d_k - d_j`` that appears in (*).  Adding it to the
         mosaic value transfers the mosaic's full-resolution texture to every
         plane.

The two are blended with weight ``w``; ``w = 1`` reproduces the pure texture
transfer, ``w = 0`` the pure per-plane interpolation.  The reliability of the
transfered correction degrades when the scene texture contaminates the
diagonal combination ``(a + c - b - d) / 2``; the amount of that
contamination is measured on the mosaic itself and used to pick ``w``
adaptively, so that strongly polarized but heavily textured scenes fall back
towards plain interpolation while clean scenes make full use of the mosaic
texture.

Only the mosaic of the current frame and the constants in this file are used.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Lattice offsets (row, col) of the plane that owns each parity class:
#   0 deg  -> odd row,  odd col
#  45 deg  -> even row, odd col
#  90 deg  -> even row, even col
# 135 deg  -> odd row,  even col
_OFFSETS = ((1, 1), (0, 1), (0, 0), (1, 0))

_LP_SIGMA = 0.7     # width (half-resolution samples) of the lattice low-pass
_SPLINE_ORDER = 2   # local polynomial order of the per-plane interpolation
_W_A = 1.30         # adaptive blend: w = clip(_W_A - _W_B * contamination, ...)
_W_B = 0.50
_W_MIN = 0.05
_W_MAX = 0.62


def _shift_sample(f, dr, dc):
    """Bilinear sample of a half-resolution field at (i + dr, j + dc).

    ``dr`` and ``dc`` are half-resolution units; only -0.5, 0 and +0.5 occur
    here.  Coordinates outside the grid are clamped to the border.
    """
    if dr == 0.0 and dc == 0.0:
        return f
    a = f
    if dr != 0.0:
        b = np.empty_like(a)
        if dr > 0.0:
            b[:-1] = 0.5 * (a[:-1] + a[1:])
            b[-1] = a[-1]
        else:
            b[0] = a[0]
            b[1:] = 0.5 * (a[:-1] + a[1:])
        a = b
    if dc != 0.0:
        b = np.empty_like(a)
        if dc > 0.0:
            b[:, :-1] = 0.5 * (a[:, :-1] + a[:, 1:])
            b[:, -1] = a[:, -1]
        else:
            b[:, 0] = a[:, 0]
            b[:, 1:] = 0.5 * (a[:, :-1] + a[:, 1:])
        a = b
    return a


def _contamination(x):
    """Amount of scene texture leaking into the diagonal plane combination.

    ``(a + c - b - d) / 2`` cancels the smooth polarization part but not the
    texture, so the high-frequency residue of that combination measures how
    noisy the polarization estimate is going to be.
    """
    fa = x[1::2, 1::2]
    fb = x[0::2, 1::2]
    fc = x[0::2, 0::2]
    fd = x[1::2, 0::2]
    comb = 0.5 * (fa + fc - fb - fd)
    if comb.shape[0] < 2 or comb.shape[1] < 2:
        return 0.0
    resid = comb - gaussian_filter(comb, 1.0, mode="nearest")
    return float(np.sqrt(np.mean(resid.astype(np.float64) ** 2)))


def reconstruct(raw):
    """Reconstruct the four polarization planes from a mosaic frame.

    Parameters
    ----------
    raw : array_like, shape (H, W)
        Mosaic intensity frame, usually ``uint8``.

    Returns
    -------
    numpy.ndarray, shape (4, H, W), dtype float32
        The 0, 45, 90 and 135 degree intensity planes, aligned with the
        mosaic and in the same intensity units.
    """
    x = np.asarray(raw)
    if x.ndim != 2:
        raise ValueError("expected a 2-D mosaic")
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    H0, W0 = x.shape
    H, W = H0, W0
    if (H % 2) or (W % 2):
        # The 2x2 mosaic pattern needs even dimensions; mirror the border.
        ph = H % 2
        pw = W % 2
        x = np.pad(x, ((0, ph), (0, pw)), mode="edge")
        H, W = x.shape
    hh, ww = H // 2, W // 2

    # Per-plane half-resolution lattices.
    lat = (x[1::2, 1::2], x[0::2, 1::2], x[0::2, 0::2], x[1::2, 0::2])

    # Wide low-pass of every lattice: tracks the common intensity plus the
    # smooth polarization deviation of that plane.
    low = [gaussian_filter(f, _LP_SIGMA, mode="nearest") for f in lat]

    # How much the scene texture contaminates the polarization estimate.
    w = _W_A - _W_B * _contamination(x)
    w = float(min(max(w, _W_MIN), _W_MAX))

    ar = np.arange(hh, dtype=np.float32)
    ac = np.arange(ww, dtype=np.float32)

    out = np.empty((4, H, W), dtype=np.float32)
    for k in range(4):
        prk, pck = _OFFSETS[k]
        for j in range(4):
            prj, pcj = _OFFSETS[j]
            # Half-resolution offset between the two lattices.
            dr = 0.5 * (prj - prk)
            dc = 0.5 * (pcj - pck)
            if j == k:
                # The plane is observed here: the interpolation reproduces the
                # measured value exactly, no need to evaluate it.
                base = x[prj::2, pcj::2]
            else:
                coords = np.empty((2, hh, ww), dtype=np.float32)
                coords[0] = (ar + dr)[:, None]
                coords[1] = (ac + dc)[None, :]
                base = map_coordinates(lat[k], coords, order=_SPLINE_ORDER,
                                       mode="nearest")
            # Smooth estimate of d_k - d_j, evaluated at the same pixel.
            delta = _shift_sample(low[k], dr, dc) - low[j]
            out[k, prj::2, pcj::2] = base + w * (
                x[prj::2, pcj::2] + delta - base)
    if out.shape[1:] != (H0, W0):
        out = out[:, :H0, :W0]
    return out
