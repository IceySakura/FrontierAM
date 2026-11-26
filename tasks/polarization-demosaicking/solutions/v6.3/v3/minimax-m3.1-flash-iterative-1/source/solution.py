"""Polarisation mosaic demosaicing.

Input : single-channel uint8 mosaic of shape (H, W).
Output: float32 array of shape (4, H, W) with the intensity images at
        0 deg, 45 deg, 90 deg and 135 deg (same units as the input).

Method
------
The mosaic repeats every two rows / two columns:

        | even col | odd col |
        |---------|---------|
        | even row|  90 |  45 |
        | odd row | 135 |  0  |

so every 2x2 block carries one sample of each of the four polarisation
planes.  The four planes are four views of the same scene, each sampled
on one coset of the lattice 2Z x 2Z: the task is a super-resolution
problem with a fixed 2x2 sub-pixel phase pattern.

Two facts drive the design:

* The scenes are very smooth (more than 90 % of the power of a plane
  lies below 0.05 cycles/pixel), so a cubic (Catmull-Rom) interpolation
  of every plane on its own lattice already removes the low frequency
  part of the error.
* The residual error is the content *above* the lattice Nyquist limit
  (0.25 cycles/pixel).  It is small but it is exactly what is left, and
  it is partly shared between the four planes, so it can be recovered
  from the samples of the other planes - information a per-plane
  interpolation throws away.

The estimator therefore uses, for every (plane, sub-pixel phase) pair, a
small linear filter over a 9x9 window of the mosaic, so that the
samples of all four planes are used.  The filters are least-squares
fitted on the public development scenes and are shipped in
`filters.npy`; nothing scene specific is hard coded.  The local mean is
removed before filtering and added back afterwards, which makes the
filters insensitive to the absolute brightness of a scene.

The filters rely on the four planes sharing their high frequency
content.  A few scenes do not share it, and there the filters are worse
than a plain interpolation; the estimator therefore measures the
similarity of the four planes from the mosaic alone and mixes the
filtered result with the plain per-plane cubic interpolation when the
measurement says the filters cannot be trusted.
"""

import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# mosaic phase of every plane: (row parity, col parity) of its samples
_PHASE = ((1, 1), (0, 1), (0, 0), (1, 0))   # 0, 45, 90, 135 degrees

# weight of the learned filter (the remainder is the plain cubic
# interpolation of the plane on its own lattice).
GAMMA = 1.0

# If ADAPTIVE is set, the weight is chosen per scene from the measured
# similarity of the four planes: when they share little high frequency
# content the learned filters are unreliable and the cubic interpolation
# is used instead (GAMMA_LO .. GAMMA_HI is the transition range of the
# statistic).
ADAPTIVE = True
GAMMA_LO = 0.20
GAMMA_HI = 0.40

# The fitted filters are also embedded below so that this file works on
# its own, even if it is copied without `filters.npy`.
_FILTERS_BLOB = (
    "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"
)

try:
    _FILTERS = np.load(os.path.join(_HERE, "filters.npy")).astype(np.float32)
    _R = (_FILTERS.shape[-1] - 1) // 2
except Exception:                                    # pragma: no cover
    try:
        import base64
        import zlib
        _FILTERS = np.frombuffer(
            zlib.decompress(base64.b64decode(_FILTERS_BLOB)),
            dtype=np.float32).reshape(4, 4, 9, 9)
        _R = (_FILTERS.shape[-1] - 1) // 2
    except Exception:                                # pragma: no cover
        _FILTERS = None
        _R = 2


def _box_mean(a, size):
    """Separable box filter of width `size` with edge replication (float32)."""
    if size <= 1:
        return a
    k = size // 2
    h, w = a.shape
    p = np.pad(a, ((k, k), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=0, dtype=np.float32)
    c = np.concatenate((np.zeros((1, w), np.float32), c), axis=0)
    a = c[size:size + h] - c[0:h]
    p = np.pad(a, ((0, 0), (k, k)), mode="edge")
    c = np.cumsum(p, axis=1, dtype=np.float32)
    c = np.concatenate((np.zeros((h, 1), np.float32), c), axis=1)
    a = c[:, size:size + w] - c[:, 0:w]
    return a * np.float32(1.0 / (size * size))


def _catmull(t):
    a = -0.5
    t = np.abs(t)
    return np.where(t < 1.0, (a + 2.0) * t ** 3 - (a + 3.0) * t ** 2 + 1.0,
                    np.where(t < 2.0, a * t ** 3 - 5.0 * a * t ** 2 + 8.0 * a * t - 4.0 * a, 0.0))


def _cubic_lattice(a, oy, ox):
    """Catmull-Rom interpolation of a lattice (h, w) onto the full grid.

    Output pixel (r, c) receives the value of the lattice at the
    continuous position ((r - oy) / 2, (c - ox) / 2); the samples
    themselves (r == oy mod 2, c == ox mod 2) are reproduced exactly.
    """
    m = np.arange(-2, 3)
    w0 = _catmull(-m.astype(np.float64)).astype(np.float32)   # phase 0 (exact)
    w5 = _catmull(0.5 - m.astype(np.float64)).astype(np.float32)  # phase 1/2
    k = 2

    def interp_axis(arr, phase0_first):
        n = arr.shape[0]
        pad = np.pad(arr, ((k + 1, k), (0, 0)), mode="edge")
        f0 = np.zeros_like(arr)
        f5 = np.zeros_like(arr)
        off = k + 1 if phase0_first else k
        for i, mm in enumerate(range(-k, k + 1)):
            f0 += w0[i] * pad[k + 1 + mm:k + 1 + mm + n]
            f5 += w5[i] * pad[off + mm:off + mm + n]
        out = np.empty((2 * n, arr.shape[1]), dtype=arr.dtype)
        if phase0_first:
            out[0::2] = f0
            out[1::2] = f5
        else:
            out[0::2] = f5
            out[1::2] = f0
        return out

    a = np.ascontiguousarray(a, dtype=np.float32)
    y = interp_axis(a, oy == 0)          # rows
    z = interp_axis(y.T, ox == 0).T       # columns
    return z


def _fallback(raw):
    """Simple, always-available reconstruction (2x2 block average)."""
    h, w = raw.shape
    img = np.asarray(raw, dtype=np.float32)
    p = np.pad(img, ((0, h % 2), (0, w % 2)), mode="edge")
    blk = p.reshape((h + 1) // 2, 2, (w + 1) // 2, 2)
    mean = blk.mean(axis=(1, 3))
    out = np.repeat(np.repeat(mean, 2, axis=0), 2, axis=1)[:h, :w]
    res = np.empty((4, h, w), dtype=np.float32)
    for i in range(4):
        res[i] = out
    return res


def _cross_corr(planes):
    """How much the four planes share their high frequency content.

    The four planes are four polarisations of the same scene, so the part
    of them that lies above the lattice Nyquist limit is largely common.
    The learned filters recover that common part from the samples of the
    other three planes; when the common part is missing (some scenes have
    four almost unrelated planes) the filters are worse than a plain
    interpolation and such a scene has to be treated differently.

    The statistic is the mean pairwise correlation of the high frequency
    parts of the four planes.  It is measured on the cubic interpolations
    of the mosaic, so it does not depend on the filters and is available
    for every scene at run time.
    """
    hp = []
    for c in planes:
        d = c - _box_mean(c, 5)
        hp.append(d.ravel()[::2])
    cors = []
    for i in range(4):
        for j in range(i + 1, 4):
            a = hp[i]
            b = hp[j]
            sa = a.std()
            sb = b.std()
            if not (sa > 1e-6) or not (sb > 1e-6):
                continue
            a = a - a.mean()
            b = b - b.mean()
            v = float((a * b).mean() / (sa * sb))
            if not np.isfinite(v):
                continue
            cors.append(v)
    if not cors:
        return 1.0
    val = float(np.mean(cors))
    return val if np.isfinite(val) else 1.0


def reconstruct(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raw = raw.reshape(raw.shape[-2:])
    h, w = raw.shape
    if _FILTERS is None:
        return _fallback(raw.astype(np.float32))

    try:
        img = raw.astype(np.float32, copy=False)
        k = 2 * _R + 1
        mean = _box_mean(img, 9)
        norm = img - mean
        pad = np.pad(norm, _R, mode="edge")
        patches = np.lib.stride_tricks.sliding_window_view(pad, (k, k))

        # --- learned linear filter over the mosaic window ---
        out = np.empty((4, h, w), dtype=np.float32)
        for pr in (0, 1):
            for pc in (0, 1):
                # the three planes that are missing at this phase share the
                # same mosaic window -> one copy, one matrix product
                cols, idx = [], []
                for ci, (oy, ox) in enumerate(_PHASE):
                    if pr == oy and pc == ox:
                        continue                      # observed: exact
                    cols.append(_FILTERS[ci, pr * 2 + pc].reshape(-1))
                    idx.append(ci)
                if not idx:
                    continue
                sub = patches[pr::2, pc::2]
                nr, nc = sub.shape[0], sub.shape[1]
                pred = sub.reshape(-1, k * k) @ np.stack(cols, axis=1)
                for j, ci in enumerate(idx):
                    out[ci][pr::2, pc::2] = pred[:, j].reshape(nr, nc)

        for ci, (oy, ox) in enumerate(_PHASE):
            out[ci][oy::2, ox::2] = norm[oy::2, ox::2]
            out[ci] += mean

        # --- how much do the four planes share their high frequencies? ---
        # Measured on the plain cubic interpolations of the four planes
        # (a central crop is enough: the statistic only needs the global
        # structure of the scene).  Using the interpolations and not the
        # filtered planes keeps the measurement independent of the filters.
        if ADAPTIVE:
            small = []
            for (oy, ox) in _PHASE:
                lat = raw[oy::2, ox::2].astype(np.float32)
                lh, lw = lat.shape
                ch = min(256, lh)
                cw = min(256, lw)
                r0 = (lh - ch) // 2
                c0 = (lw - cw) // 2
                small.append(_cubic_lattice(lat[r0:r0 + ch, c0:c0 + cw], oy, ox))
            stat = _cross_corr(small)
            gamma = (stat - GAMMA_LO) / (GAMMA_HI - GAMMA_LO)
            gamma = 0.0 if gamma < 0.0 else (1.0 if gamma > 1.0 else gamma)
            if not np.isfinite(gamma):
                gamma = 1.0
        else:
            gamma = GAMMA

        # --- mix with the plain cubic interpolation of the plane ---
        if gamma < 1.0:
            for ci, (oy, ox) in enumerate(_PHASE):
                lat = raw[oy::2, ox::2].astype(np.float32)
                cub = _cubic_lattice(lat, oy, ox)[:h, :w]
                out[ci] = gamma * out[ci] + (1.0 - gamma) * cub
    except Exception:                                    # pragma: no cover
        # something unexpected: fall back to the plain cubic interpolation
        try:
            res = np.empty((4, h, w), dtype=np.float32)
            for ci, (oy, ox) in enumerate(_PHASE):
                lat = raw[oy::2, ox::2].astype(np.float32)
                res[ci] = _cubic_lattice(lat, oy, ox)[:h, :w]
            np.nan_to_num(res, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
            np.clip(res, 0.0, 255.0, out=res)
            return res
        except Exception:                            # pragma: no cover
            return _fallback(raw.astype(np.float32))

    np.nan_to_num(out, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
    np.clip(out, 0.0, 255.0, out=out)
    if not np.isfinite(out).all():
        # the filter produced something invalid (should not happen): fall
        # back to the plain cubic interpolation of every plane
        try:
            for ci, (oy, ox) in enumerate(_PHASE):
                lat = raw[oy::2, ox::2].astype(np.float32)
                out[ci] = _cubic_lattice(lat, oy, ox)[:h, :w]
            np.nan_to_num(out, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
            np.clip(out, 0.0, 255.0, out=out)
        except Exception:                            # pragma: no cover
            return _fallback(raw.astype(np.float32))
    return out
