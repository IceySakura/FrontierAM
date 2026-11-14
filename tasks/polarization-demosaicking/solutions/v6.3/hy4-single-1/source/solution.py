"""Fast polarization-mosaic reconstruction.

The learned parameters in params.npz are scene-independent and were fitted
offline on the two public Wen et al. sample scenes.  At run time this module
uses only the supplied mosaic and those parameters.
"""
import os

# Keep numerical libraries single-threaded: the evaluator allows one logical CPU.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_PARAMS = os.path.join(_HERE, "params.npz")
if not os.path.exists(_PARAMS):
    _PARAMS = "params.npz"
with np.load(_PARAMS) as _p:
    W = np.ascontiguousarray(_p["W"], dtype=np.float32)
    ALPHA = np.ascontiguousarray(_p["alpha"], dtype=np.float32)
    BETA = np.ascontiguousarray(_p["beta"], dtype=np.float32)
    K = int(_p["K"])
    NB = int(_p["NB"])

# Top-left 2x2 tile is [[90, 45], [135, 0]]; channel order is 0, 45, 90, 135.
PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))
OBSERVED = (2, 1, 3, 0)
OTHERS = tuple(tuple(c for c in range(4) if c != kc) for kc in OBSERVED)
PROBS = np.arange(1, NB, dtype=np.float64) / NB
RAD = K // 2


def _box3(a):
    p = np.pad(a, 1, mode="edge")
    return ((p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
             + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
             + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
            * (1.0 / 9.0)).astype(np.float32)


def _texture(r):
    """Equal-mass bins of the gradient of a 3x3 box-filtered mosaic."""
    b = _box3(r)
    gx = np.empty_like(b)
    gy = np.empty_like(b)
    gx[:, 1:-1] = np.abs(b[:, 2:] - b[:, :-2])
    gx[:, 0] = np.abs(b[:, 1] - b[:, 0])
    gx[:, -1] = np.abs(b[:, -1] - b[:, -2])
    gy[1:-1, :] = np.abs(b[2:, :] - b[:-2, :])
    gy[0, :] = np.abs(b[1, :] - b[0, :])
    gy[-1, :] = np.abs(b[-1, :] - b[-2, :])
    g = gx + gy
    qs = np.quantile(g, PROBS)
    return np.searchsorted(qs, g, side="right").astype(np.uint8), b


def _bilinear_grid(g, H, Wd, oy, ox):
    """Bilinear upsampling of one 2x2-phase lattice, with replicated edges."""
    ny, nx = g.shape
    ty = np.clip((np.arange(H, dtype=np.float32) - oy) * 0.5, 0, ny - 1)
    tx = np.clip((np.arange(Wd, dtype=np.float32) - ox) * 0.5, 0, nx - 1)
    iy = ty.astype(np.intp)
    ix = tx.astype(np.intp)
    fy = ty - iy
    fx = tx - ix
    iy1 = np.minimum(iy + 1, ny - 1)
    ix1 = np.minimum(ix + 1, nx - 1)
    gy = g[iy, :] * (1.0 - fy)[:, None] + g[iy1, :] * fy[:, None]
    return (gy[:, ix] * (1.0 - fx)[None, :]
            + gy[:, ix1] * fx[None, :]).astype(np.float32)


def _bilinear_channels(r):
    H, Wd = r.shape
    a = np.empty((4, H, Wd), dtype=np.float32)
    for k, (oy, ox) in enumerate(PHASES):
        a[OBSERVED[k]] = _bilinear_grid(r[oy::2, ox::2], H, Wd, oy, ox)
    return a


def reconstruct(raw):
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError("raw must be a two-dimensional array")
    H, Wd = raw.shape
    if H < 2 or Wd < 2:
        raise ValueError("raw is too small")
    r = raw.astype(np.float32)
    labels, blur = _texture(r)
    mu = np.array([r[oy::2, ox::2].mean() for oy, ox in PHASES], dtype=np.float64)

    out = np.empty((4, H, Wd), dtype=np.float32)
    out[:] = r

    padded = np.pad(r, RAD, mode="reflect")
    windows = sliding_window_view(padded, (K, K))
    nf = K * K
    for k, (oy, ox) in enumerate(PHASES):
        kc = OBSERVED[k]
        center = r[oy::2, ox::2]
        lp = labels[oy::2, ox::2]
        # Patch values relative to the center; this removes the dominant
        # scene signal and makes the learned filters insensitive to exposure.
        X = np.ascontiguousarray(windows[oy::2, ox::2] - center[..., None, None],
                                  dtype=np.float32).reshape(-1, nf)
        for b in range(NB):
            mask = lp == b
            corr = X[mask.ravel()] @ W[k, b]
            base = center[mask]
            aa = ALPHA[k, :, b]
            bb = BETA[k, :, b]
            for j, c in enumerate(OTHERS[k]):
                vals = base + (mu[c] - mu[kc]) + aa[c] * corr[:, j] + bb[c]
                out[c, oy::2, ox::2][mask] = vals

    # A cheap modulation statistic detects scenes whose polarization signal
    # is much stronger and smoother than the camera data used to fit W.
    ny, nx = H // 2, Wd // 2
    b00 = blur[0:2 * ny:2, 0:2 * nx:2]
    b01 = blur[0:2 * ny:2, 1:2 * nx:2]
    b10 = blur[1:2 * ny:2, 0:2 * nx:2]
    b11 = blur[1:2 * ny:2, 1:2 * nx:2]
    q = _box3((b11 - b00) * 0.5)
    u = _box3((b01 - b10) * 0.5)
    d = _box3((b11 + b00 - b01 - b10) * 0.25)
    cheap_pol = np.std(q) + np.std(u) + np.std(d)
    if cheap_pol > 4.5:
        a = _bilinear_channels(r)
        pol = 0.5 * (np.std(a[0] - a[2]) + np.std(a[1] - a[3])
                     + np.std(a[0] + a[2] - a[1] - a[3]))
        bw = float(np.clip((pol - 8.0) / 10.0, 0.0, 1.0))
        if bw > 0.0:
            out *= np.float32(1.0 - bw)
            out += np.float32(bw) * a
            for k, (oy, ox) in enumerate(PHASES):
                out[OBSERVED[k], oy::2, ox::2] = r[oy::2, ox::2]
    np.clip(out, 0.0, 255.0, out=out)
    return out
