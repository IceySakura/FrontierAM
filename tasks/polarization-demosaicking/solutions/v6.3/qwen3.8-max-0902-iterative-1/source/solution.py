"""Polarisation mosaic demosaicking -> I(0deg), I(45deg), I(90deg), I(135deg).

Sensor tile (rows/cols counted from 0 at the top-left corner)::

    even row, even col -> 90 deg      even row, odd col -> 45 deg
    odd  row, even col -> 135 deg     odd  row, odd col ->  0 deg

Physical model.  With I(t) the intensity transmitted by a linear polariser at
angle t and (S0, S1, S2) the linear Stokes parameters,

    I(0) = T + a,  I(90) = T - a,  I(45) = T + b,  I(135) = T - b,
    T = S0/2,  a = S1/2,  b = S2/2.

So every pixel reports  raw = T + alpha*a + beta*b  with alpha = -1 on
(even,even), +1 on (odd,odd) and 0 elsewhere, beta = +1 on (even,odd),
-1 on (odd,even) and 0 elsewhere.

Estimator.  We minimise the quadratic functional

    J(a,b) = (raw - s)^T Phi_T (raw - s) + a^T Phi_a a + b^T Phi_b b
    s = alpha*a + beta*b,   Phi_* = c1*L + c2*L*L

with L the 4-neighbour grid Laplacian.  The data term is the only place `raw`
enters: it demands that the intensity T = raw - s be smooth, while Phi_a and
Phi_b (about ten times stiffer) demand that the polarisation fields be much
smoother still.  T therefore keeps the image detail and a, b absorb only the
slowly varying polarisation contrast - which is exactly the redundancy that
makes a 4x-sampled Stokes mosaic recoverable.

alpha, beta and every Phi are 2x2 periodic / translation invariant, so the
normal equations block-diagonalise in the polyphase Fourier basis: splitting
each full-resolution field into its four 2x2 sub-lattices (order 00, 01, 10,
11) turns every operator into a 4x4 matrix symbol and multiplication by
alpha / beta into a diagonal one.  The result is one 8x8 complex system per
frequency bin.  That system is inverted once, offline (build.sh), into a
(8, 4) transfer map from the polyphase DFT of `raw` to the polyphase DFTs of
(a, b); at run time the reconstruction is 4 forward rfft2, one 8x4 complex
matrix-vector product per bin, 8 inverse rfft2 and a re-interleave.

Finally T = raw - s and I_c = T + s_c, so every measured pixel is reproduced
exactly and only the three unmeasured directions per pixel carry error.

Only NumPy is used at run time.  No randomness, no per-scene state.
"""
import os

import numpy as np

# ---------------------------------------------------------------- parameters
# Phi_T = sum_k PRIOR_T[k-1]*L^k ; Phi_a = Phi_b likewise.
PRIOR_T = (1.0, 0.2)
PRIOR_A = (6.0, -0.3)
# Exponent on the Laplacian in Phi_a: Phi_a = PRIOR_A[0] * L**PRIOR_P.
PRIOR_P = 1.0
# Multiple of the diagonal (corner) Laplacian added to L (0 = 4-neighbour only).
PRIOR_DIAG = 0.0
# Reflect padding: kills the periodic wrap-around artefact at the image border.
PAD = 16
# Largest polyphase grid (Mh * (Mw//2+1)) for which the transfer map may still
# be built inside a scene process; bigger grids fall back to the cheap estimator
# so the 1 s budget can never be blown by an unexpected input size.
MAX_BUILD_BINS = 150000
# Polyphase component order used throughout: (0,0), (0,1), (1,0), (1,1)
_ALPHA = np.array([-1.0, 0.0, 0.0, 1.0])
_BETA = np.array([0.0, 1.0, -1.0, 0.0])

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()


# ------------------------------------------------------------- filter symbol
def _lap_symbol(Mh, Mw, dtype=np.complex64, diag_w=0.0):
    """4x4 polyphase matrix symbol of the periodic grid Laplacian.

    diag_w adds that multiple of the diagonal (4-corner) Laplacian, which
    couples the two sub-lattices a lives on (and the two b lives on) directly.
    """
    k1 = np.arange(Mh, dtype=np.float64)[:, None]
    k2 = np.arange(Mw // 2 + 1, dtype=np.float64)[None, :]
    u1 = np.exp(-2j * np.pi * k1 / Mh)
    u2 = np.exp(-2j * np.pi * k2 / Mw)
    d1 = 1 + u1
    d2 = 1 + u2
    e1 = np.conj(d1)
    e2 = np.conj(d2)
    shape = np.broadcast(u1, u2).shape
    Lh = np.zeros(shape + (4, 4), dtype)
    for i in range(4):
        Lh[..., i, i] = 4.0
    Lh[..., 0, 1] = -d2
    Lh[..., 0, 2] = -d1
    Lh[..., 1, 0] = -e2
    Lh[..., 1, 3] = -d1
    Lh[..., 2, 0] = -e1
    Lh[..., 2, 3] = -d2
    Lh[..., 3, 1] = -e1
    Lh[..., 3, 2] = -e2
    if diag_w:
        Lh[..., 0, 0] += 4.0 * diag_w
        Lh[..., 1, 1] += 4.0 * diag_w
        Lh[..., 2, 2] += 4.0 * diag_w
        Lh[..., 3, 3] += 4.0 * diag_w
        Lh[..., 0, 3] -= diag_w * d1 * d2
        Lh[..., 3, 0] -= diag_w * e1 * e2
        Lh[..., 1, 2] -= diag_w * d1 * e2
        Lh[..., 2, 1] -= diag_w * e1 * d2
    return Lh


def _phi(Lh, coeffs, p=1.0):
    """sum_k coeffs[k-1] * L^k, or coeffs[0] * L^p when p != 1."""
    if p != 1.0:
        # Hermitian PSD symbol -> fractional power through its eigendecomposition
        w, V = np.linalg.eigh(Lh)
        w = np.clip(w.real, 0.0, None) ** p
        return (coeffs[0] * (V * w[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2)))
    out = np.zeros(Lh.shape, np.complex128)
    pw = np.array(Lh, dtype=np.complex128)
    for c in coeffs:
        if c:
            out += c * pw
        pw = pw @ Lh
    return out


def build_map(Mh, Mw, wT=PRIOR_T, wa=PRIOR_A, wb=None, pa=PRIOR_P, diag_w=PRIOR_DIAG):
    """(8, 4, Mh, Mw//2+1) complex64 transfer map for an Mh*2 x Mw*2 grid."""
    if wb is None:
        wb = wa
    Lh = _lap_symbol(Mh, Mw, dtype=np.complex128, diag_w=diag_w)
    PhiT = _phi(Lh, wT)
    Phia = _phi(Lh, wa, pa)
    Phib = _phi(Lh, wb)
    Da = np.diag(_ALPHA)
    Db = np.diag(_BETA)
    H = np.zeros(Lh.shape[:-2] + (8, 8), np.complex128)
    H[..., :4, :4] = Phia + Da @ PhiT @ Da
    H[..., :4, 4:] = Da @ PhiT @ Db
    H[..., 4:, :4] = Db @ PhiT @ Da
    H[..., 4:, 4:] = Phib + Db @ PhiT @ Db
    R = np.zeros(Lh.shape[:-2] + (8, 4), np.complex128)
    R[..., :4, :] = Da @ PhiT
    R[..., 4:, :] = Db @ PhiT
    m = np.linalg.solve(H, R)
    del H, R, Lh, PhiT, Phia, Phib
    return np.ascontiguousarray(np.moveaxis(m, (2, 3), (0, 1))).astype(np.complex64)


_MAPS = {}


def _map_path(Mh, Mw):
    return os.path.join(_HERE, "filter_map_%d_%d.npy" % (Mh, Mw))


def _get_map(Mh, Mw):
    key = (Mh, Mw)
    sym = _MAPS.get(key)
    if sym is None:
        path = _map_path(Mh, Mw)
        if os.path.exists(path):
            try:
                sym = np.load(path)
                if sym.shape != (8, 4, Mh, Mw // 2 + 1):
                    sym = None
            except Exception:
                sym = None
        if sym is None:
            if Mh * (Mw // 2 + 1) > MAX_BUILD_BINS:
                raise MemoryError("grid too large to build the transfer map")
            sym = build_map(Mh, Mw)
            try:
                np.save(path, sym)
            except Exception:
                pass
        _MAPS[key] = sym
    return sym


# ------------------------------------------------------------- reconstruction
def _assemble(x, S):
    """Rebuild the four channels from raw `x` and the polyphase parts of a, b."""
    a00, a01, a10, a11, b00, b01, b10, b11 = S
    H, W = x.shape
    out = np.empty((4, H, W), np.float32)
    ee = x[0::2, 0::2]
    oo = x[1::2, 1::2]
    eo = x[0::2, 1::2]
    oe = x[1::2, 0::2]
    # (even,even): 90 deg measured, T = x + a00
    out[2, 0::2, 0::2] = ee
    out[0, 0::2, 0::2] = ee + 2 * a00
    out[1, 0::2, 0::2] = ee + a00 + b00
    out[3, 0::2, 0::2] = ee + a00 - b00
    # (odd,odd): 0 deg measured, T = x - a11
    out[0, 1::2, 1::2] = oo
    out[2, 1::2, 1::2] = oo - 2 * a11
    out[1, 1::2, 1::2] = oo - a11 + b11
    out[3, 1::2, 1::2] = oo - a11 - b11
    # (even,odd): 45 deg measured, T = x - b01
    out[1, 0::2, 1::2] = eo
    out[3, 0::2, 1::2] = eo - 2 * b01
    out[0, 0::2, 1::2] = eo - b01 + a01
    out[2, 0::2, 1::2] = eo - b01 - a01
    # (odd,even): 135 deg measured, T = x + b10
    out[3, 1::2, 0::2] = oe
    out[1, 1::2, 0::2] = oe + 2 * b10
    out[0, 1::2, 0::2] = oe + b10 + a10
    out[2, 1::2, 0::2] = oe + b10 - a10
    return out


def _solve(x, pad):
    """Run the filter on a float32 mosaic `x` (already even-sized)."""
    if pad:
        x = np.pad(x, pad, mode="reflect")
    H, W = x.shape
    Mh, Mw = H // 2, W // 2
    sym = _get_map(Mh, Mw)
    P = np.empty((4, Mh, Mw), np.float32)
    P[0] = x[0::2, 0::2]
    P[1] = x[0::2, 1::2]
    P[2] = x[1::2, 0::2]
    P[3] = x[1::2, 1::2]
    X = np.fft.rfft2(P)
    del P
    Y = np.empty((8,) + X.shape[1:], np.complex64)
    for i in range(8):
        acc = sym[i, 0] * X[0]
        acc += sym[i, 1] * X[1]
        acc += sym[i, 2] * X[2]
        acc += sym[i, 3] * X[3]
        Y[i] = acc
    del X, sym
    S = np.fft.irfft2(Y, s=(Mh, Mw)).astype(np.float32)
    del Y
    out = _assemble(x, S)
    if pad:
        out = out[:, pad:pad + (H - 2 * pad), pad:pad + (W - 2 * pad)]
    return out


def _fallback(raw):
    """Cheap Stokes block estimate, used only if the FFT path is unavailable."""
    x = np.asarray(raw, np.float32)
    h0, w0 = x.shape
    if h0 % 2:
        x = np.concatenate([x, x[-1:]], axis=0)
    if w0 % 2:
        x = np.concatenate([x, x[:, -1:]], axis=1)
    A = x[0::2, 0::2]
    B = x[0::2, 1::2]
    C = x[1::2, 0::2]
    D = x[1::2, 1::2]
    P = 0.25 * (A + B + C + D)
    U = 0.5 * (D - A)
    V = 0.5 * (B - C)
    Pp = np.pad(P, 1, mode="edge")
    gx = 0.25 * (Pp[1:-1, 2:] - Pp[1:-1, :-2])
    gy = 0.25 * (Pp[2:, 1:-1] - Pp[:-2, 1:-1])
    U = U - 0.5 * (gx + gy)
    V = V - 0.5 * (gx - gy)

    def up(Tb):
        Q = np.pad(Tb, 1, mode="edge")
        c = Q[1:-1, 1:-1]
        Y = np.empty((2 * Tb.shape[0], Tb.shape[1]), np.float32)
        Y[0::2] = 0.75 * c + 0.25 * Q[:-2, 1:-1]
        Y[1::2] = 0.75 * c + 0.25 * Q[2:, 1:-1]
        Q = np.pad(Y, ((0, 0), (1, 1)), mode="edge")
        c = Q[:, 1:-1]
        X = np.empty((Y.shape[0], 2 * Y.shape[1]), np.float32)
        X[:, 0::2] = 0.75 * c + 0.25 * Q[:, :-2]
        X[:, 1::2] = 0.75 * c + 0.25 * Q[:, 2:]
        return X

    a = up(U)
    b = up(V)
    H, W = x.shape
    out = np.empty((4, H, W), np.float32)
    T = x.copy()
    T[0::2, 0::2] += a[0::2, 0::2]
    T[1::2, 1::2] -= a[1::2, 1::2]
    T[0::2, 1::2] -= b[0::2, 1::2]
    T[1::2, 0::2] += b[1::2, 0::2]
    out[0] = T + a
    out[2] = T - a
    out[1] = T + b
    out[3] = T - b
    out = out[:, :h0, :w0]
    np.clip(out, 0.0, 255.0, out=out)
    if not np.isfinite(out).all():
        np.nan_to_num(out, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
    return out


def reconstruct(raw):
    """Reconstruct (4, H, W) float32 = I(0), I(45), I(90), I(135)."""
    x = np.asarray(raw)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("expected a 2-D mosaic of at least 2x2")
    h, w = x.shape
    pad = max(2, min(PAD, min(h, w) // 8))
    if h % 2:                       # keep the mosaic phase intact by padding
        x = np.concatenate([x, x[-1:]], axis=0)
    if w % 2:
        x = np.concatenate([x, x[:, -1:]], axis=1)
    try:
        out = _solve(x, pad)
        if h % 2:
            out = out[:, :h]
        if w % 2:
            out = out[:, :, :w]
        # the reference lives in the sensor's 0..255 range; overshoot is pure error
        np.clip(out, 0.0, 255.0, out=out)
        if not np.isfinite(out).all():
            np.nan_to_num(out, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
        return out
    except Exception:
        try:
            return _fallback(np.asarray(raw, dtype=np.float32))
        except Exception:
            # last resort: never fail a scene, every measured pixel is exact
            o = np.repeat(np.asarray(raw, np.float32)[None], 4, 0)
            return np.nan_to_num(o, nan=0.0, posinf=255.0, neginf=0.0)
