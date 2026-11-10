"""Four-angle polarization demosaicking (0/45/90/135 deg) from a 2x2 DoFP mosaic.

Mosaic (row, col parity): even rows [90, 45], odd rows [135, 0].
With X = (-1)^col and Y = (-1)^row the raw image is exactly

    r = L - X*C1 - Y*C2,   L = S0/2,  C1 = (S1+S2)/4,  C2 = (S1-S2)/4,

so the two polarization "chroma" signals sit on the (pi,0) and (0,pi) carriers.
C1 and C2 are estimated from the demodulated mosaics -X*r and -Y*r with learned
11x11 linear filters whose responses are steered per pixel by the local gradient
orientation (structure tensor), then blended with a ratio-domain estimate
(C/L averaged locally with edge-aware weights, re-multiplied by L).
Finally L = r + X*C1 + Y*C2 and

    I0 = L+C1+C2,  I45 = L+C1-C2,  I90 = L-C1-C2,  I135 = L-C1+C2,

plus a strongly smoothed (pi,pi) term D (channel gain mismatch, zero if ideal).
Every observed pixel is reproduced exactly.  numpy only (fast start-up).
"""
import os

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # single-threaded run

import numpy as np

import base64

# Learned 11x11 kernels, shape (2 chroma, 5 features, 11, 11), little-endian float64.
# (Same values as kernels.npy, embedded so that solution.py is self-contained.)
_K_B64 = (
    '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'
    'I3xaeP/XWr/u7iryC41Xv1smczy0F12/ZZIXYRTkZT/+En0TpGITP3geO6Y8V2A/5sxke/VgTD9r'
    'nj2PiRdWv6uoiqrwTlk/zFjmtd4xVb9dltsQUo4zP2qnUiyVtFC/w4S/BKYiYb+P+L7UBAJsv0WR'
    'u5GHsGg/QaqeLWv+Cb9d/VMG1OVsv7Hb8JvWcnK//jCLC1cCcr//4KdqZVxSPxK3t5QdpHs/Z5El'
    'lWevdD83KB1Ecs1VP1IBcyhSBEa/PZB/iu2ccb/eJULLRTMvP3oGLmZjTEO/rS2oeok6db/8hhi1'
    'L6qKvyBkqtGwtpO/nC7QMi4pLD816XqT622ZP2nMQgGgn4c/HIrKtFQMbT8UVHrtBZojP6E39D7L'
    'MmS/PKJPmaBUY78K3td3q0JBvwmLLZKJ+Vu/gswCBzY9ib/9Rfyp2IGev1uS0GrobWa/77Mf39NL'
    'oT9VCVezNwONP4UWS8oic24/VFeCiu7fVb+vrl8XPIo2P1Rdlvce80O/TClCXUN7Xr9yR0WWHpNU'
    'P0dKZjFhIHQ/ynjPhjdXKj88Xzs57wB+vyUGpNM2kl0/NAgmk6Vaaj/qdD13T5ZJP6KYDSXcAk+/'
    'RuDMT8RJRb9pJkvMO0xVP5A34LXiuGO/ieXP/xo5Zz96vqoAnDqRPzW3AvLiJp8/qlUa/V9Gbb+e'
    'LtSAY0Wcv7KUfuwFvIi/ucsssguEYb+uCZtQbFhAP86mNifyl1a/KZLCI6gqYr+ho9BVSxlRP7gO'
    '2SYTlWg/Ba8r2ba3iD8yb6LfKT2WP//NcE5tU2M/xtfrKuJ3k7+AnQa2wsuHvxP35oPDrnS/qKpz'
    '4NjUTj/aIvdiR91QPzUXO7pEiHC/vms7fO/nHL+6rnWkNEU1v/wrhywIn3k/eRl6c8Zabz/5gnpH'
    'LSxtP/kyCA24Mnq/0K8hhpbBdL9LUzz5S9dgv76fuU6wCPI+H7pvFibKaD+1hO91+39qv6JTlWSd'
    '6EG/yXLey79EZr+EEEa8cpg7PxmZ/QUU1Ee/7rnuxJZJST85m4REhQxBvythENfs0EC/ahX9P4lK'
    'WT84SS0Wp2o9P6mI/qzy7lk/vubvqUYPSL9+mW0/PYRUv7l2iAO462O/c1aNsuwGNr/Lgo0ZxYIx'
    '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'
)
_K = np.frombuffer(base64.b64decode(''.join(_K_B64)), dtype='<f8').reshape(2, 5, 11, 11).copy()
_R = (_K.shape[-1] - 1) // 2
_T = 16.0      # orientation confidence threshold (smoothed gradient energy)
_BETA = 0.6    # weight of the ratio-domain chroma estimate
_DELTA = 1.0   # regulariser of the local ratio C/L
_TAU = 6.0     # gradient scale of the edge-aware ratio weights
_D_DILATIONS = (1, 2, 4, 8)   # a-trous smoothing of the (pi,pi) ratio


def _good_size(n):
    """Smallest 2^a 3^b 5^c 7^d >= n, with a few sizes that are known to be fast."""
    if 1024 < n <= 1040:
        return 1040
    best = None
    p2 = 1
    while p2 < 2 * n:
        p3 = p2
        while p3 < 2 * n:
            p5 = p3
            while p5 < 2 * n:
                p7 = p5
                while p7 < 2 * n:
                    if p7 >= n and (best is None or p7 < best):
                        best = p7
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return best if best is not None else n


def _smooth14641(a, dil=1):
    """Separable [1,4,6,4,1]/16 smoothing over the last two axes, mirror boundary.

    Equivalent to two [1,2,1]/4 passes per axis with whole-sample symmetric
    boundaries.  Implemented as 4 two-tap sums per axis on a mirror-padded array.
    dil > 1 spaces the taps dil pixels apart (a-trous smoothing).
    """
    q = 2 * dil
    if min(a.shape[-2:]) <= q:
        dil, q = 1, 2
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(q, q), (q, q)], mode='reflect')
    for _ in range(4):
        p = p[..., :-dil, :] + p[..., dil:, :]
    for _ in range(4):
        p = p[..., :, :-dil] + p[..., :, dil:]
    p *= 1.0 / 256.0
    return p


def _b121(a, axis):
    """[1,2,1]/4 smoothing along one axis, mirror boundary."""
    o = np.empty_like(a)
    if axis == 0:
        np.add(a[:-2], a[2:], out=o[1:-1])
        o[1:-1] += a[1:-1]
        o[1:-1] += a[1:-1]
        o[1:-1] *= 0.25
        o[0] = (a[0] + a[1]) * 0.5
        o[-1] = (a[-1] + a[-2]) * 0.5
    else:
        np.add(a[:, :-2], a[:, 2:], out=o[:, 1:-1])
        o[:, 1:-1] += a[:, 1:-1]
        o[:, 1:-1] += a[:, 1:-1]
        o[:, 1:-1] *= 0.25
        o[:, 0] = (a[:, 0] + a[:, 1]) * 0.5
        o[:, -1] = (a[:, -1] + a[:, -2]) * 0.5
    return o


def _features(r32):
    gx = np.zeros_like(r32)
    gy = np.zeros_like(r32)
    np.subtract(r32[:, 2:], r32[:, :-2], out=gx[:, 1:-1])
    np.subtract(r32[2:], r32[:-2], out=gy[1:-1])
    gx *= 0.5
    gy *= 0.5
    E = np.empty((3,) + r32.shape, np.float32)
    np.multiply(gx, gx, out=E[0])
    np.multiply(gy, gy, out=E[1])
    np.multiply(gx, gy, out=E[2])
    S = _smooth14641(E)
    inv = 1.0 / (S[0] + S[1] + np.float32(_T))
    c = (S[0] - S[1]) * inv
    s = (2.0 * inv) * S[2]
    return c, s, E[0], E[1]


def _kernel_spectra(Ny, Nx):
    # Spectrum of the zero-padded correlation kernels on an Ny x Nx periodic grid.
    k = np.arange(-_R, _R + 1)
    Ey = np.exp((2j * np.pi / Ny) * np.outer(np.arange(Ny), k))
    Ex = np.exp((2j * np.pi / Nx) * np.outer(np.arange(Nx // 2 + 1), k))
    M = np.matmul(_K.astype(np.complex128), Ex.T)        # (2, J, 2R+1, Nx/2+1)
    return np.matmul(Ey, M)                               # (2, J, Ny, Nx/2+1)


def _fallback(r):
    # Very small inputs: nearest same-channel sample inside the 2x2 cell.
    H, W = r.shape
    out = np.empty((4, H, W))
    for dy, dx, ch in ((0, 0, 2), (0, 1, 1), (1, 0, 3), (1, 1, 0)):
        sy = dy if H > dy else 0
        sx = dx if W > dx else 0
        sub = r[sy::2, sx::2]
        rep = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
        rep = np.pad(rep, ((0, max(0, H - rep.shape[0])), (0, max(0, W - rep.shape[1]))), mode='edge')
        out[ch] = rep[:H, :W]
    for dy, dx, ch in ((0, 0, 2), (0, 1, 1), (1, 0, 3), (1, 1, 0)):
        out[ch, dy::2, dx::2] = r[dy::2, dx::2]
    return out


def reconstruct(raw):
    r = np.asarray(raw, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError('expected a 2-D mosaic')
    H, W = r.shape
    R = _R
    if H < 2 * R + 2 or W < 2 * R + 2:
        return np.clip(_fallback(r), 0.0, 255.0).astype(np.float32)

    X = np.where(np.arange(W) % 2 == 0, 1.0, -1.0)[None, :]
    Y = np.where(np.arange(H) % 2 == 0, 1.0, -1.0)[:, None]
    J = _K.shape[1]
    c, s, gx2, gy2 = _features(r.astype(np.float32))
    feats = [None, c, s]
    if J >= 5:
        feats += [c * c - s * s, (2.0 * c) * s]

    Ny, Nx = _good_size(H + 2 * R), _good_size(W + 2 * R)
    FK = _kernel_spectra(Ny, Nx)
    rp = np.pad(r, R, mode='reflect')            # mirror padding keeps the 2x2 phase
    Xp = np.where(np.arange(-R, W + R) % 2 == 0, -1.0, 1.0)[None, :]
    Yp = np.where(np.arange(-R, H + R) % 2 == 0, -1.0, 1.0)[:, None]
    C = []
    for ch, mod in enumerate((Xp, Yp)):
        d = np.zeros((Ny, Nx))
        np.multiply(rp, mod, out=d[:H + 2 * R, :W + 2 * R])   # demodulated mosaic
        D = np.fft.rfft2(d)
        acc = None
        for j in range(J):
            o = np.fft.irfft2(D * FK[ch, j], s=(Ny, Nx))[R:R + H, R:R + W]
            if feats[j] is not None:
                o = o * feats[j]
            if acc is None:
                acc = o.copy() if feats[j] is None else o
            else:
                acc += o
        C.append(acc)
    C1, C2 = C

    L = r + X * C1 + Y * C2
    if _BETA > 0:
        # The polarization ratio C/L varies more slowly than C itself: blend in
        # the estimate obtained by averaging the ratio locally and re-applying L.
        # Pixels next to strong edges across the carrier direction (horizontal
        # gradients for C1, vertical ones for C2) are the ones contaminated by
        # intensity leakage, so they get small weights in that average.
        k = np.float32(1.0 / (_TAU * _TAU))
        w1 = 1.0 / (1.0 + k * _b121(gx2, 0))
        w2 = 1.0 / (1.0 + k * _b121(gy2, 1))
        L32 = L.astype(np.float32)
        LL = L32 * L32
        P = np.empty((4, H, W), np.float32)
        np.multiply(C1, L32 * w1, out=P[0], casting='unsafe')
        np.multiply(LL, w1, out=P[1])
        np.multiply(C2, L32 * w2, out=P[2], casting='unsafe')
        np.multiply(LL, w2, out=P[3])
        M = _smooth14641(P)
        bl = np.float32(_BETA) * L32
        C1 = (1.0 - _BETA) * C1 + (bl / (M[1] + np.float32(_DELTA))) * M[0]
        C2 = (1.0 - _BETA) * C2 + (bl / (M[3] + np.float32(_DELTA))) * M[2]
        L = r + X * C1 + Y * C2

    # (pi,pi) carrier: D = (I0+I90-I45-I135)/4 is zero for ideal polarizers but
    # absorbs smooth gain differences between the angle channels.  It is
    # estimated as a strongly smoothed ratio D/L (wide windows keep the cost of
    # intensity leakage negligible when D really is zero).
    XY = X * Y
    L32 = L.astype(np.float32)
    d0 = _b121(_b121((XY * r).astype(np.float32), 0), 1)
    P = np.stack((d0 * L32, L32 * L32))
    for dil in _D_DILATIONS:
        P = _smooth14641(P, dil)
    Dm = L32 * P[0] / (P[1] + np.float32(_DELTA))
    L = L - XY * Dm

    out = np.empty((4, H, W), dtype=np.float32)
    a = L + C1
    b = L - C1
    p = C2 + Dm
    m = C2 - Dm
    np.add(a, p, out=out[0], casting='unsafe')
    np.subtract(a, p, out=out[1], casting='unsafe')
    np.subtract(b, m, out=out[2], casting='unsafe')
    np.add(b, m, out=out[3], casting='unsafe')
    np.clip(out, 0.0, 255.0, out=out)
    return out
