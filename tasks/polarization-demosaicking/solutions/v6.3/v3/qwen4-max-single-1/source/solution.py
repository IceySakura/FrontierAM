import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


def _limit_threads():
    try:
        import ctypes
        import glob
        base = os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs")
        for path in glob.glob(os.path.join(base, "libopenblas*")):
            try:
                lib = ctypes.CDLL(path)
                for name in ("openblas_set_num_threads64_", "openblas_set_num_threads"):
                    if hasattr(lib, name):
                        getattr(lib, name)(1)
                        return
            except OSError:
                pass
    except Exception:
        pass


_limit_threads()

_WEIGHTS = np.load(os.path.join(os.path.dirname(__file__), "weights.npy"))
_OFF = ((1, 1), (0, 1), (0, 0), (1, 0))
_OBS = ((0, 0, 2), (0, 1, 1), (1, 0, 3), (1, 1, 0))
_GW = np.exp(-0.5 * np.arange(-3, 4, dtype=np.float32) ** 2)
_GW /= _GW.sum()


def _sub(r):
    return [r[dr::2, dc::2] for dr, dc in _OFF]


def _up_linear(a, dr, dc, H, W):
    p = np.pad(a, ((1, 1), (0, 0)), mode="edge")
    tmp = np.empty((H, a.shape[1]), np.float32)
    tmp[dr::2] = a
    opp = 1 - dr
    lo = (np.arange(opp, H, 2) - dr) // 2
    tmp[opp::2] = 0.5 * (p[lo + 1] + p[lo + 2])
    q = np.pad(tmp, ((0, 0), (1, 1)), mode="edge")
    out = np.empty((H, W), np.float32)
    out[:, dc::2] = tmp
    oppc = 1 - dc
    lo = (np.arange(oppc, W, 2) - dc) // 2
    out[:, oppc::2] = 0.5 * (q[:, lo + 1] + q[:, lo + 2])
    return out


def _up_cubic(a, dr, dc, H, W):
    p = np.pad(a, ((2, 2), (0, 0)), mode="edge")
    tmp = np.empty((H, a.shape[1]), np.float32)
    tmp[dr::2] = a
    opp = 1 - dr
    lo = (np.arange(opp, H, 2) - dr) // 2
    tmp[opp::2] = (-p[lo + 1] + 9 * p[lo + 2] + 9 * p[lo + 3] - p[lo + 4]) / 16
    q = np.pad(tmp, ((0, 0), (2, 2)), mode="edge")
    out = np.empty((H, W), np.float32)
    out[:, dc::2] = tmp
    oppc = 1 - dc
    lo = (np.arange(oppc, W, 2) - dc) // 2
    out[:, oppc::2] = (-q[:, lo + 1] + 9 * q[:, lo + 2] + 9 * q[:, lo + 3] - q[:, lo + 4]) / 16
    return out


def _interp(r, kind):
    H, W = r.shape
    f = _up_linear if kind == "linear" else _up_cubic
    return np.stack([f(a, dr, dc, H, W) for a, (dr, dc) in zip(_sub(r), _OFF)])


def _blockfeat(r, k=5):
    H, W = r.shape
    b = r.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(H // 2, W // 2, 4)
    p = np.pad(b, ((k // 2, k // 2), (k // 2, k // 2), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(p, (k, k), axis=(0, 1))
    return win.reshape((H // 2) * (W // 2), k * k * 4)


def _from_block(Y, H, W):
    Y = Y.reshape(H // 2, W // 2, 4, 4)
    out = np.empty((4, H, W), np.float32)
    for c in range(4):
        out[c] = Y[:, :, :, c].reshape(H // 2, W // 2, 2, 2).transpose(0, 2, 1, 3).reshape(H, W)
    return out


def _ridge(r):
    H, W = r.shape
    X = _blockfeat(r)
    X = np.concatenate([X, np.ones((X.shape[0], 1), np.float32)], axis=1)
    out = _from_block(X @ _WEIGHTS, H, W)
    for c, (dr, dc) in enumerate(_OFF):
        out[c, dr::2, dc::2] = r[dr::2, dc::2]
    return out


def _smooth(r, P):
    S = _gaussian(P)
    out = np.empty_like(P)
    for odr, odc, oc in _OBS:
        ro = r[odr::2, odc::2]
        So = S[oc, odr::2, odc::2]
        for c in range(4):
            out[c, odr::2, odc::2] = ro + (S[c, odr::2, odc::2] - So)
    for c, (dr, dc) in enumerate(_OFF):
        out[c, dr::2, dc::2] = r[dr::2, dc::2]
    return out


def _gaussian(P):
    H, W = P.shape[1:]
    p = np.pad(P, ((0, 0), (0, 0), (3, 3)), mode="edge")
    h = np.lib.stride_tricks.sliding_window_view(p, 7, axis=2) @ _GW
    p = np.pad(h, ((0, 0), (3, 3), (0, 0)), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(p, 7, axis=1) @ _GW


def reconstruct(raw):
    r = np.asarray(raw, dtype=np.float32)
    H, W = r.shape
    P = _interp(r, "linear")
    M = P.mean(axis=0)
    ratio = float((P - M).std() / max(M.std(), 1e-6))
    alpha = min(1.0, max(0.0, (0.14 - ratio) / 0.06))
    if alpha <= 0.0:
        out = _interp(r, "cubic")
    else:
        sm = _smooth(r, P)
        ridge = _ridge(r)
        hybrid = 0.5 * ridge + 0.5 * sm
        if alpha >= 1.0:
            out = hybrid
        else:
            out = alpha * hybrid + (1.0 - alpha) * _interp(r, "cubic")
    for c, (dr, dc) in enumerate(_OFF):
        out[c, dr::2, dc::2] = r[dr::2, dc::2]
    return out
