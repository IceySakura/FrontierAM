import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_z = np.load(os.path.join(os.path.dirname(__file__), "weights.npz"))
_W1T = np.ascontiguousarray(_z["arr_0"].T.astype(np.float32))
_b1 = _z["arr_1"].astype(np.float32)
_W2T = np.ascontiguousarray(_z["arr_2"].T.astype(np.float32))
_b2 = _z["arr_3"].astype(np.float32)
_H = _W1T.shape[1]


def reconstruct(raw):
    raw = np.asarray(raw)
    H, W = raw.shape
    rp = np.pad(raw.astype(np.float32), 2, mode="reflect")
    X = sliding_window_view(rp, (5, 5)).reshape(-1, 25)
    X *= np.float32(1.0 / 255.0)
    np.sqrt(X, out=X)
    parity = np.empty((H * W, 2), np.float32)
    parity[:, 0] = np.repeat(np.arange(H, dtype=np.float32) % 2, W)
    parity[:, 1] = np.tile(np.arange(W, dtype=np.float32) % 2, H)
    X = np.concatenate([X, parity], axis=1)
    h = np.empty((H * W, _H), np.float32)
    np.matmul(X, _W1T, out=h)
    h += _b1
    np.maximum(h, 0, out=h)
    y = np.matmul(h, _W2T)
    y += _b2
    np.maximum(y, 0.0, out=y)
    y *= y
    out = y.reshape(H, W, 4).transpose(2, 0, 1) * 255.0
    np.clip(out, 0.0, 255.0, out=out)
    out[1, 0::2, 1::2] = raw[0::2, 1::2]
    return out.astype(np.float32)
