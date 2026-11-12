"""Entry point: python3 run_lrcov.py INPUT.npy OUTPUT.npy"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _valid(H, d):
    if H is None or H.shape != (d, d) or not np.all(np.isfinite(H)):
        return None
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    H = (V * np.maximum(w, 0.0)) @ V.T
    H = 0.5 * (H + H.T)
    return H if np.all(np.isfinite(H)) else None


def _batch_means(X):
    n, d = X.shape
    b = max(1, int(np.sqrt(n)))
    a = n // b
    Y = X[: a * b].reshape(a, b, d).mean(axis=1)
    Yc = Y - Y.mean(axis=0)
    return b * Yc.T @ Yc / max(a - 1, 1)


def main():
    inp, out = sys.argv[1], sys.argv[2]
    X = np.asarray(np.load(inp, allow_pickle=False), dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    d = X.shape[1]
    H = None
    import lrcov
    for f in (lambda: lrcov.estimate2(X, power=1.0),
              lambda: lrcov.estimate(X, resid_kernel=False),
              lambda: _batch_means(X),
              lambda: np.cov(X.T).reshape(d, d)):
        try:
            H = _valid(np.asarray(f(), dtype=np.float64), d)
        except Exception:
            H = None
        if H is not None:
            break
    if H is None:
        H = np.zeros((d, d))
    np.save(out, np.ascontiguousarray(H, dtype="<f8"))


if __name__ == "__main__":
    main()
