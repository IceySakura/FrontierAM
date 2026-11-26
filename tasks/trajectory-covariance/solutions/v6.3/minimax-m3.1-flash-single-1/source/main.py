#!/usr/bin/env python3
"""Entry point: read a trajectory, write the estimated long-run covariance.

Usage:  python3 main.py INPUT.npy OUTPUT.npy
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lrc_est import estimate_lrc  # noqa: E402


def fallback(X):
    """Sample covariance (used only if the main estimator fails)."""
    M, d = X.shape
    Xc = X - X.mean(axis=0)
    H = (Xc.T @ Xc) / max(M - 1, 1)
    H = 0.5 * (H + H.T)
    return H


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: main.py INPUT.npy OUTPUT.npy\n")
        return 2
    in_path, out_path = argv[1], argv[2]
    X = np.load(in_path, allow_pickle=False)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    M, d = X.shape
    if M < 2 or d < 1:
        np.save(out_path, np.zeros((max(d, 0), max(d, 0)), dtype="<f8"))
        return 0
    if not np.all(np.isfinite(X)):
        # replace non finite entries by the column mean (should not happen)
        for j in range(d):
            col = X[:, j]
            good = np.isfinite(col)
            if not good.any():
                col[:] = 0.0
            elif not good.all():
                col[~good] = col[good].mean()
    try:
        H = estimate_lrc(X)
    except Exception as exc:  # pragma: no cover - safety net
        sys.stderr.write("estimator failed (%r), using sample covariance\n" % (exc,))
        try:
            H = fallback(X)
        except Exception:
            H = np.zeros((d, d))
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (d, d):
        H = np.zeros((d, d))
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H = 0.5 * (H + H.T)
    H = np.ascontiguousarray(H, dtype="<f8")
    tmp = out_path + ".tmp.npy"
    np.save(tmp, H)
    os.replace(tmp, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
