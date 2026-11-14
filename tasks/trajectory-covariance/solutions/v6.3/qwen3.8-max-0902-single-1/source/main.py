"""Entry point: ./run.sh INPUT OUTPUT  ->  long-run covariance estimate."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrcov import estimate  # noqa: E402


def main(inp, outp):
    x = np.load(inp, allow_pickle=False)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    M, d = x.shape
    try:
        H = estimate(x)
    except Exception:
        H = np.atleast_2d(np.cov(x.T, bias=True))
    H = np.asarray(H, dtype=np.float64)
    H = 0.5 * (H + H.T)
    if H.shape != (d, d) or not np.all(np.isfinite(H)):
        H = np.zeros((d, d))
    ev, U = np.linalg.eigh(H)
    if ev[0] < 0:
        H = (U * np.maximum(ev, 0.0)) @ U.T
        H = 0.5 * (H + H.T)
    np.save(outp, np.ascontiguousarray(H, dtype="<f8"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
