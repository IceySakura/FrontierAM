#!/usr/bin/env python3
"""Entry point: read an (M, d) trajectory, write the (d, d) long-run covariance."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lrv import estimate_long_run_covariance  # noqa: E402


def _save_npy(path, arr):
    """Write a .npy file to the exact path given (no extension is appended)."""
    arr = np.ascontiguousarray(arr, dtype='<f8')
    with open(path, 'wb') as f:
        np.lib.format.write_array(f, arr, version=(1, 0))


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: main.py INPUT OUTPUT\n")
        return 2
    inp, outp = argv[1], argv[2]
    try:
        x = np.load(inp, allow_pickle=False)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write("cannot read input: %s\n" % exc)
        return 1
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        sys.stderr.write("unexpected input shape %r\n" % (x.shape,))
        return 1
    d = x.shape[1]
    try:
        H = estimate_long_run_covariance(x)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write("estimation failed: %s\n" % exc)
        H = np.zeros((d, d))
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (d, d):
        H = np.zeros((d, d))
    H = 0.5 * (H + H.T)
    H = np.where(np.isfinite(H), H, 0.0)
    # keep the matrix positive semi-definite (zero eigenvalues are allowed)
    try:
        w, v = np.linalg.eigh(H)
        if w[0] < 0:
            w = np.maximum(w, 0.0)
            H = (v * w) @ v.T
            H = 0.5 * (H + H.T)
    except np.linalg.LinAlgError:
        pass
    outdir = os.path.dirname(os.path.abspath(outp))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    _save_npy(outp, H)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
