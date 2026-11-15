#!/usr/bin/env python3
"""Fallback FAMEXP2 solver (pure numpy/scipy).

Used only when the native binary could not be built.  Same algorithm as
solver.c: block Chebyshev expansion of exp and phi1 on an interval that
contains the spectrum of -t*L, both right hand sides propagated together.
"""
import json
import struct
import sys

import numpy as np
from scipy.fft import dct
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

IN = sys.stdin.buffer
OUT = sys.stdout.buffer


def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = IN.read(n - len(buf))
        if not chunk:
            raise SystemExit(2)
        buf.extend(chunk)
    return bytes(buf)


def coeffs(a, b, m):
    """Chebyshev coefficients (f = c0/2 + sum c_k T_k) of exp and phi1 on [a,b]."""
    c = 0.5 * (a + b)
    d = 0.5 * (b - a)
    N = 64
    while N < m + 512:
        N <<= 1
    j = np.arange(N)
    x = np.cos(np.pi * (j + 0.5) / N)
    z = c + d * x
    ez = np.exp(z)
    zz = np.where(np.abs(z) < 1e-30, 1e-30, z)
    ph = np.where(np.abs(z) < 1e-6, 1.0 + z / 2 + z * z / 6 + z ** 3 / 24, np.expm1(zz) / zz)
    ae = dct(ez.astype(np.float64), type=2) / N
    ap = dct(ph.astype(np.float64), type=2) / N
    ae[0] *= 0.5
    ap[0] *= 0.5
    return ae[:m + 1], ap[:m + 1], c, d


def main():
    OUT.write(b"READY\n")
    OUT.flush()

    magic = read_exact(8)
    if magic != b"FAMEXP2\n":
        raise SystemExit(2)
    jlen = struct.unpack("<I", read_exact(4))[0]
    meta = json.loads(read_exact(jlen).decode())
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    requests = int(meta["requests"])
    t = float(meta["t"])
    rtol = float(meta["rtol"])
    atol = float(meta["atol"])

    indptr = np.frombuffer(read_exact(8 * (n + 1)), dtype="<u8")
    indices = np.frombuffer(read_exact(4 * nnz), dtype="<u4")
    data = np.frombuffer(read_exact(8 * nnz), dtype="<f8")
    L = csr_matrix((data, indices, indptr), shape=(n, n))

    absrow = np.abs(L).sum(axis=1).A1 if nnz else np.zeros(n)
    Linf = float(absrow.max()) if n else 0.0
    Rinf = t * Linf

    ip64 = indptr.astype(np.int64)
    rowid = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip64))
    diag_only = bool(nnz) and bool(np.all(indices.astype(np.int64) == rowid))
    if diag_only:
        dv = np.zeros(n)
        np.add.at(dv, indices, data)
    else:
        dv = None

    if rtol >= 1e-5:
        Cdeg, Kdeg = 3.45, 12.0
    elif rtol >= 1e-8:
        Cdeg, Kdeg = 4.25, 16.0
    else:
        Cdeg, Kdeg = 5.05, 22.0

    mode = 3
    if t == 0.0:
        mode = 0
    elif not (Rinf > 0.0):
        mode = 1
    elif diag_only:
        mode = 2

    R = Rinf
    ae = ap = None
    m = 0
    if mode == 3:
        m = int(Cdeg * np.sqrt(Rinf) + Kdeg) + 1
        if m >= 24 and n > 1:
            try:
                est = float(eigsh(L, k=1, which="LA", return_eigenvectors=False,
                                  maxiter=200 * n, tol=1e-10)[0])
                Rtry = t * est * 1.01
                if 0.0 < Rtry < Rinf:
                    R = Rtry
            except Exception:
                pass
            m = int(Cdeg * np.sqrt(R) + Kdeg) + 1
        ae, ap, c, d = coeffs(-R, R * 1e-8, m)
        c1 = -2.0 * t / d
        c2 = -2.0 * c / d

    for _ in range(requests):
        b0 = np.frombuffer(read_exact(8 * n), dtype="<f8").astype(np.float64)
        b1 = np.frombuffer(read_exact(8 * n), dtype="<f8").astype(np.float64)
        if mode == 0:
            y = b0.copy()
        elif mode == 1:
            y = b0 + t * b1
        elif mode == 2:
            z = -t * dv
            ez = np.exp(z)
            ph = np.where(np.abs(z) < 1e-8, 1.0 + z / 2 + z * z / 6, np.expm1(z) / np.where(z == 0, 1, z))
            y = ez * b0 + t * ph * b1
        else:
            V = np.empty((n, 2))
            V[:, 0] = b0
            V[:, 1] = b1
            Y = ae[0] * b0 + t * ap[0] * b1
            P = np.zeros((n, 2))

            def step(X, W, k1, k2):
                Z = k1 * (L @ X) + k2 * X - W
                return Z

            U = step(V, P, 0.5 * c1, 0.5 * c2)
            Y = Y + ae[1] * U[:, 0] + t * ap[1] * U[:, 1]
            for k in range(2, m + 1):
                W = step(U, V, c1, c2)
                V = U
                U = W
                Y += ae[k] * U[:, 0] + t * ap[k] * U[:, 1]
            y = Y
        y = np.where(np.isfinite(y), y, 0.0)
        OUT.write(np.ascontiguousarray(y, dtype="<f8").tobytes())
        OUT.flush()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
