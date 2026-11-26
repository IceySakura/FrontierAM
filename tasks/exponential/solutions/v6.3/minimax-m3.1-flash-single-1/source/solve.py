#!/usr/bin/env python3
"""FAMEXP2: exponential propagation for sparse symmetric PSD operators.

    y_j = exp(-t L) b0_j + t * phi1(-t L) b1_j ,   phi1(z) = sum_k z^k/(k+1)!

Algorithm
---------
Symmetric (Lanczos) Krylov projection of the operator with full
reorthogonalisation.  The projected matrix is symmetric tridiagonal, so both
matrix functions are evaluated from its spectral decomposition; the Krylov
degree is chosen adaptively by comparing approximations of increasing degree
and stopping when they agree well inside the requested tolerance.
"""
import json
import struct
import sys
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import csr_matrix

MAGIC = b"FAMEXP2\n"


# ----------------------------------------------------------------- protocol
def _read_exact(stream, n):
    parts = bytearray()
    while len(parts) < n:
        blk = stream.read(n - len(parts))
        if not blk:
            raise EOFError("unexpected end of input")
        parts.extend(blk)
    return bytes(parts)


def _write_all(stream, data):
    mv = memoryview(data)
    while mv:
        w = stream.write(mv)
        if not w:
            raise IOError("no progress")
        mv = mv[w:]


# ------------------------------------------------------------- scalar func
def phi1_neg(z):
    """phi1(z) for z <= 0 (i.e. phi1(-x) with x >= 0)."""
    out = np.empty_like(z)
    small = z > -0.5
    if small.any():
        zs = z[small]
        term = np.ones_like(zs)
        s = np.ones_like(zs)
        for k in range(1, 45):
            term = term * (zs / (k + 1))
            s = s + term
        out[small] = s
    big = ~small
    if big.any():
        zb = z[big]
        out[big] = np.expm1(zb) / zb
    return out


# --------------------------------------------------------------- krylov
def _proj_eval(V, alpha, beta, m, t, kind, nb):
    """Evaluate the projected matrix function and map back to the original space."""
    d = alpha[:m]
    e = beta[: m - 1]
    try:
        th, Q = eigh_tridiagonal(d, e)
    except Exception:
        T = np.diag(d)
        if m > 1:
            T += np.diag(e, 1) + np.diag(e, -1)
        th, Q = np.linalg.eigh(T)
    if kind == 0:
        fv = np.exp(t * th)
    else:
        fv = t * phi1_neg(t * th)
    z = Q @ (fv * (nb * Q[0, :]))
    return V[:, :m] @ z


def krylov_solve(A, b, t, rtol, atol, kind, deadline, mmax, scale_box):
    """exp(-tL) b (kind 0) or t*phi1(-tL) b (kind 1), adaptive Krylov degree."""
    n = b.size
    if not np.any(b):
        return np.zeros(n)
    V = np.zeros((n, mmax + 1), order="F")
    chunk = max(1, min(mmax + 1, int(8.0e6 / (8.0 * max(n, 1)))))
    alpha = np.zeros(mmax)
    beta = np.zeros(mmax)
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return np.zeros(n)
    V[:, 0] = b / nb
    m = 0
    prev = None
    checkpoint = 8
    last = None
    while True:
        target = min(checkpoint, mmax)
        stop = False
        while m < target:
            if time.perf_counter() > deadline:
                stop = True
                break
            j = m
            w = A @ V[:, j]
            aj = float(V[:, j] @ w)
            alpha[j] = aj
            w -= aj * V[:, j]
            if j:
                w -= beta[j - 1] * V[:, j - 1]
            # Project against the whole basis in cache-sized chunks: the
            # arithmetic is identical to a single BLAS-2 call but the second
            # pass over each chunk is served from cache.
            Vj = V[:, : j + 1]
            before = float(np.linalg.norm(w))
            step = chunk
            for c0 in range(0, j + 1, step):
                Vc = Vj[:, c0:c0 + step]
                c = Vc.T @ w
                w -= Vc @ c
            after = float(np.linalg.norm(w))
            if after < 0.717 * before:
                c = Vj.T @ w
                w -= Vj @ c
                after = float(np.linalg.norm(w))
            beta[j] = after
            m = j + 1
            if after <= 1e-14 * (abs(aj) + 1.0):
                stop = True
                break
            V[:, j + 1] = w / after
        if m > 0:
            y = _proj_eval(V, alpha, beta, m, t, kind, nb)
            last = y
            if prev is not None:
                cur = max(float(np.max(np.abs(y))), float(np.max(np.abs(prev))))
                scale = max(cur, scale_box[0])
                tol = atol + rtol * scale
                diff = float(np.max(np.abs(y - prev)))
                # A vanishing current scale means the Krylov space has not yet
                # resolved the slow modes (the exponential underflows); such a
                # degenerate agreement carries no information.
                if cur > 0.0 and diff <= 0.05 * tol:
                    scale_box[0] = max(scale_box[0], cur)
                    sys.stderr.write("krylov m=%d kind=%d diff=%.3e tol=%.3e\n"
                                     % (m, kind, diff, tol))
                    return y
                scale_box[0] = max(scale_box[0], cur)
            prev = y
        if stop or m >= mmax:
            break
        checkpoint = min(max(int(checkpoint * 1.5) + 1, m + 4), mmax)
    if last is None:
        return np.zeros(n)
    return last


# ----------------------------------------------------------------- driver
def main():
    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    out.write(b"READY\n")
    out.flush()

    magic = _read_exact(inp, len(MAGIC))
    if magic != MAGIC:
        return 1
    (jlen,) = struct.unpack("<I", _read_exact(inp, 4))
    meta = json.loads(_read_exact(inp, jlen).decode("utf-8"))
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    t = float(meta["t"])
    rtol = float(meta["rtol"])
    atol = float(meta["atol"])
    requests = int(meta["requests"])
    scenario = meta.get("scenario", "one_shot")

    indptr = np.frombuffer(_read_exact(inp, 8 * (n + 1)), dtype="<u8")
    indices = np.frombuffer(_read_exact(inp, 4 * nnz), dtype="<u4")
    data = np.frombuffer(_read_exact(inp, 8 * nnz), dtype="<f8")

    start = time.perf_counter()
    budget = 135.0 if scenario == "reuse" else 13.5
    deadline = start + budget

    L = csr_matrix((data, indices, indptr), shape=(n, n))
    diag = None
    if nnz == n and np.array_equal(indices, np.arange(n, dtype=np.uint32)):
        diag = np.asarray(L.diagonal(), dtype=np.float64)
    A = (-L).tocsr()
    A.sort_indices()
    del L, data, indices, indptr

    # memory cap for the Krylov basis: 8 * n * (mmax+1) bytes
    mcap = int(1.5e9 / (8.0 * max(n, 1)))
    mmax = int(min(3000, max(64, mcap)))

    for r in range(requests):
        b0 = np.frombuffer(_read_exact(inp, 8 * n), dtype="<f8").copy()
        b1 = np.frombuffer(_read_exact(inp, 8 * n), dtype="<f8").copy()
        now = time.perf_counter()
        left = max(0.05, deadline - now)
        per = left / max(1, requests - r)
        rdeadline = min(deadline, now + per)
        b0scale = float(np.max(np.abs(b0))) if b0.size else 0.0
        b1scale = t * float(np.max(np.abs(b1))) if b1.size else 0.0
        first_is_b1 = b1scale > b0scale
        if diag is not None:
            # diagonal operator: the action of every matrix function is local
            if t == 0.0:
                y = b0.copy()
            else:
                tl = -t * diag
                y = np.exp(tl) * b0 + (t * phi1_neg(tl)) * b1
        elif t == 0.0:
            y = b0.copy()
        else:
            y = np.zeros(n)
            box = [0.0]
            parts = []
            if first_is_b1:
                parts = [(1, b1), (0, b0)]
            else:
                parts = [(0, b0), (1, b1)]
            for kind, bv in parts:
                if not np.any(bv):
                    continue
                share = rdeadline - now
                pdeadline = now + share * (0.6 if len(parts) == 2 else 1.0)
                y += krylov_solve(A, bv, t, rtol, atol, kind,
                                  min(pdeadline, rdeadline), mmax, box)
        if not np.all(np.isfinite(y)):
            y = np.where(np.isfinite(y), y, 0.0)
        _write_all(out, np.ascontiguousarray(y, dtype="<f8").tobytes())
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
