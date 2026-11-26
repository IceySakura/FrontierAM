#!/usr/bin/env python3
import os
import sys
import json
import math
import struct
import ctypes

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy.special as sps
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from numpy.polynomial.legendre import leggauss
HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(HERE, "fame_core.so"))
I32P = ctypes.POINTER(ctypes.c_int32)
DP = ctypes.POINTER(ctypes.c_double)
LDP = ctypes.POINTER(ctypes.c_longdouble)

lib.row_bound.argtypes = [ctypes.c_int64, I32P, I32P, DP, I32P]
lib.row_bound.restype = ctypes.c_double
lib.cheb_apply.argtypes = [ctypes.c_int64, I32P, I32P, DP, ctypes.c_double,
                           ctypes.c_int64, DP, DP, DP, DP, DP]
lib.cheb_apply_mix.argtypes = [ctypes.c_int64, I32P, I32P, DP, ctypes.c_double,
                               ctypes.c_int64, DP, LDP, DP, DP, DP]

stdin = sys.stdin.buffer
stdout = sys.stdout.buffer


def read_bytes(n):
    parts = bytearray()
    while len(parts) < n:
        chunk = stdin.read(n - len(parts))
        if not chunk:
            raise EOFError("unexpected end of input")
        parts.extend(chunk)
    return bytes(parts)


def read_array(dtype, count):
    arr = np.empty(count, dtype=dtype)
    view = memoryview(arr).cast("B")
    pos = 0
    while pos < view.nbytes:
        got = stdin.readinto(view[pos:])
        if not got:
            raise EOFError("unexpected end of input")
        pos += got
    return arr


def write_result(y):
    stdout.write(np.asarray(y, dtype="<f8").tobytes())
    stdout.flush()


def phi_scaled(t, lam):
    z = t * lam
    h = np.full_like(z, t, dtype=np.float64)
    nz = z > 0
    h[nz] = -t * np.expm1(-z[nz]) / z[nz]
    return np.exp(-z), h


def phi_scaled_ld(t, lam):
    z = np.longdouble(t) * lam
    h = np.full_like(z, np.longdouble(t))
    nz = z > 0
    h[nz] = -np.longdouble(t) * np.expm1(-z[nz]) / z[nz]
    return np.exp(-z), h


def composite_F(d, k):
    r = math.sqrt(d)
    edges = [0.0] + [e for e in (1.0, 10.0, 50.0, 100.0) if e < r] + [r]
    nodes = []
    weights = []
    for aa, bb in zip(edges[:-1], edges[1:]):
        if bb <= aa:
            continue
        ss, ww = leggauss(32)
        nodes.append((aa + bb) / 2 + (bb - aa) / 2 * ss)
        weights.append((bb - aa) / 2 * ww)
    s = np.concatenate(nodes)
    w = np.concatenate(weights)
    u = s * s
    vals = sps.ive(k[:, None], u[None, :])
    return (2 * s * vals) @ w


def make_coeffs(t, b, n, atol, high):
    eta = t * b
    d = eta / 2
    target = atol * 0.05
    if d == 0:
        return np.array([1.0]), np.array([t]), 0
    M = max(64, int(math.ceil(math.sqrt(max(eta, 1.0) * math.log(1.0 / target)) * 1.4 + 40)))
    sqrt_n = math.sqrt(n)
    while True:
        k = np.arange(M + 1)
        signs = np.where(k % 2 == 0, 1.0, -1.0)
        ive_d = sps.ive(k, d)
        ce = signs * 2 * ive_d
        ce[0] *= 0.5
        F = composite_F(d, k)
        ch = signs * (4.0 / b) * F
        ch[0] *= 0.5
        te = np.cumsum(np.abs(ce)[::-1])[::-1]
        th = np.cumsum(np.abs(ch)[::-1])[::-1]
        bound = (te[1:] + th[1:]) * sqrt_n
        idx = np.flatnonzero(bound <= target)
        if len(idx):
            m = int(idx[0])
            return ce[:m + 1].copy(), ch[:m + 1].copy(), m
        M = int(M * 1.5) + 20
        if M > 20000:
            m = M - 1
            return ce[:m + 1].copy(), ch[:m + 1].copy(), m


def exact_diagonal(state, b0, b1):
    return state["f"] * b0 + state["h"] * b1


def prepare_small(L, t):
    n = L.shape[0]
    labels, sizes = None, None
    ncomp, labels = connected_components(L, directed=False, return_labels=True)
    sizes = np.bincount(labels)
    maxs = int(sizes.max())
    if maxs <= 2:
        diag = L.diagonal()
        f, h = phi_scaled_ld(t, diag.astype(np.longdouble))
        rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(L.indptr))
        mask = L.indices > rows
        i = rows[mask].astype(np.int64)
        j = L.indices[mask].astype(np.int64)
        c = L.data[mask].astype(np.longdouble)
        a = diag[i].astype(np.longdouble)
        d = diag[j].astype(np.longdouble)
        tr = (a + d) / 2
        diff = (a - d) / 2
        r = np.sqrt(diff * diff + c * c)
        f1, h1 = phi_scaled_ld(t, tr - r)
        f2, h2 = phi_scaled_ld(t, tr + r)
        ae = (f1 + f2) / 2
        ah = (h1 + h2) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            be = np.where(r > 0, (f2 - f1) / (2 * r), 0.0)
            bh = np.where(r > 0, (h2 - h1) / (2 * r), 0.0)
        return {"kind": "pair", "f": f, "h": h, "i": i, "j": j,
                "c": c, "diff": diff, "ae": ae, "be": be, "ah": ah, "bh": bh}
    if maxs <= 64:
        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(ncomp + 1))
        blocks = []
        for cid in range(ncomp):
            idx = order[bounds[cid]:bounds[cid + 1]]
            sub = L[idx][:, idx].toarray()
            w, V = np.linalg.eigh(sub)
            f, h = phi_scaled(t, w)
            blocks.append((idx, V, f, h))
        return {"kind": "blocks", "blocks": blocks}
    return None


def solve_small(state, b0, b1):
    if state["kind"] == "pair":
        y = state["f"] * b0 + state["h"] * b1
        i = state["i"]; j = state["j"]; c = state["c"]; diff = state["diff"]
        ae = state["ae"]; be = state["be"]; ah = state["ah"]; bh = state["bh"]
        y[i] = ae * b0[i] + be * (diff * b0[i] + c * b0[j]) + ah * b1[i] + bh * (diff * b1[i] + c * b1[j])
        y[j] = ae * b0[j] + be * (c * b0[i] - diff * b0[j]) + ah * b1[j] + bh * (c * b1[i] - diff * b1[j])
        return np.asarray(y, dtype=np.float64)
    y = np.empty_like(b0)
    for idx, V, f, h in state["blocks"]:
        bb0 = b0[idx]; bb1 = b1[idx]
        y[idx] = V @ (f * (V.T @ bb0)) + V @ (h * (V.T @ bb1))
    return y


def main():
    stdout.write(b"READY\n")
    stdout.flush()
    magic = read_bytes(8)
    if magic != b"FAMEXP2\n":
        raise ValueError("bad magic")
    hlen = struct.unpack("<I", read_bytes(4))[0]
    h = json.loads(read_bytes(hlen).decode("utf-8"))
    n = int(h["n"]); nnz = int(h["nnz"]); requests = int(h["requests"])
    t = float(h["t"]); rtol = float(h["rtol"]); atol = float(h["atol"])
    indptr64 = read_array("<u8", n + 1)
    indices = read_array("<u4", nnz)
    data = read_array("<f8", nnz)
    indptr = indptr64.astype(np.int32)
    del indptr64
    indices32 = indices.astype(np.int32)
    del indices

    if t == 0:
        for _ in range(requests):
            b0 = read_array("<f8", n)
            read_array("<f8", n)
            write_result(b0)
        return

    diag_flag = ctypes.c_int32(0)
    b = lib.row_bound(n, indptr.ctypes.data_as(I32P), indices32.ctypes.data_as(I32P),
                      data.ctypes.data_as(DP), ctypes.byref(diag_flag))
    if b == 0:
        for _ in range(requests):
            b0 = read_array("<f8", n)
            b1 = read_array("<f8", n)
            write_result(b0 + t * b1)
        return

    if diag_flag.value:
        f, hcoef = phi_scaled(t, data)
        state = {"f": f, "h": hcoef}
        for _ in range(requests):
            b0 = read_array("<f8", n)
            b1 = read_array("<f8", n)
            write_result(exact_diagonal(state, b0, b1))
        return

    small = None
    if nnz <= 4 * n or n <= 2000:
        L = sp.csr_matrix((data, indices32, indptr), shape=(n, n), copy=False)
        small = prepare_small(L, t)
    if small is not None:
        for _ in range(requests):
            b0 = read_array("<f8", n)
            b1 = read_array("<f8", n)
            write_result(solve_small(small, b0, b1))
        return

    high = rtol <= 1e-10
    ce, ch, m = make_coeffs(t, b, n, atol, high)
    if high:
        ch_ld = ch.astype(np.longdouble)
        for _ in range(requests):
            b0 = read_array("<f8", n)
            b1 = read_array("<f8", n)
            y = np.empty(n, dtype=np.float64)
            lib.cheb_apply_mix(n, indptr.ctypes.data_as(I32P), indices32.ctypes.data_as(I32P),
                               data.ctypes.data_as(DP), b, m,
                               ce.ctypes.data_as(DP), ch_ld.ctypes.data_as(LDP),
                               b0.ctypes.data_as(DP), b1.ctypes.data_as(DP),
                               y.ctypes.data_as(DP))
            write_result(y)
    else:
        for _ in range(requests):
            b0 = read_array("<f8", n)
            b1 = read_array("<f8", n)
            y = np.empty(n, dtype=np.float64)
            lib.cheb_apply(n, indptr.ctypes.data_as(I32P), indices32.ctypes.data_as(I32P),
                           data.ctypes.data_as(DP), b, m,
                           ce.ctypes.data_as(DP), ch.ctypes.data_as(DP),
                           b0.ctypes.data_as(DP), b1.ctypes.data_as(DP),
                           y.ctypes.data_as(DP))
            write_result(y)


if __name__ == "__main__":
    main()
