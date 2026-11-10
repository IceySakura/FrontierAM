"""Affine union-find presolve and adjoint recovery on the equality forest."""
import ctypes
import os
from types import SimpleNamespace

import numpy as np
from scipy import sparse

_lib = None


def native():
    global _lib
    if _lib is None:
        path = os.path.join(os.path.dirname(__file__), "libldl.so")
        lib = ctypes.CDLL(path)
        ip, dp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
        bp = ctypes.POINTER(ctypes.c_ubyte)
        lib.affine_analyze.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, dp, dp,
                                      bp, ip, dp, dp]
        lib.affine_analyze.restype = ctypes.c_int
        lib.affine_dual.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, dp,
                                   ip, ip, dp, dp, dp]
        lib.affine_dual.restype = ctypes.c_int
        lib.sparse_basis.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, dp, bp, ip, ip]
        lib.sparse_basis.restype = ctypes.c_int
        _lib = lib
    return _lib


def numeric_basis(A, cheap):
    A = A.tocsr()
    cheap = np.ascontiguousarray(cheap, dtype=np.uint8)
    rows = np.empty(min(A.shape), dtype=np.int32)
    cols = np.empty_like(rows)
    rank = native().sparse_basis(A.shape[1], A.shape[0], ptr(A.indptr),
                                 ptr(A.indices), ptr(A.data), ptr(cheap),
                                 ptr(rows), ptr(cols))
    if rank < 0:
        raise RuntimeError("Sparse numerical basis")
    return rows[:rank], cols[:rank]


def ptr(a):
    t = ctypes.c_double if a.dtype == np.float64 else (
        ctypes.c_ubyte if a.dtype == np.uint8 else ctypes.c_int)
    return a.ctypes.data_as(ctypes.POINTER(t))


def affine_presolve(p):
    rows = np.flatnonzero(p.equal)
    if p.n < 300 or not rows.size:
        return None
    Ae = p.A[rows].tocsr()
    Ae.sort_indices()
    selected = np.zeros(rows.size, dtype=np.uint8)
    roots = np.empty(p.n, dtype=np.int32)
    weight, offset = np.empty(p.n), np.empty(p.n)
    bb = p.b[rows].copy()
    try:
        count = native().affine_analyze(p.n, rows.size, ptr(Ae.indptr), ptr(Ae.indices),
                                        ptr(Ae.data), ptr(bb), ptr(selected),
                                        ptr(roots), ptr(weight), ptr(offset))
    except (OSError, AttributeError):
        return None
    if count <= max(3, 0.005 * p.n):
        return None
    valid = roots >= 0
    rr = np.unique(roots[valid])
    mapping = np.full(p.n, -1, dtype=np.int32)
    mapping[rr] = np.arange(rr.size, dtype=np.int32)
    variables = np.flatnonzero(valid)
    T = sparse.csc_matrix((weight[valid], (variables, mapping[roots[valid]])),
                          shape=(p.n, rr.size))
    eliminated = rows[selected.astype(bool)]
    keep = np.ones(p.m, dtype=bool)
    keep[eliminated] = False
    keep = np.flatnonzero(keep)
    A = (p.A[keep] @ T).tocsc()
    b = p.b[keep] - p.A[keep] @ offset
    P = (T.T @ (p.P @ T)).tocsc()
    P = ((P + P.T) * 0.5).tocsc()
    Poffset = p.P @ offset
    q = np.asarray(T.T @ (p.q + Poffset))
    core_offset = (getattr(p, "core_offset", 0.0) + p.q @ offset
                   + 0.5 * offset @ Poffset)
    A.eliminate_zeros()
    P.eliminate_zeros()
    equal = p.equal[keep]
    removed = np.zeros(p.m, dtype=bool)
    removed[eliminated] = True
    blocks, start = [], 0
    for k, d, a, bb in p.blocks:
        width = bb - a - np.count_nonzero(removed[a:bb])
        if width:
            blocks.append((k, width if k == "zero" else d, start, start + width))
            start += width
    child = SimpleNamespace(A=A, b=b, P=P, q=q, equal=equal, blocks=blocks,
                            n=rr.size, m=keep.size, core_offset=float(core_offset))
    chosen = p.A[eliminated].tocsr()
    chosen.sort_indices()
    chosen_csc = chosen.tocsc()
    record = (p, T, offset, keep, eliminated, chosen, chosen_csc)
    return child, record


def affine_restore(result, records):
    status = result["status"]
    if status == "unresolved":
        return result
    for p, T, offset, keep, eliminated, R, Rc in reversed(records):
        if status in ("solved", "dual_infeasible"):
            x = T @ result["x"]
            if status == "solved":
                x += offset
            s = np.zeros(p.m)
            s[keep] = result["s"]
            result["x"], result["s"] = x, s
        if status in ("solved", "primal_infeasible"):
            z = np.zeros(p.m)
            z[keep] = result["z"]
            g = p.A.T @ z
            if status == "solved":
                g += p.P @ x + p.q
            zz = np.zeros(eliminated.size)
            error = native().affine_dual(p.n, eliminated.size, ptr(Rc.indptr),
                                         ptr(Rc.indices), ptr(Rc.data), ptr(R.indptr),
                                         ptr(R.indices), ptr(R.data), ptr(g), ptr(zz))
            if error:
                return {"status": "unresolved"}
            z[eliminated] = zz
            result["z"] = z
    return result
