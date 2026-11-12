"""Variable elimination and exact multiplier recovery for affine equalities."""

import ctypes
import os

import numpy as np
from scipy import sparse

from linear import library, ptrd


def ptri(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


class AffineReduction:
    def __init__(self, P, q, A, b, cones, eq):
        self.P0, self.q0, self.A0, self.b0 = P, q, A, b
        self.eq = eq
        self.lib = library()
        pi, pd, pv = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_void_p
        self.lib.affine_create.argtypes = [ctypes.c_int, ctypes.c_int, pi, pi, pd, pd]
        self.lib.affine_create.restype = pv
        self.lib.affine_destroy.argtypes = [pv]
        self.lib.affine_map.argtypes = [pv, pi, pd, pd, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.affine_map.restype = ctypes.c_int
        self.lib.affine_dual.argtypes = [pv, pd, pd]
        self.lib.affine_dual.restype = ctypes.c_int
        E = A[eq, :].tocsr()
        ap, ai = np.asarray(E.indptr, dtype=np.int32), np.asarray(E.indices, dtype=np.int32)
        bb = np.ascontiguousarray(b[eq])
        self.handle = self.lib.affine_create(len(q), len(eq), ptri(ap), ptri(ai), ptrd(E.data), ptrd(bb))
        if not self.handle:
            raise RuntimeError("Could not allocate affine presolve")
        roots, coef, offset = np.empty(len(q), dtype=np.int32), np.empty(len(q)), np.empty(len(q))
        removed = np.empty(len(eq), dtype=np.uint8)
        if self.lib.affine_map(self.handle, ptri(roots), ptrd(coef), ptrd(offset),
                               removed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))):
            raise RuntimeError("Affine presolve overflow")
        self.changed = bool(np.any(removed))
        if not self.changed:
            return
        free = roots >= 0
        rootmask = np.zeros(len(q), dtype=bool)
        rootmask[roots[free]] = True
        remap = np.cumsum(rootmask, dtype=np.int32) - 1
        self.T = sparse.csc_matrix((coef[free], (np.flatnonzero(free), remap[roots[free]])),
                                   shape=(len(q), int(rootmask.sum())))
        self.offset = offset
        keep = np.ones(len(b), dtype=bool)
        keep[eq[removed != 0]] = False
        self.keep = np.flatnonzero(keep)
        self.cones = []
        off = 0
        for c in cones:
            kind, d = c["type"], c["dim"]
            w = d * (d + 1) // 2 if kind == "psd" else d
            if kind == "zero":
                dim = int(np.count_nonzero(keep[off:off + w]))
                if dim:
                    self.cones.append({"type": kind, "dim": dim})
            else:
                self.cones.append(dict(c))
            off += w
        self.P = (self.T.T @ P @ self.T).tocsc()
        self.q = self.T.T @ (P @ offset + q)
        self.A = ((A @ self.T)[self.keep, :]).tocsc()
        self.b = (b - A @ offset)[self.keep]
        for mat in (self.P, self.A):
            mat.sum_duplicates()
            mat.eliminate_zeros()
            mat.sort_indices()
        if not all(np.isfinite(v).all() for v in (self.P.data, self.A.data, self.q, self.b)):
            raise RuntimeError("Affine transformed data overflow")

    def recover(self, result):
        status = result["status"]
        if status == "unresolved":
            return result
        out = {"status": status}
        if "x" in result:
            x = self.T @ result["x"]
            if status == "solved":
                x += self.offset
            s = np.zeros(len(self.b0))
            s[self.keep] = result["s"]
            out["x"], out["s"] = x, s
        if "z" in result:
            z = np.zeros(len(self.b0))
            z[self.keep] = result["z"]
            gradient = self.A0.T @ z
            if status == "solved":
                gradient += self.P0 @ out["x"] + self.q0
            gradient = np.ascontiguousarray(gradient)
            multipliers = np.empty(len(self.eq))
            if self.lib.affine_dual(self.handle, ptrd(gradient), ptrd(multipliers)):
                return {"status": "unresolved"}
            z[self.eq] += multipliers
            out["z"] = z
        return out

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.affine_destroy(self.handle)
            self.handle = None


def reduce(P, q, A, b, cones):
    mode = os.getenv("AFFINE_PRESOLVE", "linear")
    if mode == "0" or not len(q) or (mode == "linear" and P.nnz):
        return None
    if not P.nnz and all(c["type"] in ("zero", "nonnegative") for c in cones):
        return None
    parts, off = [], 0
    for c in cones:
        kind, d = c["type"], c["dim"]
        w = d * (d + 1) // 2 if kind == "psd" else d
        if kind == "zero":
            parts.append(np.arange(off, off + w))
        off += w
    if not parts:
        return None
    eq = np.concatenate(parts)
    counts = np.bincount(A.indices, minlength=len(b))[eq]
    if np.count_nonzero((counts > 0) & (counts <= 2)) < max(1, len(q) // 100):
        return None
    try:
        result = AffineReduction(P, q, A, b, cones, eq)
        return result if result.changed else None
    except (OSError, RuntimeError, ValueError, MemoryError):
        return None
