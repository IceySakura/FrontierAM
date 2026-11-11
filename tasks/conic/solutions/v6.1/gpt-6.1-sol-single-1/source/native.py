"""Small single-threaded kernels; all inputs use explicit fixed-width arrays."""

import ctypes
import os
import numpy as np


LIB = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libkernels.so"))
LIB.affine_product.restype = ctypes.c_int64


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def affine_product(A, T, keep, capacity):
    ad, ai, ap, ac = A
    tp, ti, tv = T
    ri, ci = np.empty(capacity, np.int32), np.empty(capacity, np.int32)
    val = np.empty(capacity)
    used = LIB.affine_product(
        ctypes.c_int(len(ap) - 1), ptr(ap.astype(np.int64, copy=False)),
        ptr(ai.astype(np.int32, copy=False)), ptr(ad),
        ptr(tp), ptr(ti), ptr(tv), ptr(keep),
        ptr(ri), ptr(ci), ptr(val))
    return ri[:used], ci[:used], val[:used]


class Derivatives:
    def __init__(self, mp, mi, mv, cp, ci, cv, B):
        self.mp = np.asarray(mp, dtype=np.int64)
        self.mi = np.asarray(mi, dtype=np.int32)
        self.mv = np.asarray(mv, dtype=np.float64)
        self.cp = np.asarray(cp, dtype=np.int64)
        self.ci = np.asarray(ci, dtype=np.int32)
        self.cv = np.asarray(cv, dtype=np.float64)
        self.B = np.ascontiguousarray(B, dtype=np.float64)
        self.m, self.n = B.shape
        self.terms = len(mp) - 1

    def __call__(self, ev, weight):
        J, H = np.empty_like(self.B), np.empty((self.n, self.n))
        LIB.gp_derivatives(
            ctypes.c_int(self.terms), ctypes.c_int(self.n), ctypes.c_int(self.m),
            ptr(self.mp), ptr(self.mi), ptr(self.mv),
            ptr(self.cp), ptr(self.ci), ptr(self.cv),
            ptr(ev), ptr(weight), ptr(self.B), ptr(J), ptr(H))
        return J, H

    def evaluate(self, x, s, z, qb, qo, offset, yc, rhs, target):
        theta, ev, weight = (np.empty(self.terms) for _ in range(3))
        rd, rp, rc = np.empty(self.n), np.empty(self.m), np.empty(self.m)
        status = LIB.gp_evaluate(
            ctypes.c_int(self.terms), ctypes.c_int(self.n), ctypes.c_int(self.m),
            ptr(self.mp), ptr(self.mi), ptr(self.mv), ptr(self.cp), ptr(self.ci), ptr(self.cv),
            ptr(self.B), ptr(x), ptr(s), ptr(z), ptr(qb), ptr(qo), ptr(offset),
            ptr(yc), ptr(rhs), ctypes.c_double(target),
            ptr(theta), ptr(ev), ptr(weight), ptr(rd), ptr(rp), ptr(rc))
        if status:
            raise FloatingPointError("GP exponential overflow")
        return theta, ev, weight, rd, rp, rc


class Lasso:
    def __init__(self, F, diagonal, positive, negative, rhs):
        self.ptr = F.indptr.astype(np.int64)
        self.rows = F.indices.astype(np.int32, copy=False)
        self.data = F.data
        self.diagonal = np.ascontiguousarray(diagonal)
        self.weighted = self.data * self.diagonal[self.rows]
        self.norm = np.bincount(
            np.repeat(np.arange(F.shape[1]), np.diff(self.ptr)),
            weights=self.data * self.weighted, minlength=F.shape[1])
        self.positive, self.negative = np.ascontiguousarray(positive), np.ascontiguousarray(negative)
        self.residual = np.array(rhs, dtype=float, copy=True)
        self.x = np.zeros(F.shape[1])
        self.n, self.m = F.shape[1], F.shape[0]

    def run(self, sweeps):
        LIB.lasso_sweeps(
            ctypes.c_int(self.n), ptr(self.ptr), ptr(self.rows), ptr(self.data),
            ptr(self.weighted), ptr(self.norm), ptr(self.positive), ptr(self.negative),
            ptr(self.x), ptr(self.residual), ctypes.c_int(sweeps))

    def metrics(self):
        out = np.empty(3)
        LIB.lasso_metrics(
            ctypes.c_int(self.n), ctypes.c_int(self.m), ptr(self.ptr), ptr(self.rows),
            ptr(self.weighted), ptr(self.diagonal), ptr(self.positive), ptr(self.negative),
            ptr(self.x), ptr(self.residual), ptr(out))
        return out
