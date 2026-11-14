#!/usr/bin/env python3
"""Sparse exponential propagation for the FAMEXP2 session interface.

Session problem
---------------
Given a real symmetric positive semidefinite sparse L, a time t and an error
tolerance, answer one or sixteen requests

    y = exp(-t L) b0 + t * phi1(-t L) b1,      phi1(z) = (e^z - 1)/z,

the state at s = t of u'(s) = -L u(s) + b1, u(0) = b0.  With
f0(x) = exp(-t x) and f1(x) = (1 - exp(-t x))/x (f1(0) = t) this is simply

    y = f0(L) b0 + f1(L) b1.

Method
------
f0 and f1 are entire, so both are expanded in Chebyshev series on a spectral
interval [a, b] containing the spectrum of L and evaluated with the three-term
matrix recursion

    T_0 v = v,   T_1 v = H v,   T_{k+1} v = 2 H T_k v - T_{k-1} v,
    H = (2 L - (a + b) I) / (b - a),

one sparse matrix-vector product per step serving both right-hand sides at
once.  For exp(-x) on [0, eta] the Chebyshev coefficients behave like
exp(-k^2 / eta), so the degree grows only like sqrt(t * lambda_max * log(1/eps)):
a few thousand matvecs at the stiffest admissible session (t*||L||_inf = 1e5)
and a handful at the non-stiff end.  No substepping, no matrix factorisation,
no assumption beyond the given spectral range -- singular L, zero L, t = 0,
positive off-diagonal entries and zero right-hand sides are all handled by the
same code path.

Interval: b comes from a power iteration on L capped by the rigorous bound
||L||_inf; a is the Gershgorin lower bound when that is positive (a genuine
spectral gap makes the series converge much faster), otherwise a small negative
pad absorbing rounding-level negative eigenvalues.

Truncation: because |T_k| <= 1 on [-1, 1], the discarded tail obeys the
rigorous bound ||sum_{k>d} c_k T_k(H) b||_2 <= (sum_{k>d} |c_k|) * ||b||_2.
The kernel stops as soon as that bound falls below 0.1 * (atol + rtol * max|y|),
so the degree adapts to the realised size of the answer.

Reuse: L, t and the tolerance are fixed for the session, so the spectral
interval, the Chebyshev coefficients, their tails and every work buffer are
computed once before the first request and shared by all sixteen pairs.

Only wire reading is done in Python; the inner loops live in expkernel.c and
are reached through ctypes.  A NumPy/SciPy path implements the identical
algorithm and is used automatically if the shared object cannot be loaded.
"""

import ctypes
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MAGIC = b"FAMEXP2\n"
MAX_DEGREE = 20000          # hard cap on the Chebyshev degree
LAMBDA_PAD = 1e-3           # relative safety margin on the lambda_max estimate
LOW_PAD = 1e-4              # relative pad below the spectral lower bound
STOP_FRAC = 0.1             # truncation error aimed at, as a fraction of tol

_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int32)
_Q = ctypes.POINTER(ctypes.c_int64)
_NULL_D = _D()


def _p(arr, kind):
    return arr.ctypes.data_as(kind)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
def _load_kernel():
    path = os.path.join(HERE, "libexp.so")
    if not os.path.exists(path):
        return None
    try:
        lib = ctypes.CDLL(path)
        prep, spmv, cheb = lib.exp_prep, lib.exp_spmv, lib.exp_cheb
    except (OSError, AttributeError):
        return None
    prep.argtypes = [_Q, _I, _D, ctypes.c_int64, _D]
    prep.restype = None
    spmv.argtypes = [_Q, _I, _D, ctypes.c_int64, _D, _D]
    spmv.restype = None
    cheb.argtypes = [_Q, _I, _D, ctypes.c_int64, ctypes.c_double, ctypes.c_double,
                     _D, _D, ctypes.c_int64, _D, _D, _D, _D,
                     ctypes.c_double, ctypes.c_double,
                     ctypes.c_double, ctypes.c_double, ctypes.c_double,
                     _D, _D, ctypes.POINTER(ctypes.c_int64), _D]
    cheb.restype = ctypes.c_int64
    return prep, spmv, cheb


KERNEL = _load_kernel()


# --------------------------------------------------------------------------
# wire protocol
# --------------------------------------------------------------------------
def _read_exact(stream, count):
    buf = stream.read(count)
    while len(buf) < count:
        chunk = stream.read(count - len(buf))
        if not chunk:
            raise EOFError("truncated session input")
        buf += chunk
    return buf


# --------------------------------------------------------------------------
# Chebyshev coefficients of f0 and f1 on [a, b]
# --------------------------------------------------------------------------
def _coefficients(a, b, t, npts):
    from scipy.fft import dct

    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    u = np.cos(np.pi * np.arange(npts + 1) / npts)          # +1 ... -1
    x = mid + half * u
    s = t * x
    with np.errstate(over="ignore", under="ignore"):
        f0 = np.exp(-s)
    small = np.abs(s) < 1e-8
    xs = np.where(small, 1.0, x)
    ss = np.where(small, 1.0, s)
    f1 = np.where(small, t * (1.0 - 0.5 * s + s * s / 6.0), -np.expm1(-ss) / xs)
    c0 = dct(f0, type=1) / npts
    c1 = dct(f1, type=1) / npts
    for c in (c0, c1):
        c[0] *= 0.5
        c[-1] *= 0.5
    return c0, c1


class Session:
    """Everything that depends only on (L, t, tolerance) -- built once."""

    def __init__(self, indptr, indices, data, n, t, rtol, atol):
        self.n, self.t, self.rtol, self.atol = n, t, rtol, atol
        self.indptr, self.indices, self.data = indptr, indices, data
        self.lib = KERNEL
        self.last_degree = 0
        self._work = None
        self._csr = None

        if t == 0.0:
            self.mode = "zero_time"
            return
        pinf, gersh = self._prep()
        if pinf <= 0.0:
            self.mode = "zero_operator"
            return
        self.mode = "chebyshev"

        lo = gersh if gersh > 1e-12 * pinf else 0.0
        hi = self._lambda_max(pinf)
        if lo > 0.0:
            a = lo - LOW_PAD * hi
        else:
            a = -LOW_PAD * hi
        b = hi * (1.0 + LAMBDA_PAD)
        if not np.isfinite(a) or not np.isfinite(b) or not b > a:
            a, b = -LOW_PAD * pinf, pinf * (1.0 + LAMBDA_PAD)
        self.a, self.b = a, b
        self.scale = 2.0 / (b - a)
        self.shift = (a + b) / (b - a)
        self._build_coefficients()

    # -- operator statistics ---------------------------------------------
    def _prep(self):
        out = np.zeros(2)
        if self.lib is not None:
            self.lib[0](_p(self.indptr, _Q), _p(self.indices, _I),
                        _p(self.data, _D), self.n, _p(out, _D))
            return float(out[0]), float(out[1])
        rs = self._rowsums_abs()
        pinf = float(rs.max()) if rs.size else 0.0
        dg = self._diagonal()
        return pinf, float(np.min(dg - (rs - np.abs(dg))))

    def _rowsums_abs(self):
        n, indptr, data = self.n, self.indptr, self.data
        rs = np.zeros(n)
        if data.size == 0:
            return rs
        counts = np.diff(indptr)
        rowid = np.repeat(np.arange(n, dtype=np.int64), counts)
        np.add.at(rs, rowid, np.abs(data))
        return rs

    def _diagonal(self):
        n, indptr, indices, data = self.n, self.indptr, self.indices, self.data
        dg = np.zeros(n)
        if data.size == 0:
            return dg
        ar = np.arange(n, dtype=np.int32)
        lo, hi = indptr[:-1], indptr[1:]
        has = hi > lo
        idx = np.flatnonzero(has)
        for i in idx:
            s, e = int(lo[i]), int(hi[i])
            pos = np.searchsorted(indices[s:e], i)
            if pos < e - s and indices[s + pos] == i:
                dg[i] = data[s + pos]
        return dg

    def _spmv(self, x, out=None):
        if self.lib is not None:
            y = np.empty(self.n) if out is None else out
            self.lib[1](_p(self.indptr, _Q), _p(self.indices, _I),
                        _p(self.data, _D), self.n, _p(x, _D), _p(y, _D))
            return y
        if self._csr is None:
            from scipy.sparse import csr_matrix
            self._csr = csr_matrix((self.data, self.indices, self.indptr),
                                   shape=(self.n, self.n))
        r = self._csr @ x
        if out is None:
            return r
        out[:] = r
        return out

    def _lambda_max(self, pinf):
        """Power iteration, capped by the rigorous ||L||_inf bound."""
        n = self.n
        v = np.random.default_rng(20240517).standard_normal(n)
        w = np.empty(n)
        nv = float(np.linalg.norm(v))
        if nv == 0.0:
            return pinf
        v /= nv
        iters = 60 if self.t * pinf > 1.0 else 10
        rho, rho_prev, stable = pinf, 0.0, 0
        for _ in range(iters):
            self._spmv(v, w)
            vv = float(np.dot(v, v))
            rho = float(np.dot(v, w)) / vv if vv > 0.0 else 0.0
            nw = float(np.linalg.norm(w))
            if nw == 0.0:
                return pinf
            v, w = w, v
            v /= nw
            if rho_prev > 0.0 and abs(rho - rho_prev) <= 1e-13 * rho:
                stable += 1
            else:
                stable = 0
            rho_prev = rho
            if stable >= 3:
                break
        if not np.isfinite(rho) or rho <= 0.0:
            return pinf
        return min(pinf, rho * (1.0 + LAMBDA_PAD))

    # -- series ------------------------------------------------------------
    def _build_coefficients(self):
        npts = 1024
        c0 = c1 = None
        for _ in range(5):
            c0, c1 = _coefficients(self.a, self.b, self.t, npts)
            d = self._degree(c0, c1)
            if d <= npts // 3:
                break
            npts *= 4
        d = min(d, MAX_DEGREE)
        self.dmax = d
        c0 = np.ascontiguousarray(c0[:d + 2])
        c1 = np.ascontiguousarray(c1[:d + 2])
        if c0.size < d + 3:
            c0 = np.pad(c0, (0, d + 3 - c0.size))
            c1 = np.pad(c1, (0, d + 3 - c1.size))
        self.c0, self.c1 = c0, c1
        self.tail0 = self._tail(c0, d)
        self.tail1 = self._tail(c1, d)
        self.s0 = float(np.abs(c0).sum())
        self.s1 = float(np.abs(c1).sum())

    @staticmethod
    def _tail(c, d):
        t = np.zeros(d + 3)
        seg = np.abs(c[:d + 2])
        t[:d + 2] = np.cumsum(seg[::-1])[::-1]
        return np.ascontiguousarray(t)

    def _degree(self, c0, c1):
        """Degree whose discarded tail is at double-precision noise level."""
        lim = min(c0.size // 2, MAX_DEGREE)
        a0 = np.abs(c0[:lim + 2])
        a1 = np.abs(c1[:lim + 2])
        t0 = np.cumsum(a0[::-1])[::-1]
        t1 = np.cumsum(a1[::-1])[::-1]
        target = max(STOP_FRAC * self.atol,
                     1e-17 * (float(a0.sum()) + float(a1.sum())))
        est = t0 + t1
        hit = np.flatnonzero(est[1:] <= target)
        return int(hit[0]) + 1 if hit.size else lim

    # -- one request -------------------------------------------------------
    def solve(self, b0, b1):
        n, t = self.n, self.t
        if self.mode == "zero_time":
            return np.array(b0, dtype="<f8")
        if self.mode == "zero_operator":
            return np.asarray(b0, dtype="<f8") + t * np.asarray(b1, dtype="<f8")
        b0 = np.ascontiguousarray(b0, dtype="<f8")
        b1 = np.ascontiguousarray(b1, dtype="<f8")
        nz0 = bool(np.any(b0))
        nz1 = bool(np.any(b1))
        if not nz0 and not nz1:
            return np.zeros(n, dtype="<f8")
        if nz0 and nz1:
            return self._run(b0, b1, self.c0, self.c1, self.tail0, self.tail1,
                             self.s0 + self.s1)
        if nz0:
            return self._run(b0, None, self.c0, self.c0, self.tail0, self.tail0,
                             self.s0)
        return self._run(b1, None, self.c1, self.c1, self.tail1, self.tail1,
                         self.s1)

    def _run(self, v0, v1, c0, c1, t0, t1, cscale):
        n = self.n
        two = v1 is not None
        nb0 = float(np.linalg.norm(v0))
        nb1 = float(np.linalg.norm(v1)) if two else 0.0
        sa = STOP_FRAC * self.atol
        sr = STOP_FRAC * self.rtol
        tfloor = 1e-18 * cscale * max(nb0, nb1, 1.0)
        y = np.empty(n, dtype="<f8")
        if self.lib is not None:
            need = 4 * n * (2 if two else 1)
            if self._work is None or self._work.size < need:
                self._work = np.empty(need)
            used = ctypes.c_int64(0)
            ym = ctypes.c_double(0.0)
            self.lib[2](_p(self.indptr, _Q), _p(self.indices, _I),
                        _p(self.data, _D), n, self.scale, self.shift,
                        _p(c0, _D), _p(c1, _D), self.dmax,
                        _p(t0, _D), _p(t1, _D),
                        _p(v0, _D), _p(v1, _D) if two else _NULL_D,
                        nb0, nb1, sa, sr, tfloor,
                        _p(y, _D), _p(self._work, _D),
                        ctypes.byref(used), _p(np.array([0.0]), _D))
            self.last_degree = int(used.value)
            return y
        return self._run_numpy(v0, v1 if two else None, c0, c1, t0, t1,
                               nb0, nb1, sa, sr, tfloor)

    def _hop(self, v, out):
        """out = H v = scale*(L v) - shift*v"""
        self._spmv(v, out)
        out *= self.scale
        out -= self.shift * v

    def _run_numpy(self, v0, v1, c0, c1, t0, t1, nb0, nb1, sa, sr, tfloor):
        A0 = v0.copy()
        y = c0[0] * A0
        if v1 is not None:
            A1 = v1.copy()
            y += c1[0] * A1
        else:
            A1 = None
        B0 = np.empty(self.n)
        self._hop(A0, B0)
        y += c0[1] * B0
        if A1 is not None:
            B1 = np.empty(self.n)
            self._hop(A1, B1)
            y += c1[1] * B1
        W = np.empty(self.n)
        k = 1
        while k < self.dmax:
            est = t0[k + 1] * nb0 + (t1[k + 1] * nb1 if A1 is not None else 0.0)
            if est <= sa + sr * float(np.max(np.abs(y))) or est <= tfloor:
                break
            self._hop(B0, W)
            C0 = W.copy()
            C0 *= 2.0
            C0 -= A0
            y += c0[k + 1] * C0
            A0, B0 = B0, C0
            if A1 is not None:
                self._hop(B1, W)
                C1 = W.copy()
                C1 *= 2.0
                C1 -= A1
                y += c1[k + 1] * C1
                A1, B1 = B1, C1
            k += 1
        self.last_degree = k
        return y


# --------------------------------------------------------------------------
def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    stdout.write(b"READY\n")
    stdout.flush()

    if _read_exact(stdin, len(MAGIC)) != MAGIC:
        raise SystemExit("bad wire magic")
    ln = struct.unpack("<I", _read_exact(stdin, 4))[0]
    meta = json.loads(_read_exact(stdin, ln).decode("utf-8"))
    n, nnz = int(meta["n"]), int(meta["nnz"])
    requests = int(meta["requests"])

    indptr = np.frombuffer(_read_exact(stdin, 8 * (n + 1)), dtype="<i8")
    indices = np.frombuffer(_read_exact(stdin, 4 * nnz), dtype="<i4")
    data = np.frombuffer(_read_exact(stdin, 8 * nnz), dtype="<f8")

    sess = Session(np.ascontiguousarray(indptr), np.ascontiguousarray(indices),
                   np.ascontiguousarray(data), n, float(meta["t"]),
                   float(meta["rtol"]), float(meta["atol"]))

    for _ in range(requests):
        b0 = np.frombuffer(_read_exact(stdin, 8 * n), dtype="<f8")
        b1 = np.frombuffer(_read_exact(stdin, 8 * n), dtype="<f8")
        y = sess.solve(b0, b1)
        stdout.write(np.ascontiguousarray(y, dtype="<f8").tobytes())
        stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
