#!/usr/bin/env python3
"""FAMEXP2 solver: exact tiny-block propagation or Chebyshev propagation.

Protocol (see API.md): after READY, read header + full CSR + K requests of
(b0, b1), writing exactly one n-vector of float64 per request.

Main path.  L is real symmetric PSD, so its spectrum lies in [0, lmax] with
lmax a rigorous upper bound obtained from a Collatz-Wielandt estimate of
rho(|L|).  With S = 2L/lmax - I (spectrum in [-1, 1]) and a = t*lmax/2,

    exp(-t x)     = sum_k c_k T_k(2x/lmax - 1),
    t phi1(-t x)  = sum_k d_k T_k(2x/lmax - 1),

    c_0 = ive(0,a),  c_k = 2(-1)^k ive(k,a),
    d_0 = (2/lmax) J_0,  d_k = (4/lmax)(-1)^k J_k,
    J_k = int_0^a ive(k,u) du.

J is evaluated by two cumulative sums of the (positive) Bessel sequence,
equivalent to the stable backward second-difference recurrence.  Both
functions share ONE Clenshaw recurrence (source c_k*b0 + d_k*b1), so each
degree costs a single sparse matvec.

Exact path.  If the sparsity graph splits into many tiny independent blocks
(diagonal / log-spectrum operators), each block is diagonalised once and the
matrix function is then applied exactly in O(nnz) per request.
"""
import ctypes
import json
import math
import os
import struct
import sys

import numpy as np
from scipy import special

# ---------------------------------------------------------------- library
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_library():
    path = os.path.join(_HERE, "libfamexp.so")
    if not os.path.exists(path):
        os.system("cd \"%s\" && sh build.sh >/dev/null 2>&1" % _HERE)
    return ctypes.CDLL(path)


_lib = _load_library()
_PU64 = ctypes.POINTER(ctypes.c_int64)
_PU32 = ctypes.POINTER(ctypes.c_uint32)
_PD = ctypes.POINTER(ctypes.c_double)
_PL = ctypes.POINTER(ctypes.c_longdouble)
_PI32 = ctypes.POINTER(ctypes.c_int32)
_PI = ctypes.POINTER(ctypes.c_int)

_lib.famexp_lmax_bound.argtypes = [_PU64, _PU32, _PD, ctypes.c_int,
                                   ctypes.c_int, _PD]
_lib.famexp_lmax_bound.restype = ctypes.c_double
_lib.famexp_pattern_stats.argtypes = [_PU64, _PU32, ctypes.c_int, _PI, _PI]
_lib.famexp_components.argtypes = [_PU64, _PU32, ctypes.c_int, _PI32, _PI32, _PI]
_lib.famexp_components.restype = ctypes.c_int
_CLENSHAW_ARGS = [_PU64, _PU32, _PD, ctypes.c_int, ctypes.c_int,
                 _PD, _PD, _PD, _PD, _PD, _PD]
_lib.famexp_clenshaw_d.argtypes = _CLENSHAW_ARGS
_lib.famexp_clenshaw_d4.argtypes = _CLENSHAW_ARGS
_lib.famexp_clenshaw_d5.argtypes = _CLENSHAW_ARGS
_lib.famexp_clenshaw_l.argtypes = [_PU64, _PU32, _PD, ctypes.c_int, ctypes.c_int,
                                   _PD, _PD, _PD, _PD, _PD, _PL]
_lib.famexp_scale.argtypes = [_PD, ctypes.c_int, ctypes.c_double]


def _ptr(arr, typ):
    return ctypes.cast(arr.ctypes.data, typ)


# ---------------------------------------------------------------- wire I/O
def _read_exact(stream, count):
    buf = bytearray(count)
    view = memoryview(buf)
    pos = 0
    while pos < count:
        got = stream.readinto(view[pos:])
        if not got:
            raise EOFError("unexpected end of input")
        pos += got
    return buf


def read_header(stream):
    if bytes(_read_exact(stream, 8)) != b"FAMEXP2\n":
        raise ValueError("bad magic")
    (length,) = struct.unpack("<I", _read_exact(stream, 4))
    meta = json.loads(bytes(_read_exact(stream, length)).decode("utf-8"))
    n, nnz = int(meta["n"]), int(meta["nnz"])
    indptr = np.frombuffer(_read_exact(stream, 8 * (n + 1)), dtype="<u8")
    indices = np.frombuffer(_read_exact(stream, 4 * nnz), dtype="<u4")
    data_bytes = _read_exact(stream, 8 * nnz)
    data = np.frombuffer(data_bytes, dtype="<f8")
    return meta, indptr, indices, data


# ---------------------------------------------------------------- polynomials
def make_coefficients(t, lmax, K):
    """Chebyshev coefficients of exp(-t x) and t*phi1(-t x) on [0, lmax]."""
    a = 0.5 * t * lmax
    M = K + 96
    kk = np.arange(M + 2, dtype=np.float64)
    iv = special.ive(kk, a).astype(np.longdouble)
    sign = np.ones(K + 1, dtype=np.float64)
    sign[1::2] = -1.0

    c = np.empty(K + 1, dtype=np.float64)
    c[0] = float(iv[0])
    c[1:] = 2.0 * sign[1:] * np.asarray(iv[1:K + 1], dtype=np.float64)

    # J_k via J_{k-1} - 2J_k + J_{k+1} = 2 ive(k,a), J_M = J_{M+1} = 0.
    tail = np.cumsum(iv[::-1])[::-1]
    delta = 2.0 * (tail - iv)
    delta[M:] = 0.0
    J = np.cumsum(delta[::-1])[::-1]

    d = np.empty(K + 1, dtype=np.float64)
    d[0] = float((np.longdouble(2.0) / np.longdouble(lmax)) * J[0])
    d[1:] = (4.0 / lmax) * sign[1:] * np.asarray(J[1:K + 1], dtype=np.float64)
    return c, d


def _tails(c, d):
    tc = np.cumsum(np.abs(c)[::-1])[::-1]
    td = np.cumsum(np.abs(d)[::-1])[::-1]
    K = len(c) - 1
    out_c = np.zeros(K + 1, dtype=np.float64)
    out_d = np.zeros(K + 1, dtype=np.float64)
    out_c[:K] = tc[1:]
    out_d[:K] = td[1:]
    return out_c, out_d


# ---------------------------------------------------------------- small blocks
def _scalar_functions(z, t):
    """exp(-z) and (1-exp(-z))*t/z, stable at z=0 and for negative z."""
    f0 = np.exp(-z)
    f1 = np.empty_like(z)
    small = np.abs(z) < 1e-8
    zs = z[small]
    f1[small] = t * (1.0 - 0.5 * zs + zs * zs / 6.0)
    zn = z[~small]
    f1[~small] = -np.expm1(-zn) * t / zn
    return f0, f1


def _try_diagonal(n, indptr, indices, data, t):
    """Direct O(n) evaluation when L is diagonal."""
    counts = np.diff(indptr)
    rows = np.flatnonzero(counts).astype(np.int64)
    if rows.size != indices.size or not np.array_equal(indices, rows):
        return None
    diag = np.zeros(n, dtype=np.float64)
    diag[rows] = data
    f0, f1 = _scalar_functions(t * diag, t)
    return (np.ascontiguousarray(f0), np.ascontiguousarray(f1))


def _try_exact_blocks(n, nnz, indptr, indices, data, t):
    """Return a list of (rows, V, f0, f1) groups, or None if inapplicable."""
    label = np.empty(n, dtype=np.int32)
    sizes = np.empty(n, dtype=np.int32)
    maxsize = ctypes.c_int()
    nc = _lib.famexp_components(indptr.ctypes.data_as(_PU64),
                                indices.ctypes.data_as(_PU32), n,
                                label.ctypes.data_as(_PI32),
                                sizes.ctypes.data_as(_PI32),
                                ctypes.byref(maxsize))
    if nc < 0 or maxsize.value > 32:
        return None
    sizes = sizes[:nc].astype(np.int64)
    total_cubic = float(np.sum(sizes.astype(np.float64) ** 3))
    if total_cubic > 6e7:
        return None

    order = np.argsort(label, kind="stable")
    starts = np.concatenate(([0], np.cumsum(sizes)))
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n, dtype=np.int64)
    inv = pos - starts[label]

    bmax = int(maxsize.value)
    Aflat = np.zeros((n, bmax), dtype=np.float64)
    counts = np.diff(indptr)
    row_of_nnz = np.repeat(np.arange(n, dtype=np.int32), counts)
    Aflat[row_of_nnz, inv[indices]] = data

    groups = []
    for b in np.unique(sizes):
        b = int(b)
        comp_ids = np.flatnonzero(sizes == b)
        rows = order[starts[comp_ids][:, None] + np.arange(b)[None, :]]
        blocks = Aflat[rows, :b]
        blocks = 0.5 * (blocks + blocks.transpose(0, 2, 1))
        w, V = np.linalg.eigh(blocks)
        f0, f1 = _scalar_functions(t * w, t)
        groups.append((rows, np.ascontiguousarray(V),
                       np.ascontiguousarray(f0), np.ascontiguousarray(f1)))
    return groups


def _exact_solve(groups, b0, b1, n):
    y = np.empty(n, dtype=np.float64)
    for rows, V, f0, f1 in groups:
        q0 = np.einsum("cij,cj->ci", V.transpose(0, 2, 1), b0[rows])
        q1 = np.einsum("cij,cj->ci", V.transpose(0, 2, 1), b1[rows])
        y[rows] = np.einsum("cij,cj->ci", V, f0 * q0 + f1 * q1)
    return y


# ---------------------------------------------------------------- session
class Session:
    def __init__(self, meta, indptr, indices, data):
        self.n = int(meta["n"])
        self.nnz = int(meta["nnz"])
        self.t = float(meta["t"])
        self.rtol = float(meta["rtol"])
        self.atol = float(meta["atol"])
        self.requests = int(meta["requests"])
        self.indptr = indptr.view(np.int64)
        self.indices = indices
        self.ip_ptr = _ptr(self.indptr, _PU64)
        self.ix_ptr = _ptr(self.indices, _PU32)
        self.mode = "zero"
        self.K = 0
        self.groups = None
        self.ybuf = np.zeros(self.n, dtype=np.float64)
        self.work_d = None
        self.work_l = None
        self.use_long = False

        if self.t == 0.0:
            self.mode = "t0"
            return
        if self.nnz == 0:
            return

        self.data = np.array(data, dtype=np.float64, copy=True)
        self.d_ptr = _ptr(self.data, _PD)

        # Row-sum bound (one matvec) and cheap structural description.
        work = np.empty(2 * self.n, dtype=np.float64)
        gersh = _lib.famexp_lmax_bound(self.ip_ptr, self.ix_ptr, self.d_ptr,
                                       self.n, 1, _ptr(work, _PD))
        eta_g = self.t * gersh
        maxrow = ctypes.c_int()
        maxband = ctypes.c_int()
        _lib.famexp_pattern_stats(self.ip_ptr, self.ix_ptr, self.n,
                                  ctypes.byref(maxrow), ctypes.byref(maxband))

        # Diagonal matrices: an exact O(n) formula beats any iterative method.
        if maxrow.value <= 1:
            diag = _try_diagonal(self.n, self.indptr, self.indices, self.data, self.t)
            if diag is not None:
                self.f0_diag, self.f1_diag = diag
                self.mode = "diagonal"
                return

        # Many tiny independent blocks: dense diagonalisation once, then
        # O(nnz) per request.  Only worth its setup when the polynomial degree
        # would otherwise be large.
        m_est = math.sqrt(max(eta_g, 0.0) * (math.log(1.0 / self.rtol) + 4.0)) + 10.0
        if (maxrow.value <= 40 and maxband.value <= 64 and m_est > 80.0):
            try:
                groups = _try_exact_blocks(self.n, self.nnz, self.indptr,
                                           self.indices, self.data, self.t)
            except Exception:
                groups = None
            if groups is not None:
                self.groups = groups
                self.mode = "exact"
                return

        # Tighten the spectral bound only where it matters.
        if eta_g < 3.0:
            bound = gersh
        else:
            iters = 12 if eta_g < 300.0 else 32
            bound = _lib.famexp_lmax_bound(self.ip_ptr, self.ix_ptr, self.d_ptr,
                                           self.n, iters, _ptr(work, _PD))
        self.lmax = bound * (1.0 + 2e-8) + 1e-300
        if not (self.lmax > 0.0) or not np.isfinite(self.lmax):
            return
        eta = self.t * self.lmax
        # Only a numerically zero matrix may skip the polynomial: even for
        # tiny t*lmax the O(eta) correction to b0 + t*b1 can exceed atol.
        if eta < 1e-28:
            return

        K = int(math.ceil(math.sqrt(eta * 85.0))) + 120
        K = max(K, 32)
        K = min(K, 12000)
        self.K = K
        self.c_arr, self.d_arr = make_coefficients(self.t, self.lmax, K)
        self.c_arr = np.ascontiguousarray(self.c_arr, dtype=np.float64)
        self.d_arr = np.ascontiguousarray(self.d_arr, dtype=np.float64)
        self.cptr = _ptr(self.c_arr, _PD)
        self.dptr = _ptr(self.d_arr, _PD)
        self.tail_c, self.tail_d = _tails(self.c_arr, self.d_arr)

        log_inv = math.log(1.0 / self.rtol)
        # p10 has the tightest absolute floor; the looser levels safely use a
        # smaller Chebyshev degree.  The a-posteriori check below is the final
        # guard for unusually small results.
        margin = 6.0 if self.rtol <= 1e-10 else 4.0
        self.m1 = min(K, int(math.ceil(math.sqrt(eta * (log_inv + margin)))) + 10)

        self.work_d = np.empty(3 * self.n, dtype=np.float64)
        self.work_l = None  # allocated lazily only for extreme cancellation

        # Once the working vector outgrows L2, two independent row chains
        # hide gather latency on irregular graphs; otherwise four short rows
        # per iteration is best (and never worse than the scalar loop).
        avg_deg = float(self.nnz) / max(1, self.n)
        if self.n >= 65000 and (avg_deg >= 7.0 or maxrow.value >= 24):
            self._kernel_d = _lib.famexp_clenshaw_d4
        else:
            self._kernel_d = _lib.famexp_clenshaw_d5

        _lib.famexp_scale(self.d_ptr, self.nnz, 4.0 / self.lmax)
        self.mode = "cheb"

    def _clenshaw(self, b0, b1, m, long=False):
        y = self.ybuf
        if long:
            if self.work_l is None:
                self.work_l = np.empty(3 * self.n, dtype=np.longdouble)
            _lib.famexp_clenshaw_l(self.ip_ptr, self.ix_ptr, self.d_ptr,
                                   self.n, m, self.cptr, self.dptr,
                                   _ptr(b0, _PD), _ptr(b1, _PD),
                                   _ptr(y, _PD), _ptr(self.work_l, _PL))
        else:
            self._kernel_d(self.ip_ptr, self.ix_ptr, self.d_ptr,
                           self.n, m, self.cptr, self.dptr,
                           _ptr(b0, _PD), _ptr(b1, _PD),
                           _ptr(y, _PD), _ptr(self.work_d, _PD))
        return y

    def solve(self, b0, b1):
        mode = self.mode
        if mode == "t0":
            return b0
        if mode == "zero":
            return b0 + self.t * b1
        if mode == "exact":
            return _exact_solve(self.groups,
                                np.asarray(b0, dtype=np.float64),
                                np.asarray(b1, dtype=np.float64), self.n)
        if mode == "diagonal":
            return self.f0_diag * b0 + self.f1_diag * b1

        b0 = np.ascontiguousarray(b0, dtype=np.float64)
        b1 = np.ascontiguousarray(b1, dtype=np.float64)
        B = max(float(np.max(np.abs(b0))), float(np.max(np.abs(b1))))
        if B == 0.0:
            return np.zeros(self.n, dtype=np.float64)
        if B != 1.0:
            b0 = b0 / B
            b1 = b1 / B
        B0 = float(np.max(np.abs(b0)))
        B1 = float(np.max(np.abs(b1)))
        kappa = 4.0
        weighted_tail = self.tail_c * B0 + self.tail_d * B1

        # For mild stiffness the absolute-tolerance floor alone determines a
        # much smaller degree than the worst-case formula.  Only do this when
        # the series converges quickly enough that a low-degree value is still
        # a usable estimate of max|y|.
        m = self.m1
        if self.t * self.lmax < 100.0:
            floor = 0.2 * (self.atol / B) / kappa
            m_tail = int(np.searchsorted(-weighted_tail, -floor))
            m = min(m, max(m_tail, 2))

        # Evaluate, then check the truncation bound against the measured
        # tolerance.  Re-evaluating at a higher degree is rare and always
        # converges (the bound decreases monotonically with m).
        for _ in range(3):
            y = self._clenshaw(b0, b1, m).copy()
            ymax = float(np.max(np.abs(y)))
            tol = self.atol / B + self.rtol * ymax
            err = kappa * weighted_tail[m]
            if err <= 0.5 * tol:
                break
            limit = 0.2 * tol / kappa
            m2 = int(np.searchsorted(-weighted_tail, -limit))
            m2 = min(max(m2, m + 1), self.K)
            if os.environ.get("FAMEXP_DEBUG"):
                sys.stderr.write("refine %d -> %d eta=%.4g tol=%.3g err=%.3g\n"
                                 % (m, m2, self.t * self.lmax, tol, err))
                sys.stderr.flush()
            if m2 <= m:
                break
            m = m2
        if B != 1.0:
            y *= B
        return y


# ---------------------------------------------------------------- main
def main():
    dbg = bool(os.environ.get("FAMEXP_DEBUG"))
    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    out.write(b"READY\n")
    out.flush()

    meta, indptr, indices, data = read_header(inp)
    sess = Session(meta, indptr, indices, data)
    if dbg:
        sys.stderr.write("mode=%s K=%d m1=%d\n" %
                         (sess.mode, sess.K, getattr(sess, "m1", 0)))
        sys.stderr.flush()
    n = sess.n
    for _ in range(sess.requests):
        b0 = np.frombuffer(_read_exact(inp, 8 * n), dtype="<f8")
        b1 = np.frombuffer(_read_exact(inp, 8 * n), dtype="<f8")
        y = sess.solve(b0, b1)
        out.write(np.asarray(y, dtype="<f8", order="C").tobytes(order="C"))
        out.flush()
    # Skip interpreter finalisation (atexit / NumPy-SciPy teardown): that
    # would add ~30 ms to the timed session.  All results are already flushed.
    sys.stdout.buffer.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
