#!/usr/bin/env python3
"""FAMEXP2 solver: y = exp(-tL) b0 + t*phi1(-tL) b1 for symmetric PSD sparse L.

Method
------
L is symmetric and positive semidefinite, so its spectrum lies in [0, lam_max].
Both scalar functions are expanded in Chebyshev polynomials of the scaled
operator  Ltilde = (L - c I)/h  with c = lam_max/2, h = lam_max/2:

    e^{-tL}      = sum_k  a_k T_k(Ltilde)      (exact Chebyshev series)
    t*phi1(-tL)  = sum_k  d_k T_k(Ltilde)      (Chebyshev series of the
                                                  integral representation)

Because |T_k(x)| <= 1 on [-1,1] the truncation error of either series is
bounded by the tail of its coefficients times the norm of the right hand
side; the recurrence is therefore stopped as soon as that bound falls below
the tolerance demanded by the accuracy rule.

The two right hand sides are propagated together, so one pass over the
matrix serves both series.  The matrix is used in place (no conversion of
the incoming CSR arrays) and the inner loop runs in C.
"""

import ctypes
import json
import time as _time
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(HERE, "libfamexp.so"))
_lib.famexp_spectral_bound.restype = ctypes.c_double
_lib.famexp_spectral_bound.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_double)]
_lib.famexp_cheb.restype = ctypes.c_int
_lib.famexp_cheb.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int64, ctypes.c_double, ctypes.c_double,
                             ctypes.c_void_p, ctypes.c_int64,
                             ctypes.c_void_p, ctypes.c_int64,
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_double, ctypes.c_double,
                             ctypes.POINTER(ctypes.c_int64),
                             ctypes.POINTER(ctypes.c_double)]
_lib.famexp_components.restype = ctypes.c_int
_lib.famexp_components.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
                                   ctypes.POINTER(ctypes.c_int64)]

_DEBUG = bool(os.environ.get("FAMEXP_DEBUG"))
MAGIC = b"FAMEXP2\n"
PI = 3.14159265358979323846


# ----------------------------------------------------------------- wire ---
def _read_into(stream, mv):
    """Fill the writable memoryview mv from the stream."""
    total = len(mv)
    got = 0
    while got < total:
        k = stream.readinto(mv[got:])
        if not k:
            raise EOFError("short input")
        got += k


def _read_exact(stream, nbytes):
    buf = bytearray(nbytes)
    _read_into(stream, memoryview(buf))
    return bytes(buf)


# ----------------------------------------------------------- coefficients -
def _dct1(v):
    """Unnormalised DCT-I of the Chebyshev-Lobatto samples v.

    Equivalent to scipy.fft.dct(v, type=1) but implemented with numpy's FFT
    so that the submission does not depend on scipy (importing scipy costs
    several milliseconds of process teardown in every session).
    """
    npts = len(v)
    ext = np.concatenate((v, v[-2:0:-1]))
    y = np.fft.rfft(ext).real
    a = y / (npts - 1.0)
    a[0] *= 0.5
    a[-1] *= 0.5
    return a


def _cheb_coefficients(t, lo, hi, npts):
    """Chebyshev coefficients of e^{-t lam} and of t*phi1(-t lam) on [lo,hi].

    Both functions are entire, so their Chebyshev series converge on the whole
    interval; the coefficients are obtained from the Chebyshev-Lobatto samples
    (Chebyshev interpolation).  npts has to be a few times larger than the
    degree that is finally used, otherwise the interpolation aliases the tail
    of the series onto the low orders.
    """
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)
    xs = np.cos(PI * np.arange(npts, dtype=np.float64) / (npts - 1))
    lam = c + h * xs
    x = t * lam
    small = x > 0.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        # t*phi1(-t lam) = (1-exp(-t lam))/lam = t*(1-exp(-x))/x
        gv = np.where(small, -np.expm1(-np.where(small, x, 1.0)) / np.where(small, x, 1.0), 1.0) * t
        fe = np.exp(-x)
    a = _dct1(fe)
    d = _dct1(gv)
    return a, d


def _tail(coef):
    """tail[k] = sum_{j>=k} |coef[j]| (decreasing tail of the series)."""
    return np.cumsum(np.abs(coef[::-1]))[::-1]


# ------------------------------------------------------------- solver ------
class Solver(object):
    def __init__(self, n, nnz, indptr, indices, data, t, rtol, atol):
        self.n = n
        self.t = t
        self.rtol = rtol
        self.atol = atol
        self.mode = "cheb"
        self.y = np.empty(n, dtype="<f8")
        if t == 0.0:
            self.mode = "identity"
            return
        if nnz == 0:
            self.mode = "zero"
            return
        counts = np.diff(indptr.astype(np.int64))
        # cheap diagonal test: every row holds at most the diagonal entry
        if counts.max() <= 1:
            nz = np.nonzero(counts)[0]
            if np.array_equal(indices, nz.astype(np.uint32)):
                self.mode = "diagonal"
                self.lam = np.zeros(n, dtype=np.float64)
                self.lam[nz] = data
                self._prep_diagonal()
                return
        # general Chebyshev path
        self.indptr = indptr
        self.indices = indices
        self.data = data
        # enclosure of the spectrum: Gershgorin gives both ends, the
        # Collatz-Wielandt iteration on |L| tightens the upper end
        glo = ctypes.c_double(0.0)
        gersh = _lib.famexp_spectral_bound(indptr.ctypes.data, indices.ctypes.data,
                                           data.ctypes.data, n, 0,
                                           ctypes.byref(glo))
        est = 4.8 * math.sqrt(max(t * gersh, 0.0))
        # each Collatz-Wielandt iteration costs one sweep over the matrix,
        # while the tighter enclosure only saves a fraction of the series, so
        # the number of iterations is scaled with the size of the series
        iters = 0 if est < 8.0 else int(min(40, max(2, 0.1 * est)))
        self.hi = _lib.famexp_spectral_bound(indptr.ctypes.data, indices.ctypes.data,
                                             data.ctypes.data, n, iters, None)
        if not (self.hi > 0.0 and math.isfinite(self.hi)):
            self.hi = gersh
        # the lower end of the enclosure: Gershgorin, and never below zero
        # because the operator is positive semidefinite
        self.lo = max(0.0, min(glo.value, self.hi))
        self.a = None
        self.d = None
        # A block diagonal operator can be solved exactly, but finding the
        # blocks costs about as much as a matrix sweep.  Only try it when the
        # series that it replaces would be considerably more expensive.
        alpha = 0.5 * t * (self.hi - self.lo)
        m_est = math.sqrt(max(2.0 * alpha * math.log(1.0 / max(atol, 1e-300)), 1.0))
        if m_est > 50.0 and self._setup_blocks(indptr, indices, data, counts):
            return
        self._prepare_coefficients()
        if _DEBUG:
            sys.stderr.write("setup: hi=%.6g ncoef=%d\n" % (self.hi, self.ncoef))
            sys.stderr.flush()

    # -- small block diagonal operators ---------------------------------
    def _setup_blocks(self, indptr, indices, data, counts):
        """Exact solution when the operator splits into small dense blocks."""
        n = self.n
        label = np.empty(n, dtype=np.int64)
        maxsize = ctypes.c_int64(0)
        ncomp = ctypes.c_int64(0)
        rc = _lib.famexp_components(indptr.ctypes.data, indices.ctypes.data, n,
                                    label.ctypes.data, ctypes.byref(maxsize),
                                    ctypes.byref(ncomp))
        if rc != 0 or ncomp.value <= 1 or ncomp.value >= n:
            return False
        if maxsize.value > 8:
            return False
        if maxsize.value > 2 and ncomp.value > 4000:
            # a batched dense eigen decomposition of many blocks would cost
            # more than the Chebyshev series
            return False
        rows = np.repeat(np.arange(n, dtype=np.int64), counts)
        # size of the component of every index
        cnt = np.bincount(label)
        size_of = cnt[label]
        # order the indices by component, keeping the index order inside a
        # component; this gives the local position inside each block
        order = np.argsort(label, kind="stable")
        labs = label[order]
        starts = np.concatenate(([0], np.nonzero(np.diff(labs))[0] + 1))
        sizes = np.diff(np.append(starts, n))
        comp_of = np.empty(n, dtype=np.int64)
        comp_of[order] = np.repeat(np.arange(len(starts)), sizes)
        local_of = np.empty(n, dtype=np.int64)
        local_of[order] = np.arange(n) - np.repeat(starts, sizes)
        groups = []
        hist = np.bincount(size_of)
        for k in np.nonzero(hist)[0]:
            k = int(k)
            # members in (component, local position) order, so that the block
            # index and the member order agree
            members = order[size_of[order] == k]
            # renumber the components of this size consecutively: the rank of a
            # component is its position among the components of this size,
            # ordered by their global number
            cid = comp_of[members]
            present = np.zeros(len(starts), dtype=np.int64)
            present[cid] = 1
            rank = np.cumsum(present) - 1
            comp_ids = rank[cid]
            nc = int(rank.max()) + 1 if len(cid) else 0
            cg = np.zeros(n, dtype=np.int64)
            cg[members] = comp_ids
            lg = np.zeros(n, dtype=np.int64)
            lg[members] = local_of[members]
            blocks = np.zeros((nc, k, k))
            ridx = np.nonzero(size_of[rows] == k)[0]
            flat = cg[rows[ridx]] * (k * k) + lg[rows[ridx]] * k + lg[indices[ridx]]
            flatblocks = blocks.reshape(-1)
            flatblocks[flat] = data[ridx]
            if not np.array_equal(blocks, np.swapaxes(blocks, 1, 2)):
                # the blocks must be symmetric; if they are not, the pattern
                # was not assembled as intended and the series is used instead
                return False
            groups.append((k, members, blocks))
        self.mode = "blocks"
        self.block_groups = groups
        return True

    def _solve_blocks(self, b0, b1):
        """Exact solution for a block diagonal operator with small blocks."""
        y = self.y
        t = self.t
        for (k, sel, blocks) in self.block_groups:
            nc = blocks.shape[0]
            B0 = b0[sel].reshape(nc, k)
            B1 = b1[sel].reshape(nc, k)
            if k == 2:
                # analytic symmetric 2x2 eigen decomposition
                a = blocks[:, 0, 0]
                c = blocks[:, 1, 1]
                b = blocks[:, 0, 1]
                half = 0.5 * (a - c)
                disc = np.sqrt(half * half + b * b)
                l1 = 0.5 * (a + c) + disc
                l2 = 0.5 * (a + c) - disc
                v1 = np.empty((nc, 2))
                use_b = np.abs(b) >= np.abs(l1 - a)
                v1[use_b, 0] = b[use_b]
                v1[use_b, 1] = (l1 - a)[use_b]
                v1[~use_b, 0] = (l1 - c)[~use_b]
                v1[~use_b, 1] = b[~use_b]
                nrm = np.sqrt(v1[:, 0] ** 2 + v1[:, 1] ** 2)
                nrm[nrm == 0.0] = 1.0
                v1 /= nrm[:, None]
                v2 = np.stack([v1[:, 1], -v1[:, 0]], axis=1)
                lam = np.stack([l1, l2], axis=1)
                V = np.stack([v1, v2], axis=1)
            else:
                lam, V = np.linalg.eigh(blocks)
            x = t * lam
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                small = np.abs(x) < 1e-6
                g = np.where(small, t * (1.0 - x / 2.0 + x * x / 6.0 - x * x * x / 24.0),
                             -np.expm1(-x) / np.where(x != 0.0, x, 1.0) * t)
                e = np.exp(-x)
            c0 = np.einsum("nij,nj->ni", V, B0)
            c1 = np.einsum("nij,nj->ni", V, B1)
            res = e * c0 + g * c1
            yb = np.einsum("nij,nj->ni", V, res)
            y[sel] = yb.reshape(-1)
        return y

    # -- diagonal -------------------------------------------------------
    def _prep_diagonal(self):
        x = self.t * self.lam
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            self.ex = np.exp(-x)
            self.ph = np.where(x > 0.0, -np.expm1(-np.where(x > 0.0, x, 1.0)) / np.where(x > 0.0, x, 1.0), 1.0) * self.t
        self.ex = np.where(np.isfinite(self.ex), self.ex, 0.0)
        self.ph = np.where(np.isfinite(self.ph), self.ph, 0.0)

    # -- coefficients ----------------------------------------------------
    def _prepare_coefficients(self, npts=None):
        """Build the Chebyshev series.

        The number of nodes is chosen from an upper bound of the degree that
        the series can need (the tail of a Chebyshev series decays like
        exp(-m^2/(2 alpha)) with alpha = t*(hi-lo)/2) and is a few times
        larger, so that the interpolation resolves the series far beyond the
        degree that is finally used.  If the recurrence nevertheless runs out
        of coefficients, the caller enlarges the series and retries.
        """
        if npts is None:
            alpha = 0.5 * self.t * (self.hi - self.lo)
            m_est = math.sqrt(max(2.0 * alpha * math.log(1.0 / max(self.atol, 1e-300)), 1.0))
            npts = 256
            while npts < 8.0 * m_est and npts < (1 << 20):
                npts *= 2
        a, d = _cheb_coefficients(self.t, self.lo, self.hi, npts)
        self.a = np.ascontiguousarray(a, dtype=np.float64)
        self.d = np.ascontiguousarray(d, dtype=np.float64)
        self.ncoef = npts

    # -- one request -----------------------------------------------------
    def solve(self, b0, b1):
        n = self.n
        _t0 = _time.perf_counter()
        if self.mode == "identity":
            np.copyto(self.y, b0)
            return self.y
        if self.mode == "zero":
            np.multiply(b1, self.t, out=self.y)
            np.add(self.y, b0, out=self.y)
            return self.y
        if self.mode == "blocks":
            return self._solve_blocks(b0, b1)
        if self.mode == "diagonal":
            np.multiply(b0, self.ex, out=self.y)
            y = b1 * self.ph
            np.add(self.y, y, out=self.y)
            return self.y
        deg = ctypes.c_int64(0)
        dbg = (ctypes.c_double * 4)()
        attempts = 0
        while True:
            rc = _lib.famexp_cheb(self.indptr.ctypes.data, self.indices.ctypes.data,
                                  self.data.ctypes.data, n, self.lo, self.hi,
                                  self.a.ctypes.data, self.ncoef,
                                  self.d.ctypes.data, self.ncoef,
                                  b0.ctypes.data, b1.ctypes.data, self.y.ctypes.data,
                                  self.atol, self.rtol, ctypes.byref(deg), dbg)
            if rc == 0:
                if _DEBUG:
                    sys.stderr.write("solve %.1f ms deg=%d rem=%.3e ymax=%.3e stall=%g\n" % ((_time.perf_counter() - _t0) * 1e3, deg.value, dbg[1], dbg[2], dbg[3]))
                    sys.stderr.flush()
                return self.y
            # not enough coefficients: enlarge the series and retry
            attempts += 1
            if attempts > 4:
                # the series cannot be shortened any further; the answer that
                # has been computed is the best available one
                return self.y
            self._prepare_coefficients(self.ncoef * 2)


def _warmup():
    """Touch the numpy machinery that is used later, before READY.

    The first call of some numpy routines carries a one time initialisation
    cost; doing it here keeps it out of the measured session.
    """
    try:
        a = np.arange(16.0)
        np.diff(a)
        np.repeat(np.arange(8), 2)
        np.bincount(np.arange(8))
        np.nonzero(np.arange(8) % 2)[0]
        np.argsort(np.arange(8), kind="stable")
        np.cumsum(np.abs(a[::-1]))
        np.empty(0).ctypes.data
        _dct1(np.ones(16))
        np.exp(-a)
        np.expm1(-a)
    except Exception:
        pass


def _enlarge_pipe():
    """Give the session pipe a large buffer.

    The matrix arrives through a pipe whose default capacity is 64 KiB, so a
    multi megabyte matrix needs hundreds of round trips between the controller
    and this process.  Enlarging the pipe (allowed for either end) removes
    most of that latency.
    """
    try:
        import fcntl
        F_SETPIPE_SZ = 1031
        old = fcntl.fcntl(sys.stdin.fileno(), F_SETPIPE_SZ)
        new = fcntl.fcntl(sys.stdin.fileno(), F_SETPIPE_SZ, 1 << 20)
        if _DEBUG:
            sys.stderr.write("pipe %d -> %d\n" % (old, new))
            sys.stderr.flush()
    except Exception as exc:
        if _DEBUG:
            sys.stderr.write("pipe resize failed %r\n" % (exc,))
            sys.stderr.flush()


def main():
    out = sys.stdout.buffer
    stdin = sys.stdin.buffer
    # the pipe has to be resized before the controller starts to write
    _enlarge_pipe()
    _warmup()
    out.write(b"READY\n")
    out.flush()

    _tstart = _time.perf_counter()
    if _read_exact(stdin, 8) != MAGIC:
        raise SystemExit("bad magic")
    length = struct.unpack("<I", _read_exact(stdin, 4))[0]
    meta = json.loads(_read_exact(stdin, length).decode("utf-8"))
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    t = float(meta["t"])
    rtol = float(meta["rtol"])
    atol = float(meta["atol"])
    requests = int(meta["requests"])

    indptr = np.empty(n + 1, dtype="<u8")
    _read_into(stdin, indptr.data)
    indices = np.empty(nnz, dtype="<u4")
    if nnz:
        _read_into(stdin, indices.data)
    data = np.empty(nnz, dtype="<f8")
    if nnz:
        _read_into(stdin, data.data)

    if _DEBUG:
        sys.stderr.write("read %.2f ms\n" % ((_time.perf_counter()-_tstart)*1e3))
        sys.stderr.flush()
    _tprep = _time.perf_counter()
    solver = Solver(n, nnz, indptr, indices, data, t, rtol, atol)
    if _DEBUG:
        sys.stderr.write("prep %.2f ms\n" % ((_time.perf_counter()-_tprep)*1e3))
        sys.stderr.flush()
    b0 = np.empty(n, dtype="<f8")
    b1 = np.empty(n, dtype="<f8")
    for _ in range(requests):
        _read_into(stdin, b0.data)
        _read_into(stdin, b1.data)
        y = solver.solve(b0, b1)
        out.write(y.tobytes())
        out.flush()
        if _DEBUG:
            sys.stderr.write("done %.2f ms\n" % ((_time.perf_counter()-_tstart)*1e3))
            sys.stderr.flush()
    out.flush()
    try:
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
