"""FAMEXP2 solver: y = exp(-tL) b0 + t*phi1(-tL) b1 for sparse symmetric PSD L.

Main method: symmetric Lanczos (stored basis, C kernel) per nonzero right-hand
side with a-posteriori stopping; exact paths for t=0, zero, diagonal and tiny
matrices.  See README.md.
"""
import ctypes
import fcntl
import json
import math
import os
import struct
import subprocess
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import ive
from scipy.fft import dct

HERE = os.path.dirname(os.path.abspath(__file__))
F_SETPIPE_SZ = 1031
DENSE_MAX_N = 200
BASIS_BYTES = 2.4e9          # memory budget for the stored Lanczos bases


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# native kernels
# ----------------------------------------------------------------------------
def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _try_load(path):
    try:
        return _bind(ctypes.CDLL(path))
    except Exception:
        return None


def _bind(lib):
    i64, p = ctypes.c_int64, ctypes.c_void_p
    lib.spmv.argtypes = [i64, p, p, p, p, p]
    lib.spmv.restype = None
    lib.lanczos_steps.argtypes = [i64, p, p, p, p, i64, p, i64, i64, p, p,
                                  ctypes.c_double, p, p, p, p]
    lib.lanczos_steps.restype = i64
    lib.csr_inf_norm.argtypes = [i64, p, p]
    lib.csr_inf_norm.restype = ctypes.c_double
    lib.csr_stats.argtypes = [i64, p, p, p, p]
    lib.csr_stats.restype = None
    lib.csr_gersh_low.argtypes = [i64, p, p, p]
    lib.csr_gersh_low.restype = ctypes.c_double
    lib.csr_is_diagonal.argtypes = [i64, p, p]
    lib.csr_is_diagonal.restype = ctypes.c_int
    lib.cheb_tridiag.argtypes = [i64, p, p, ctypes.c_double, ctypes.c_double, p, i64, p, p]
    lib.cheb_tridiag.restype = None
    lib.lanczos2_steps.argtypes = [i64, p, p, p, p, i64, p, p, p, p, i64, p, p, p,
                                   i64, i64, ctypes.c_double, p]
    lib.lanczos2_steps.restype = i64
    return lib


def load_native():
    flags = _cpu_flags()
    names = []
    if "avx2" in flags and "fma" in flags:
        names.append("libk_avx2.so")
    names.append("libk_base.so")
    for name in names:
        path = os.path.join(HERE, "build", name)
        if os.path.exists(path):
            lib = _try_load(path)
            if lib is not None:
                return lib
    # build missing: compile into /tmp (input independent, before READY)
    src = os.path.join(HERE, "kernels.c")
    out = os.path.join("/tmp", "libk_%d.so" % os.getpid())
    try:
        subprocess.run(["gcc", "-O3", "-ffp-contract=off", "-shared", "-fPIC", "-o", out, src, "-lm"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=10)
        lib = _try_load(out)
        try:
            os.unlink(out)
        except OSError:
            pass
        return lib
    except Exception:
        return None


def ptr(a):
    return a.ctypes.data


# ----------------------------------------------------------------------------
# scalar functions
# ----------------------------------------------------------------------------
def f_exp(t, th):
    return np.exp(-t * th)


def f_phi(t, th):
    # t*phi1(-t*th) = (1 - exp(-t th)) / th
    x = t * th
    out = np.empty_like(th)
    small = np.abs(x) < 1e-5
    big = ~small
    out[big] = -np.expm1(-x[big]) / th[big]
    xs = x[small]
    out[small] = t * (1.0 - xs / 2.0 + xs * xs / 6.0)
    return out


def cheb_tail(rho):
    """Tail sums of |Chebyshev coefficients| of exp(-x) on [0, rho]."""
    z = 0.5 * rho
    kmax = int(math.sqrt(2.0 * z * 45.0) + 60)
    k = np.arange(kmax + 1, dtype=float)
    c = 2.0 * ive(k, z)
    c[0] *= 0.5
    tail = np.cumsum(c[::-1])[::-1]
    return tail


def cheb_degree(tail, eps):
    idx = np.nonzero(tail <= eps)[0]
    return int(idx[0]) if idx.size else len(tail) + 10


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------
class Solver:
    def __init__(self, lib, n, indptr, indices, data, t, rtol, atol):
        self.lib = lib
        self.n = n
        self.t = t
        self.rtol = rtol
        self.atol = atol
        self.ip = indptr.view(np.int64)
        self.ix = indices
        self.a = data
        self.mode = "lanczos"
        nnz = data.size
        if t == 0.0:
            self.mode = "identity"
            return
        if nnz == 0 or not np.any(data):
            self.mode = "zero"
            return
        self.scatter = 0.0
        if lib is not None:
            st = np.zeros(4)
            lib.csr_stats(n, ptr(self.ip), ptr(indices), ptr(data), ptr(st))
            self.normL = float(st[0])
            self.lam_low = float(st[1])
            self.scatter = float(st[2])
            diag = bool(st[3])
        else:
            rows = np.repeat(np.arange(n), np.diff(self.ip))
            self.normL = float(np.bincount(rows, weights=np.abs(data), minlength=n).max())
            diag = bool(np.all(indices.astype(np.int64) == rows))
            on = indices.astype(np.int64) == rows
            dg = np.bincount(rows[on], weights=data[on], minlength=n)
            off = np.bincount(rows[~on], weights=np.abs(data[~on]), minlength=n)
            self.lam_low = float((dg - off).min())
        if diag:
            self.mode = "diagonal"
            d = np.zeros(n)
            rows = np.repeat(np.arange(n), np.diff(self.ip))
            d[rows] = data
            self.dexp = f_exp(t, d)
            self.dphi = f_phi(t, d)
            return
        if n <= DENSE_MAX_N:
            self.mode = "dense"
            A = np.zeros((n, n))
            rows = np.repeat(np.arange(n), np.diff(self.ip))
            A[rows, indices.astype(np.int64)] = data
            A = 0.5 * (A + A.T)
            w, Q = np.linalg.eigh(A)
            self.Q = Q
            self.wexp = f_exp(t, w)
            self.wphi = f_phi(t, w)
            return
        self.rho = t * self.normL
        self.perm = None
        self.pool = []
        self.reorder_checked = False
        self.nreq = 1
        self.tail = cheb_tail(self.rho)
        self.brk = 1e-13 * self.normL
        if lib is None:
            import scipy.sparse as sp
            self.A = sp.csr_matrix((data, indices, self.ip), shape=(n, n))
        self.x2 = np.empty(2 * n)

    # -- Lanczos kernels ---------------------------------------------------------
    def steps(self, r, k0, k1, acc=None, coef=None, acc2=None, coef2=None):
        n = self.n
        V, lim, ring, alpha, beta = r.V, r.lim, r.ring, r.alpha, r.beta
        if self.lib is not None:
            return self.lib.lanczos_steps(
                n, ptr(self.ip), ptr(self.ix), ptr(self.a), ptr(V), lim, ptr(ring),
                k0, k1, ptr(alpha), ptr(beta), self.brk,
                ptr(acc) if acc is not None else None,
                ptr(coef) if coef is not None else None,
                ptr(acc2) if acc2 is not None else None,
                ptr(coef2) if coef2 is not None else None)

        def row(j):
            return V[j] if j < lim else ring[j % 3]
        for k in range(k0, k1):
            v = row(k)
            w = self.A @ v
            if k > 0:
                w -= beta[k - 1] * row(k - 1)
            al = float(v @ w)
            w -= al * v
            nr = float(np.sqrt(w @ w))
            alpha[k] = al
            beta[k] = nr
            if not nr > self.brk:
                row(k + 1)[:] = w
                return k + 1
            w /= nr
            row(k + 1)[:] = w
            if acc is not None:
                acc += coef[k + 1] * w
            if acc2 is not None:
                acc2 += coef2[k + 1] * w
        return k1

    def steps2(self, r0, r1, k0, k1):
        if self.lib is None:
            e0 = self.steps(r0, k0, k1)
            e1 = self.steps(r1, k0, k1)
            return min(e0, e1)
        return self.lib.lanczos2_steps(
            self.n, ptr(self.ip), ptr(self.ix), ptr(self.a),
            ptr(r0.V), r0.lim, ptr(r0.ring), ptr(r0.alpha), ptr(r0.beta),
            ptr(r1.V), r1.lim, ptr(r1.ring), ptr(r1.alpha), ptr(r1.beta),
            k0, k1, self.brk, ptr(self.x2))

    def coeffs(self, r, k, exact=False):
        """Return (c, noise, theta_min) with c = beta0 * f(T_k) e1.

        exact=True (or small k) uses the tridiagonal eigendecomposition; otherwise
        a Chebyshev series on T_k whose rounding floor 'noise' is returned too."""
        alpha, beta = r.alpha, r.beta
        t = self.t
        if k == 1:
            return r.fn(t, alpha[:1].copy()) * r.beta0, 0.0, float(alpha[0])
        if exact or self.lib is None or k < 40:
            th, S = eigh_tridiagonal(alpha[:k], beta[:k - 1])
            return S @ (r.fn(t, th) * S[0]) * r.beta0, 0.0, float(th[0])
        top = eigh_tridiagonal(alpha[:k], beta[:k - 1], eigvals_only=True,
                               select="i", select_range=(k - 1, k - 1))
        bot = eigh_tridiagonal(alpha[:k], beta[:k - 1], eigvals_only=True,
                               select="i", select_range=(0, 0))
        marg = 1e-9 * self.normL
        hi = float(top[0]) + marg
        lo = min(float(bot[0]), 0.0) - marg
        if not (lo > -1e-7 * self.normL and t * (hi - lo) < 1e6 and hi > lo):
            return self.coeffs(r, k, exact=True)
        g = self.cheb_coef(r.kind, lo, hi)
        out = np.empty(k)
        work = np.empty(3 * k)
        self.lib.cheb_tridiag(k, ptr(alpha), ptr(beta), lo, hi, ptr(g), g.size,
                              ptr(out), ptr(work))
        noise = max(4e-16 * float(np.abs(g).sum()) * math.sqrt(g.size) * r.beta0, 1e-300)
        return out * r.beta0, noise, float(bot[0])

    def cheb_coef(self, kind, lo, hi):
        t = self.t
        half = 0.5 * (hi - lo)
        a = t * half
        jmax = int(math.sqrt(2.0 * a * 45.0) + 40)
        j = np.arange(jmax + 1, dtype=float)
        ev = ive(j, a)
        tail = np.cumsum(ev[::-1])[::-1]
        cut = np.nonzero(tail <= 1e-18 * ev[0] + 1e-300)[0]
        m = int(cut[0]) + 2 if cut.size else jmax + 1
        m = min(m, jmax + 1)
        if kind == "exp":
            g = 2.0 * ev[:m] * math.exp(-t * lo)
            g[1::2] *= -1.0
            g[0] *= 0.5
            return g
        N = 64
        while N < 2 * m + 32:
            N *= 2
        th = math.pi * (np.arange(N) + 0.5) / N
        lam = lo + half * (np.cos(th) + 1.0)
        fv = f_phi(t, lam)
        g = dct(fv, type=2) / N
        g[0] *= 0.5
        return np.ascontiguousarray(g[:min(N, m + 8)])

    def combine(self, r, c, k, extra=None):
        """Return V_k c (and V_k extra), recomputing rows beyond the stored limit."""
        V, lim = r.V, r.lim
        kk = min(k, lim)
        if extra is None:
            y = c[:kk] @ V[:kk]
            d = None
        else:
            Y = np.stack([c[:kk], extra[:kk]]) @ V[:kk]
            y, d = Y[0], Y[1]
        if k > lim:
            # rows lim..k are regenerated from the stored rows lim-2, lim-1
            # (bitwise identical to the first pass), accumulating on the fly
            coef = np.zeros(k + 2)
            coef[:k] = c[:k]
            acc = np.zeros(self.n)
            if extra is None:
                self.steps(r, lim - 1, k, acc=acc, coef=coef)
            else:
                coef2 = np.zeros(k + 2)
                coef2[:k] = extra[:k]
                acc2 = np.zeros(self.n)
                self.steps(r, lim - 1, k, acc=acc, coef=coef, acc2=acc2, coef2=coef2)
                d = d + acc2
            y = y + acc
        return y, d

    def get_basis(self, rows):
        """Basis storage of `rows` rows, reused across requests so the pages are
        not zero-filled again for every request.  Only arrays at most 25% larger
        are reused and unsuitable pooled arrays are freed before allocating, so
        resident memory stays bounded by the basis budget plus 25%."""
        pool = self.pool
        best = None
        for i, arr in enumerate(pool):
            if rows <= arr.shape[0] <= rows + max(64, rows // 4):
                if best is None or arr.shape[0] < pool[best].shape[0]:
                    best = i
        if best is not None:
            return pool.pop(best)[:rows]
        pool.clear()
        return np.empty((rows, self.n))

    def release_basis(self, V):
        base = V.base if V.base is not None else V
        if base.ndim == 2 and base.shape[1] == self.n and base.flags.c_contiguous:
            self.pool.append(base)
            while len(self.pool) > 2:
                self.pool.pop(0)

    def new_run(self, kind, b, nruns):
        n = self.n
        r = Run()
        r.kind = kind
        r.fn = f_exp if kind == "exp" else f_phi
        scale = 1.0 if kind == "exp" else self.t
        r.beta0 = float(np.sqrt(b @ b))
        binf = float(np.abs(b).max())
        kcap = cheb_degree(self.tail, max(self.atol / (2.0 * r.beta0 * scale), 1e-17))
        kcap = int(kcap * 1.15) + 20
        ksched = max(1, cheb_degree(self.tail, max(self.rtol * binf / (2.0 * r.beta0), 1e-17)))
        kcap = max(kcap, ksched + 5)
        r.kcap = kcap
        r.lim = int(min(kcap + 2, max(8, BASIS_BYTES / nruns // (8 * n))))
        r.V = self.get_basis(r.lim)
        r.V[0] = b / r.beta0
        r.ring = np.empty((3, n))
        r.alpha = np.zeros(kcap + 3)
        r.beta = np.zeros(kcap + 3)
        r.step = max(1, int(math.ceil(0.06 * ksched)))
        r.first = min(max(1, int(0.42 * ksched)), kcap)
        r.target = min(16, r.first)
        r.k = 0
        r.c_prev = None
        r.k_prev = 0
        r.m_known = 0.0
        r.m_est = 0.0
        r.e_last = None
        r.exact = False
        r.est_last = None
        r.prev_ok = False
        r.done = False
        r.y = None
        return r

    def check(self, r, broke, m_other):
        """Checkpoint at r.k; sets r.done/r.y when converged.

        Chebyshev-evaluated coefficients (cheap, but with a rounding floor) only
        decide when to switch to exact evaluation; every acceptance decision
        uses coefficients from the tridiagonal eigendecomposition."""
        k = r.k
        rtol, atol = self.rtol, self.atol
        if broke or k >= r.kcap:
            c, _, _ = self.coeffs(r, k, exact=True)
            r.y, _ = self.combine(r, c, k)
            r.done = True
            return
        c, noise, thmin = self.coeffs(r, k, exact=r.exact)
        if r.c_prev is None and k > 1:
            kp = max(1, k - max(1, r.step // 2))
            r.c_prev, _, _ = self.coeffs(r, kp, exact=r.exact)
            r.k_prev = kp
        if r.c_prev is not None:
            dc = c.copy()
            dc[:r.k_prev] -= r.c_prev
            est1 = float(np.abs(dc).sum())
            est2 = float(np.sqrt(dc @ dc))
            cn = float(np.sqrt(c @ c))
            r.m_est = max(r.m_known, 0.5 * cn / math.sqrt(self.n))
            # exp(-tL)b: iterates are unreliable until the Krylov space reaches
            # the part of the spectrum where exp(-t lambda) is not negligible
            reached = r.kind != "exp" or self.t * thmin <= 60.0
            exact_now = noise == 0.0
            if not exact_now and reached:
                opt = 0.25 * (atol + rtol * max(cn, m_other, r.m_known))
                stagn = (r.est_last is not None and est1 > 0.3 * r.est_last
                         and est1 <= 1e-8 * float(np.abs(c).sum()))
                if stagn or k >= 0.8 * r.kcap:
                    r.exact = True
                if r.exact or est2 <= opt:
                    c, noise, thmin = self.coeffs(r, k, exact=True)
                    r.c_prev, _, _ = self.coeffs(r, r.k_prev, exact=True)
                    dc = c.copy()
                    dc[:r.k_prev] -= r.c_prev
                    est1 = float(np.abs(dc).sum())
                    est2 = float(np.sqrt(dc @ dc))
                    cn = float(np.sqrt(c @ c))
                    exact_now = True
            r.est_last = est1
            if exact_now:
                self.accept_test(r, c, dc, est1, est2, cn, thmin, m_other)
                if r.done:
                    return
        r.c_prev = c
        r.k_prev = k
        if k < r.first:
            r.target = min(2 * k, r.first)
        else:
            r.target = min(k + r.step, r.kcap)

    def accept_test(self, r, c, dc, est1, est2, cn, thmin, m_other):
        k = r.k
        rtol, atol = self.rtol, self.atol
        reached = r.kind != "exp" or self.t * thmin <= 60.0
        r.m_est = max(r.m_known, 0.5 * cn / math.sqrt(self.n))
        m_low = max(r.m_est, m_other)
        early = k < r.first
        ok1 = reached and est1 <= 0.25 * (atol + rtol * m_low)
        if ok1 and (not early or r.prev_ok):
            r.y, _ = self.combine(r, c, k)
            r.done = True
            return
        r.prev_ok = ok1
        go = reached and not early and est2 <= 0.25 * (atol + rtol * max(cn, m_other))
        if go and r.e_last is not None:
            pred = r.e_last[0] * est2 / max(r.e_last[1], 1e-300)
            go = pred <= 0.25 * (atol + rtol * max(r.m_known, m_other))
        if go:
            y, d = self.combine(r, c, k, extra=dc)
            M = float(np.abs(y).max())
            einf = float(np.abs(d).max())
            if einf <= 0.25 * (atol + rtol * max(M, m_other)):
                r.y = y
                r.done = True
                return
            r.m_known = M
            r.m_est = max(r.m_est, M)
            r.e_last = (einf, est2)

    def run_parts(self, parts):
        runs = [self.new_run(kind, b, len(parts)) for kind, b in parts]
        k = 0
        while True:
            active = [r for r in runs if not r.done]
            if not active:
                break
            target = min(r.target for r in active)
            if len(active) == 2:
                kend = self.steps2(active[0], active[1], k, target)
            else:
                kend = self.steps(active[0], k, target)
            k = kend
            for r in active:
                r.k = k
            for r in active:
                broke = not r.beta[k - 1] > self.brk
                if broke or k >= r.target:
                    others = [q.m_est if not q.done else q.m_final for q in runs if q is not r]
                    m_other = 0.5 * max(others) if others else 0.0
                    self.check(r, broke, m_other)
                    if r.done:
                        r.m_final = float(np.abs(r.y).max())
                        self.release_basis(r.V)
                        r.V = None
        return runs

    def maybe_reorder(self, parts):
        """RCM-reorder a scattered matrix when the expected Lanczos work is large."""
        self.reorder_checked = True
        if self.lib is None or self.scatter < 0.02:
            return
        nnz = self.a.size
        work = 0.0
        for kind, b in parts:
            beta0 = float(np.sqrt(b @ b))
            binf = float(np.abs(b).max())
            work += 0.6 * cheb_degree(self.tail, max(self.rtol * binf / (2.0 * beta0), 1e-17))
        work *= self.nreq
        # reordering costs roughly 40 matrix passes; it saves 15-25% per step
        if work < 250 or nnz < 50000:
            return
        try:
            import scipy.sparse as sp
            from scipy.sparse.csgraph import reverse_cuthill_mckee
            n = self.n
            A = sp.csr_matrix((self.a, self.ix, self.ip), shape=(n, n))
            perm = np.asarray(reverse_cuthill_mckee(A, symmetric_mode=True), dtype=np.int64)
            if perm.size != n:
                return
            iperm = np.empty(n, dtype=np.int64)
            iperm[perm] = np.arange(n, dtype=np.int64)
            rowlen = np.diff(self.ip)
            newlen = rowlen[perm]
            nip = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(newlen, out=nip[1:])
            src = np.repeat(self.ip[:-1][perm] - nip[:-1], newlen) + np.arange(nnz, dtype=np.int64)
            nix = iperm[self.ix[src]].astype(np.uint32)
            na = self.a[src]
            self.ip, self.ix, self.a = nip, nix, na
            self.perm, self.iperm = perm, iperm
        except Exception as exc:
            log("reorder skipped:", repr(exc))

    def solve(self, b0, b1):
        t = self.t
        mode = self.mode
        if mode == "identity":
            return b0.copy()
        if mode == "zero":
            return b0 + t * b1
        if mode == "diagonal":
            return self.dexp * b0 + self.dphi * b1
        if mode == "dense":
            Q = self.Q
            return Q @ (self.wexp * (Q.T @ b0) + self.wphi * (Q.T @ b1))
        parts = []
        # ||exp(-tL) b0|| <= exp(-t*lam_low) ||b0|| (lam_low <= lambda_min by
        # Gershgorin) and ||t*phi1(-tL) b1|| <= t ||b1||: parts whose bound is
        # below 0.05*atol are skipped
        nb0 = float(np.sqrt(b0 @ b0))
        if nb0 > 0.0 and not self.t * self.lam_low >= math.log(20.0 * nb0 / self.atol):
            parts.append(("exp", b0))
        if np.any(b1) and not t * float(np.sqrt(b1 @ b1)) <= 0.05 * self.atol:
            parts.append(("phi", b1))
        y = np.zeros(self.n)
        if not parts:
            return y
        if not self.reorder_checked:
            self.maybe_reorder(parts)
        if self.perm is not None:
            parts = [(kind, b[self.perm]) for kind, b in parts]
        for r in self.run_parts(parts):
            y += r.y
        if self.perm is not None:
            y = y[self.iperm]
        return y


class Run:
    pass


# ----------------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------------
def read_into(fin, arr):
    mv = memoryview(arr).cast("B")
    total = len(mv)
    pos = 0
    while pos < total:
        r = fin.readinto(mv[pos:])
        if not r:
            raise EOFError("unexpected end of input")
        pos += r


def write_all(fd, arr):
    mv = memoryview(arr).cast("B")
    pos = 0
    total = len(mv)
    while pos < total:
        pos += os.write(fd, mv[pos:])


def main():
    for fd in (0, 1):
        try:
            fcntl.fcntl(fd, F_SETPIPE_SZ, 1 << 20)
        except OSError:
            pass
    lib = load_native()
    # warm up lazily-initialised library paths
    eigh_tridiagonal(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5]))
    ive(np.arange(3.0), 1.0)
    dct(np.ones(8), type=2)
    fin = open(0, "rb", buffering=0, closefd=False)
    write_all(1, np.frombuffer(b"READY\n", dtype=np.uint8))

    head = bytearray(12)
    read_into(fin, head)
    if bytes(head[:8]) != b"FAMEXP2\n":
        raise SystemExit("bad magic")
    hlen = struct.unpack("<I", bytes(head[8:12]))[0]
    hbuf = bytearray(hlen)
    read_into(fin, hbuf)
    meta = json.loads(hbuf.decode("utf-8"))
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    nreq = int(meta["requests"])
    t = float(meta["t"])
    rtol = float(meta["rtol"])
    atol = float(meta["atol"])
    indptr = np.empty(n + 1, dtype="<u8")
    read_into(fin, indptr)
    indices = np.empty(nnz, dtype="<u4")
    if nnz:
        read_into(fin, indices)
    data = np.empty(nnz, dtype="<f8")
    if nnz:
        read_into(fin, data)
    try:
        solver = Solver(lib, n, indptr, indices, data, t, rtol, atol)
        solver.nreq = nreq
    except Exception as exc:
        log("setup failed:", repr(exc))
        solver = None
    bb = np.empty(2 * n, dtype="<f8")
    for _ in range(nreq):
        read_into(fin, bb)
        b0 = bb[:n].copy()
        b1 = bb[n:].copy()
        try:
            if solver is None:
                raise RuntimeError("no solver")
            y = solver.solve(b0, b1)
            if not np.all(np.isfinite(y)):
                raise FloatingPointError("nonfinite result")
        except Exception as exc:  # last-resort robust path
            log("primary solver failed:", repr(exc))
            y = fallback(n, indptr, indices, data, t, b0, b1)
        write_all(1, np.ascontiguousarray(y, dtype="<f8"))
    return 0


def fallback(n, indptr, indices, data, t, b0, b1):
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply
    if t == 0.0:
        return b0.copy()
    A = sp.csr_matrix((data, indices.astype(np.int64), indptr.astype(np.int64)), shape=(n, n))
    B = sp.bmat([[-t * A, sp.csr_matrix(t * b1.reshape(-1, 1))],
                 [None, sp.csr_matrix((1, 1))]], format="csr")
    v = np.concatenate([b0, [1.0]])
    return expm_multiply(B, v)[:n]


if __name__ == "__main__":
    code = main()
    os._exit(code)
