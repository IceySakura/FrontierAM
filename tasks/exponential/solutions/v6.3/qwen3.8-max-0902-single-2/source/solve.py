#!/usr/bin/env python3
"""Fast exponential propagation for symmetric PSD sparse operators.

Computes, for each request pair (b0, b1),

    y = exp(-t*L) b0 + t*phi1(-t*L) b1,     phi1(z) = (exp(z)-1)/z

with two independent symmetric Lanczos processes (one per right hand side).
The stopping test uses the *exact* difference between successive Krylov
approximations, which is available for free in the small projected space
because the Lanczos basis is orthonormal:

    ||y_m - y_{m'}||_2 = beta * || s_m - pad(s_{m'}) ||_2

A geometric extrapolation of those differences gives the remaining error.
Periodic full re-orthogonalisation keeps the basis orthonormal without paying
O(m n) every step.
"""

import json
import math
import struct
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eigh_tridiagonal

STDIN = sys.stdin.buffer
STDOUT = sys.stdout.buffer


def hphi1(x):
    """h(x) = phi1(-x) = (1-exp(-x))/x, with h(0)=1, elementwise."""
    out = np.empty_like(x)
    big = np.abs(x) > 1e-8
    xb = x[big]
    out[big] = -np.expm1(-xb) / xb
    xs = x[~big]
    out[~big] = 1.0 - 0.5 * xs + xs * xs / 6.0
    return out


class Term(object):
    """Krylov approximation of one of the two terms."""

    __slots__ = ("A", "n", "t", "kind", "beta0", "V", "al", "be", "m", "mcap",
                 "active", "exact", "s", "sprev", "mprev", "est", "rho",
                 "since", "nnz", "ascale", "k")

    def __init__(self, A, b, t, kind, n, mcap, nnz):
        self.A = A
        self.n = n
        self.t = t
        self.kind = kind            # 0 -> exp(-tL)b ; 1 -> t*phi1(-tL)b
        self.nnz = nnz
        self.beta0 = math.sqrt(float(b @ b))
        self.mcap = mcap
        rows = min(mcap, 16)
        self.V = np.empty((rows, n))
        self.al = np.empty(mcap)
        self.be = np.empty(mcap)
        np.divide(b, self.beta0, out=self.V[0])
        self.m = 1                  # basis vectors currently held
        self.active = True
        self.exact = False
        self.s = None
        self.sprev = None
        self.mprev = 0
        self.est = np.inf
        self.rho = 0.5
        self.since = 0
        self.ascale = 0.0
        self.k = 0

    # ------------------------------------------------------------------ step
    def step(self):
        m = self.m
        if m >= self.mcap:
            self.k = m
            self.active = False
            return
        V = self.V
        v = V[m - 1]
        w = self.A._matmul_vector(v)
        a = float(w @ v)
        self.al[m - 1] = a
        self.k = m                # order of the projected tridiagonal matrix
        w -= a * v
        if m > 1:
            w -= self.be[m - 2] * V[m - 2]

        c = V[:m] @ w
        w -= c @ V[:m]
        da = c[m - 1]
        bn0 = math.sqrt(float(w @ w))
        if abs(da) > 0.05 * (bn0 + 1e-300):
            c2 = V[:m] @ w
            w -= c2 @ V[:m]
            da += c2[m - 1]
        self.al[m - 1] = a + da
        aa = abs(a + da)
        if aa > self.ascale:
            self.ascale = aa

        bn = math.sqrt(float(w @ w))
        if bn <= 1e-13 * (self.ascale + 1e-300):
            self.exact = True       # invariant subspace: answer is exact
            self.active = False
            self.est = 0.0
            return
        self.be[m - 1] = bn
        self.m = m + 1
        if self.m > V.shape[0]:
            if self.m > self.mcap:
                self.m = m
                self.k = m
                self.active = False
                return
            newr = min(self.mcap, max(2 * V.shape[0], self.m))
            Vn = np.empty((newr, self.n))
            Vn[:V.shape[0]] = V
            self.V = Vn
            V = Vn
        np.divide(w, bn, out=V[m])
        self._maybe_estimate()

    def _reorth_interval(self, m):
        """Amortise re-orthogonalisation to ~50% of one SpMV."""
        r = 8.0 * m * self.n / max(6.0 * self.nnz, 1.0)
        return 3 if r < 3.0 else (10 if r > 10.0 else int(r))

    # ------------------------------------------------------- small problems
    def small_s(self):
        """s_m with f(-t T_m) e1 * beta, for the current Krylov dimension."""
        m = self.k                # order of the projected tridiagonal matrix
        theta, Q = eigh_tridiagonal(self.al[:m], self.be[:m - 1],
                                    check_finite=False)
        np.maximum(theta, 0.0, out=theta)
        q1 = Q[0, :]
        if self.kind == 0:
            f = np.exp(-self.t * theta)
        else:
            f = self.t * hphi1(self.t * theta)
        return self.beta0 * (Q @ (f * q1))

    def _maybe_estimate(self):
        m = self.k
        freq = 1 if m <= 24 else (2 if m <= 64 else 4)
        if (m % freq) == 0:
            self._estimate()

    def _estimate(self):
        s = self.small_s()
        sp = self.s
        if sp is None:
            d = s
            gap = len(s)
        else:
            gap = len(s) - self.mprev
            d = s.copy()
            d[:len(sp)] -= sp
        e = math.sqrt(float(d @ d))
        if sp is not None and gap >= 1 and e > 0.0 and self.est < np.inf and self.est > 0.0:
            r = (e / self.est) ** (1.0 / gap)
            self.rho = min(max(r, 0.05), 0.75)
        self.est = e
        self.sprev = sp
        self.s = s
        self.mprev = len(s)

    def error_estimate(self):
        if self.exact:
            return 0.0
        f = self.rho / (1.0 - self.rho)
        return max(f, 4.0) * self.est

    # ---------------------------------------------------------------- output
    def accumulate(self, y):
        s = self.s
        if s is None or len(s) != self.k:
            s = self.small_s()
            self.s = s
            self.mprev = len(s)
        y += s @ self.V[:len(s)]


def solve_pair(A, b0, b1, t, atol, rtol, n, mcap, nnz):
    if t == 0.0:
        return b0.copy()
    terms = []
    if float(b0 @ b0) > 0.0:
        terms.append(Term(A, b0, t, 0, n, mcap, nnz))
    if float(b1 @ b1) > 0.0:
        terms.append(Term(A, b1, t, 1, n, mcap, nnz))
    if not terms:
        return np.zeros(n)

    S = 0.0                          # running estimate of ||y||_inf
    cps = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    guard = 0
    while True:
        live = [tm for tm in terms if tm.active]
        if not live:
            break
        for tm in live:
            tm.step()
            if tm.active and tm.error_estimate() <= 0.1 * (atol + rtol * S):
                tm.active = False
        guard += 1
        if guard in cps:
            y = np.zeros(n)
            for tm in terms:
                tm.accumulate(y)
            S = float(np.max(np.abs(y)))
        if guard > 3000:
            break

    y = np.zeros(n)
    for tm in terms:
        tm.accumulate(y)
    # safety net: the final tolerance uses the true ||y||_inf
    S2 = float(np.max(np.abs(y)))
    tol = 0.1 * (atol + rtol * S2)
    if S2 < S and any(tm.error_estimate() > tol for tm in terms):
        for tm in terms:
            if not tm.exact and tm.error_estimate() > tol:
                tm.active = True
        while True:
            live = [tm for tm in terms if tm.active]
            if not live:
                break
            for tm in live:
                tm.step()
                if tm.error_estimate() <= tol:
                    tm.active = False
            guard += 1
            if guard > 6000:
                break
        y = np.zeros(n)
        for tm in terms:
            tm.accumulate(y)
    return y


def main():
    STDOUT.write(b"READY\n")
    STDOUT.flush()
    rd = STDIN.read
    if rd(8) != b"FAMEXP2\n":
        raise SystemExit("bad magic")
    ln = struct.unpack("<I", rd(4))[0]
    meta = json.loads(rd(ln))
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    reqs = int(meta["requests"])
    t = float(meta["t"])
    rtol = float(meta["rtol"])
    atol = float(meta["atol"])

    indptr = np.frombuffer(rd(8 * (n + 1)), dtype="<u8")
    indices = np.frombuffer(rd(4 * nnz), dtype="<u4")
    data = np.frombuffer(rd(8 * nnz), dtype="<f8")

    if nnz == 0:
        A = csr_matrix((n, n))
    else:
        A = csr_matrix((np.ascontiguousarray(data, dtype=np.float64),
                        np.ascontiguousarray(indices, dtype=np.int32),
                        np.ascontiguousarray(indptr, dtype=np.int32)),
                       shape=(n, n))

    dense = n <= 1000
    if dense:
        Ld = np.asarray(A.todense(), dtype=np.float64)
        Ld = 0.5 * (Ld + Ld.T)
        theta, Q = np.linalg.eigh(Ld)
        expt = np.exp(-t * theta)
        phit = (t * hphi1(t * theta)) if t != 0.0 else None

    budget = 2.2e9
    mcap = int(budget / max(2.0 * n * 8.0, 1.0))
    mcap = max(24, min(mcap, 1500))

    write = STDOUT.write
    flush = STDOUT.flush
    for _ in range(reqs):
        b0 = np.frombuffer(rd(8 * n), dtype="<f8").astype(np.float64)
        b1 = np.frombuffer(rd(8 * n), dtype="<f8").astype(np.float64)
        if dense:
            if t == 0.0:
                y = b0.copy()
            else:
                y = Q @ (expt * (Q.T @ b0)) + Q @ (phit * (Q.T @ b1))
        else:
            y = solve_pair(A, b0, b1, t, atol, rtol, n, mcap, nnz)
        write(np.ascontiguousarray(y, dtype="<f8").tobytes())
        flush()


if __name__ == "__main__":
    main()
