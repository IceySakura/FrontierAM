"""FAMEXP2 solver: y = exp(-tL) b0 + t*phi1(-tL) b1 by a fused Chebyshev expansion (see README.md)."""
import ctypes
import json
import os
import struct
import sys

import numpy as np
from scipy.fft import dct
from scipy.linalg import eigh_tridiagonal
from scipy.special import ive

HERE = os.path.dirname(os.path.abspath(__file__))
_dp = ctypes.POINTER(ctypes.c_double)


def _load_lib():
    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line
                    break
    except OSError:
        pass
    names = []
    if " avx2" in flags and " fma" in flags:
        names.append("libcheb_avx2.so")
    names.append("libcheb_generic.so")
    for name in names:
        path = os.path.join(HERE, name)
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        lib.spmv1.restype = None
        lib.rowabsmax.restype = ctypes.c_double
        lib.cheb_run.restype = ctypes.c_int64
        lib.bounds.restype = None
        lib.lanczos_steps.restype = ctypes.c_int64
        return lib
    return None


LIB = None if os.environ.get("FAMEXP_NOLIB") else _load_lib()


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class Operator:
    def __init__(self, n, indptr, indices, data):
        self.n, self.indptr, self.indices, self.data = n, indptr, indices, data
        self.nnz = data.size
        # values exactly representable in float32 (unweighted/integer/signed graphs): stream half the bytes
        self.data32 = None
        if self.nnz:
            d32 = data.astype(np.float32)
            if np.array_equal(d32.astype(np.float64), data):
                self.data32 = d32
        if LIB is None:
            from scipy.sparse import csr_matrix
            self.csr = csr_matrix((data, indices.astype(np.int64), indptr.astype(np.int64)), shape=(n, n))

    def matvec(self, x, out=None):
        if out is None:
            out = np.empty(self.n)
        if LIB is not None:
            LIB.spmv1(ctypes.c_int64(self.n), _p(self.indptr), _p(self.indices), _p(self.data), _p(x), _p(out))
        else:
            out[:] = self.csr @ x
        return out

    def gersh(self):
        if self.nnz == 0:
            return 0.0
        if LIB is not None:
            return float(LIB.rowabsmax(ctypes.c_int64(self.n), _p(self.indptr), _p(self.data)))
        return float(abs(self.csr).sum(axis=1).max())


def lanczos_bound(op, m_max, t, tol_rel=1e-4, m_low=150, ub_safe=np.inf):
    """Lanczos estimates of the extreme eigenvalues (no reorthogonalisation; only extremal Ritz
    values are used).  Returns (theta_max, r_max, beta_m, theta_min, r_min) where r are the
    residual norms of the extreme Ritz pairs.  Iterates until the top pair has converged and,
    when t*theta_min is large, also the bottom pair (or the iteration caps are reached)."""
    n = op.n
    rng = np.random.default_rng(20261112)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    vold = np.zeros(n)
    w = np.empty(n)
    st = np.zeros(1)
    cap = min(max(m_max, m_low), n)
    al = np.zeros(cap)
    be = np.zeros(cap)
    done = 0
    res = None
    while done < cap:
        chunk = min(10 if done < 10 else 5, cap - done)
        if LIB is not None:
            got = int(LIB.lanczos_steps(ctypes.c_int64(n), _p(op.indptr), _p(op.indices), _p(op.data),
                                        ctypes.c_int64(chunk), _p(v), _p(vold), _p(w), _p(st),
                                        _p(al[done:]), _p(be[done:])))
        else:
            got = 0
            for _ in range(chunk):
                op.matvec(v, w)
                alpha = float(np.dot(w, v))
                w -= alpha * v + st[0] * vold
                beta = float(np.linalg.norm(w))
                al[done + got], be[done + got] = alpha, beta
                got += 1
                if not beta > 1e-13 * max(abs(alpha), 1e-300):
                    break
                vold, v = v, w / beta
                st[0] = beta
        done += got
        a_, b_ = al[:done], be[:done]
        if not b_[-1] > 1e-13 * max(abs(a_[-1]), 1e-300):
            th = eigh_tridiagonal(a_, b_[:-1], eigvals_only=True)
            return float(th[-1]), 0.0, float(b_[-1]), float(th[0]), 0.0
        th, S = eigh_tridiagonal(a_, b_[:-1])
        beta = float(b_[-1])
        res = (float(th[-1]), abs(beta * S[-1, -1]), beta, float(th[0]), abs(beta * S[-1, 0]))
        if done >= 10:
            # stop early when the top is converged or cannot improve on the rigorous bound
            top_ok = res[1] <= tol_rel * abs(res[0]) or done >= m_max or res[0] >= 0.98 * ub_safe
            low_needed = t * res[3] > 8.0 and res[4] > 1e-3 * res[3]
            if top_ok and not low_needed:
                break
    return res


def cheb_coeffs(t, lb, ub):
    """Chebyshev coefficients on [lb,ub] of exp(-t x) and (1-exp(-t x))/x (= t*phi1(-t x))."""
    wid = ub - lb
    tau = 0.5 * t * wid
    N = 64
    need = 14.0 * np.sqrt(tau) + 1.3 * tau ** 0.75 + 80.0
    while N < need:
        N *= 2
    while True:
        k = np.arange(N + 1)
        c0 = 2.0 * ive(k, tau) * np.exp(-t * lb)
        c0[1::2] *= -1.0
        c0[0] *= 0.5
        s = np.cos(np.pi * k / N)
        x = lb + 0.5 * wid * (1.0 + s)
        x[-1] = lb
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = -np.expm1(-t * x) / x
        small = t * x < 1e-8
        f1[small] = t * (1.0 - 0.5 * t * x[small])
        c1 = dct(f1, type=1) / N
        c1[0] *= 0.5
        c1[-1] *= 0.5
        scale0 = max(np.max(np.abs(c0)), 1e-300)
        noise1 = 4e-16 * np.max(np.abs(f1)) * (1.0 + np.log2(N))
        tail0 = np.max(np.abs(c0[-N // 8:]))
        tail1 = np.max(np.abs(c1[-N // 8:]))
        if (tail0 <= 1e-19 * scale0 and tail1 <= noise1) or N >= 1 << 18:
            break
        N *= 2
    big = np.nonzero(np.abs(c1) > noise1)[0]
    c1[(big[-1] + 1 if big.size else 0):] = 0.0
    big = np.nonzero(np.abs(c0) > 1e-19 * scale0)[0]
    c0[(big[-1] + 1 if big.size else 0):] = 0.0
    return c0, c1


def read_exact_into(stream, arr):
    mv = memoryview(arr).cast("B")
    got = 0
    total = mv.nbytes
    while got < total:
        r = stream.readinto(mv[got:])
        if not r:
            raise EOFError("unexpected end of input")
        got += r


def read_exact(stream, nbytes):
    buf = bytearray(nbytes)
    read_exact_into(stream, buf)
    return bytes(buf)


class Solver:
    def __init__(self, op, t, rtol, atol, requests=1):
        self.op, self.t, self.rtol, self.atol = op, t, rtol, atol
        self.n = op.n
        self.trivial = None
        if t == 0.0:
            self.trivial = "t0"
            return
        if op.nnz == 0 or not np.any(op.data):
            self.trivial = "zero"
            return
        gu, gl, am, dmarg = self._bounds()
        ub = gu
        if dmarg >= -1e-10:
            ub = min(ub, am * (1.0 + 1e-9) + 1e-300)
        lb = max(0.0, gl) if dmarg >= 0.0 else 0.0
        lb = max(0.0, lb - 1e-12 * gu)
        self.safe_ub = ub * (1.0 + 1e-12)
        self.safe_lb = lb
        self.lb = lb
        self.ub = self.safe_ub
        self.work = np.empty(6 * self.n)
        self.info = np.zeros(8)
        self.frac = 0.05
        self.safety = 2.0
        self._setup()
        # tighten the interval by a short Lanczos run when the expansion is long
        if self.K_est > 40 and self.n > 20:
            m = 80 if self.K_est > 300 else 50
            # extra steps for the bottom Ritz pair: affordable share of the expansion work
            m_low = int(min(1000, max(2 * m, (1.0 if requests > 1 else 0.1) * self.K_est)))
            try:
                theta, r1, beta, tmin, rmin = lanczos_bound(op, m, t, m_low=m_low, ub_safe=self.safe_ub)
                if r1 <= 2e-3 * abs(theta):
                    ub2 = theta + 3.0 * r1 + 1e-7 * abs(theta)
                else:
                    ub2 = theta + beta
                lb2 = tmin - 3.0 * rmin - 1e-7 * abs(theta)
                changed = False
                if np.isfinite(ub2) and lb < ub2 < 0.99 * self.safe_ub:
                    self.ub = ub2
                    changed = True
                # a raised lower end only matters (for rounding) when exp(-t*lambda_min) is tiny
                if np.isfinite(lb2) and t * lb2 > 4.0 and lb2 > self.lb and lb2 < 0.5 * self.ub:
                    self.lb = lb2
                    changed = True
                if changed:
                    self._setup()
                print(f"lanczos theta={theta:.10g} r1={r1:.3g} beta={beta:.4g} tmin={tmin:.6g} rmin={rmin:.3g} "
                      f"lb={self.lb:.6g} ub={self.ub:.10g} safe={self.safe_ub:.10g}", file=sys.stderr)
            except Exception as exc:  # keep the safe bounds
                print("lanczos failed:", exc, file=sys.stderr)

    def _bounds(self):
        op = self.op
        if LIB is not None:
            diag = np.empty(self.n)
            out = np.zeros(4)
            LIB.bounds(ctypes.c_int64(self.n), _p(op.indptr), _p(op.indices), _p(op.data), _p(diag), _p(out))
            return float(out[0]), float(out[1]), float(out[2]), float(out[3])
        A = op.csr
        d = A.diagonal()
        R = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(d)
        gu = float(np.max(np.abs(d) + R))
        return gu, float(np.min(d - R)), gu, -1.0

    def _setup(self):
        if self.ub - self.lb < 1e-12 * self.ub:
            self.lb = max(0.0, self.ub * (1.0 - 1e-12))
        c0, c1 = cheb_coeffs(self.t, self.lb, self.ub)
        self.c0 = np.ascontiguousarray(c0)
        self.c1 = np.ascontiguousarray(c1)
        a0 = np.abs(c0)
        a1 = np.abs(c1)
        self.S0 = np.ascontiguousarray(np.concatenate([np.cumsum(a0[::-1])[::-1][1:], [0.0]]))
        self.S1 = np.ascontiguousarray(np.concatenate([np.cumsum(a1[::-1])[::-1][1:], [0.0]]))
        self.K = len(c0) - 1
        # rough length estimate (unit vectors, unit-size result)
        thr = 0.05 * (self.atol + self.rtol)
        idx = np.nonzero(self.S0 + self.S1 > thr)[0]
        self.K_est = int(idx[-1]) + 1 if idx.size else 0

    def _run(self, b0, b1, ub, c0, c1, S0, S1):
        op = self.op
        nb0 = bool(np.any(b0))
        nb1 = bool(np.any(b1))
        if nb0 and nb1:
            nvec, x0, x1, cc0, cc1, SS0, SS1 = 2, b0, b1, c0, c1, S0, S1
        elif nb1:
            nvec, x0, x1, cc0, cc1, SS0, SS1 = 1, b1, b1, c1, c1, S1, np.zeros_like(S1)
        else:
            nvec, x0, x1, cc0, cc1, SS0, SS1 = 1, b0, b0, c0, c0, S0, np.zeros_like(S0)
        y = np.empty(self.n)
        use32 = op.data32 is not None
        r = LIB.cheb_run(ctypes.c_int64(self.n), _p(op.indptr), _p(op.indices),
                         _p(op.data32 if use32 else op.data), ctypes.c_int(1 if use32 else 0),
                         ctypes.c_double(self.lb), ctypes.c_double(ub), ctypes.c_int64(self.K), _p(cc0), _p(cc1), _p(SS0), _p(SS1),
                         ctypes.c_double(self.atol), ctypes.c_double(self.rtol),
                         ctypes.c_double(self.frac), ctypes.c_double(self.safety),
                         ctypes.c_double(1.001 * np.linalg.norm(x0) + 1e-300),
                         ctypes.c_double(1.001 * np.linalg.norm(x1) + 1e-300), ctypes.c_int(nvec),
                         _p(x0), _p(x1), _p(y), _p(self.work), _p(self.info))
        return r, y

    def solve(self, b0, b1):
        if self.trivial == "t0":
            return b0.copy()
        if self.trivial == "zero":
            return b0 + self.t * b1
        if LIB is None:
            return self._solve_py(b0, b1)
        r, y = self._run(b0, b1, self.ub, self.c0, self.c1, self.S0, self.S1)
        print(f"cheb terms={r} lb={self.lb:.6g} ub={self.ub:.6g} safe={self.safe_ub:.6g} info={self.info[:4]}", file=sys.stderr)
        if r < 0 and (self.ub < self.safe_ub or self.lb > self.safe_lb):
            # divergence or no convergence: fall back permanently to the rigorous interval
            self.ub = self.safe_ub
            self.lb = self.safe_lb
            self._setup()
            r, y = self._run(b0, b1, self.ub, self.c0, self.c1, self.S0, self.S1)
            print(f"fallback cheb terms={r}", file=sys.stderr)
        return y

    def _solve_py(self, b0, b1):
        # pure NumPy/SciPy fallback (only if the compiled kernel is unavailable)
        op, ub = self.op, self.ub
        wid = ub - self.lb
        sh = 1.0 + 2.0 * self.lb / wid
        y = self.c0[0] * b0 + self.c1[0] * b1
        A = np.stack([b0, b1], axis=1)
        B = (2.0 / wid) * (op.csr @ A) - sh * A
        y += self.c0[1] * B[:, 0] + self.c1[1] * B[:, 1]
        M0 = max(np.abs(b0).max(), np.abs(B[:, 0]).max())
        M1 = max(np.abs(b1).max(), np.abs(B[:, 1]).max())
        k = 1
        while True:
            tol = self.atol + self.rtol * np.abs(y).max()
            if self.safety * (self.S0[k] * M0 + self.S1[k] * M1) <= self.frac * tol or k >= self.K:
                break
            k += 1
            C = (4.0 / wid) * (op.csr @ B) - (2.0 * sh) * B - A
            y += self.c0[k] * C[:, 0] + self.c1[k] * C[:, 1]
            M0 = max(M0, np.abs(C[:, 0]).max())
            M1 = max(M1, np.abs(C[:, 1]).max())
            A, B = B, C
        self.info[0] = k
        return y


def maybe_reorder(n, indptr, indices, data, t, rtol, atol, requests):
    """Reverse Cuthill-McKee reordering for long expansions on matrices with poor locality."""
    ip = indptr.astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ip))
    far = np.count_nonzero(np.abs(rows - indices.astype(np.int64)) > 4096)
    if far < 0.2 * indices.size:
        return None, indptr, indices, data
    # cheap length estimate from the Gershgorin bound
    gu = float(np.max(np.bincount(rows, weights=np.abs(data), minlength=n)))
    tau = 0.5 * t * gu
    k_est = np.sqrt(2.0 * tau * max(np.log(1.0 / rtol), 1.0)) + 10.0 if tau > 1 else 10.0
    if k_est * requests < 300:
        return None, indptr, indices, data
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    A = csr_matrix((data, indices.astype(np.int32), ip.astype(np.int32 if indices.size < 2**31 else np.int64)),
                   shape=(n, n))
    p = np.asarray(reverse_cuthill_mckee(A, symmetric_mode=True), dtype=np.int64)
    if p.size != n:
        return None, indptr, indices, data
    B = A[p][:, p].tocsr()
    B.sort_indices()
    print(f"rcm applied far={far / indices.size:.2f} k_est={k_est:.0f}", file=sys.stderr)
    return (p, np.ascontiguousarray(B.indptr.astype(np.uint64)), np.ascontiguousarray(B.indices.astype(np.uint32)),
            np.ascontiguousarray(B.data.astype(np.float64)))


def _warmup():
    try:
        cheb_coeffs(1.0, 0.0, 4.0)
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        A = csr_matrix((np.ones(3), np.array([0, 1, 2], dtype=np.int32), np.array([0, 1, 2, 3], dtype=np.int32)), shape=(3, 3))
        reverse_cuthill_mckee(A, symmetric_mode=True)
        A[np.array([2, 1, 0])][:, np.array([2, 1, 0])].tocsr().sort_indices()
        eigh_tridiagonal(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5]))
    except Exception:
        pass


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    _warmup()
    stdout.write(b"READY\n")
    stdout.flush()
    magic = read_exact(stdin, 8)
    if magic != b"FAMEXP2\n":
        raise SystemExit("bad magic")
    (length,) = struct.unpack("<I", read_exact(stdin, 4))
    meta = json.loads(read_exact(stdin, length).decode("utf-8"))
    n, nnz, requests = int(meta["n"]), int(meta["nnz"]), int(meta["requests"])
    t, rtol, atol = float(meta["t"]), float(meta["rtol"]), float(meta["atol"])
    indptr = np.empty(n + 1, dtype=np.uint64)
    indices = np.empty(max(nnz, 1), dtype=np.uint32)[:nnz]
    data = np.empty(max(nnz, 1), dtype=np.float64)[:nnz]
    read_exact_into(stdin, indptr)
    if nnz:
        read_exact_into(stdin, indices)
        read_exact_into(stdin, data)
    perm = None
    if nnz >= 50000 and t > 0:
        try:
            perm, indptr, indices, data = maybe_reorder(n, indptr, indices, data, t, rtol, atol, requests)
        except Exception as exc:  # keep the original ordering
            print("reorder failed:", exc, file=sys.stderr)
            perm = None
    op = Operator(n, indptr, indices, data)
    solver = Solver(op, t, rtol, atol, requests)
    b = np.empty(2 * n, dtype=np.float64)
    for _ in range(requests):
        read_exact_into(stdin, b)
        if perm is None:
            b0 = b[:n].copy()
            b1 = b[n:].copy()
            y = solver.solve(b0, b1)
        else:
            b0 = b[:n][perm]
            b1 = b[n:][perm]
            yp = solver.solve(b0, b1)
            y = np.empty(n)
            y[perm] = yp
        y = np.ascontiguousarray(y, dtype="<f8")
        stdout.write(memoryview(y).cast("B"))
        stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
