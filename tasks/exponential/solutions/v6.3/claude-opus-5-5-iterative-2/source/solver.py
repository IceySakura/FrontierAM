"""FAMEXP2 session solver: y = exp(-tL) b0 + t*phi1(-tL) b1, L sparse symmetric PSD.

Strategy (see README.md):
  * t == 0 or L == 0: closed form.
  * all connected components tiny: exact per-block eigendecomposition (Jacobi, in C).
  * small n: dense eigendecomposition.
  * otherwise: Chebyshev expansion on [lo, hi] (hi from Gershgorin, refined by Lanczos),
    fused C kernel, adaptive degree with a tail-based error estimate and a divergence
    monitor that falls back to the guaranteed Gershgorin interval.
"""
import ctypes
import json
import os
import struct
import sys

import numpy as np
from scipy.special import ive
from scipy.fft import dct
from scipy.linalg import eigh_tridiagonal

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = np.finfo(float).eps


DEBUG = bool(os.environ.get("EXPSOLVER_DEBUG"))
import time as _time
_T0 = [0.0]


def log(*a):
    if DEBUG:
        print("%.2fms" % (1e3 * (_time.perf_counter() - _T0[0])), *a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- C library
def _cpu_has_v3():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    fl = set(line.split(":", 1)[1].split())
                    return {"avx2", "fma", "bmi2"} <= fl
    except OSError:
        pass
    return False


def _load_lib():
    names = (["libexpkern_v3.so"] if _cpu_has_v3() else []) + ["libexpkern_gen.so"]
    for name in names:
        try:
            return ctypes.CDLL(os.path.join(HERE, name))
        except OSError:
            pass
    # last resort: compile into /tmp (gcc is part of the image)
    try:
        import subprocess
        import tempfile
        d = tempfile.mkdtemp(prefix="expk")
        out = os.path.join(d, "libexpkern.so")
        subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", out,
                        os.path.join(HERE, "expkern.c"), "-lm"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ctypes.CDLL(out)
    except Exception:
        return None


LIB = _load_lib()
I64, F64, VP = ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
if LIB is not None:
    LIB.gersh.argtypes = [I64, VP, VP, VP, VP]
    LIB.cheb1.argtypes = [I64, VP, VP, VP, F64, F64, VP, VP, I64, VP, VP, VP, VP,
                          F64, F64, F64, F64, VP]
    LIB.cheb1.restype = I64
    LIB.cheb2.argtypes = [I64, VP, VP, VP, F64, F64, VP, VP, VP, VP, I64, VP, VP, VP, VP, VP,
                          F64, F64, F64, F64, VP]
    LIB.cheb2.restype = I64
    LIB.lanczos.argtypes = [I64, VP, VP, VP, I64, VP, VP, VP, VP]
    LIB.lanczos.restype = I64
    LIB.components.argtypes = [I64, VP, VP, VP, VP, VP, VP, I64]
    LIB.components.restype = I64
    LIB.block_eig.argtypes = [I64, VP, VP, VP, I64, VP, VP, VP, VP, VP, VP, VP]
    LIB.block_apply.argtypes = [I64, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP]
    LIB.permute_csr.argtypes = [I64, VP, VP, VP, VP, VP, VP, VP, VP]
    LIB.permute_csr.restype = I64
    LIB.gather.argtypes = [I64, VP, VP, VP]
    LIB.scatter.argtypes = [I64, VP, VP, VP]


def P(a):
    return a.ctypes.data


# ----------------------------------------------------------------------------- wire I/O
def _setup_pipes():
    try:
        import fcntl
        for fd in (0, 1):
            try:
                fcntl.fcntl(fd, 1031, 1 << 20)  # F_SETPIPE_SZ
            except OSError:
                pass
    except ImportError:
        pass


def read_into(buf):
    mv = memoryview(buf).cast("B")
    total = mv.nbytes
    pos = 0
    while pos < total:
        r = os.readv(0, [mv[pos:]])
        if r <= 0:
            raise EOFError("unexpected end of input")
        pos += r


def read_bytes(k):
    b = bytearray(k)
    read_into(b)
    return bytes(b)


def write_all(arr):
    mv = memoryview(arr).cast("B")
    pos = 0
    while pos < mv.nbytes:
        pos += os.write(1, mv[pos:])


# ----------------------------------------------------------------------------- numerics
def g_of(x, t):
    """t*phi1(-t x) = (1 - exp(-t x)) / x, = t at x == 0."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    nz = x != 0.0
    out[nz] = -np.expm1(-t * x[nz]) / x[nz]
    out[~nz] = t
    return out


def f_of(x, t):
    return np.exp(-t * np.asarray(x, dtype=float))


def cheb_coeffs(t, lo, hi, need_c, need_d):
    """Chebyshev coefficients on [lo, hi] of exp(-t x) (c) and (1-exp(-t x))/x (d)."""
    h = 0.5 * (hi - lo)
    m = 0.5 * (hi + lo)
    beta = t * h
    K = int(np.sqrt(110.0 * beta) + 40)
    c = d = None
    if need_c:
        k = np.arange(K + 1)
        with np.errstate(over="ignore", under="ignore"):
            c = 2.0 * ive(k, beta) * np.exp(-t * lo)
        c[0] *= 0.5
        c[1::2] *= -1.0
        c[~np.isfinite(c)] = 0.0
    if need_d:
        N = 64
        while N < 2 * K + 2:
            N *= 2
        theta = np.pi * (np.arange(N) + 0.5) / N
        x = m + h * np.cos(theta)
        gv = g_of(x, t)
        dd = dct(gv, type=2) / N
        dd[0] *= 0.5
        d = dd[:K + 1].copy()
        floor = 32.0 * EPS * float(np.max(np.abs(gv)))
        big = np.nonzero(np.abs(d) > floor)[0]
        last = int(big[-1]) if big.size else 0
        d[last + 1:] = 0.0
    # trim trailing negligible coefficients
    last = 0
    if c is not None:
        sc = float(np.sum(np.abs(c)))
        nzc = np.nonzero(np.abs(c) > 1e-30 * max(sc, 1e-300))[0]
        last = max(last, int(nzc[-1]) if nzc.size else 0)
    if d is not None:
        nzd = np.nonzero(d)[0]
        last = max(last, int(nzd[-1]) if nzd.size else 0)
    K = last + 1
    if c is None:
        c = np.zeros(K + 1)
    if d is None:
        d = np.zeros(K + 1)
    c = np.ascontiguousarray(c[:K + 1])
    d = np.ascontiguousarray(d[:K + 1])
    if c.size < K + 1:
        c = np.concatenate([c, np.zeros(K + 1 - c.size)])
    if d.size < K + 1:
        d = np.concatenate([d, np.zeros(K + 1 - d.size)])

    def tails(v):
        a = np.abs(v)[::-1].cumsum()[::-1]
        return np.ascontiguousarray(np.concatenate([a, [0.0]]))

    return K, c, tails(c), d, tails(d)


def degree_estimate(tc, td, t, rtol):
    """Rough degree needed (for planning only)."""
    k1 = int(np.searchsorted(-tc, -1e-3 * rtol * max(tc[0], 1e-300)))
    k2 = int(np.searchsorted(-td, -1e-3 * rtol * max(td[0], 1e-300)))
    return max(k1, k2, 1)


class Session:
    def __init__(self, meta, ip, ix, a):
        self.n = n = int(meta["n"])
        self.nnz = int(meta["nnz"])
        self.t = float(meta["t"])
        self.rtol = float(meta["rtol"])
        self.atol = float(meta["atol"])
        self.requests = int(meta["requests"])
        self.ip, self.ix, self.a = ip, ix, a
        self.mode = None
        self.prepare()

    # ------------------------------------------------------------------ preparation
    def prepare(self):
        n, t = self.n, self.t
        if t == 0.0:
            self.mode = "identity"
            return
        gi = np.zeros(6)
        LIB.gersh(n, P(self.ip), P(self.ix), P(self.a), P(gi))
        self.G, self.glo, self.noff, self.dmax = float(gi[0]), float(gi[1]), int(gi[2]), float(gi[3])
        log("G", self.G, "glo", self.glo, "noff", self.noff)
        if self.G == 0.0:
            self.mode = "zero"
            return
        if n <= 300:
            self.prepare_dense()
            return
        # connected components -> exact small blocks
        label = np.empty(n, dtype=np.int64)
        queue = np.empty(n, dtype=np.int64)
        mx = np.zeros(1, dtype=np.int64)
        nc = int(LIB.components(n, P(self.ip), P(self.ix), P(self.a), P(label), P(queue), P(mx), 24))
        smax = int(mx[0])
        log("components", nc, "max", smax)
        lo, hi = self.interval(self.G)
        K, c, tc, d, td = cheb_coeffs(t, lo, hi, True, True)
        kest = degree_estimate(tc, td / max(t, 1e-300), t, self.rtol)
        self.kest = kest
        cheb_cost = self.requests * kest * (self.nnz * 1.6 + n * 6.0)
        if nc > 0 and smax <= 24:
            block_cost = n * (smax * smax * 30.0 + 20.0) + self.requests * n * (smax * 6.0 + 10)
            log("block_cost", block_cost, "cheb_cost", cheb_cost)
            if block_cost < cheb_cost:
                self.prepare_blocks(label, nc, smax)
                return
        self.prepare_cheb(kest)

    def interval(self, hi):
        lo = self.glo * (1.0 - 1e-12) if self.glo > 0 else -1e-13 * hi
        hi = hi * (1.0 + 1e-12) + 1e-300
        return lo, hi

    def prepare_dense(self):
        n = self.n
        A = np.zeros((n, n))
        rows = np.repeat(np.arange(n), np.diff(self.ip.astype(np.int64)))
        np.add.at(A, (rows, self.ix.astype(np.int64)), self.a)
        A = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(A)
        self.dw, self.dV = w, V
        self.fw, self.gw = f_of(w, self.t), g_of(w, self.t)
        self.mode = "dense"

    def prepare_blocks(self, label, nc, smax):
        n = self.n
        order = np.argsort(label, kind="stable").astype(np.int64)
        counts = np.bincount(label, minlength=nc).astype(np.int64)
        comp_off = np.zeros(nc + 1, dtype=np.int64)
        np.cumsum(counts, out=comp_off[1:])
        vec_off = np.zeros(nc + 1, dtype=np.int64)
        np.cumsum(counts * counts, out=vec_off[1:])
        loc = np.empty(n, dtype=np.int64)
        lam = np.empty(n)
        vec = np.empty(int(vec_off[-1]))
        work = np.empty(2 * smax * smax + 16)
        LIB.block_eig(n, P(self.ip), P(self.ix), P(self.a), nc, P(order), P(comp_off),
                      P(vec_off), P(loc), P(lam), P(vec), P(work))
        self.b_order, self.b_off, self.b_voff, self.b_vec = order, comp_off, vec_off, vec
        self.b_nc = nc
        self.b_f = f_of(lam, self.t)
        self.b_g = g_of(lam, self.t)
        self.b_work = np.empty(smax + 16)
        self.mode = "blocks"

    def prepare_cheb(self, kest):
        n, t = self.n, self.t
        hi = self.G
        work_units = self.requests * kest
        if work_units >= 120 and n > 1000:
            m = int(min(80, max(16, 0.08 * work_units))) if self.requests > 1 else int(min(48, max(16, 0.1 * kest)))
            hi = self.lanczos_bound(m)
        self.lo, self.hi = self.interval(hi)
        self.setup_coeffs()
        self.y = np.empty(n)
        self.wa = np.empty(2 * n)
        self.wb = np.empty(2 * n)
        self.info = np.zeros(4)
        self.mode = "cheb"

    def setup_coeffs(self):
        K, c, tc, d, td = cheb_coeffs(self.t, self.lo, self.hi, True, True)
        self.K, self.c, self.tc, self.d, self.td = K, c, tc, d, td
        h = 0.5 * (self.hi - self.lo)
        mid = 0.5 * (self.hi + self.lo)
        self.sa = 1.0 / h
        self.sb = -mid / h
        log("interval", self.lo, self.hi, "K", K)

    def lanczos_bound(self, m):
        n = self.n
        rng = np.random.default_rng(1234567)
        v0 = rng.standard_normal(n)
        v0 /= np.linalg.norm(v0)
        alpha = np.zeros(m)
        beta = np.zeros(m)
        work = np.empty(3 * n)
        mm = int(LIB.lanczos(n, P(self.ip), P(self.ix), P(self.a), m, P(v0), P(alpha), P(beta), P(work)))
        al, be = alpha[:mm], beta[:mm]
        if mm == 1:
            theta = np.array([al[0]])
            S = np.ones((1, 1))
        else:
            theta, S = eigh_tridiagonal(al, be[:mm - 1])
        th1 = float(theta[-1])
        r1 = float(abs(be[mm - 1] * S[-1, -1]))
        est = th1 + r1 + 1e-6 * abs(th1)
        log("lanczos m", mm, "theta1", th1, "r1", r1, "G", self.G, "est", est)
        return min(self.G, est)

    # ------------------------------------------------------------------ requests
    def solve(self, b0, b1):
        mode, t = self.mode, self.t
        if mode == "identity":
            return b0
        if mode == "zero":
            return b0 + t * b1
        if mode == "dense":
            V = self.dV
            return V @ (self.fw * (V.T @ b0) + self.gw * (V.T @ b1))
        if mode == "blocks":
            y = np.empty(self.n)
            LIB.block_apply(self.b_nc, P(self.b_order), P(self.b_off), P(self.b_voff), P(self.b_vec),
                            P(self.b_f), P(self.b_g), P(b0), P(b1), P(y), P(self.b_work))
            return y
        return self.solve_cheb(b0, b1)

    def solve_cheb(self, b0, b1):
        n = self.n
        z0 = not b0.any()
        z1 = not b1.any()
        if z0 and z1:
            return np.zeros(n)
        safety = 20.0
        growlim = 1.25
        for attempt in range(3):
            y, info = self.y, self.info
            if z0:
                k = LIB.cheb1(n, P(self.ip), P(self.ix), P(self.a), self.sa, self.sb, P(self.d), P(self.td),
                              self.K, P(b1), P(y), P(self.wa), P(self.wb), self.atol, self.rtol,
                              safety, growlim, P(info))
            elif z1:
                k = LIB.cheb1(n, P(self.ip), P(self.ix), P(self.a), self.sa, self.sb, P(self.c), P(self.tc),
                              self.K, P(b0), P(y), P(self.wa), P(self.wb), self.atol, self.rtol,
                              safety, growlim, P(info))
            else:
                k = LIB.cheb2(n, P(self.ip), P(self.ix), P(self.a), self.sa, self.sb, P(self.c), P(self.tc),
                              P(self.d), P(self.td), self.K, P(b0), P(b1), P(y), P(self.wa), P(self.wb),
                              self.atol, self.rtol, safety, growlim, P(info))
            log("cheb degree", k, "of", self.K, "tail", info[0], "M", info[1], "ymax", info[2])
            if k >= 0:
                return y
            # diverged: interval too small -> guaranteed Gershgorin interval
            # (PSD is guaranteed, so only round-off can push eigenvalues below 0)
            G = self.G
            self.lo, self.hi = -1e-13 * G, G * (1.0 + 1e-10) + 1e-300
            if attempt == 1:
                self.hi *= 1.01
                self.lo = -1e-8 * G
            self.setup_coeffs()
        return y


def main():
    _setup_pipes()
    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()
    head = read_bytes(12)
    _T0[0] = _time.perf_counter()
    if head[:8] != b"FAMEXP2\n":
        raise SystemExit(2)
    (jl,) = struct.unpack("<I", head[8:12])
    meta = json.loads(read_bytes(jl).decode("utf-8"))
    n, nnz = int(meta["n"]), int(meta["nnz"])
    ip = np.empty(n + 1, dtype=np.int64)
    ix = np.empty(nnz, dtype=np.int32)
    a = np.empty(nnz, dtype=np.float64)
    read_into(ip)
    read_into(ix)
    read_into(a)
    log("matrix read", n, nnz)
    sess = Session(meta, ip, ix, a)
    log("prepared", sess.mode)
    b = np.empty(2 * n)
    for _ in range(int(meta["requests"])):
        read_into(b)
        log("request read")
        b0 = b[:n]
        b1 = b[n:]
        y = sess.solve(b0, b1)
        y = np.ascontiguousarray(y, dtype=np.float64)
        write_all(y)
        log("written")
    os._exit(0)


if __name__ == "__main__":
    main()
