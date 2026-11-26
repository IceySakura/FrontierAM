import os
import sys
import json
import struct
import math
import ctypes

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.special import ive

HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(HERE, "libkernel.so"))


def _setup(name, argtypes, restype):
    f = getattr(lib, name)
    f.argtypes = argtypes
    f.restype = restype
    return f


I64P = ctypes.POINTER(ctypes.c_int64)
I32P = ctypes.POINTER(ctypes.c_int32)
DP = ctypes.POINTER(ctypes.c_double)

_spmv = _setup("fam_spmv", [ctypes.c_int, I64P, I32P, DP, DP, DP], None)
_inf = _setup("fam_inf_norm", [ctypes.c_int, I64P, I32P, DP], ctypes.c_double)
_isdiag = _setup("fam_is_diagonal", [ctypes.c_int, I64P, I32P, DP], ctypes.c_int)
_power = _setup("fam_power_upper", [ctypes.c_int, I64P, I32P, DP, ctypes.c_double, ctypes.c_int, DP], ctypes.c_double)
_cheb = _setup("fam_cheb", [ctypes.c_int, I64P, I32P, DP, ctypes.c_double, ctypes.c_double, DP, DP, ctypes.c_int, DP, DP, DP, DP], None)
_taylor = _setup("fam_taylor", [ctypes.c_int, I64P, I32P, DP, ctypes.c_double, DP, DP, ctypes.c_int, DP, DP, DP, DP], None)
_to_ell = _setup("fam_to_ell", [ctypes.c_int, I64P, I32P, DP, ctypes.c_int, I32P, DP], None)
_cheb_ell = _setup("fam_cheb_ell", [ctypes.c_int, ctypes.c_int, I32P, DP, ctypes.c_double, ctypes.c_double, DP, DP, ctypes.c_int, DP, DP, DP, DP], None)
_taylor_ell = _setup("fam_taylor_ell", [ctypes.c_int, ctypes.c_int, I32P, DP, ctypes.c_double, DP, DP, ctypes.c_int, DP, DP, DP, DP], None)
_gersh = _setup("fam_gersh_lower", [ctypes.c_int, I64P, I32P, DP], ctypes.c_double)
_inv = _setup("fam_inv_cheb", [ctypes.c_int, I64P, I32P, DP, ctypes.c_double, ctypes.c_double, DP, ctypes.c_int, DP, DP, DP], None)
_inv_ell = _setup("fam_inv_cheb_ell", [ctypes.c_int, ctypes.c_int, I32P, DP, ctypes.c_double, ctypes.c_double, DP, ctypes.c_int, DP, DP, DP], None)
_abs_upper = _setup("fam_abs_upper", [ctypes.c_int, I64P, I32P, DP, ctypes.c_double, ctypes.c_int, DP], ctypes.c_double)
_sell_plan = _setup("fam_sell_plan", [ctypes.c_int, I64P, I32P, DP, ctypes.c_int, I32P, I32P, I64P], ctypes.c_int64)
_sell_fill = _setup("fam_sell_fill", [ctypes.c_int, I64P, I32P, DP, I32P, I32P, I64P, I32P, DP], None)
_cheb_sell = _setup("fam_cheb_sell", [ctypes.c_int, I32P, I32P, DP, I32P, I64P, ctypes.c_double, ctypes.c_double, DP, DP, ctypes.c_int, DP, DP, DP, DP], None)


def ptr(a):
    return a.ctypes.data_as(DP)


def iptr(a):
    return a.ctypes.data_as(I64P)


def jptr(a):
    return a.ctypes.data_as(I32P)


def read_exact(f, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = f.readinto(view[got:])
        if not r:
            raise EOFError("short input")
        got += r
    return buf


def read_header():
    f = sys.stdin.buffer
    if read_exact(f, 8) != b"FAMEXP2\n":
        raise ValueError("bad magic")
    ln = struct.unpack("<I", read_exact(f, 4))[0]
    h = json.loads(read_exact(f, ln))
    n, nnz = h["n"], h["nnz"]
    ap = np.frombuffer(read_exact(f, 8 * (n + 1)), dtype="<u8").astype(np.int64, copy=False)
    aj = np.frombuffer(read_exact(f, 4 * nnz), dtype="<u4").astype(np.int32, copy=False)
    ax = np.frombuffer(read_exact(f, 8 * nnz), dtype="<f8")
    return h, ap, aj, ax


def cheb_coeffs_full(eta, eps, nmax=6000):
    # Standard Chebyshev convention: f(x) = c0/2 + sum_{k>=1} c_k T_k(x).
    N = min(nmax, max(8, int(math.ceil(10.0 * math.sqrt(max(eta, 1e-300)))) + 40))
    k = np.arange(N + 1, dtype=np.float64)
    A = 2.0 * ive(k, 0.5 * eta)
    # The phi coefficients involve differences of large suffix sums.  Compute
    # them in extended precision so that multiplication by a large t does not
    # amplify float64 cancellation noise.
    A_ld = np.array(A, dtype=np.longdouble)
    k_ld = np.array(k, dtype=np.longdouble)
    eta_ld = np.longdouble(eta)
    s1 = np.cumsum(A_ld[::-1])[::-1]
    s2 = np.cumsum((A_ld * k_ld)[::-1])[::-1]
    B_ld = np.zeros(N + 1, dtype=np.longdouble)
    B_ld[:-1] = (4.0 / eta_ld) * (s2[1:] - k_ld[:-1] * s1[1:])
    B_ld[0] = 2.0 * (1.0 - B_ld[1:].sum())
    B = np.array(B_ld, dtype=np.float64)
    tailA = np.cumsum(A[::-1])[::-1]
    tailB = np.cumsum(B[::-1])[::-1]
    # Find degree whose omitted tails are below eps.  Include coefficient 0 only in value.
    ok = np.flatnonzero((tailA[1:] <= eps) & (tailB[1:] <= eps))
    if ok.size:
        deg = int(ok[0]) + 1
    else:
        deg = N
    ca = A.copy()
    cb = B.copy()
    ca[0] *= 0.5
    cb[0] *= 0.5
    tailA = np.concatenate([tailA, [0.0]])
    tailB = np.concatenate([tailB, [0.0]])
    return ca, cb, deg, tailA, tailB


def cheb_coeffs(eta, eps, nmax=6000):
    ca, cb, deg, _, _ = cheb_coeffs_full(eta, eps, nmax)
    return ca[: deg + 1], cb[: deg + 1], deg


def taylor_coeffs(eta, eps):
    K = 0
    term = 1.0
    tail = 0.0
    c = [1.0]
    d = [1.0]  # includes t later; here coefficient of t*b1
    # exp tail and phi tail bounds at z=eta
    while K < 200:
        K += 1
        term *= eta / K
        # remainder exp <= term * eta/(K+1) / (1 - eta/(K+2)) when K large
        if K + 2 > eta:
            rem = term * (eta / (K + 1)) / max(1e-300, 1.0 - eta / (K + 2))
        else:
            rem = math.inf
        if rem <= eps:
            break
    c = [1.0 / math.factorial(k) for k in range(K + 1)]
    d = [1.0 / math.factorial(k + 1) for k in range(K + 1)]
    return np.array(c), np.array(d), K


def main():
    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()
    h, ap, aj, ax = read_header()
    n = h["n"]
    t = float(h["t"])
    rtol = float(h["rtol"])
    atol = float(h["atol"])
    requests = h["requests"]
    infnorm = _inf(n, iptr(ap), jptr(aj), ptr(ax))
    diag = bool(_isdiag(n, iptr(ap), jptr(aj), ptr(ax)))
    work = np.empty(3 * n, dtype=np.float64)
    state = None
    if t == 0.0:
        state = ("zero", 0.0)
    elif infnorm == 0.0:
        state = ("zero", t)
    elif diag:
        d = np.zeros(n, dtype=np.float64)
        # CSR diagonal entries are the only nonzero off-diagonal-free entries.
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ap))
        mask = aj == rows
        d[rows[mask]] = ax[mask]
        z = t * d
        e = np.exp(-z)
        phi = np.ones(n, dtype=np.float64)
        nz = z > 0
        phi[nz] = -np.expm1(-z[nz]) / z[nz]
        state = ("diag", e, t * phi)
    elif n <= 128:
        Ldense = np.zeros((n, n), dtype=np.float64)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ap))
        Ldense[rows, aj] = ax
        w, V = np.linalg.eigh(Ldense)
        z = t * w
        e = np.exp(-z)
        phi = np.ones(n, dtype=np.float64)
        nz = z > 0
        phi[nz] = -np.expm1(-z[nz]) / z[nz]
        state = ("dense", V, e, t * phi)
    else:
        # A Collatz-Wielandt bound on |L| gives a rigorous spectral upper
        # bound, often tighter than the raw infinity norm.
        if t * infnorm < 100.0:
            lam = infnorm
        else:
            iters = 10 if t * infnorm >= 1000.0 else 3
            lam = _abs_upper(n, iptr(ap), jptr(aj), ptr(ax), infnorm, iters, ptr(work))
        eta = t * lam
        scale = 1.0
        eps = 0.02 * (atol + rtol * scale) / math.sqrt(max(n, 1))
        lower = _gersh(n, iptr(ap), jptr(aj), ptr(ax)) if t * infnorm >= 40.0 else -1.0
        sqrt_n = math.sqrt(n)
        inv_state = None
        if lower > 0.0 and lam > lower:
            tl = t * lower
            decay = math.exp(-tl) if tl < 700.0 else 0.0
            if decay * sqrt_n <= 0.02 * atol and decay / lower * sqrt_n <= 0.02 * atol:
                hi = lam
                sqlo = math.sqrt(lower)
                sqhi = math.sqrt(hi)
                q = (sqhi - sqlo) / (sqhi + sqlo)
                eps_poly = 0.02 * (atol + rtol) / sqrt_n
                if q <= 0.0:
                    M = 0
                else:
                    M = max(0, int(math.ceil(math.log(eps_poly * (1.0 - q) * sqlo * sqhi / 2.0) / math.log(q) - 1.0)))
                if M <= 0.8 * (6.0 * math.sqrt(eta) + 30.0):
                    k = np.arange(M + 1, dtype=np.float64)
                    c = (2.0 / (sqlo * sqhi)) * ((-q) ** k)
                    c[0] *= 0.5
                    inv_state = (c, M, lower, hi)
        # Optional ELLPACK storage for short, fairly uniform rows.
        lens = np.diff(ap)
        maxk = int(lens.max()) if n else 0
        use_ell = False
        if maxk <= 16 and maxk * n <= 1.8 * max(len(ax), 1) and (requests >= 4 or t * infnorm >= 20.0):
            use_ell = True
            eaj = np.empty(n * maxk, dtype=np.int32)
            eax = np.empty(n * maxk, dtype=np.float64)
            _to_ell(n, iptr(ap), jptr(aj), ptr(ax), maxk, jptr(eaj), ptr(eax))
        else:
            eaj = eax = None
        sell = None
        if inv_state is None and eta > 4.0 and not use_ell and (requests >= 4 or t * infnorm >= 1000.0):
            nch = (n + 31) // 32
            rowperm = np.empty(n, dtype=np.int32)
            chunk_max = np.empty(nch, dtype=np.int32)
            offsets = np.empty(nch + 1, dtype=np.int64)
            total = _sell_plan(n, iptr(ap), jptr(aj), ptr(ax), 1024, jptr(rowperm), jptr(chunk_max), iptr(offsets))
            if 0 < total <= 1.5 * max(len(ax), 1):
                saj = np.empty(total, dtype=np.int32)
                sax = np.empty(total, dtype=np.float64)
                _sell_fill(n, iptr(ap), jptr(aj), ptr(ax), jptr(rowperm), jptr(chunk_max), iptr(offsets), jptr(saj), ptr(sax))
                sell = (rowperm, saj, sax, chunk_max, offsets)
        if inv_state is not None:
            c, M, lo, hi = inv_state
            state = ("inv", c, M, lo, hi, use_ell, eaj, eax, maxk)
        elif eta <= 4.0:
            c, d, K = taylor_coeffs(eta, eps)
            d *= t
            state = ("taylor", c, d, K, use_ell, eaj, eax, maxk)
        else:
            ca, cb, Nmax, tailA, tailB = cheb_coeffs_full(eta, eps)
            state = ("cheb", ca, cb, Nmax, eta, tailA, tailB, use_ell, eaj, eax, maxk, sell)
    out = sys.stdout.buffer
    for _ in range(requests):
        b0 = np.frombuffer(read_exact(sys.stdin.buffer, 8 * n), dtype="<f8")
        b1 = np.frombuffer(read_exact(sys.stdin.buffer, 8 * n), dtype="<f8")
        y = np.empty(n, dtype=np.float64)
        kind = state[0]
        if kind == "zero":
            if state[1] == 0.0:
                y[:] = b0
            else:
                y[:] = b0 + state[1] * b1
        elif kind == "diag":
            y[:] = state[1] * b0 + state[2] * b1
        elif kind == "dense":
            _, V, e, tph = state
            y[:] = V @ (e * (V.T @ b0)) + V @ (tph * (V.T @ b1))
        elif kind == "inv":
            _, c, M, lo, hi, use_ell, eaj, eax, maxk = state
            if use_ell:
                _inv_ell(n, maxk, jptr(eaj), ptr(eax), lo, hi, ptr(c), M, ptr(b1), ptr(y), ptr(work))
            else:
                _inv(n, iptr(ap), jptr(aj), ptr(ax), lo, hi, ptr(c), M, ptr(b1), ptr(y), ptr(work))
        elif kind == "taylor":
            _, c, d, K, use_ell, eaj, eax, maxk = state
            if use_ell:
                _taylor_ell(n, maxk, jptr(eaj), ptr(eax), t, ptr(c), ptr(d), K, ptr(b0), ptr(b1), ptr(y), ptr(work))
            else:
                _taylor(n, iptr(ap), jptr(aj), ptr(ax), t, ptr(c), ptr(d), K, ptr(b0), ptr(b1), ptr(y), ptr(work))
        else:
            _, ca, cb, Nmax, eta, tailA, tailB, use_ell, eaj, eax, maxk, sell = state
            if n < 4096:
                N = Nmax
            else:
                nb0 = float(np.linalg.norm(b0))
                nb1 = float(np.linalg.norm(b1))
                scale = max(float(np.max(np.abs(b0))), float(np.max(np.abs(b1))))
                target = 0.02 * (atol + rtol * scale)
                if nb0 == 0.0 and nb1 == 0.0:
                    y[:] = 0.0
                    out.write(y.tobytes()); out.flush(); continue
                err_after = tailA[1:] * nb0 + t * tailB[1:] * nb1
                idx = int(np.searchsorted(-err_after, -target))
                N = min(idx, Nmax)
            if sell is not None:
                rowperm, saj, sax, chunk_max, offsets = sell
                _cheb_sell(n, jptr(rowperm), jptr(saj), ptr(sax), jptr(chunk_max), iptr(offsets), t, eta, ptr(ca), ptr(cb), N, ptr(b0), ptr(b1), ptr(y), ptr(work))
            elif use_ell:
                _cheb_ell(n, maxk, jptr(eaj), ptr(eax), t, eta, ptr(ca), ptr(cb), N, ptr(b0), ptr(b1), ptr(y), ptr(work))
            else:
                _cheb(n, iptr(ap), jptr(aj), ptr(ax), t, eta, ptr(ca), ptr(cb), N, ptr(b0), ptr(b1), ptr(y), ptr(work))
            if n >= 4096:
                yinf = float(np.max(np.abs(y)))
                if yinf < 0.01 * scale:
                    target2 = 0.002 * (atol + rtol * yinf)
                    idx2 = int(np.searchsorted(-err_after, -target2))
                    N2 = min(idx2, Nmax)
                    if N2 > N:
                        N = N2
                        if sell is not None:
                            rowperm, saj, sax, chunk_max, offsets = sell
                            _cheb_sell(n, jptr(rowperm), jptr(saj), ptr(sax), jptr(chunk_max), iptr(offsets), t, eta, ptr(ca), ptr(cb), N, ptr(b0), ptr(b1), ptr(y), ptr(work))
                        elif use_ell:
                            _cheb_ell(n, maxk, jptr(eaj), ptr(eax), t, eta, ptr(ca), ptr(cb), N, ptr(b0), ptr(b1), ptr(y), ptr(work))
                        else:
                            _cheb(n, iptr(ap), jptr(aj), ptr(ax), t, eta, ptr(ca), ptr(cb), N, ptr(b0), ptr(b1), ptr(y), ptr(work))
        out.write(y.tobytes())
        out.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
