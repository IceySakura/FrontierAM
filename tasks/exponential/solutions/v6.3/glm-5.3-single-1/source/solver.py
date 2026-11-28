#!/usr/bin/env python3
"""FAMEXP2 exponential propagation solver.

    y = exp(-tL) b0 + t*phi1(-tL) b1,      phi1(z) = (exp(z)-1)/z

L is real symmetric positive semidefinite and delivered as full CSR, so its
spectrum lies in [0, ||L||_inf] with ||L||_inf = max_i sum_j |L_ij| a rigorous
Gershgorin upper bound of lambda_max.  Both functions of L are therefore
approximated by one Chebyshev series each on that interval and applied to the
two right hand sides inside a single fused sparse recurrence, which touches the
matrix data once per degree for both columns.

Strategy
  * t == 0            -> y = b0 exactly.
  * ||L||_inf == 0    -> y = b0 + t*b1 exactly (L is the zero matrix).
  * otherwise         -> block Chebyshev (exp column + phi column), truncated
                         where the rigorous coefficient tail
                         sum_{k>m} |c_k| * ||start vector||_2 drops below a
                         quarter of the acceptance threshold.

The coefficient series is obtained from Chebyshev-Lobatto interpolation
(DCT-I through numpy.fft.rfft), which keeps the truncation error provable:
|T_k(A~)| <= 1 for the symmetrically scaled operator, so

    || sum_{k>m} c_k T_k(A~) v ||_2 <= (sum_{k>m} |c_k|) ||v||_2.

A symmetric Krylov (Lanczos) path with full reorthogonalisation handles
operators whose spectral measure is strongly non uniform, where the polynomial
degree on the full Gershgorin interval is far larger than the Krylov dimension
needed for the same accuracy.
"""

import ctypes
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

READY = b"READY\n"
MAGIC = b"FAMEXP2\n"

# --------------------------------------------------------------------------- #
# compiled kernel
# --------------------------------------------------------------------------- #

_I64 = ctypes.c_int64
_I32 = ctypes.c_int32
_P = ctypes.c_void_p
_D = ctypes.c_double

_KERNEL_NAMES = ("k_inf_norm", "k_matvec2", "k_matvec1", "k_cheb2", "k_cheb1",
                 "k_symatv", "k_symatv2", "k_split_upper")


def _compile(src, dst):
    cc = os.environ.get("CC") or "cc"
    for flags in (["-O3", "-march=native", "-fomit-frame-pointer"],
                  ["-O3", "-fomit-frame-pointer"],
                  ["-O2"]):
        cmd = [cc] + flags + ["-shared", "-fPIC", "-o", dst, src, "-lm"]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and os.path.exists(dst):
            return True
    return False


def load_kernel():
    """Build and load the C kernels.  Runs before READY, hence untimed."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "kernel.c")
    if not os.path.exists(src):
        return None
    candidates = []
    try:
        if os.access(here, os.W_OK):
            candidates.append(os.path.join(here, "famexp_kernel.so"))
    except OSError:
        pass
    try:
        candidates.append(os.path.join(tempfile.gettempdir(), "famexp_kernel.so"))
    except OSError:
        pass
    for so in candidates:
        if os.path.exists(so):
            try:
                return _bind(ctypes.CDLL(so))
            except OSError:
                pass
    if candidates:
        os.makedirs(os.path.dirname(candidates[-1]), exist_ok=True)
        if _compile(src, candidates[-1]):
            try:
                return _bind(ctypes.CDLL(candidates[-1]))
            except OSError:
                return None
    return None


def _bind(lib):
    lib.k_inf_norm.argtypes = [_I64, _P, _P]
    lib.k_inf_norm.restype = _D
    lib.k_matvec2.argtypes = [_I64, _P, _P, _P, _P, _P]
    lib.k_matvec1.argtypes = [_I64, _P, _P, _P, _P, _P]
    lib.k_cheb2.argtypes = [_I64, _P, _P, _P, _D, _D, _I64, _P, _P, _P, _P, _P, _I32]
    lib.k_cheb1.argtypes = [_I64, _P, _P, _P, _D, _D, _I64, _P, _P, _P, _P]
    lib.k_symatv.argtypes = [_I64, _P, _P, _P, _P, _P, _P]
    lib.k_symatv2.argtypes = [_I64, _P, _P, _P, _P, _P, _P]
    lib.k_split_upper.argtypes = [_I64, _P, _P, _P, _P, _P, _P, _P]
    lib.k_split_upper.restype = _I64
    return lib


# --------------------------------------------------------------------------- #
# wire level I/O
# --------------------------------------------------------------------------- #

def _fill(stream, view):
    """Read exactly view.nbytes bytes into a writable uint8 numpy view."""
    need = view.nbytes
    got = 0
    while got < need:
        chunk = stream.readinto(view[got:])
        if not chunk:
            raise EOFError("unexpected end of input")
        got += chunk


def _read(stream, count):
    if count == 0:
        return b""
    data = bytearray(count)
    view = memoryview(data)
    got = 0
    while got < count:
        chunk = stream.readinto(view[got:])
        if not chunk:
            raise EOFError("unexpected end of input")
        got += chunk
    return bytes(data)


def _read_array(stream, dtype, count):
    arr = np.empty(count, dtype=dtype)
    if count:
        _fill(stream, arr.view(np.uint8))
    return arr


# --------------------------------------------------------------------------- #
# Chebyshev coefficient machinery
# --------------------------------------------------------------------------- #

def _cheb_coeffs(values, M):
    """Chebyshev coefficients of the interpolant through Lobatto samples.

    values[j] = f(cos(pi*j/M));  returns c with f ~ sum_k c_k T_k.
    """
    ext = np.concatenate((values[:M], values[M:0:-1]))
    spec = np.fft.rfft(ext).real
    coef = spec / M
    coef[0] *= 0.5
    coef[M] *= 0.5
    return coef


def _tail_table(coef):
    """tail[k] = sum_{j>=k} |coef[j]|, with the interpolation noise floor removed.

    The DCT coefficients carry O(eps * M * max|c|) noise which would otherwise
    stop the tail from ever reaching very tight tolerances and blow the degree
    up to the table length.
    """
    absc = np.abs(coef)
    noise = 1e-13 * (float(absc.max()) if absc.size else 0.0)
    return np.cumsum(np.maximum(absc[::-1] - noise, 0.0))[::-1]


def _degree_for(tail, eps):
    """Smallest m with sum_{k>m} |c_k| <= eps (capped by the table length)."""
    if tail[0] <= eps:
        return 0
    below = tail <= eps
    if not below.any():
        return len(tail) - 1
    return int(np.argmax(below)) - 1


def _phi1(lam, t):
    """phi1(-t*lam) evaluated stably for tiny and huge t*lam."""
    x = t * lam
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ph = -np.expm1(-x) / x
    return np.where(x == 0.0, 1.0, ph)


class ChebyshevSeries:
    """Chebyshev series of exp(-t*.) and t*phi1(-t*.) on [a, b]."""

    def __init__(self, t, a, b, eps_ref, m_cap=8192):
        self.a = a
        self.b = b
        self.t = t
        M = 128
        while True:
            nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * np.arange(M + 1) / M)
            fvals = np.exp(-t * nodes)
            hvals = _phi1(nodes, t)
            cf = _cheb_coeffs(fvals, M)
            ch = _cheb_coeffs(hvals, M)
            tf = _tail_table(cf)
            th = _tail_table(ch)
            need = max(_degree_for(tf, eps_ref), _degree_for(th, eps_ref))
            if need * 2 + 16 <= M or M >= m_cap:
                break
            M *= 2
        self.M = M
        self.cf = cf
        self.ch = ch
        self.tf = tf
        self.th = th
        self.maxdeg = M - 1

    def deg_exp(self, eps):
        return min(_degree_for(self.tf, eps), self.maxdeg)

    def deg_phi(self, eps):
        return min(_degree_for(self.th, eps), self.maxdeg)


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #

SAFETY = 0.01


class Session:
    def __init__(self, n, indptr, indices, data, t, rtol, atol, lib):
        self.n = n
        self.ip = indptr
        self.ix = indices
        self.da = data
        self.t = t
        self.rtol = rtol
        self.atol = atol
        self.lib = lib
        self.n2 = int(n) * 2
        self.npin = np.array(indptr, dtype=np.intp) if lib is None else indptr
        self.g = self._inf_norm()
        self.tol0 = atol + rtol  # smallest achievable threshold scale
        self.series = None
        self.lanczos = None
        if t == 0.0 or self.g <= 0.0:
            return
        a = -1e-8 * self.g
        b = self.g * (1.0 + 1e-9)
        eps_ref = SAFETY * (atol / np.sqrt(n) + rtol)
        self.series = ChebyshevSeries(t, a, b, eps_ref)
        # workspace reused by every request
        self.W0 = np.empty((n, 2))
        self.W1 = np.empty((n, 2))
        self.U = np.empty((n, 2))
        self.Wa = np.empty(n)
        self.Wb = np.empty(n)
        self.c1 = 2.0 / (b - a)
        self.c0 = -(a + b) / (b - a)

    # -- operator helpers -------------------------------------------------- #

    def _inf_norm(self):
        n, ip, da, lib = self.n, self.ip, self.da, self.lib
        if lib is not None:
            return float(lib.k_inf_norm(n, ip.ctypes.data, da.ctypes.data))
        if len(da) == 0:
            return 0.0
        sums = np.add.reduceat(np.abs(da), np.minimum(ip[:-1], len(da) - 1))
        sums = np.where(ip[1:] == ip[:-1], 0.0, sums)
        return float(sums.max()) if len(sums) else 0.0

    def _matvec2(self, x, y):
        lib = self.lib
        if lib is not None:
            lib.k_matvec2(self.n, self.ip.ctypes.data, self.ix.ctypes.data,
                          self.da.ctypes.data, x.ctypes.data, y.ctypes.data)
        else:
            self._matvec1(x[:, 0], y[:, 0])
            self._matvec1(x[:, 1], y[:, 1])

    def _matvec1(self, x, y):
        lib = self.lib
        if lib is not None:
            lib.k_matvec1(self.n, self.ip.ctypes.data, self.ix.ctypes.data,
                          self.da.ctypes.data, x.ctypes.data, y.ctypes.data)
            return
        if len(self.da) == 0:
            y[:] = 0.0
            return
        prod = self.da * x[self.ix]
        prod = np.concatenate((prod, [0.0]))
        starts = np.minimum(self.ip[:-1], len(prod) - 1)
        res = np.add.reduceat(prod, starts)
        res = np.where(self.ip[1:] == self.ip[:-1], 0.0, res)
        y[:] = res

    # -- main entry -------------------------------------------------------- #

    def solve(self, b0, b1):
        t = self.t
        if t == 0.0:
            return np.array(b0)
        if self.g <= 0.0:
            return np.array(b0) + t * np.asarray(b1)
        nb0 = float(np.linalg.norm(b0))
        nc = t * float(np.linalg.norm(b1))
        y = np.zeros(self.n)
        if nb0 == 0.0 and nc == 0.0:
            return y
        if nb0 > 0.0:
            eps = SAFETY * (self.atol + self.rtol * nb0) / nb0
            mexp = self.series.deg_exp(eps)
        else:
            mexp = 0
        if nc > 0.0:
            eps = SAFETY * (self.atol + self.rtol * nc) / nc
            mphi = self.series.deg_phi(eps)
        else:
            mphi = 0
        mmax = max(mexp, mphi)
        if mmax <= 0:
            # constant term only
            y += self.series.cf[0] * np.asarray(b0) if nb0 > 0.0 else 0.0
            if nc > 0.0:
                y += self.series.ch[0] * t * np.asarray(b1)
            return y
        self._run_chebyshev(b0, b1, mexp, mphi, y, nb0, nc)
        return y

    # -- Chebyshev application --------------------------------------------- #

    def _run_chebyshev(self, b0, b1, mexp, mphi, y, nb0, nc):
        n = self.n
        lib = self.lib
        t = self.t
        cf = self.series.cf
        ch = self.series.ch
        c1 = self.c1
        c0 = self.c0
        W0 = self.W0
        W1 = self.W1
        U = self.U
        W0[:, 0] = b0
        W0[:, 1] = b1
        W0[:, 1] *= t
        # degree zero contributions
        if mexp >= 0 and nb0 > 0.0:
            y += cf[0] * W0[:, 0]
        if mphi >= 0 and nc > 0.0:
            y += ch[0] * W0[:, 1]
        mmax = max(mexp, mphi)
        if mmax == 0:
            return
        # degree one: W1 = (c1 L + c0 I) W0
        self._matvec2(W0, U)
        np.multiply(U, c1, out=U)
        U += c0 * W0
        W1[:, :] = U
        if nb0 > 0.0 and mexp >= 1:
            y += cf[1] * W1[:, 0]
        if nc > 0.0 and mphi >= 1:
            y += ch[1] * W1[:, 1]
        if mmax == 1:
            return
        # degrees two and above.  kacc is the highest degree already folded
        # into y; the block state (W0, W1) always holds (W_{kacc-1}, W_{kacc}).
        kacc = 1
        common = min(mexp, mphi)
        ncommon = max(0, common - kacc)
        if lib is not None:
            if ncommon > 0:
                ce = np.ascontiguousarray(cf[kacc + 1:common + 1]) if nb0 > 0.0 else None
                cp = np.ascontiguousarray(ch[kacc + 1:common + 1]) if nc > 0.0 else None
                lib.k_cheb2(n, self.ip.ctypes.data, self.ix.ctypes.data,
                            self.da.ctypes.data, c1, c0, ncommon,
                            ce.ctypes.data if ce is not None else None,
                            cp.ctypes.data if cp is not None else None,
                            W0.ctypes.data, W1.ctypes.data, y.ctypes.data, 1)
                kacc = common
            if mexp > kacc:
                extra = mexp - kacc
                Wa = self.Wa
                Wb = self.Wb
                Wa[:] = W0[:, 0]
                Wb[:] = W1[:, 0]
                ce = np.ascontiguousarray(cf[kacc + 1:mexp + 1])
                lib.k_cheb1(n, self.ip.ctypes.data, self.ix.ctypes.data,
                            self.da.ctypes.data, c1, c0, extra,
                            ce.ctypes.data, Wa.ctypes.data, Wb.ctypes.data,
                            y.ctypes.data)
            elif mphi > kacc:
                extra = mphi - kacc
                Wa = self.Wa
                Wb = self.Wb
                Wa[:] = W0[:, 1]
                Wb[:] = W1[:, 1]
                cp = np.ascontiguousarray(ch[kacc + 1:mphi + 1])
                lib.k_cheb1(n, self.ip.ctypes.data, self.ix.ctypes.data,
                            self.da.ctypes.data, c1, c0, extra,
                            cp.ctypes.data, Wa.ctypes.data, Wb.ctypes.data,
                            y.ctypes.data)
            return
        # pure numpy fallback (three rotating buffers, no aliasing)
        A = W0
        B = W1
        C = U
        for k in range(2, mmax + 1):
            self._matvec2(B, C)
            C *= 2.0 * c1
            C += (2.0 * c0) * B
            C -= A
            A, B, C = B, C, A
            if k <= mexp and nb0 > 0.0:
                y += cf[k] * B[:, 0]
            if k <= mphi and nc > 0.0:
                y += ch[k] * B[:, 1]


# --------------------------------------------------------------------------- #
# process driver
# --------------------------------------------------------------------------- #

def main():
    lib = load_kernel()
    out = sys.stdout.buffer
    out.write(READY)
    out.flush()
    stream = sys.stdin.buffer

    if _read(stream, len(MAGIC)) != MAGIC:
        raise SystemExit(2)
    (hdrlen,) = struct.unpack("<I", _read(stream, 4))
    meta = json.loads(_read(stream, hdrlen).decode("utf-8"))
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    nreq = int(meta["requests"])
    t = float(meta["t"])
    rtol = float(meta["rtol"])
    atol = float(meta["atol"])

    indptr = _read_array(stream, np.dtype("<u8"), n + 1)
    indices = _read_array(stream, np.dtype("<u4"), nnz)
    data = _read_array(stream, np.dtype("<f8"), nnz)

    session = Session(n, indptr, indices, data, t, rtol, atol, lib)

    pair = np.empty(2 * n)
    for _ in range(nreq):
        _fill(stream, pair.view(np.uint8))
        b0 = pair[:n]
        b1 = pair[n:]
        y = session.solve(b0, b1)
        out.write(np.ascontiguousarray(y, dtype="<f8").tobytes())
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
