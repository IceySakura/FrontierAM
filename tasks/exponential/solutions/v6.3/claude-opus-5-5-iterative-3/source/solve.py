"""FAMEXP2 solver: y = exp(-tL) b0 + t*phi1(-tL) b1 for sparse symmetric PSD L.

See README.md for the algorithm.  Diagnostics go to stderr only.
"""
import os
import sys
import json
import struct
import ctypes
import math

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG = bool(os.environ.get("EXPO_DEBUG"))


def log(*a):
    if DEBUG:
        print(*a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# native kernels
# ----------------------------------------------------------------------------
def _load_lib():
    names = []
    try:
        flags = open("/proc/cpuinfo").read()
        if " avx2" in flags and " fma" in flags:
            names.append("libexpo_v3.so")
    except OSError:
        pass
    names.append("libexpo_generic.so")
    extra = os.environ.get("EXPO_LIBDIR")
    dirs = ([extra] if extra else []) + [HERE]
    for d in dirs:
        for nm in names:
            p = os.path.join(d, nm)
            if os.path.exists(p):
                try:
                    return ctypes.CDLL(p)
                except OSError:
                    continue
    # last resort: compile from source into a temporary directory
    try:
        import subprocess
        import tempfile
        out = os.path.join(tempfile.mkdtemp(prefix="expo"), "libexpo.so")
        subprocess.run(["gcc", "-O3", "-fPIC", "-shared", "-ffp-contract=off", "-fno-math-errno",
                        "-o", out, os.path.join(HERE, "expo.c"), "-lm"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        return ctypes.CDLL(out)
    except Exception:
        return None


LIB = _load_lib()
P = ctypes.c_void_p
I64 = ctypes.c_int64
if LIB is not None:
    LIB.lanczos_run.restype = I64
    LIB.lanczos_run.argtypes = [I64, ctypes.c_int, P, P, P, P, I64, I64, P, P, P, P, I64, P]
    LIB.lanczos_run3.restype = I64
    LIB.lanczos_run3.argtypes = [I64, ctypes.c_int, P, P, P, P, I64, I64, P, P, P, P, I64, P]
    LIB.lanczos_replay_from.restype = None
    LIB.lanczos_replay_from.argtypes = [I64, ctypes.c_int, P, P, P, P, I64, I64, P, P, P, P]
    LIB.combine.restype = None
    LIB.combine.argtypes = [I64, ctypes.c_int, P, I64, P, P]


def ptr(a):
    return a.ctypes.data


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
IN = sys.stdin.buffer.raw if hasattr(sys.stdin.buffer, "raw") else sys.stdin.buffer
OUT_FD = sys.stdout.fileno()


def read_into(arr):
    mv = memoryview(arr).cast("B")
    total = len(mv)
    got = 0
    while got < total:
        r = IN.readinto(mv[got:])
        if not r:
            raise EOFError("unexpected end of input")
        got += r


def read_bytes(k):
    buf = bytearray(k)
    read_into(buf)
    return bytes(buf)


def write_all(arr):
    mv = memoryview(np.ascontiguousarray(arr, dtype="<f8")).cast("B")
    while len(mv):
        w = os.write(OUT_FD, mv)
        mv = mv[w:]


def enlarge_pipes():
    try:
        import fcntl
        F_SETPIPE_SZ = 1031
        for fd in (0, 1):
            try:
                fcntl.fcntl(fd, F_SETPIPE_SZ, 1 << 20)
            except OSError:
                pass
    except Exception:
        pass


# ----------------------------------------------------------------------------
# scalar functions
# ----------------------------------------------------------------------------
def f_exp(theta, t):
    return np.exp(-t * theta)


def f_phi(theta, t):
    """t * phi1(-t*theta) = (1 - exp(-t theta)) / theta, stable near 0."""
    x = t * np.asarray(theta, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = 1.0 - xs / 2.0 + xs * xs / 6.0 - xs * xs * xs / 24.0
    xl = x[~small]
    out[~small] = -np.expm1(-xl) / xl
    return t * out


# ----------------------------------------------------------------------------
# eigen-based solvers (exact up to rounding)
# ----------------------------------------------------------------------------
class EigSolver:
    """Dense eigendecomposition of a whole (small) matrix."""

    def __init__(self, L, t):
        A = L.toarray()
        A = 0.5 * (A + A.T)
        w, Q = np.linalg.eigh(A)
        self.Q = Q
        self.fe = f_exp(w, t)
        self.fp = f_phi(w, t)

    def solve(self, b0, b1):
        Q = self.Q
        return Q @ (self.fe * (Q.T @ b0) + self.fp * (Q.T @ b1))


class BlockEigSolver:
    """Block-diagonal (after permutation) matrix with small components."""

    def __init__(self, L, t, labels, ncomp):
        n = L.shape[0]
        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=ncomp)
        starts = np.zeros(ncomp + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        self.groups = []
        Lc = L.tocsr()
        for s in np.unique(counts):
            comps = np.flatnonzero(counts == s)
            G = comps.size
            # node indices of these components, shape (G, s)
            idx = order[(starts[comps][:, None] + np.arange(s)[None, :])]
            if s == 1:
                d = np.asarray(Lc[idx[:, 0], idx[:, 0]]).ravel()
                self.groups.append((idx, None, f_exp(d, t), f_phi(d, t)))
                continue
            flat = idx.ravel()
            sub = Lc[flat][:, flat]
            # extract diagonal blocks
            blocks = np.empty((G, s, s))
            sub = sub.tocoo()
            r, c, v = sub.row, sub.col, sub.data
            gr, gc = r // s, c // s
            m = gr == gc
            blocks[:] = 0.0
            blocks[gr[m], r[m] % s, c[m] % s] = v[m]
            blocks = 0.5 * (blocks + blocks.transpose(0, 2, 1))
            w, Q = np.linalg.eigh(blocks)
            self.groups.append((idx, Q, f_exp(w, t), f_phi(w, t)))
        self.n = n

    def solve(self, b0, b1):
        y = np.empty(self.n)
        for idx, Q, fe, fp in self.groups:
            if Q is None:
                i = idx[:, 0]
                y[i] = fe * b0[i] + fp * b1[i]
                continue
            B0 = b0[idx]
            B1 = b1[idx]
            c = fe * np.einsum("gji,gj->gi", Q, B0) + fp * np.einsum("gji,gj->gi", Q, B1)
            y[idx] = np.einsum("gij,gj->gi", Q, c)
        return y


# ----------------------------------------------------------------------------
# Lanczos
# ----------------------------------------------------------------------------
MEM_BUDGET = 2.4e9  # bytes for stored Krylov basis


class LanczosSolver:
    def __init__(self, n, indptr, indices, data, t, rtol, atol, normest):
        self.n = n
        self.ip, self.ix, self.a = indptr, indices, data
        self.t, self.rtol, self.atol = t, rtol, atol
        self.normest = normest
        ns = min(n, 512)
        self.samp = np.unique(np.linspace(0, n - 1, ns).astype(np.int64))
        self.ns = self.samp.size
        self.stats = []

    def solve(self, b0, b1):
        n, t = self.n, self.t
        chains = []
        for b, fn in ((b0, f_exp), (b1, f_phi)):
            nb = float(np.linalg.norm(b))
            if nb > 0.0:
                chains.append((b, fn, nb))
        y = np.zeros(n)
        if not chains:
            return y
        c = len(chains)
        fns = [ch[1] for ch in chains]
        nbs = [ch[2] for ch in chains]
        ln = c * n
        cap = int(max(16, min(n + 2, MEM_BUDGET // (8 * ln))))
        cap_total = 20000
        V = np.empty((min(cap, 64) + 1, ln))
        Vv = V
        for j, ch in enumerate(chains):
            V[0, j::c] = ch[0] / ch[2]
        alpha = np.zeros((cap_total + 1, c))
        beta = np.zeros((cap_total + 1, c))
        vmax = np.zeros((cap_total + 2, c))
        S = np.zeros((cap_total + 2, self.ns, c))
        for j in range(c):
            vmax[0, j] = np.max(np.abs(V[0, j::c]))
            S[0, :, j] = V[0, j::c][self.samp]
        m = 0           # number of completed steps; basis has m+1 vectors
        stored = 0      # vectors stored in V (V[0..stored-1]); grows by chunks
        W3 = None       # rotating buffers when out of memory
        mode = "store"
        prev_u = None
        prev_m = 0
        prev_est = None
        target = 6
        brk = False
        tol_est = None
        while True:
            target = min(target, cap_total)
            # advance iteration to `target` steps
            while m < target and not brk:
                if mode == "store":
                    if m + 1 >= Vv.shape[0]:
                        newcap = min(cap, max(Vv.shape[0] * 2, m + 2 + 64))
                        if newcap <= m + 1:
                            mode = "rot"
                            W3 = np.empty((3, ln))
                            W3[(m - 1) % 3] = Vv[m - 1]
                            W3[m % 3] = Vv[m]
                            stored = m + 1
                            continue
                        Vn = np.empty((newcap + 1, ln))
                        Vn[: m + 1] = Vv[: m + 1]
                        Vv = Vn
                    k1 = min(target, Vv.shape[0] - 1)
                    done = LIB.lanczos_run(n, c, ptr(self.ip), ptr(self.ix), ptr(self.a), ptr(Vv),
                                           m, k1, ptr(alpha), ptr(beta), ptr(vmax),
                                           ptr(self.samp), self.ns, ptr(S))
                else:
                    k1 = target
                    done = LIB.lanczos_run3(n, c, ptr(self.ip), ptr(self.ix), ptr(self.a), ptr(W3),
                                            m, k1, ptr(alpha), ptr(beta), ptr(vmax),
                                            ptr(self.samp), self.ns, ptr(S))
                m += done
                # breakdown check: all chains beta tiny
                bt = 1e-13 * self.normest
                if np.all(beta[m - 1] <= bt):
                    brk = True
                    break
            # coefficients for T_m (m x m), basis vectors v_0..v_{m-1}
            us = []
            for j in range(c):
                a_ = alpha[:m, j]
                b_ = beta[: m - 1, j].copy()
                # truncate at first breakdown of this chain
                bt = 1e-13 * self.normest
                cut = np.flatnonzero(b_ <= bt)
                mj = m if cut.size == 0 else int(cut[0]) + 1
                if mj == 1:
                    th = a_[:1]
                    u = np.array([fns[j](th, t)[0] * nbs[j]])
                else:
                    th, Q = eigh_tridiagonal(a_[:mj], b_[: mj - 1])
                    u = Q @ (fns[j](th, t) * Q[0, :]) * nbs[j]
                uu = np.zeros(m)
                uu[:mj] = u
                us.append(uu)
            # sample-based lower bound of max|y|
            ysamp = np.zeros(self.ns)
            for j in range(c):
                ysamp += S[:m, :, j].T @ us[j]
            ynorm2 = math.sqrt(sum(float(u @ u) for u in us))
            ymax_lb = max(float(np.max(np.abs(ysamp))), ynorm2 / math.sqrt(n))
            tol = self.atol + self.rtol * ymax_lb
            if prev_u is not None:
                est = 0.0
                for j in range(c):
                    d = us[j].copy()
                    d[:prev_m] -= prev_u[j]
                    est += float(np.abs(d) @ vmax[:m, j])
                # est ~ error at prev_m; extrapolate to m
                rate = None
                if prev_est is not None and prev_est > 0 and est > 0 and est < prev_est and prev_m > pprev_m:
                    rate = (est / prev_est) ** (1.0 / (prev_m - pprev_m))
                err_now = est if rate is None else est * rate ** (m - prev_m)
                log(f"  m={m} est={est:.3e} err_now={err_now:.3e} tol={tol:.3e}")
                if brk or err_now <= 0.3 * tol or m >= cap_total:
                    break
                # choose next check
                if rate is not None and rate < 1:
                    need = math.log(max(0.3 * tol, 1e-300) / err_now) / math.log(rate)
                    step = int(need * 1.05) + 2
                    step = max(step, 3, int(0.03 * m))
                    step = min(step, max(4, int(0.25 * m)))
                else:
                    step = max(4, int(0.3 * m))
                pprev_m = prev_m
                prev_est = est
            else:
                if brk:
                    break
                step = max(4, m)
                pprev_m = 0
            prev_u = us
            prev_m = m
            target = m + step
        # assemble y = sum_k coef[k] v_k
        coef = np.zeros((m, c))
        for j in range(c):
            coef[:, j] = us[j]
        coef = np.ascontiguousarray(coef)
        if mode == "store":
            LIB.combine(n, c, ptr(Vv), m, ptr(coef), ptr(y))
        else:
            LIB.combine(n, c, ptr(Vv), stored, ptr(coef), ptr(y))
            # replay from stored-1, stored
            W = np.empty((3, ln))
            W[(stored - 2) % 3] = Vv[stored - 2]
            W[(stored - 1) % 3] = Vv[stored - 1]
            y2 = np.zeros(n)
            # replay generates v_stored .. v_{m-1}
            LIB.lanczos_replay_from(n, c, ptr(self.ip), ptr(self.ix), ptr(self.a), ptr(W),
                                    stored - 1, m, ptr(alpha), ptr(beta), ptr(coef), ptr(y2))
            y += y2
        self.stats.append(m)
        return y


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    enlarge_pipes()
    os.write(OUT_FD, b"READY\n")
    magic = read_bytes(8)
    if magic != b"FAMEXP2\n":
        raise SystemExit("bad magic")
    (hl,) = struct.unpack("<I", read_bytes(4))
    meta = json.loads(read_bytes(hl).decode("utf-8"))
    n = int(meta["n"])
    nnz = int(meta["nnz"])
    nreq = int(meta["requests"])
    t = float(meta["t"])
    rtol = float(meta["rtol"])
    atol = float(meta["atol"])
    indptr = np.empty(n + 1, dtype="<u8")
    indices = np.empty(nnz, dtype="<u4")
    data = np.empty(nnz, dtype="<f8")
    read_into(indptr)
    if nnz:
        read_into(indices)
        read_into(data)
    rhs = np.empty((2, n), dtype="<f8")

    solver = None
    kind = None
    normest = 0.0
    if nnz:
        rows = np.repeat(np.arange(n), np.diff(indptr).astype(np.int64)) if False else None
        absrow = np.add.reduceat(np.abs(data), indptr[:-1].astype(np.int64)) if nnz else np.zeros(n)
        # reduceat on empty rows returns the next element; fix rows with no entries
        empty = indptr[1:] == indptr[:-1]
        if np.any(empty):
            absrow[empty] = 0.0
        normest = float(absrow.max()) if n else 0.0
    if t == 0.0 or normest == 0.0:
        kind = "trivial"
    else:
        L = None
        if n <= 300:
            L = sp.csr_matrix((data, indices.astype(np.int64), indptr.astype(np.int64)), shape=(n, n))
            solver = EigSolver(L, t)
            kind = "dense"
        else:
            # diagonal / small-component detection
            L = sp.csr_matrix((data, indices.astype(np.int32), indptr.astype(np.int32)), shape=(n, n))
            ncomp, labels = connected_components(L, directed=False)
            maxsize = int(np.bincount(labels).max())
            if maxsize <= 64:
                solver = BlockEigSolver(L, t, labels, ncomp)
                kind = "blocks"
            else:
                solver = LanczosSolver(n, indptr, indices, data, t, rtol, atol, normest)
                kind = "lanczos"
    log(f"n={n} nnz={nnz} t={t} rtol={rtol} tnorm={t*normest:.3g} kind={kind}")
    for _ in range(nreq):
        read_into(rhs)
        b0 = rhs[0]
        b1 = rhs[1]
        if kind == "trivial":
            y = b0 + t * b1
        else:
            y = solver.solve(b0, b1)
        write_all(y)
    if kind == "lanczos":
        log("iters", solver.stats)
    os._exit(0)


if __name__ == "__main__":
    main()
