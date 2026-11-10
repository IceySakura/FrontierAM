"""Homogeneous-embedding primal-dual interior point solver for
min 1/2 x'Px + q'x  s.t. Ax + s = b, s in K  (zero, nonneg, soc, exp, psd).
Linear algebra: sparse quasi-definite LDL^T (C kernel, kkt.c) on the KKT
system with a minimum degree ordering obtained from SuperLU (MMD on A+A')."""
import os
import sys
import json
import math
import time
import ctypes as ct
import numpy as np

T_START = time.perf_counter()
DEBUG = bool(os.environ.get("SOLVER_DEBUG"))
REF_IT = int(os.environ.get("REF_IT", "10"))
REF_ABS = float(os.environ.get("REF_ABS", "1e-12"))
REF_REL = float(os.environ.get("REF_REL", "1e-13"))
EXP_FRAC = float(os.environ.get("EXP_FRAC", "0.05"))
EXP_FTB = float(os.environ.get("EXP_FTB", "0.02"))
CAPF = float(os.environ.get("CAPF", "3"))
STEPF = float(os.environ.get("STEPF", "0.99"))
USE_QR = os.environ.get("USE_QR", "0") == "1"
SEQ_H = os.environ.get("SEQ_H", "1") == "1"
PSD_BT = os.environ.get("PSD_BT", "0") == "1"
SOC_LIN = os.environ.get("SOC_LIN", "1") == "1"
DYN_EPS = float(os.environ.get("DYN_EPS", "1e-13"))
DYN_DELTA = float(os.environ.get("DYN_DELTA", "2e-7"))
HERE = os.path.dirname(os.path.abspath(__file__))
EXP_E = math.e
VP = ct.c_void_p


def ptr(a):
    return a.ctypes.data


# ----------------------------------------------------------------------------
# native kernels
# ----------------------------------------------------------------------------
def _bind(lib):
    lib.csc_matvec.argtypes = [ct.c_int, VP, VP, VP, VP, VP]
    lib.csc_matvec.restype = None
    lib.csc_matvec_t.argtypes = [ct.c_int, VP, VP, VP, VP, VP]
    lib.csc_matvec_t.restype = None
    lib.ldl_etree.argtypes = [ct.c_int, VP, VP, VP, VP, VP]
    lib.ldl_etree.restype = ct.c_long
    lib.ldl_factor.argtypes = [ct.c_int, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP,
                               ct.c_double, ct.c_double, VP, VP]
    lib.ldl_factor.restype = ct.c_int
    lib.ldl_solve_refine.argtypes = [ct.c_int, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP, VP,
                                     ct.c_int, ct.c_double, ct.c_double]
    lib.ldl_solve_refine.restype = ct.c_int
    lib.coo_to_csc.argtypes = [ct.c_int, ct.c_long, VP, VP, VP, VP, VP]
    lib.coo_to_csc.restype = None
    lib.ldl_count.argtypes = [ct.c_int, VP, VP, VP, VP, VP, ct.c_long]
    lib.ldl_count.restype = ct.c_long
    return lib


def load_lib():
    path = os.path.join(HERE, "libkkt.so")
    if os.path.exists(path):
        try:
            return _bind(ct.CDLL(path))
        except Exception:
            pass
    tmp = os.environ.get("TMPDIR", "/tmp")
    path = os.path.join(tmp, "libkkt_%d.so" % os.getpid())
    os.system("gcc -O3 -fPIC -shared -o '%s' '%s' -lm" % (path, os.path.join(HERE, "kkt.c")))
    return _bind(ct.CDLL(path))


try:
    LIB = load_lib()
except Exception:
    LIB = None


def load_superlu():
    import importlib.util
    import glob
    spec = importlib.util.find_spec("scipy")
    if spec is None or not spec.submodule_search_locations:
        return None
    base = list(spec.submodule_search_locations)[0]
    cands = glob.glob(os.path.join(base, "sparse", "linalg", "_dsolve", "_superlu*.so"))
    if not cands:
        return None
    s2 = importlib.util.spec_from_file_location("_superlu", cands[0])
    mod = importlib.util.module_from_spec(s2)
    s2.loader.exec_module(mod)
    return mod


class CSC:
    """minimal CSC matrix (int32 indices, duplicates allowed)"""

    def __init__(self, nrow, ncol, indptr, indices, data):
        self.nrow, self.ncol = nrow, ncol
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int32)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.cols = np.repeat(np.arange(ncol, dtype=np.int32), np.diff(self.indptr))

    def mv(self, x, out=None):
        y = np.zeros(self.nrow) if out is None else out
        LIB.csc_matvec(self.ncol, ptr(self.indptr), ptr(self.indices), ptr(self.data), ptr(x), ptr(y))
        return y

    def tmv(self, x):
        y = np.zeros(self.ncol)
        LIB.csc_matvec_t(self.ncol, ptr(self.indptr), ptr(self.indices), ptr(self.data), ptr(x), ptr(y))
        return y


def ninf(v):
    return float(np.max(np.abs(v))) if v.size else 0.0


# ----------------------------------------------------------------------------
# KKT factorization
# ----------------------------------------------------------------------------
class KKT:
    """K = [[P + d I, A'], [A, -(H + d I)]] with H block diagonal (cone scaling)."""

    def __init__(self, n, m, P, a_rows, a_cols, a_vals, blocks_rows, blocks_cols):
        # entries of upper triangle: list of (row, col) in natural order.
        self.n, self.m = n, m
        N = n + m
        self.N = N
        Pr, Pc = P.indices, P.cols
        up = Pr <= Pc
        r_list = [Pr[up].astype(np.int64), np.arange(n, dtype=np.int64),
                  a_cols.astype(np.int64), np.arange(n, N, dtype=np.int64)]
        c_list = [Pc[up].astype(np.int64), np.arange(n, dtype=np.int64),
                  a_rows.astype(np.int64) + n, np.arange(n, N, dtype=np.int64)]
        # off-diagonal (strict upper) entries of dense cone blocks
        r_list.append(blocks_rows.astype(np.int64) + n)
        c_list.append(blocks_cols.astype(np.int64) + n)
        rows = np.concatenate(r_list)
        cols = np.concatenate(c_list)
        self.nP = int(up.sum())
        self.off_diagx = self.nP
        self.off_A = self.nP + n
        self.off_diagz = self.off_A + a_vals.size
        self.off_blk = self.off_diagz + m
        self.nent = rows.size
        self.vals = np.zeros(self.nent)
        self.vals[:self.nP] = P.data[up]
        self.vals[self.off_A:self.off_diagz] = a_vals
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        cols = np.ascontiguousarray(cols, dtype=np.int64)
        work = np.zeros(N, dtype=np.int32)
        self.Lnz = np.zeros(N, dtype=np.int32)
        self.etree = np.zeros(N, dtype=np.int32)
        self.Kp = np.zeros(N + 1, dtype=np.int32)
        self.Ki = np.zeros(self.nent, dtype=np.int32)
        self.order = np.zeros(self.nent, dtype=np.int64)
        # 1) natural ordering, accepted if its fill is small
        LIB.coo_to_csc(N, self.nent, ptr(rows), ptr(cols), ptr(self.Kp), ptr(self.Ki), ptr(self.order))
        cap = int(max(CAPF * self.nent, 2 * N, 1000))
        lnz = LIB.ldl_count(N, ptr(self.Kp), ptr(self.Ki), ptr(work), ptr(self.Lnz), ptr(self.etree), cap)
        perm = None
        if lnz >= 0 and N > 60:
            perm = np.arange(N)
        elif N > 60:
            # 2) constraint rows first (normal-equations-like elimination)
            permz = np.concatenate([np.arange(n, N), np.arange(n)])
            pinvz = np.empty(N, dtype=np.int64)
            pinvz[permz] = np.arange(N)
            pr, pc = pinvz[rows], pinvz[cols]
            r2 = np.ascontiguousarray(np.minimum(pr, pc))
            c2 = np.ascontiguousarray(np.maximum(pr, pc))
            del pr, pc
            LIB.coo_to_csc(N, self.nent, ptr(r2), ptr(c2), ptr(self.Kp), ptr(self.Ki), ptr(self.order))
            del r2, c2
            lnz = LIB.ldl_count(N, ptr(self.Kp), ptr(self.Ki), ptr(work), ptr(self.Lnz), ptr(self.etree), cap)
            if lnz >= 0:
                perm = permz
        if perm is None:
            perm = self.ordering(rows, cols)
            pinv = np.empty(N, dtype=np.int64)
            pinv[perm] = np.arange(N)
            pr, pc = pinv[rows], pinv[cols]
            r2 = np.ascontiguousarray(np.minimum(pr, pc))
            c2 = np.ascontiguousarray(np.maximum(pr, pc))
            del pr, pc
            LIB.coo_to_csc(N, self.nent, ptr(r2), ptr(c2), ptr(self.Kp), ptr(self.Ki), ptr(self.order))
            del r2, c2
            lnz = LIB.ldl_etree(N, ptr(self.Kp), ptr(self.Ki), ptr(work), ptr(self.Lnz), ptr(self.etree))
        pinv = np.empty(N, dtype=np.int64)
        pinv[perm] = np.arange(N)
        self.Kx = np.zeros(self.nent)
        self.pinv = pinv
        self.perm = perm
        sign = np.ones(N, dtype=np.int8)
        sign[n:] = -1
        self.sign = np.ascontiguousarray(sign[perm])
        if lnz < 0:
            raise RuntimeError("etree failure")
        self.Lp = np.zeros(N + 1, dtype=np.int32)
        self.Li = np.zeros(max(lnz, 1), dtype=np.int32)
        self.Lx = np.zeros(max(lnz, 1))
        self.D = np.zeros(N)
        self.Dinv = np.zeros(N)
        self.iwork = np.zeros(3 * N, dtype=np.int32)
        self.fwork = np.zeros(3 * N)
        self.reg = np.zeros(N)
        self.lnz = lnz

    def ordering(self, rows, cols):
        N = self.N
        if N <= 60:
            return np.arange(N)
        slu = load_superlu()
        if slu is None:
            return np.arange(N)
        # symmetric pattern, strongly diagonally dominant values
        off = rows != cols
        rr = np.concatenate([rows[off], cols[off], np.arange(N)])
        cc = np.concatenate([cols[off], rows[off], np.arange(N)])
        deg = np.bincount(rr, minlength=N).astype(float)
        vv = np.concatenate([np.full(2 * int(off.sum()), -1.0), deg + 1.0])
        key = cc * N + rr
        o = np.argsort(key, kind="stable")
        key = key[o]
        keep = np.ones(key.size, dtype=bool)
        keep[1:] = key[1:] != key[:-1]
        # sum duplicates
        grp = np.cumsum(keep) - 1
        vals = np.bincount(grp, weights=vv[o])
        ri = rr[o][keep].astype(np.intc)
        ci = cc[o][keep]
        indptr = np.zeros(N + 1, dtype=np.intc)
        np.cumsum(np.bincount(ci, minlength=N), out=indptr[1:])
        try:
            lu = slu.gstrf(N, int(vals.size), vals, ri, indptr,
                           csc_construct_func=lambda *a: None, ilu=False,
                           options=dict(ColPerm="MMD_AT_PLUS_A", DiagPivotThresh=0.0,
                                        SymmetricMode=True, Equil=False))
            pc = np.asarray(lu.perm_c, dtype=np.int64)
            del lu
            perm = np.empty(N, dtype=np.int64)
            perm[pc] = np.arange(N)
            return perm
        except Exception:
            return np.arange(N)

    def factor(self, Hdiag, Hoff, dx_reg, dz_reg, a_upd=None, a_pos=None):
        n = self.n
        v = self.vals
        if a_upd is not None:
            v[self.off_A + a_pos] = a_upd
        v[self.off_diagx:self.off_A] = dx_reg
        v[self.off_diagz:self.off_blk] = -Hdiag - dz_reg
        if Hoff is not None and Hoff.size:
            v[self.off_blk:] = -Hoff
        np.take(v, self.order, out=self.Kx)
        reg = self.reg
        reg[:] = 0.0
        regn = np.empty(self.N)
        regn[:n] = dx_reg
        regn[n:] = -dz_reg
        reg[self.pinv] = regn
        nreg = LIB.ldl_factor(self.N, ptr(self.Kp), ptr(self.Ki), ptr(self.Kx), ptr(self.Lp), ptr(self.Li),
                              ptr(self.Lx), ptr(self.D), ptr(self.Dinv), ptr(self.Lnz), ptr(self.etree),
                              ptr(self.sign), DYN_EPS, DYN_DELTA, ptr(self.iwork), ptr(self.fwork))
        return nreg

    def solve(self, rhs):
        bp = np.empty(self.N)
        bp[self.pinv] = rhs
        xp = np.empty(self.N)
        LIB.ldl_solve_refine(self.N, ptr(self.Kp), ptr(self.Ki), ptr(self.Kx), ptr(self.reg), ptr(self.Lp),
                             ptr(self.Li), ptr(self.Lx), ptr(self.Dinv), ptr(bp), ptr(xp), ptr(self.fwork),
                             REF_IT, REF_ABS, REF_REL)
        return xp[self.pinv]


# ----------------------------------------------------------------------------
# cones
# ----------------------------------------------------------------------------
EXP_INIT = np.array([-1.051383945322714, 0.556409619469370, 1.258967884768947])


class Cones:
    def __init__(self, cones, m):
        zero, nn, soc, expc, psd = [], [], [], [], []
        r = 0
        for c in cones:
            t, d = c["type"], int(c["dim"])
            if t == "zero":
                zero.append(np.arange(r, r + d)); r += d
            elif t == "nonnegative":
                nn.append(np.arange(r, r + d)); r += d
            elif t == "soc":
                soc.append((r, d)); r += d
            elif t == "exponential":
                expc.append(r); r += 3
            elif t == "psd":
                psd.append((r, d)); r += d * (d + 1) // 2
            else:
                raise ValueError(t)
        assert r == m
        self.m = m
        self.iz = np.concatenate(zero) if zero else np.zeros(0, dtype=np.int64)
        self.inn = np.concatenate(nn) if nn else np.zeros(0, dtype=np.int64)
        self.nsoc = len(soc)
        if soc:
            st = np.array([a for a, _ in soc], dtype=np.int64)
            dm = np.array([d for _, d in soc], dtype=np.int64)
            self.soc_dim = dm
            off = np.zeros(len(soc), dtype=np.int64)
            np.cumsum(dm[:-1], out=off[1:])
            self.soc_off = off  # positions of heads in concatenated array
            tot = int(dm.sum())
            seg = np.repeat(np.arange(len(soc)), dm)
            self.soc_seg = seg
            self.soc_rows = st[seg] + (np.arange(tot) - off[seg])
            self.soc_tail = np.ones(tot)
            self.soc_tail[off] = 0.0
            self.soc_J = 2 * (1 - self.soc_tail) - 1  # +1 head, -1 tail
        self.nexp = len(expc)
        if expc:
            st = np.array(expc, dtype=np.int64)
            self.iexp = np.stack([st, st + 1, st + 2], axis=1)
            iu = [(0, 1), (0, 2), (1, 2)]
            self.exp_br = np.concatenate([st + a for a, _ in iu])
            self.exp_bc = np.concatenate([st + b for _, b in iu])
        self.psd = psd
        self.npsd = len(psd)
        self.degree = self.inn.size + self.nsoc + 3 * self.nexp + sum(d for _, d in psd)

    def block_pattern(self):
        rr, cc = [], []
        if self.nexp:
            rr.append(self.exp_br); cc.append(self.exp_bc)
        if rr:
            return np.concatenate(rr), np.concatenate(cc)
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    def soc_apattern(self, Arows, Acols, Avals, n):
        """union pattern of A rows within each SOC (for W^{-1} A)."""
        m = self.m
        cone_of = np.full(m, -1, dtype=np.int64)
        cone_of[self.soc_rows] = self.soc_seg
        pos_of = np.zeros(m, dtype=np.int64)
        pos_of[self.soc_rows] = np.arange(self.soc_rows.size) - self.soc_off[self.soc_seg]
        sel = cone_of[Arows] >= 0
        r, c, v = Arows[sel], Acols[sel], Avals[sel]
        k = cone_of[r]
        i = pos_of[r]
        key = k * n + c
        uk, inv = np.unique(key, return_inverse=True)
        pk = uk // n
        pc = uk % n
        groups, er, ec = [], [], []
        base = 0
        pdim = self.soc_dim[pk]
        for d in np.unique(pdim):
            d = int(d)
            gsel = np.nonzero(pdim == d)[0]
            loc = np.full(uk.size, -1, dtype=np.int64)
            loc[gsel] = np.arange(gsel.size)
            esel = loc[inv] >= 0
            M = np.zeros((gsel.size, d))
            np.add.at(M, (loc[inv[esel]], i[esel]), v[esel])
            kk = pk[gsel]
            cones_d = np.nonzero(self.soc_dim == d)[0]
            posmap = np.zeros(self.nsoc, dtype=np.int64)
            posmap[cones_d] = np.arange(cones_d.size)
            rows_e = self.soc_rows[self.soc_off[kk]][:, None] + np.arange(d)[None, :]
            cols_e = np.repeat(pc[gsel][:, None], d, axis=1)
            groups.append((d, cones_d, posmap[kk], M, base))
            er.append(rows_e.ravel())
            ec.append(cols_e.ravel())
            base += gsel.size * d
        if er:
            return groups, np.concatenate(er), np.concatenate(ec), base
        return groups, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0

    def soc_winv_dense(self, Wb, eta, cones_d, d):
        idx = self.soc_off[cones_d][:, None] + np.arange(d)[None, :]
        w = Wb[idx]
        w0 = w[:, 0]
        w1 = w[:, 1:]
        Mi = np.empty((cones_d.size, d, d))
        Mi[:, 0, 0] = w0
        Mi[:, 0, 1:] = -w1
        Mi[:, 1:, 0] = -w1
        Mi[:, 1:, 1:] = w1[:, :, None] * w1[:, None, :] / (1.0 + w0)[:, None, None]
        ar = np.arange(1, d)
        Mi[:, ar, ar] += 1.0
        Mi /= eta[cones_d][:, None, None]
        return Mi

    def nonsym_rows(self):
        """rows belonging to blocks whose equilibration must be uniform"""
        groups = []
        if self.nsoc:
            groups.append(("seg", self.soc_rows, self.soc_seg, self.nsoc))
        if self.nexp:
            groups.append(("seg", self.iexp.reshape(-1), np.repeat(np.arange(self.nexp), 3), self.nexp))
        for a, d in self.psd:
            t = d * (d + 1) // 2
            groups.append(("seg", np.arange(a, a + t), np.zeros(t, dtype=np.int64), 1))
        return groups

    # ------------------------------------------------------------------
    def unit_init(self, v):
        v[:] = 0.0
        v[self.inn] = 1.0
        if self.nsoc:
            v[self.soc_rows[self.soc_off]] = 1.0
        if self.nexp:
            v[self.iexp] = EXP_INIT
        for a, d in self.psd:
            v[a + self.psd_diag_pos(d)] = 1.0
        return v

    @staticmethod
    def psd_diag_pos(d):
        j = np.arange(d)
        return j * (j + 1) // 2 + j

    def shift_sym(self, v):
        """shift v into int(K) along e (symmetric cones only)."""
        alpha = 0.0
        if self.inn.size:
            alpha = max(alpha, -float(np.min(v[self.inn])))
        if self.nsoc:
            S = v[self.soc_rows]
            t = np.sqrt(np.add.reduceat(S * S * self.soc_tail, self.soc_off))
            alpha = max(alpha, float(np.max(t - S[self.soc_off])))
        for a, d in self.psd:
            M = smat(v[a:a + d * (d + 1) // 2], d)
            alpha = max(alpha, -float(np.linalg.eigvalsh(M)[0]))
        alpha = 1.0 + max(alpha, 0.0)
        if self.inn.size:
            v[self.inn] += alpha
        if self.nsoc:
            v[self.soc_rows[self.soc_off]] += alpha
        for a, d in self.psd:
            v[a + self.psd_diag_pos(d)] += alpha
        v[self.iz] = 0.0
        return v


def smat(v, d):
    M = np.zeros((d, d))
    # column-wise upper triangle ordering: (i,j) i<=j, ordered by j then i
    jj, ii = np.tril_indices(d)
    M[ii, jj] = v
    M[jj, ii] = v
    off = ii != jj
    M[ii[off], jj[off]] /= math.sqrt(2.0)
    M[jj[off], ii[off]] /= math.sqrt(2.0)
    return M


def svec(M):
    d = M.shape[0]
    jj, ii = np.tril_indices(d)
    v = M[ii, jj].copy()
    v[ii != jj] *= math.sqrt(2.0)
    return v


# ----------------------------------------------------------------------------
# SOC helpers (vectorized over all SOC blocks, concatenated ordering)
# ----------------------------------------------------------------------------
def rs(v, off):
    return np.add.reduceat(v, off)


def soc_wmul(C, w, v):
    off, seg, tail = C.soc_off, C.soc_seg, C.soc_tail
    w0 = w[off]
    v0 = v[off]
    wv1 = rs(w * v * tail, off)
    coef = v0 + wv1 / (1.0 + w0)
    out = v + coef[seg] * w * tail
    out[off] = w0 * v0 + wv1
    return out


def soc_winvmul(C, w, v):
    J = C.soc_J
    return soc_wmul(C, w, v * J) * J


def soc_jprod(C, x, y):
    off, seg = C.soc_off, C.soc_seg
    out = x[off][seg] * y + y[off][seg] * x
    out[off] = rs(x * y, off)
    return out


def soc_jdiv(C, lam, v):
    off, seg, tail = C.soc_off, C.soc_seg, C.soc_tail
    l0 = lam[off]
    v0 = v[off]
    det = l0 * l0 - rs(lam * lam * tail, off)
    u0 = (l0 * v0 - rs(lam * v * tail, off)) / det
    u = (v - u0[seg] * lam) / l0[seg]
    u[off] = u0
    return u


def soc_step(C, S, dS):
    off, tail = C.soc_off, C.soc_tail
    s0, d0 = S[off], dS[off]
    tn = np.sqrt(rs(S * S * tail, off))
    a = d0 * d0 - rs(dS * dS * tail, off)
    b = 2.0 * (s0 * d0 - rs(S * dS * tail, off))
    c = (s0 - tn) * (s0 + tn)
    disc = b * b - 4.0 * a * c
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = -0.5 * (b + np.where(b >= 0, sq, -sq))
        r1 = np.where(t != 0, c / t, np.inf)
        r2 = np.where(a != 0, t / a, np.inf)
        r1 = np.where(r1 > 0, r1, np.inf)
        r2 = np.where(r2 > 0, r2, np.inf)
        cand = np.minimum(r1, r2)
        lin = np.where((a == 0) & (b < 0), -c / b, np.inf)
        cand = np.where(a == 0, lin, cand)
        cand = np.where((disc < 0) | ((a > 0) & (b > 0)), np.inf, cand)
        cand = np.where(c <= 0, 0.0, cand)
        head = np.where(d0 < 0, -s0 / d0, np.inf)
    alpha = np.minimum(cand, head)
    return float(np.min(alpha)) if alpha.size else np.inf


def soc_interior(C, S):
    off, tail = C.soc_off, C.soc_tail
    s0 = S[off]
    tn = np.sqrt(rs(S * S * tail, off))
    return bool(np.all(s0 - tn > 0))


# ----------------------------------------------------------------------------
# exponential cone helpers
# ----------------------------------------------------------------------------
def exp_primal_interior(S):
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    ok = (y > 0) & (z > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = y * np.log(z / y) - x
    return bool(np.all(ok & (val > 0)))


def exp_dual_interior(Z):
    u, v, w = Z[:, 0], Z[:, 1], Z[:, 2]
    ok = (u < 0) & (w > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = -u * (1.0 + np.log(w / (-u))) + v
    return bool(np.all(ok & (val > 0)))


def exp_primal_terms(S):
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = y * np.log(z / y) - x
    return np.stack([psi, y, z], 1)


def exp_dual_terms(Z):
    u, v, w = Z[:, 0], Z[:, 1], Z[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = -u * (1.0 + np.log(w / (-u))) + v
    return np.stack([psi, -u, w], 1)


def exp_grad_hess(S):
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    lzy = np.log(z / y)
    psi = y * lzy - x
    g = np.stack([-np.ones_like(x), lzy - 1.0, y / z], 1)
    grad = -g / psi[:, None]
    grad[:, 1] -= 1.0 / y
    grad[:, 2] -= 1.0 / z
    H = g[:, :, None] * g[:, None, :] / (psi * psi)[:, None, None]
    H[:, 1, 1] += 1.0 / (psi * y) + 1.0 / (y * y)
    H[:, 1, 2] -= 1.0 / (psi * z)
    H[:, 2, 1] -= 1.0 / (psi * z)
    H[:, 2, 2] += y / (psi * z * z) + 1.0 / (z * z)
    return grad, H


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def load_input(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        q = np.asarray(d["q"], dtype=np.float64).reshape(-1)
        b = np.asarray(d["b"], dtype=np.float64).reshape(-1)
        mats = {}
        for pre in ("P", "A"):
            mats[pre] = (d[pre + "_indptr"], d[pre + "_indices"], d[pre + "_data"])
    n, m = q.size, b.size
    P = CSC(n, n, *mats["P"])
    A = CSC(m, n, *mats["A"])
    return P, q, A, b, meta


def write_output(path, status, x=None, s=None, z=None, extra=None):
    meta = {"schema_version": 1, "status": status}
    if extra:
        meta.update(extra)
    arrays = {"metadata_json": np.array(json.dumps(meta, allow_nan=False))}
    for k, v in (("x", x), ("s", s), ("z", z)):
        if v is not None:
            arrays[k] = np.ascontiguousarray(v, dtype=np.float64)
    with open(path, "wb") as f:
        np.savez(f, **arrays)


# ----------------------------------------------------------------------------
# checks mirroring the official checker (original coordinates)
# ----------------------------------------------------------------------------
def seg_max(v, indptr):
    nseg = indptr.size - 1
    out = np.zeros(nseg)
    if v.size == 0:
        return out
    cnt = np.diff(indptr)
    nz = cnt > 0
    out[nz] = np.maximum.reduceat(v, indptr[:-1][nz])
    return out


def cone_violation(C, v, dual):
    """max normalized block defect (cheap version for zero/nn/soc/exp)."""
    worst = 0.0
    if C.iz.size and not dual:
        worst = max(worst, ninf(v[C.iz]) / max(1.0, ninf(v[C.iz])))
    if C.inn.size:
        u = v[C.inn]
        mn = float(np.min(u))
        if mn < 0:
            worst = max(worst, -mn / max(1.0, ninf(u)))  # conservative (block norm >= ...)
    if C.nsoc:
        S = v[C.soc_rows]
        nrm = np.maximum(1.0, np.maximum.reduceat(np.abs(S), C.soc_off))
        t = np.sqrt(rs(S * S * C.soc_tail, C.soc_off))
        d = (t - S[C.soc_off]) / nrm
        worst = max(worst, float(np.max(d)), 0.0)
    if C.nexp:
        V = v[C.iexp]
        nrm = np.maximum(1.0, np.max(np.abs(V), axis=1))
        U = V / nrm[:, None]
        if dual:
            U = np.stack([-U[:, 1], -U[:, 0], EXP_E * U[:, 2]], 1)
        a, bb, c = U[:, 0], U[:, 1], U[:, 2]
        B = np.maximum(np.abs(bb), np.maximum(np.maximum(a, 0), np.maximum(-c, 0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            inter = np.maximum(0.0, a - bb * (np.log(c) - np.log(bb)))
        E = np.where((bb > 0) & (c > 0), np.minimum(B, inter), B)
        worst = max(worst, float(np.max(E)))
    for a0, d in C.psd:
        u = v[a0:a0 + d * (d + 1) // 2]
        u = u / max(1.0, ninf(u))
        ev = float(np.linalg.eigvalsh(smat(u, d))[0])
        worst = max(worst, -ev)
    return worst


# ----------------------------------------------------------------------------
# interior point method
# ----------------------------------------------------------------------------
class Solver:
    def __init__(self, P, q, A, b, meta):
        self.P, self.q, self.A, self.b = P, q, A, b
        self.n, self.m = q.size, b.size
        self.C = Cones(meta["cones"], self.m)
        self.offset = float(meta.get("objective_offset", 0.0) or 0.0)
        self.tol = 1e-8

    # Ruiz equilibration of [P A'; A 0] + cost scaling
    def equilibrate(self, iters=int(os.environ.get("RUIZ", "12"))):
        P, A, C = self.P, self.A, self.C
        if A.data.size + P.data.size > 2e6:
            iters = min(iters, 6)
        n, m = self.n, self.m
        D = np.ones(n)
        E = np.ones(m)
        Pabs = np.abs(P.data)
        Aabs = np.abs(A.data)
        ro = np.argsort(A.indices, kind="stable")
        rptr = np.zeros(m + 1, dtype=np.int64)
        np.cumsum(np.bincount(A.indices, minlength=m), out=rptr[1:])
        groups = C.nonsym_rows()
        for _ in range(iters):
            vA = Aabs * E[A.indices] * D[A.cols]
            cn = seg_max(vA, A.indptr)
            if Pabs.size:
                cn = np.maximum(cn, seg_max(Pabs * D[P.indices] * D[P.cols], P.indptr))
            rn = seg_max(vA[ro], rptr)
            with np.errstate(divide="ignore"):
                dD = np.where(cn > 0, 1.0 / np.sqrt(cn), 1.0)
                dE = np.where(rn > 0, 1.0 / np.sqrt(rn), 1.0)
            np.clip(dD, 1e-4, 1e4, out=dD)
            np.clip(dE, 1e-4, 1e4, out=dE)
            for _g, rows, seg, k in groups:
                vals = np.bincount(seg, weights=dE[rows], minlength=k) / np.bincount(seg, minlength=k)
                dE[rows] = vals[seg]
            D *= dD
            E *= dE
        Psd = Pabs * D[P.indices] * D[P.cols]
        pn = seg_max(Psd, P.indptr)
        ctmp = max(float(np.mean(pn)) if n else 0.0, ninf(D * self.q))
        if ctmp < 1e-4:
            ctmp = 1.0
        ctmp = min(ctmp, 1e4)
        return D, E, 1.0 / ctmp

    def run(self):
        self.last_good = None
        self.fatal = False
        try:
            return self._run()
        except MemoryError:
            self.fatal = True
            return "unresolved", None, None, None
        except Exception:
            if DEBUG:
                import traceback
                traceback.print_exc()
            return "unresolved", None, None, None

    def _run(self):
        P, q, A, b, C = self.P, self.q, self.A, self.b, self.C
        n, m = self.n, self.m
        D, E, c = self.equilibrate()
        self.D, self.E, self.c = D, E, c
        Ps = CSC(n, n, P.indptr, P.indices, c * P.data * D[P.indices] * D[P.cols])
        As = CSC(m, n, A.indptr, A.indices, A.data * E[A.indices] * D[A.cols])
        qs = c * D * q
        bs = E * b
        br, bc = C.block_pattern()
        a_rows, a_cols, a_vals = As.indices.astype(np.int64), As.cols.astype(np.int64), As.data
        soc_groups, soc_apos = [], None
        if C.nsoc:
            is_soc = np.zeros(m, dtype=bool)
            is_soc[C.soc_rows] = True
            keep = ~is_soc[a_rows]
            soc_groups, sr, sc, nse = C.soc_apattern(a_rows, a_cols, a_vals, n)
            nk = int(keep.sum())
            # initial values: plain A (W = I)
            init_soc = np.concatenate([g[3].ravel() for g in soc_groups]) if soc_groups else np.zeros(0)
            a_rows = np.concatenate([a_rows[keep], sr])
            a_cols = np.concatenate([a_cols[keep], sc])
            a_vals = np.concatenate([a_vals[keep], init_soc])
            soc_apos = np.arange(nk, nk + nse)
            soc_aval = np.zeros(nse)
        npsd = C.npsd
        if npsd:
            kkt = NormalKKT(n, m, Ps, As, C)
            pblocks = kkt.blocks
        else:
            kkt = KKT(n, m, Ps, a_rows, a_cols, a_vals, br, bc)
            if DEBUG:
                print("KKT N %d nnzK %d nnzL %d t %.3f" % (kkt.N, kkt.nent, kkt.lnz, time.perf_counter() - T_START), file=sys.stderr)
        self.kkt = kkt
        bnorm, qnorm = ninf(b), ninf(q)
        amax = ninf(A.data) or 1.0
        hmax = ninf(P.data) or 1.0
        inn = C.inn
        regx = regz = float(os.environ.get("STAT_REG", "1e-8"))
        nsoc, nexp = C.nsoc, C.nexp
        if nsoc:
            srows, soff, sseg, J = C.soc_rows, C.soc_off, C.soc_seg, C.soc_J
        if nexp:
            erows = C.iexp
        nblk = br.size
        Hdiag = np.zeros(m)
        Hoff = np.zeros(nblk)
        nsoc_off = 0

        # ---------------- initial point
        if nexp or C.npsd:
            x = np.zeros(n)
            s = C.unit_init(np.zeros(m))
            z = C.unit_init(np.zeros(m))
        else:
            Hdiag[:] = 1.0
            Hdiag[C.iz] = 0.0
            kkt.factor(Hdiag, Hoff, regx, regz)
            sol = kkt.solve(np.concatenate([np.zeros(n), bs]))
            x = sol[:n].copy()
            s = C.shift_sym(-sol[n:].copy())
            sol = kkt.solve(np.concatenate([-qs, np.zeros(m)]))
            z = C.shift_sym(sol[n:].copy())
        tau, kap = 1.0, 1.0
        deg = C.degree
        best = None
        status = "unresolved"
        maxit = 200
        stall = 0
        prev_res = np.inf
        for it in range(maxit):
            Px = Ps.mv(x)
            Ax = As.mv(x)
            Atz = As.tmv(z)
            xPx = float(x @ Px)
            r1 = Px + Atz + qs * tau
            r2 = -Ax + bs * tau - s
            r3 = -float(qs @ x) - float(bs @ z) - xPx / tau - kap
            mu = (float(s @ z) + tau * kap) / (deg + 1)

            # ---- termination tests in original coordinates
            Et = E * tau
            cDt = c * D * tau
            xs = D * x / tau
            pres = ninf(r2 / Et) / (1.0 + max(ninf(Ax / Et), ninf(s / Et), bnorm))
            dres = ninf(r1 / cDt) / (1.0 + max(ninf(Px / cDt), qnorm, ninf(Atz / cDt)))
            xPxu = xPx / (c * tau * tau)
            zs = E * z / (c * tau)
            p0 = 0.5 * xPxu + float(q @ xs)
            d0 = -0.5 * xPxu - float(b @ zs)
            gap = abs(p0 - d0)
            mx = max(abs(p0), abs(d0))
            gapok = gap <= 0.9 * self.tol or (mx > 0 and gap / mx <= 0.9 * self.tol)
            if pres <= 0.9 * self.tol and dres <= 0.9 * self.tol and gapok:
                ss = s / Et
                if self.verify_solved(xs, ss, zs):
                    return "solved", xs, ss, zs
            near = (pres <= 0.9 * self.tol and dres <= 1e-6 and (gap <= 0.9 * self.tol or (mx > 0 and gap / mx <= 0.9 * self.tol)))
            if DEBUG:
                print("it %3d mu %.2e pres %.2e dres %.2e gap %.2e tau %.2e kap %.2e p0 %.6e t %.3f" % (
                    it, mu, pres, dres, gap, tau, kap, p0, time.perf_counter() - T_START), file=sys.stderr)
            # infeasibility certificates
            cert = self.check_infeasible(x, s, z, Px, Ax, Atz, bnorm, qnorm, amax, hmax)
            if cert is not None:
                return cert
            if it == maxit - 1:
                break
            if time.perf_counter() - T_START > self.time_cap:
                break

            # ---- scaling matrices
            Hdiag[:] = 0.0
            if inn.size:
                Hdiag[inn] = s[inn] / z[inn]
            if nsoc:
                S, Z = s[srows], z[srows]
                s0, z0 = S[soff], Z[soff]
                stn = np.sqrt(rs(S * S * C.soc_tail, soff))
                ztn = np.sqrt(rs(Z * Z * C.soc_tail, soff))
                sres = np.maximum((s0 - stn) * (s0 + stn), 1e-300)
                zres = np.maximum((z0 - ztn) * (z0 + ztn), 1e-300)
                ssc, zsc = np.sqrt(sres), np.sqrt(zres)
                Sb, Zb = S / ssc[sseg], Z / zsc[sseg]
                gam = np.sqrt(np.maximum((1.0 + rs(Sb * Zb, soff)) * 0.5, 1e-300))
                Wb = (Sb + J * Zb) / (2.0 * gam[sseg])
                eta = np.sqrt(ssc / zsc)
                eta2 = (eta * eta)[sseg]
                lam = eta[sseg] * soc_wmul(C, Wb, Z)
                Hdiag[srows] = 1.0
                for d_, cones_d, pm, Mg, base in soc_groups:
                    Wi = C.soc_winv_dense(Wb, eta, cones_d, d_)
                    soc_aval[base:base + Mg.size] = np.einsum("pij,pj->pi", Wi[pm], Mg).ravel()
            if nexp:
                SE, ZE = s[erows], z[erows]
                gradE, HE = exp_grad_hess(SE)
                try:
                    HsE = np.linalg.inv(mu * HE)
                except np.linalg.LinAlgError:
                    tr = np.trace(HE, axis1=1, axis2=2)
                    HsE = np.linalg.pinv(mu * (HE + 1e-14 * tr[:, None, None] * np.eye(3)))
                HsE = 0.5 * (HsE + HsE.transpose(0, 2, 1))
                Hdiag[erows] = np.stack([HsE[:, 0, 0], HsE[:, 1, 1], HsE[:, 2, 2]], 1)
                Hoff[nsoc_off:nsoc_off + 3 * nexp] = np.concatenate([HsE[:, 0, 1], HsE[:, 0, 2], HsE[:, 1, 2]])
            for rg in ((1.0,) if npsd else (1.0, 30.0, 1e3, 3e4)):
                if npsd:
                    kkt.set_scaling(s, z)
                    nreg = 0
                elif nsoc:
                    nreg = kkt.factor(Hdiag, Hoff, regx * rg, regz * rg, soc_aval, soc_apos)
                else:
                    nreg = kkt.factor(Hdiag, Hoff, regx * rg, regz * rg)
                if nreg == 0:
                    break
            if DEBUG and nsoc:
                print("    soc min rel sres %.2e zres %.2e  eta [%.2e,%.2e] w0max %.2e" % (
                    float(np.min(sres / s0**2)), float(np.min(zres / z0**2)), float(eta.min()), float(eta.max()), float(Wb[soff].max())), file=sys.stderr)
            def Hmul(v):
                out = Hdiag * v
                if inn.size == m:
                    return out
                if nsoc:
                    out[srows] = eta2 * soc_wmul(C, Wb, soc_wmul(C, Wb, v[srows]))
                if nexp:
                    out[erows] = np.einsum("kij,kj->ki", HsE, v[erows])
                for B in (pblocks if npsd else ()):
                    out[B.a:B.a + B.t] = B.Hmul(v[B.a:B.a + B.t])
                out[C.iz] = 0.0
                return out

            def dstilde(sigmu, corr):
                out = np.zeros(m)
                if inn.size:
                    out[inn] = (s[inn] * z[inn] + corr[0] - sigmu) / z[inn]
                if nsoc:
                    dsj = soc_jprod(C, lam, lam)
                    if corr[1] is not None:
                        dsj += corr[1]
                    dsj[soff] -= sigmu
                    out[srows] = eta[sseg] * soc_wmul(C, Wb, soc_jdiv(C, lam, dsj))
                if nexp:
                    out[erows] = np.einsum("kij,kj->ki", HsE, ZE + sigmu * gradE)
                for i_, B in enumerate(pblocks if npsd else ()):
                    cm = corr[2][i_] if (len(corr) > 2 and corr[2] is not None) else None
                    out[B.a:B.a + B.t] = B.dstilde(sigmu, cm)
                return out

            def ksolve(rhs):
                if npsd:
                    return kkt.solve(rhs)
                if nsoc:
                    rhs = rhs.copy()
                    rhs[n + srows] = soc_winvmul(C, Wb, rhs[n + srows]) / eta[sseg]
                sol = kkt.solve(rhs)
                if nsoc:
                    sol[n + srows] = soc_winvmul(C, Wb, sol[n + srows]) / eta[sseg]
                return sol

            if near:
                try:
                    xp_, zp_, sp_, r1p = x, z, s, r1
                    p_zero = not np.any(Ps.data)
                    for _pl in range(2):
                        cor = ksolve(np.concatenate([-r1p, np.zeros(m)]))
                        zp_ = zp_ + cor[n:]
                        if not p_zero:
                            xp_ = xp_ + cor[:n]
                            sp_ = sp_ - As.mv(cor[:n])
                        if self.verify_solved(D * xp_ / tau, sp_ / Et, E * zp_ / (c * tau)):
                            return "solved", D * xp_ / tau, sp_ / Et, E * zp_ / (c * tau)
                        r1p = Ps.mv(xp_) + As.tmv(zp_) + qs * tau
                except Exception:
                    pass
            rhs1 = np.concatenate([-qs, bs])
            sol1 = ksolve(rhs1)
            x1, z1 = sol1[:n], sol1[n:]
            if DEBUG:
                e1 = Ps.mv(x1) + As.tmv(z1) + qs
                e2 = As.mv(x1) - Hmul(z1) - bs
                print("    nreg %d solve err %.2e %.2e  |sol| %.2e Hmax %.2e Hmin %.2e" % (
                    nreg, ninf(e1), ninf(e2), ninf(sol1), float(np.max(Hdiag, initial=0)), float(np.min(Hdiag[Hdiag > 0])) if np.any(Hdiag > 0) else 0), file=sys.stderr)
            xi = x / tau
            Pxi = Px / tau
            qp = qs + 2.0 * Pxi
            denom = float(xi @ Pxi) + kap / tau - float(qp @ x1) - float(bs @ z1)

            def direction(sigma, sigmu, corr, dk):
                dst = dstilde(sigmu, corr)
                rhs = np.concatenate([-(1 - sigma) * r1, (1 - sigma) * r2 + dst])
                sol2 = ksolve(rhs)
                x2, z2 = sol2[:n], sol2[n:]
                dtau = (-(1 - sigma) * r3 + float(qp @ x2) + float(bs @ z2) - dk / tau) / denom
                dx = x2 + dtau * x1
                dz = z2 + dtau * z1
                if nsoc or npsd:
                    target = ninf(r1) * (1 - sigma) * 1e-3 + 1e-14
                    for _rf in range(3):
                        e1 = Ps.mv(dx) + As.tmv(dz) + qs * dtau + (1 - sigma) * r1
                        if ninf(e1) <= target:
                            break
                        cor = ksolve(np.concatenate([-e1, np.zeros(m)]))
                        dx = dx + cor[:n]
                        dz = dz + cor[n:]
                ds = -dst - Hmul(dz)
                if (nsoc and SOC_LIN) or npsd:
                    lin = (1 - sigma) * r2 - As.mv(dx) + bs * dtau
                    if nsoc:
                        ds[srows] = lin[srows]
                    for B in (pblocks if npsd else ()):
                        ds[B.a:B.a + B.t] = lin[B.a:B.a + B.t]
                dkap = -(dk + kap * dtau) / tau
                return dx, ds, dz, dtau, dkap

            def max_step(ds, dz, dtau, dkap, amax_):
                a = amax_
                if inn.size:
                    d = ds[inn]
                    neg = d < 0
                    if np.any(neg):
                        a = min(a, float(np.min(-s[inn][neg] / d[neg])))
                    d = dz[inn]
                    neg = d < 0
                    if np.any(neg):
                        a = min(a, float(np.min(-z[inn][neg] / d[neg])))
                if nsoc:
                    a = min(a, soc_step(C, S, ds[srows]), soc_step(C, Z, dz[srows]))
                if dtau < 0:
                    a = min(a, -tau / dtau)
                if dkap < 0:
                    a = min(a, -kap / dkap)
                for B in (pblocks if npsd else ()):
                    sl = slice(B.a, B.a + B.t)
                    a = min(a, B.max_step(ds[sl], True), B.max_step(dz[sl], False))
                return a

            def exp_backtrack(a, ds, dz, frac):
                for _ in range(60):
                    Sn = SE + a * ds[erows]
                    Zn = ZE + a * dz[erows]
                    if exp_primal_interior(Sn) and exp_dual_interior(Zn):
                        if frac is None:
                            return a
                        if not (np.all(exp_primal_terms(Sn) >= EXP_FTB * pt0) and np.all(exp_dual_terms(Zn) >= EXP_FTB * dt0)):
                            a *= 0.8
                            continue
                        sz = np.sum(Sn * Zn, axis=1)
                        mun = (float(s @ z) + a * float(ds @ z + s @ dz) + a * a * float(ds @ dz)
                               + (tau + a * dtau_c[0]) * (kap + a * dtau_c[1])) / (deg + 1)
                        if np.all(sz >= frac * 3 * mun):
                            return a
                    a *= 0.8
                return 0.0

            # affine predictor
            corr0 = (0.0, None)
            dx, ds, dz, dtau, dkap = direction(0.0, 0.0, corr0, tau * kap)
            a_aff = max_step(ds, dz, dtau, dkap, 1.0)
            if nexp:
                dtau_c = (dtau, dkap)
                a_aff = exp_backtrack(a_aff, ds, dz, None)
            sigma = (1.0 - a_aff) ** 3
            sigma = min(max(sigma, 0.0), 1.0)
            # corrector
            cnn = ds[inn] * dz[inn] if inn.size else 0.0
            csoc = None
            if nsoc and os.environ.get("NOCORR") != "1":
                csoc = soc_jprod(C, soc_winvmul(C, Wb, ds[srows]), soc_wmul(C, Wb, dz[srows]))
            if nexp:
                # no higher order correction for exp cones; use smaller centering
                pass
            cpsd = None
            if npsd:
                cpsd = [B.corr(ds[B.a:B.a + B.t], dz[B.a:B.a + B.t]) for B in pblocks]
            dk = tau * kap + dtau * dkap - sigma * mu
            dx, ds, dz, dtau, dkap = direction(sigma, sigma * mu, (cnn, csoc, cpsd), dk)
            a = max_step(ds, dz, dtau, dkap, 1e30)
            if DEBUG:
                comps = []
                if inn.size:
                    d1 = ds[inn]; d2 = dz[inn]
                    comps.append(("nn_s", float(np.min(np.where(d1 < 0, -s[inn] / np.where(d1 < 0, d1, -1), np.inf)))))
                    comps.append(("nn_z", float(np.min(np.where(d2 < 0, -z[inn] / np.where(d2 < 0, d2, -1), np.inf)))))
                if nsoc:
                    comps.append(("soc_s", soc_step(C, S, ds[srows])))
                    comps.append(("soc_z", soc_step(C, Z, dz[srows])))
                comps.append(("tau", -tau / dtau if dtau < 0 else np.inf))
                comps.append(("kap", -kap / dkap if dkap < 0 else np.inf))
                print("    blocking", " ".join("%s=%.2e" % c for c in comps), file=sys.stderr)
            a = min(1.0, 0.99 * a) if (nsoc == 0 and nexp == 0 and npsd == 0) else STEPF * min(1.0, a)
            if nsoc:
                for _bt in range(30):
                    if soc_interior(C, S + a * ds[srows]) and soc_interior(C, Z + a * dz[srows]):
                        break
                    a *= 0.5
            if npsd and PSD_BT:
                for _bt in range(30):
                    if all(B.interior(s[B.a:B.a + B.t] + a * ds[B.a:B.a + B.t]) and
                           B.interior(z[B.a:B.a + B.t] + a * dz[B.a:B.a + B.t]) for B in pblocks):
                        break
                    a *= 0.5
            if nexp:
                dtau_c = (dtau, dkap)
                pt0 = exp_primal_terms(SE)
                dt0 = exp_dual_terms(ZE)
                a = exp_backtrack(a, ds, dz, EXP_FRAC)
            if DEBUG:
                print("    a_aff %.3f sigma %.3f a %.3e" % (a_aff, sigma, a), file=sys.stderr)
            if not np.isfinite(a) or a <= 1e-12:
                stall += 1
                if stall > 3:
                    if DEBUG:
                        print("break: stall", file=sys.stderr)
                    break
                a = 0.0
            x = x + a * dx
            s = s + a * ds
            z = z + a * dz
            tau = tau + a * dtau
            kap = kap + a * dkap
            if not (np.isfinite(tau) and np.isfinite(kap)) or tau <= 0 or kap <= 0:
                if DEBUG:
                    print("break: tau %r kap %r dtau %r dkap %r a %r" % (tau, kap, dtau, dkap, a), file=sys.stderr)
                break
            # rescale homogeneous variables when they drift
            if tau > 1e6 or tau < 1e-6 and kap < 1e-6:
                pass
        return "unresolved", None, None, None

    time_cap = 1e9

    _polish_inv = None

    def dual_polish(self, x, z):
        n, m = self.n, self.m
        if n > 3000 or n * m > 4e7:
            return None
        try:
            if self._polish_inv is None:
                Ad = np.zeros((m, n))
                np.add.at(Ad, (self.A.indices, self.A.cols), self.A.data)
                G = Ad.T @ Ad
                G[np.diag_indices(n)] += 1e-14 * max(1.0, float(np.max(np.diag(G))))
                self._polish_A = Ad
                self._polish_inv = np.linalg.inv(G)
            rd = self.P.mv(x) + self.q + self.A.tmv(z)
            w = self._polish_inv @ rd
            dz = -(self._polish_A @ w)
            zn = z + dz
            rd2 = self.P.mv(x) + self.q + self.A.tmv(zn)
            if ninf(rd2) < ninf(rd):
                # one refinement step
                w2 = self._polish_inv @ rd2
                zn = zn - self._polish_A @ w2
            return self.dual_into_cone(zn)
        except Exception:
            return None

    def dual_into_cone(self, z):
        C = self.C
        z = z.copy()
        if C.inn.size:
            z[C.inn] = np.maximum(z[C.inn], 0.0)
        if C.nsoc:
            S = z[C.soc_rows]
            tn = np.sqrt(rs(S * S * C.soc_tail, C.soc_off))
            h = S[C.soc_off]
            z[C.soc_rows[C.soc_off]] = np.maximum(h, tn)
        for a0, d in C.psd:
            t = d * (d + 1) // 2
            M = smat(z[a0:a0 + t], d)
            ev = float(np.linalg.eigvalsh(M)[0])
            if ev < 0:
                M[np.diag_indices(d)] += -ev * (1 + 1e-6)
                z[a0:a0 + t] = svec(M)
        return z

    def verify_solved(self, x, s, z):
        P, q, A, b, C = self.P, self.q, self.A, self.b, self.C
        Ax = A.mv(x)
        Px = P.mv(x)
        Atz = A.tmv(z)
        tol = self.tol
        rp = ninf(Ax + s - b) / (1 + max(ninf(Ax), ninf(s), ninf(b)))
        rd = ninf(Px + q + Atz) / (1 + max(ninf(Px), ninf(q), ninf(Atz)))
        if DEBUG:
            print("    verify rp %.2e rd %.2e" % (rp, rd), file=sys.stderr)
        if not (rp <= 0.9 * tol and rd <= 0.9 * tol):
            return False
        xPx = float(x @ Px)
        p0 = 0.5 * xPx + float(q @ x)
        d0 = -0.5 * xPx - float(b @ z)
        g = abs(p0 - d0)
        mx = max(abs(p0), abs(d0))
        if DEBUG:
            print("    verify gap %.2e rel %.2e cones %.2e %.2e" % (g, g / max(mx, 1e-300), cone_violation(C, s, False), cone_violation(C, z, True)), file=sys.stderr)
        if not (g <= 0.9 * tol or (mx > 0 and g / mx <= 0.9 * tol)):
            return False
        if cone_violation(C, s, False) > 0.5 * tol or cone_violation(C, z, True) > 0.5 * tol:
            return False
        return True

    def check_infeasible(self, x, s, z, Px, Ax, Atz, bnorm, qnorm, amax, hmax):
        D, E, c, C = self.D, self.E, self.c, self.C
        tol = self.tol
        # primal infeasibility: z direction
        if bnorm > 0:
            zu = E * z
            zn = ninf(zu)
            if zn > 0:
                zh = zu / zn
                beta = -float(self.b @ zh) / bnorm
                if beta > 10 * tol:
                    # A' zu = D^{-1} As' z
                    atz = (Atz / D) / zn
                    if ninf(atz) / amax / beta <= 0.5 * tol:
                        zz = zh.copy()
                        if cone_violation(C, zz, True) / beta <= 0.5 * tol:
                            # exact re-check
                            atz2 = self.A.tmv(zh)
                            if ninf(atz2) / amax / beta <= 0.8 * tol:
                                return ("primal_infeasible", None, None, zh)
        if qnorm > 0:
            xu = D * x
            xn = ninf(xu)
            if xn > 0:
                xh = xu / xn
                alpha = -float(self.q @ xh) / qnorm
                if DEBUG:
                    sh_ = (s / E) / xn
                    print("    dinf: alpha %.3e |Ph| %.3e |Ax+s| %.3e cone %.3e" % (alpha, ninf(self.P.mv(xh)) / hmax / max(alpha, 1e-300),
                          ninf(self.A.mv(xh) + sh_) / amax / max(alpha, 1e-300), cone_violation(C, sh_ / amax, False) / max(alpha, 1e-300)), file=sys.stderr)
                if alpha > 10 * tol:
                    sh = (s / E) / xn
                    pxh = self.P.mv(xh)
                    if ninf(pxh) / hmax / alpha <= 0.5 * tol:
                        axs = self.A.mv(xh) + sh
                        if ninf(axs) / amax / alpha <= 0.5 * tol:
                            if cone_violation(C, sh / amax, False) / alpha <= 0.5 * tol:
                                return ("dual_infeasible", xh, sh, None)
        return None


def main():
    argv = sys.argv[1:]
    inp = out = None
    for i, a in enumerate(argv):
        if a == "--input":
            inp = argv[i + 1]
        elif a == "--output":
            out = argv[i + 1]
        elif a.startswith("--input="):
            inp = a.split("=", 1)[1]
        elif a.startswith("--output="):
            out = a.split("=", 1)[1]
    status, x, s, z = "unresolved", None, None, None
    extra = None
    try:
        P, q, A, b, meta = load_input(inp)
        attempts = [{},
                    {"EXP_FRAC": 0.02, "EXP_FTB": 0.1, "STEPF": 0.95, "SOC_LIN": False, "SEQ_H": False},
                    {"EXP_FRAC": 0.1, "EXP_FTB": 0.05, "USE_QR": True, "STEPF": 0.9}]
        g = globals()
        for k, over in enumerate(attempts):
            el = time.perf_counter() - T_START
            if k > 0 and el > (100.0 if k == 1 else 130.0):
                break
            g.update(over)
            solver = None
            solver = Solver(P, q, A, b, meta)
            if solver.C.npsd and (solver.C.iz.size or solver.C.nsoc or solver.C.nexp):
                break
            status, x, s, z = solver.run()
            if DEBUG:
                print("attempt %d -> %s (%.2fs)" % (k, status, time.perf_counter() - T_START), file=sys.stderr)
            if status != "unresolved" or getattr(solver, "fatal", False):
                break
    except Exception as e:  # noqa
        if DEBUG:
            import traceback
            traceback.print_exc()
        status, x, s, z = "unresolved", None, None, None
    if status == "unresolved":
        x = s = z = None
    write_output(out, status, x, s, z, extra)
    if os.environ.get("SOLVER_DEBUG"):
        print("status", status, "time %.3f" % (time.perf_counter() - T_START), file=sys.stderr)



# ----------------------------------------------------------------------------
# PSD support: NT scaling + normal equations (dense Cholesky)
# ----------------------------------------------------------------------------
_SV_CACHE = {}


def sv_index(d):
    r = _SV_CACHE.get(d)
    if r is None:
        jj, ii = np.tril_indices(d)
        sc = np.where(ii != jj, math.sqrt(2.0), 1.0)
        r = (ii, jj, sc)
        _SV_CACHE[d] = r
    return r


def smat_b(v, d):
    """batched smat: v (..., t) -> (..., d, d)"""
    ii, jj, sc = sv_index(d)
    M = np.zeros(v.shape[:-1] + (d, d))
    w = v / sc
    M[..., ii, jj] = w
    M[..., jj, ii] = w
    return M


def svec_b(M, d):
    ii, jj, sc = sv_index(d)
    return M[..., ii, jj] * sc


class PSDBlock:
    def __init__(self, a, d):
        self.a, self.d = a, d
        self.t = d * (d + 1) // 2

    def scaling(self, s, z):
        d = self.d
        S = smat(s, d)
        Z = smat(z, d)
        Ls = _robust_chol(S)
        Lz = _robust_chol(Z)
        U, lam, Vt = np.linalg.svd(Lz.T @ Ls)
        R = Ls @ Vt.T / np.sqrt(lam)[None, :]
        self.R = R
        self.Rinv = np.linalg.inv(R)
        self.G = R @ R.T
        self.Ginv = self.Rinv.T @ self.Rinv
        self.lam = lam
        self.Ls = Ls
        self.Lz = Lz

    def Hmul(self, v):
        X = smat(v, self.d)
        if SEQ_H:
            Y = self.R.T @ X @ self.R
            return svec(self.R @ Y @ self.R.T)
        return svec(self.G @ X @ self.G)

    def Hinv(self, v):
        X = smat(v, self.d)
        if SEQ_H:
            Y = self.Rinv @ X @ self.Rinv.T
            return svec(self.Rinv.T @ Y @ self.Rinv)
        return svec(self.Ginv @ X @ self.Ginv)

    def Wmul(self, v):  # R' X R
        X = smat(v, self.d)
        return self.R.T @ X @ self.R

    def WinvT(self, v):  # R^{-1} X R^{-T}
        X = smat(v, self.d)
        return self.Rinv @ X @ self.Rinv.T

    def dstilde(self, sigmu, corrM):
        lam = self.lam
        dsj = np.diag(lam * lam - sigmu)
        if corrM is not None:
            dsj = dsj + corrM
        U = 2.0 * dsj / (lam[:, None] + lam[None, :])
        return svec(self.R @ U @ self.R.T)

    def corr(self, ds, dz):
        X = self.WinvT(ds)
        Y = self.Wmul(dz)
        return 0.5 * (X @ Y + Y @ X)

    def max_step(self, dv, primal):
        X = self.WinvT(dv) if primal else self.Wmul(dv)
        il = 1.0 / np.sqrt(self.lam)
        M = X * il[:, None] * il[None, :]
        ev = np.linalg.eigvalsh(0.5 * (M + M.T))[0]
        return np.inf if ev >= 0 else -1.0 / ev

    def interior(self, v):
        try:
            np.linalg.cholesky(smat(v, self.d))
            return True
        except np.linalg.LinAlgError:
            return False


def _robust_chol(S):
    for k in range(8):
        try:
            return np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            ev = np.linalg.eigvalsh(S)
            S = S + (max(-ev[0], 0.0) + 1e-15 * max(1.0, ev[-1]) * 10 ** k) * np.eye(S.shape[0])
    return np.linalg.cholesky(S)


class NormalKKT:
    """Solves [[P, A'], [A, -H]] [dx; dz] = [r1; r2] by the normal equations
    (P + A' H^{-1} A) dx = r1 + A' H^{-1} r2 with dense Cholesky.
    Supports nonnegative and PSD cones (no zero cones)."""

    def __init__(self, n, m, Ps, As, C):
        self.n, self.m, self.C = n, m, C
        est = 8.0 * (m * n + n * n * 3 + sum(3 * n * d * d for _, d in C.psd))
        if est > 3.0e9:
            raise MemoryError("PSD problem too large for dense normal equations")
        self.Pd = np.zeros((n, n))
        np.add.at(self.Pd, (Ps.indices, Ps.cols), Ps.data)
        self.As = As
        Ad = np.zeros((m, n))
        np.add.at(Ad, (As.indices, As.cols), As.data)
        self.Ad = Ad
        self.blocks = [PSDBlock(a, d) for a, d in C.psd]
        # dense matrix stacks for the PSD rows
        self.Amats = []
        for B in self.blocks:
            sub = Ad[B.a:B.a + B.t, :].T  # (n, t)
            self.Amats.append(smat_b(sub, B.d))  # (n, d, d)
        self.inn = C.inn
        self.reg = 1e-10

    def set_identity(self):
        self.hnn = np.ones(self.inn.size)
        for B in self.blocks:
            d = B.d
            B.R = np.eye(d); B.Rinv = np.eye(d); B.G = np.eye(d); B.Ginv = np.eye(d)
            B.lam = np.ones(d)
        self.build()

    def set_scaling(self, s, z):
        inn = self.inn
        self.hnn = s[inn] / z[inn]
        for B in self.blocks:
            B.scaling(s[B.a:B.a + B.t], z[B.a:B.a + B.t])
        self.build()

    def build(self):
        n = self.n
        if USE_QR and not np.any(self.Pd):
            Fs = []
            inn = self.inn
            if inn.size:
                Fs.append(self.Ad[inn] / np.sqrt(self.hnn)[:, None])
            for B, Am in zip(self.blocks, self.Amats):
                T = B.Rinv @ Am @ B.Rinv.T
                Fs.append(svec_b(T, B.d).T)
            F = np.vstack(Fs)
            Rq = np.linalg.qr(F, mode="r")
            dg = np.abs(np.diag(Rq))
            dmax = float(dg.max()) if dg.size else 1.0
            fix = dg < 1e-13 * max(dmax, 1e-300)
            if np.any(fix):
                Rq[np.diag_indices(n)] = np.where(fix, 1e-13 * dmax, np.diag(Rq))
            Ri = np.linalg.inv(Rq)
            self.Minv = Ri @ Ri.T
            return
        M = self.Pd.copy()
        inn = self.inn
        if inn.size:
            An = self.Ad[inn]
            M += An.T @ (An / self.hnn[:, None])
        for B, Am in zip(self.blocks, self.Amats):
            T = B.Rinv @ Am @ B.Rinv.T  # (n, d, d)
            F = svec_b(T, B.d)  # (n, t)
            M += F @ F.T
        dmax = float(np.max(np.abs(np.diag(M)))) if n else 1.0
        M[np.diag_indices(n)] += self.reg * max(1.0, dmax)
        self.Minv = np.linalg.inv(M)

    def Hmul(self, v):
        out = np.zeros(self.m)
        inn = self.inn
        out[inn] = self.hnn * v[inn]
        for B in self.blocks:
            out[B.a:B.a + B.t] = B.Hmul(v[B.a:B.a + B.t])
        return out

    def Hinv(self, v):
        out = np.zeros(self.m)
        inn = self.inn
        out[inn] = v[inn] / self.hnn
        for B in self.blocks:
            out[B.a:B.a + B.t] = B.Hinv(v[B.a:B.a + B.t])
        return out

    def chol_solve(self, r):
        return self.Minv @ r

    def solve1(self, r1, r2):
        u = self.Hinv(r2)
        dx = self.chol_solve(r1 + self.As.tmv(u))
        dz = self.Hinv(self.As.mv(dx) - r2)
        return dx, dz

    def solve(self, rhs):
        n = self.n
        r1, r2 = rhs[:n], rhs[n:]
        dx, dz = self.solve1(r1, r2)
        nb = ninf(rhs)
        for _ in range(4):
            e1 = r1 - (self.Pd @ dx + self.As.tmv(dz))
            e2 = r2 - (self.As.mv(dx) - self.Hmul(dz))
            en = max(ninf(e1), ninf(e2))
            if en <= 1e-14 * (1 + nb):
                break
            cx, cz = self.solve1(e1, e2)
            dx = dx + cx
            dz = dz + cz
        return np.concatenate([dx, dz])


def _trisolve(L, r):
    # solve L L' x = r
    from scipy.linalg import solve_triangular
    y = solve_triangular(L, r, lower=True, check_finite=False)
    return solve_triangular(L.T, y, lower=False, check_finite=False)


if __name__ == "__main__":
    main()
