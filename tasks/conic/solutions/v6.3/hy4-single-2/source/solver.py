#!/usr/bin/env python3
"""Self-contained primal-dual interior-point solver for conic QPs.

Problem (checker convention):

    minimize    0.5 x'Px + q'x + c0
    subject to  A x + s = b,   s in K
    dual var    z in K*,  P x + q + A'z = 0.

K is a product of zero, nonnegative, second-order, exponential and PSD cones.
The algorithm is an infeasible-start primal-dual path following method.  The
central path is written as z = -mu grad F(s) where F is the canonical
self-concordant barrier of K; the Newton system is reduced with the Hessian
scaling W = mu grad^2 F(s).  Infeasibility certificates are produced by
solving small auxiliary regularized problems with the same machinery.

Only numpy/scipy are required.  Everything is single threaded.
"""
import json
import math
import os
import sys
import time
import types

import numpy as np

# SciPy 1.16 pulls numpy.testing / unittest / importlib.metadata through
# scipy._lib._array_api on every startup.  Only plain numpy arrays are used
# here, so a tiny compatible shib is installed before importing scipy.sparse.
# This saves ~0.23 s of interpreter startup on the pinned environment.
if "scipy._lib._array_api" not in sys.modules:
    class _ShimModule(types.ModuleType):
        def __getattr__(self, name):
            # scipy.optimize (used by the LP fast path) needs the full API.
            # Load the genuine module on demand and hand over future imports.
            if name.startswith("__"):
                raise AttributeError(name)
            import importlib
            sys.modules.pop("scipy._lib._array_api", None)
            real = importlib.import_module("scipy._lib._array_api")
            return getattr(real, name)

    _shim = _ShimModule("scipy._lib._array_api")
    _shim.Array = np.ndarray
    _shim.array_namespace = lambda *a, **k: np
    _shim.is_lazy_array = lambda x: False
    _shim.is_numpy = lambda ns: True
    _shim.is_marray = lambda x: False
    _shim.xp_result_device = lambda *a, **k: None
    _shim.xp_size = lambda x: x.size
    _shim.xp_result_type = lambda *a, **k: np.result_type(*a)
    _shim._asarray = np.asarray
    _shim.ArrayLike = object
    _shim.is_cupy = lambda ns: False
    _shim.xp_copy = lambda x, *, xp=None: np.array(x, copy=True)
    _shim.xp_ravel = lambda x, /, *, xp=None: np.ravel(x)
    def _xp_promote(*args, broadcast=False, force_floating=False, xp=None):
        arrs = [np.asarray(a) for a in args if a is not None]
        dt = np.result_type(*arrs) if arrs else np.dtype(np.float64)
        x = np.asarray(arrs[0]) if arrs else np.asarray(0.0)
        if force_floating and dt.kind not in "fc":
            dt = np.dtype(np.float64)
        return x.astype(dt, copy=False)
    _shim.xp_promote = _xp_promote
    _shim.xp_capabilities = lambda **kw: (lambda f: f)
    sys.modules["scipy._lib._array_api"] = _shim

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

T0 = time.perf_counter()
SQRT2 = math.sqrt(2.0)
ISQRT2 = 1.0 / SQRT2
TOL = 1e-8
USE_NT_NN = 1
DEFAULT_MODE = dict(nt=1, split=0, corr=1, sigma=-1.0, gamma=0.995)


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def load_problem(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        P = sp.csc_matrix((d["P_data"], d["P_indices"], d["P_indptr"]),
                          shape=tuple(int(v) for v in d["P_shape"]))
        A = sp.csc_matrix((d["A_data"], d["A_indices"], d["A_indptr"]),
                          shape=tuple(int(v) for v in d["A_shape"]))
        q = np.asarray(d["q"], dtype=np.float64).ravel()
        b = np.asarray(d["b"], dtype=np.float64).ravel()
    P = P.astype(np.float64).tocsc()
    A = A.astype(np.float64).tocsc()
    P.sum_duplicates()
    A.sum_duplicates()
    P.eliminate_zeros()
    A.eliminate_zeros()
    return P, q, A, b, meta["cones"], float(meta.get("objective_offset", 0.0))


def save_result(path, status, x=None, s=None, z=None, n=0, m=0):
    arrays = {"metadata_json": np.array(
        json.dumps({"schema_version": 1, "status": status}))}
    for key, v, size in (("x", x, n), ("s", s, m), ("z", z, m)):
        if v is None:
            v = np.zeros(size, dtype=np.float64)
        arrays[key] = np.ascontiguousarray(np.asarray(v, dtype=np.float64).ravel())
    np.savez_compressed(path, **arrays)


# ----------------------------------------------------------------------------
# Cone bookkeeping.  All cone operations are vectorised over blocks of equal
# dimension; PSD blocks (which need dense linear algebra) are handled one by
# one but only over rows where A actually has entries.
# ----------------------------------------------------------------------------

def tri_indices(d):
    rows = np.concatenate([np.arange(j + 1) for j in range(d)])
    cols = np.concatenate([np.full(j + 1, j) for j in range(d)])
    scale = np.where(rows != cols, SQRT2, 1.0)
    u = np.where(rows != cols, ISQRT2, 0.5)
    return rows, cols, scale, u


class Cones:
    def __init__(self, spec, m):
        self.m = m
        self.blocks = []
        zero, nn = [], []
        soc, exp, psd = [], [], []
        pos = 0
        for c in spec:
            kind, dim = c["type"], int(c["dim"])
            width = dim * (dim + 1) // 2 if kind == "psd" else dim
            self.blocks.append((kind, pos, width, dim))
            if kind == "zero":
                zero.extend(range(pos, pos + width))
            elif kind == "nonnegative":
                nn.extend(range(pos, pos + width))
            elif kind == "soc":
                soc.append((pos, dim))
            elif kind == "exponential":
                exp.append(pos)
            elif kind == "psd":
                rows, cols, scale, u = tri_indices(dim)
                psd.append(dict(start=pos, d=dim, width=width,
                                rows=rows, cols=cols, scale=scale, u=u))
            pos += width
        if pos != m:
            raise ValueError("cone dimensions do not cover m")
        self.zero = np.asarray(zero, dtype=np.int64)
        self.nn = np.asarray(nn, dtype=np.int64)
        self.proper = np.setdiff1d(np.arange(m), self.zero, assume_unique=True)
        self.soc_groups = {}
        for start, dim in soc:
            self.soc_groups.setdefault(dim, []).append(start)
        self.soc_idx = {}
        for dim, starts in self.soc_groups.items():
            self.soc_idx[dim] = (np.asarray(starts)[:, None] +
                                 np.arange(dim)[None, :])
        self.exp_idx = (np.asarray(exp, dtype=np.int64)[:, None] +
                        np.arange(3)[None, :]) if exp else None
        self.psd = psd
        self.nu = 0
        for kind, start, width, dim in self.blocks:
            if kind == "nonnegative":
                self.nu += width
            elif kind == "soc":
                self.nu += 2
            elif kind == "exponential":
                self.nu += 3
            elif kind == "psd":
                self.nu += dim
        self.nu = max(self.nu, 1)

    def initial(self, scale=1.0):
        """Interior primal point (zero rows exactly zero)."""
        s = np.ones(self.m) * scale
        s[self.zero] = 0.0
        for dim, idx in self.soc_idx.items():
            s[idx] = 0.0
            s[idx[:, 0]] = scale * SQRT2
        if self.exp_idx is not None:
            e = self.exp_idx
            s[e[:, 0]] = -1.0 * scale
            s[e[:, 1]] = 1.0 * scale
            s[e[:, 2]] = 1.0 * scale
        for b in self.psd:
            r, c = b["rows"], b["cols"]
            s[b["start"] + np.arange(len(r))] = np.where(r == c, scale, 0.0)
        return s


def svec_to_mat(v, b):
    d = b["d"]
    M = np.empty((d, d))
    w = v / b["scale"]
    M[b["rows"], b["cols"]] = w
    M[b["cols"], b["rows"]] = w
    return M


def mat_to_svec(M, b):
    return M[b["rows"], b["cols"]] * b["scale"]


def inv_spd(S):
    try:
        cf = sla.cho_factor(S, lower=True, check_finite=False)
        C = sla.cho_solve(cf, np.eye(S.shape[0]), check_finite=False)
        return 0.5 * (C + C.T)
    except Exception:
        return 0.5 * (np.linalg.inv(S) + np.linalg.inv(S).T)


class ConeCache:
    """Per-iteration cached barrier quantities at (s, z)."""
    def __init__(self, cones, s, z=None, mu=1.0):
        self.cones = cones
        self.s = s
        self.z = z
        self.mu = mu
        self.C = []
        self.Cz = []
        self.L = []
        self.Lz = []
        for b in cones.psd:
            sl = slice(b["start"], b["start"] + b["width"])
            d = b["d"]
            try:
                cf = sla.cho_factor(svec_to_mat(s[sl], b), lower=True,
                                    check_finite=False)
                self.L.append(np.tril(cf[0]))
                self.C.append(inv_spd(svec_to_mat(s[sl], b)))
            except Exception:
                self.L.append(np.eye(d))
                self.C.append(np.eye(d))
            if z is not None:
                try:
                    cf = sla.cho_factor(svec_to_mat(z[sl], b), lower=True,
                                        check_finite=False)
                    self.Lz.append(np.tril(cf[0]))
                    self.Cz.append(inv_spd(svec_to_mat(z[sl], b)))
                except Exception:
                    self.Lz.append(np.eye(d))
                    self.Cz.append(np.eye(d))

    def grad(self):
        """grad F(s), zero on free/zero rows."""
        c, s = self.cones, self.s
        g = np.zeros(c.m)
        if c.nn.size:
            g[c.nn] = -1.0 / s[c.nn]
        for dim, idx in c.soc_idx.items():
            S = s[idx]
            t, V = S[:, 0], S[:, 1:]
            tau = t * t - np.einsum("ij,ij->i", V, V)
            G = np.empty_like(S)
            G[:, 0] = -2.0 * t / tau
            G[:, 1:] = 2.0 * V / tau[:, None]
            g[idx] = G
        if c.exp_idx is not None:
            E = s[c.exp_idx]
            u, v, w = E[:, 0], E[:, 1], E[:, 2]
            L = np.log(w) - np.log(v)
            psi = v * L - u
            ip = 1.0 / psi
            G = np.empty_like(E)
            G[:, 0] = ip
            G[:, 1] = -(L - 1.0) * ip - 1.0 / v
            G[:, 2] = -(v / w) * ip - 1.0 / w
            g[c.exp_idx] = G
        for k, b in enumerate(c.psd):
            sl = slice(b["start"], b["start"] + b["width"])
            g[sl] = -mat_to_svec(self.C[k], b)
        return g

    def hess_apply(self, v):
        """(grad^2 F(s)) v, zero on zero rows.  Does NOT include mu."""
        c, s = self.cones, self.s
        out = np.zeros(c.m)
        if c.nn.size:
            out[c.nn] = v[c.nn] / (s[c.nn] ** 2)
        for dim, idx in c.soc_idx.items():
            S = s[idx]
            D = v[idx]
            t, V = S[:, 0], S[:, 1:]
            tau = t * t - np.einsum("ij,ij->i", V, V)
            d0, dV = D[:, 0], D[:, 1:]
            dot = t * d0 - np.einsum("ij,ij->i", V, dV)
            H = np.empty_like(D)
            H[:, 0] = 4.0 * dot * t / tau ** 2 - 2.0 * d0 / tau
            H[:, 1:] = (4.0 * dot / tau ** 2)[:, None] * (-V) + (2.0 / tau)[:, None] * dV
            out[idx] = H
        if c.exp_idx is not None:
            E = s[c.exp_idx]
            D = v[c.exp_idx]
            u, vv, w = E[:, 0], E[:, 1], E[:, 2]
            L = np.log(w) - np.log(vv)
            psi = vv * L - u
            # grad psi = (-1, L-1, v/w), Hess psi[1:,1:] = [[-1/v,1/w],[1/w,-v/w^2]]
            gp0 = -np.ones(len(E))
            gp1 = L - 1.0
            gp2 = vv / w
            dot = gp0 * D[:, 0] + gp1 * D[:, 1] + gp2 * D[:, 2]
            ip = 1.0 / psi
            ip2 = ip ** 2
            H = np.empty_like(D)
            H[:, 0] = ip2 * dot * gp0
            H[:, 1] = ip2 * dot * gp1 - ip * (-D[:, 1] / vv + D[:, 2] / w) + D[:, 1] / vv ** 2
            H[:, 2] = ip2 * dot * gp2 - ip * (D[:, 1] / w - vv * D[:, 2] / w ** 2) + D[:, 2] / w ** 2
            out[c.exp_idx] = H
        for k, b in enumerate(c.psd):
            sl = slice(b["start"], b["start"] + b["width"])
            C = self.C[k]
            Dm = svec_to_mat(v[sl], b)
            out[sl] = mat_to_svec(C @ Dm @ C, b)
        return out

    def W_apply(self, v):
        """Scaling matrix W applied to v.  For the nonnegative cone this is
        the exact Nesterov-Todd / Mehrotra scaling diag(z/s); for the other
        cones it is mu * grad^2 F(s)."""
        c, s, z = self.cones, self.s, self.z
        out = np.zeros(c.m)
        if c.nn.size:
            if USE_NT_NN:
                out[c.nn] = (z[c.nn] / s[c.nn]) * v[c.nn]
            else:
                out[c.nn] = self.mu * (v[c.nn] / s[c.nn] ** 2)
        H = self.hess_apply(v)
        for dim, idx in c.soc_idx.items():
            out[idx] = self.mu * H[idx]
        if c.exp_idx is not None:
            out[c.exp_idx] = self.mu * H[c.exp_idx]
        for b in c.psd:
            sl = slice(b["start"], b["start"] + b["width"])
            out[sl] = self.mu * H[sl]
        return out

    def W_dense_small(self):
        """Dense W blocks for nonneg/SOC/exp cones, as (rows, cols, data)."""
        c, s, z = self.cones, self.s, self.z
        rr, cc, dd = [], [], []
        if c.nn.size:
            rr.append(c.nn)
            cc.append(c.nn)
            dd.append(z[c.nn] / s[c.nn] if USE_NT_NN
                      else self.mu / s[c.nn] ** 2)
        for dim, idx in c.soc_idx.items():
            S = s[idx]
            nb = S.shape[0]
            t, V = S[:, 0], S[:, 1:]
            tau = t * t - np.einsum("ij,ij->i", V, V)
            J = np.eye(dim)
            J[0, 0] = -1.0
            U = np.concatenate([t[:, None], -V], axis=1)
            H = (4.0 / tau ** 2)[:, None, None] * U[:, :, None] * U[:, None, :] \
                + (2.0 / tau)[:, None, None] * J[None, :, :]
            rr.append(np.broadcast_to(idx[:, :, None], (nb, dim, dim)).ravel())
            cc.append(np.broadcast_to(idx[:, None, :], (nb, dim, dim)).ravel())
            dd.append((self.mu * H).ravel())
        if c.exp_idx is not None:
            E = s[c.exp_idx]
            nb = E.shape[0]
            u, vv, w = E[:, 0], E[:, 1], E[:, 2]
            L = np.log(w) - np.log(vv)
            psi = vv * L - u
            gp = np.stack([-np.ones(nb), L - 1.0, vv / w], axis=1)
            H = (1.0 / psi ** 2)[:, None, None] * gp[:, :, None] * gp[:, None, :]
            for i in range(nb):
                H[i, 0, 0] += 0.0
                H[i, 1, 1] += 1.0 / (psi[i] * vv[i]) + 1.0 / vv[i] ** 2
                H[i, 1, 2] += -1.0 / (psi[i] * w[i])
                H[i, 2, 1] += -1.0 / (psi[i] * w[i])
                H[i, 2, 2] += vv[i] / (psi[i] * w[i] ** 2) + 1.0 / w[i] ** 2
                H[i, 0, 0] += 0.0
            idx = c.exp_idx
            rr.append(np.broadcast_to(idx[:, :, None], (nb, 3, 3)).ravel())
            cc.append(np.broadcast_to(idx[:, None, :], (nb, 3, 3)).ravel())
            dd.append((self.mu * H).ravel())
        return rr, cc, dd

    def psd_W_sub(self, A):
        """(rows, cols, data) for W restricted to rows touched by A, one per PSD block."""
        c = self.cones
        rr, cc, dd = [], [], []
        for k, b in enumerate(c.psd):
            sl = slice(b["start"], b["start"] + b["width"])
            Ab = A[sl, :]
            Ab.sum_duplicates()
            used = np.unique(Ab.nonzero()[0])
            if used.size == 0:
                continue
            C = self.C[k]
            p, q = b["rows"][used], b["cols"][used]
            r, ss = b["rows"][used], b["cols"][used]
            uu = b["u"][used]
            # W_ij = 2 u_i u_j (C[p_i,r_j] C[q_i,s_j] + C[p_i,s_j] C[q_i,r_j])
            P, R = p[:, None], r[None, :]
            Q, S = q[:, None], ss[None, :]
            W = 2.0 * (uu[:, None] * uu[None, :]) * (
                C[P, R] * C[Q, S] + C[P, S] * C[Q, R])
            gidx = b["start"] + used
            rr.append(np.broadcast_to(gidx[:, None], W.shape).ravel())
            cc.append(np.broadcast_to(gidx[None, :], W.shape).ravel())
            dd.append((self.mu * W).ravel())
        return rr, cc, dd


def build_W_sparse(cache, A):
    c = cache.cones
    rr, cc, dd = cache.W_dense_small()
    r2, c2, d2 = cache.psd_W_sub(A)
    rr, cc, dd = rr + r2, cc + c2, dd + d2
    if not rr:
        return sp.csr_matrix((c.m, c.m))
    W = sp.coo_matrix((np.concatenate(dd),
                       (np.concatenate(rr), np.concatenate(cc))),
                      shape=(c.m, c.m)).tocsr()
    W.sum_duplicates()
    return W


# ----------------------------------------------------------------------------
# Cone membership / maximum step length
# ----------------------------------------------------------------------------

def _affine_steps_ratio(v, dv):
    mask = dv < 0
    if not np.any(mask):
        return np.inf
    return float(np.min(-v[mask] / dv[mask]))


def _soc_step(S, D):
    t, V = S[:, 0], S[:, 1:]
    d0, dV = D[:, 0], D[:, 1:]
    a = d0 * d0 - np.einsum("ij,ij->i", dV, dV)
    bq = 2.0 * (t * d0 - np.einsum("ij,ij->i", V, dV))
    cq = t * t - np.einsum("ij,ij->i", V, V)
    disc = bq * bq - 4.0 * a * cq
    out = np.full(len(S), np.inf)
    bad_a = a < 0
    if np.any(bad_a):
        sq = np.sqrt(np.maximum(disc[bad_a], 0.0))
        root = (-bq[bad_a] - sq) / (2.0 * a[bad_a])
        out[bad_a] = np.maximum(root, 0.0)
    good = (a >= 0) & (bq < 0) & (disc >= 0)
    if np.any(good):
        sq = np.sqrt(np.maximum(disc[good], 0.0))
        root = (-bq[good] - sq) / (2.0 * a[good])
        out[good] = np.maximum(root, 0.0)
    return float(np.min(out)) if out.size else np.inf


def _exp_ok(E):
    """True where (u,v,w) is strictly inside K_exp."""
    u, v, w = E[:, 0], E[:, 1], E[:, 2]
    ok = (v > 0) & (w > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        psi = np.where(ok, v * (np.log(np.maximum(w, 1e-300)) -
                                np.log(np.maximum(v, 1e-300))) - u, -1.0)
    return ok & (psi > 0) & np.isfinite(psi)


def _exp_max_step(S, D, dual):
    if dual:
        # T(z) = (-z2, -z1, e*z3) must be in K_exp
        def at(alpha):
            Z = S + alpha * D
            return np.stack([-Z[:, 1], -Z[:, 0], math.e * Z[:, 2]], axis=1)
    else:
        def at(alpha):
            return S + alpha * D
    if np.all(_exp_ok(at(1.0))):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(52):
        mid = 0.5 * (lo + hi)
        if np.all(_exp_ok(at(mid))):
            lo = mid
        else:
            hi = mid
    return lo


def _psd_step(cache, v, dv, dual):
    """Exact maximal step preserving positive definiteness of S + alpha D."""
    c = cache.cones
    Ls = cache.Lz if dual else cache.L
    out = np.inf
    for k, b in enumerate(c.psd):
        sl = slice(b["start"], b["start"] + b["width"])
        L = Ls[k]
        D = svec_to_mat(dv[sl], b)
        M = sla.solve_triangular(L, D, lower=True, check_finite=False)
        M = sla.solve_triangular(L, M.T, lower=True, check_finite=False).T
        M = 0.5 * (M + M.T)
        lam = float(np.linalg.eigvalsh(M)[0])
        if lam < 0:
            out = min(out, -1.0 / lam)
    return out


def max_step(cache, ds, dz, gamma=0.995, split=False):
    """Largest fraction-to-boundary step for both primal and dual iterates."""
    c = cache.cones
    s, z = cache.s, cache.z
    ap, ad = np.inf, np.inf
    if c.nn.size:
        ap = min(ap, _affine_steps_ratio(s[c.nn], ds[c.nn]))
        ad = min(ad, _affine_steps_ratio(z[c.nn], dz[c.nn]))
    for dim, idx in c.soc_idx.items():
        ap = min(ap, _soc_step(s[idx], ds[idx]))
        ad = min(ad, _soc_step(z[idx], dz[idx]))
    if c.exp_idx is not None:
        ap = min(ap, _exp_max_step(s[c.exp_idx], ds[c.exp_idx], False))
        ad = min(ad, _exp_max_step(z[c.exp_idx], dz[c.exp_idx], True))
    if c.psd:
        ap = min(ap, _psd_step(cache, s, ds, False))
        ad = min(ad, _psd_step(cache, z, dz, True))
    if split:
        return (min(gamma, gamma * ap) if np.isfinite(ap) else gamma,
                min(gamma, gamma * ad) if np.isfinite(ad) else gamma)
    a = min(ap, ad)
    if not np.isfinite(a):
        return gamma
    return min(gamma, gamma * a)


# ----------------------------------------------------------------------------
# Independent (lightweight) acceptance check in original coordinates
# ----------------------------------------------------------------------------

def _nrm(v):
    return float(np.max(np.abs(v), initial=0.0))


def cone_defect(v, cones, dual):
    worst = 0.0
    for kind, start, width, dim in cones.blocks:
        blk = v[start:start + width]
        u = blk / max(1.0, float(np.max(np.abs(blk))) if blk.size else 0.0)
        if kind == "zero":
            viol = 0.0 if dual else float(np.max(np.abs(u))) if u.size else 0.0
        elif kind == "nonnegative":
            viol = max(0.0, -float(np.min(u))) if u.size else 0.0
        elif kind == "soc":
            viol = max(0.0, float(np.linalg.norm(u[1:])) - float(u[0]))
        elif kind == "exponential":
            a, b, cc = ((-u[1], -u[0], math.e * u[2]) if dual
                        else (u[0], u[1], u[2]))
            boundary = max(abs(b), max(a, 0.0), max(-cc, 0.0))
            if b > 0 and cc > 0:
                interior = max(0.0, a - b * (math.log(cc) - math.log(b)))
                viol = min(interior, boundary)
            else:
                viol = boundary
            viol = max(0.0, viol)
        else:
            M = np.zeros((dim, dim))
            p = 0
            for col in range(dim):
                vals = u[p:p + col + 1].copy()
                vals[:col] *= ISQRT2
                M[:col + 1, col] = vals
                M[col, :col + 1] = vals
                p += col + 1
            ev = np.linalg.eigvalsh(0.5 * (M + M.T))[0]
            viol = max(0.0, -float(ev))
        worst = max(worst, viol)
    return worst


def check_solved(P, q, A, b, cones, x, s, z, tol=TOL):
    Ax = A @ x
    Px = P @ x
    Atz = A.T @ z
    rp = _nrm(Ax + s - b) / (1.0 + max(_nrm(Ax), _nrm(s), _nrm(b)))
    rd = _nrm(Px + q + Atz) / (1.0 + max(_nrm(Px), _nrm(q), _nrm(Atz)))
    cp = cone_defect(s, cones, False)
    cd = cone_defect(z, cones, True)
    xPx = float(x @ Px)
    p0 = 0.5 * xPx + float(q @ x)
    d0 = -0.5 * xPx - float(b @ z)
    gap = abs(p0 - d0)
    rel = gap / max(abs(p0), abs(d0)) if max(abs(p0), abs(d0)) else 0.0
    ok = (rp <= tol and rd <= tol and cp <= tol and cd <= tol and
          (gap <= tol or rel <= tol))
    return ok, dict(rp=rp, rd=rd, cp=cp, cd=cd, gap=gap, rel=rel)


def check_primal_cert(A, b, cones, z, tol=TOL):
    nz = _nrm(z)
    nb = _nrm(b)
    if nz <= 0 or nb <= 0:
        return False
    zh = z / nz
    beta = -float((b / nb) @ zh)
    if not (beta > tol):
        return False
    a = float(np.max(np.abs(A.data))) if A.nnz else 1.0
    if a == 0:
        a = 1.0
    lin = float(np.max(np.abs((A / a).T @ zh))) / beta
    cd = cone_defect(zh, cones, True) / beta
    return lin <= tol and cd <= tol


def check_dual_cert(P, q, A, cones, x, s, tol=TOL):
    nx = _nrm(x)
    nq = _nrm(q)
    if nx <= 0 or nq <= 0:
        return False
    xh = x / nx
    sh = s / nx
    alpha = -float((q / nq) @ xh)
    if not (alpha > tol):
        return False
    a = float(np.max(np.abs(A.data))) if A.nnz else 1.0
    if a == 0:
        a = 1.0
    h = float(np.max(np.abs(P.data))) if P.nnz else 1.0
    if h == 0:
        h = 1.0
    quad = float(np.max(np.abs((P / h) @ xh))) / alpha
    lin = float(np.max(np.abs((A / a) @ xh + sh / a))) / alpha
    cp = cone_defect(sh / a, cones, False) / alpha
    return quad <= tol and lin <= tol and cp <= tol


# ----------------------------------------------------------------------------
# Scaling (Ruiz-like, cone compatible)
# ----------------------------------------------------------------------------

def make_row_groups(cones, m):
    """Groups of rows that must share one scaling factor."""
    groups = []
    for kind, start, width, dim in cones.blocks:
        if kind in ("zero", "nonnegative"):
            groups.append(np.arange(start, start + width))
        else:
            groups.append(np.array([start]))
    # For nonneg/zero rows we allow per-row scales: split into singletons
    groups = []
    for kind, start, width, dim in cones.blocks:
        if kind in ("zero", "nonnegative"):
            for i in range(start, start + width):
                groups.append(np.array([i]))
        else:
            groups.append(np.arange(start, start + width))
    return groups


def equilibrate(P, q, A, b, cones, rounds=2):
    """Return (Pt, qt, At, bt, X, R, c) for the scaled problem.

    Ruiz equilibration of A, followed by a global balancing between the
    constraint right hand side (B), the columns of x and the objective.
    A' = R A X keeps O(1) entries while R b and c X q are both O(1).
    """
    m, n = A.shape
    X0 = np.ones(n)
    R0 = np.ones(m)
    groups = make_row_groups(cones, m)
    Ac = A.tocsr().astype(np.float64)
    for _ in range(rounds):
        cn = np.asarray(np.abs(Ac).max(axis=0).toarray()).ravel() if Ac.nnz else np.zeros(n)
        cn = np.where(cn > 0, cn, 1.0)
        Xf = np.clip(1.0 / np.sqrt(cn), 1e-8, 1e8)
        Ac = Ac @ sp.diags(Xf)
        X0 *= Xf
        rn = np.asarray(np.abs(Ac).max(axis=1).toarray()).ravel() if Ac.nnz else np.zeros(m)
        Rf = np.ones(m)
        for g in groups:
            val = max(float(np.max(rn[g])), 1e-30)
            Rf[g] = 1.0 / math.sqrt(val)
        Rf = np.clip(Rf, 1e-8, 1e8)
        Ac = sp.diags(Rf) @ Ac
        R0 *= Rf
    B = max(1.0, _nrm(R0 * b))
    R = R0 / B
    X = X0 * B
    At = Ac.tocsc()                       # = R A X (B cancels)
    q0 = X0 * q
    P0 = (sp.diags(X0) @ P @ sp.diags(X0)).tocsc()
    qn = _nrm(q0)
    pn = float(np.max(np.abs(P0.data))) if P0.nnz else 0.0
    c = 1.0 / max(1.0, B * qn, B * B * pn)
    Pt = (P0 * (c * B * B)).tocsc()
    qt = q0 * (c * B)
    bt = (R0 * b) / B
    return Pt, qt, At, bt, X, R, c


# ----------------------------------------------------------------------------
# Core IPM
# ----------------------------------------------------------------------------

class Diverged(Exception):
    pass


def solve_ipm(P, q, A, b, cones, deadline, max_iter=80, verbose=False, accept=None, opts=None, init=None):
    """Infeasible-start Mehrotra-type primal-dual IPM in the supplied data.

    W = diag(z/s) on nonnegative blocks (NT / Mehrotra), W = mu grad^2 F(s)
    elsewhere.  Returns (x, s, z, info) or None; raises Diverged on breakdown.
    """
    m, n = A.shape
    zero = cones.zero
    mz = zero.size
    O = dict(DEFAULT_MODE)
    if opts:
        O.update(opts)
    global USE_NT_NN
    USE_NT_NN = int(O["nt"])
    SPLIT = int(O["split"])
    SIGMA = float(O["sigma"])
    GAMMA = float(O["gamma"])
    CORR = int(O["corr"])
    amax = max(1.0, float(np.max(np.abs(A.data))) if A.nnz else 1.0)
    scale_s = max(1.0, _nrm(b))
    scale_d = max(1.0, _nrm(q) / amax)
    mu0 = max(1.0, scale_s * scale_d)
    BAL = int(os.environ.get("IPM_BAL", "0"))
    if BAL:
        s0 = cones.initial(scale_s)
        gtmp = ConeCache(cones, s0, None, 1.0).grad()
        ztmp = -gtmp
        ztmp[zero] = 0.0
        den = _nrm(A.T @ ztmp)
        if den > 0:
            ratio = (_nrm(q) + 1e-12) / (den + 1e-12)
            mu0 *= min(1.0, max(1e-6, ratio))
    if init is not None:
        x, s, z = (np.asarray(v, dtype=np.float64).copy() for v in init)
        s[zero] = 0.0
        mu0 = max(float(s @ z) / cones.nu, 1e-16)
    else:
        s = cones.initial(scale_s)
        x = np.zeros(n)
        ci = ConeCache(cones, s, None, mu0)
        z = -mu0 * ci.grad()
        z[zero] = 0.0
    mu = mu0
    stall = 0
    best = None
    best_score = np.inf

    def _fin():
        # Prefer the current (latest) iterate when it verifies: convergence
        # is usually fastest in the final iterations, while the heuristic
        # "best score" may be stuck at an earlier point.
        if x is not None:
            ok, inf = check_solved(P, q, A, b, cones, x, s, z,
                                   tol=0.25 * TOL)
            if ok and (accept is None or accept(x, s, z)):
                return (x.copy(), s.copy(), z.copy(), inf)
        return best

    for it in range(max_iter):
        if time.perf_counter() > deadline:
            return _fin()
        mu = max(float(s @ z) / cones.nu, 1e-300)
        cache = ConeCache(cones, s, z, mu)
        g = cache.grad()
        W = build_W_sparse(cache, A)
        K11 = (P + A.T @ W @ A).tocsc()
        kscale = max(1.0, float(np.max(np.abs(K11.data))) if K11.nnz else 1.0)
        reg = 1e-13 * kscale
        if mz:
            AE = A[zero, :].tocsc()
            K = sp.bmat([[K11 + reg * sp.eye(n), AE.T],
                         [AE, -reg * sp.eye(mz)]], format="csc")
        else:
            K = (K11 + reg * sp.eye(n)).tocsc()
        try:
            lu = spla.splu(K, permc_spec="COLAMD", diag_pivot_thresh=0.1)
        except Exception:
            try:
                lu = spla.splu(K + 1e-9 * kscale * sp.eye(K.shape[0]),
                               permc_spec="COLAMD", diag_pivot_thresh=0.1)
            except Exception:
                return _fin()

        Ax = A @ x
        Px = P @ x
        Atz = A.T @ z
        r_p = Ax + s - b
        r_d = Px + q + Atz
        Atr_pW = A.T @ cache.W_apply(r_p)

        def direction(mu_t, corr=None):
            r_c = z + mu_t * g
            if corr is not None and cones.nn.size:
                r_c[cones.nn] = r_c[cones.nn] + corr[cones.nn]
            r_c[zero] = 0.0
            rhs1 = -r_d + A.T @ r_c - Atr_pW
            rhs = np.concatenate([rhs1, -r_p[zero]]) if mz else rhs1
            try:
                sol = lu.solve(rhs)
            except Exception:
                raise Diverged("solve failed")
            dx = sol[:n]
            Adx = A @ dx
            ds = (-r_p - Adx).copy()
            dz = -r_c + cache.W_apply(r_p + Adx)
            if mz:
                ds[zero] = 0.0
                dz[zero] = sol[n:]
            else:
                ds[zero] = 0.0
            return dx, ds, dz

        try:
            dxa, dsa, dza = direction(0.0)
        except Diverged:
            return _fin()
        if SPLIT:
            ap_a, ad_a = max_step(cache, dsa, dza, gamma=0.999, split=True)
            a_aff = min(ap_a, ad_a)
        else:
            a_aff = max_step(cache, dsa, dza, gamma=0.999)
        if SIGMA >= 0:
            sigma = SIGMA
        else:
            sigma = min(0.85, max(0.01, (1.0 - a_aff) ** 3))
        corr = None
        if CORR and USE_NT_NN and cones.nn.size:
            corr = np.zeros(cones.m)
            with np.errstate(all="ignore"):
                corr[cones.nn] = dsa[cones.nn] * dza[cones.nn] / s[cones.nn]
            corr[~np.isfinite(corr)] = 0.0
        try:
            dx, ds, dz = direction(sigma * mu, corr)
        except Diverged:
            return _fin()
        if SPLIT:
            ap, ad = max_step(cache, ds, dz, gamma=GAMMA, split=True)
        else:
            ap = ad = max_step(cache, ds, dz, gamma=GAMMA)
        if min(ap, ad) < 1e-6:
            sigma2 = min(0.98, sigma * 2.0 + 0.3)
            try:
                dx, ds, dz = direction(sigma2 * mu)
            except Diverged:
                return _fin()
            if SPLIT:
                ap, ad = max_step(cache, ds, dz, gamma=GAMMA, split=True)
            else:
                ap = ad = max_step(cache, ds, dz, gamma=GAMMA)
        if ap <= 0 or ad <= 0:
            return _fin()
        x = x + ap * dx
        s = s + ap * ds
        s[zero] = 0.0
        z = z + ad * dz
        stall = stall + 1 if min(ap, ad) < 1e-3 else 0
        if stall >= 6:
            return _fin()
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(s)) and
                np.all(np.isfinite(z))):
            return _fin()
        if max(_nrm(x), _nrm(z), _nrm(s)) > 1e13:
            return _fin()
        if it % 2 == 1 or it > 55:
            ok, info = check_solved(P, q, A, b, cones, x, s, z,
                                    tol=0.25 * TOL)
            score = max(info["rp"], info["rd"],
                        info["gap"] / max(1.0, abs(float(q @ x))))
            if score < best_score:
                best_score = score
                best = (x.copy(), s.copy(), z.copy(), info)
            if ok and (accept is None or accept(x, s, z)):
                return x, s, z, info
        if verbose:
            sys.stderr.write("it %d mu %.3e ap %.3f ad %.3f sig %.3f\n" %
                             (it, mu, ap, ad, sigma))
    return _fin()

# ----------------------------------------------------------------------------
# Driver: solve a problem (with scaling), try certificates when needed
# ----------------------------------------------------------------------------

def _scale_result(x, s, z, X, R, c):
    return (X * x, s / R, R * z / c)


def _clip_status(result):
    """Sanitize finite values."""
    return {k: (np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                if isinstance(v, np.ndarray) else v)
            for k, v in result.items()}



def algebraic_polish(P, q, A, b, cones, x, s, z, tol=0.9 * TOL):
    """Try to zero the linear residuals algebraically without leaving the cones.

    Only returns a triple when the independent acceptance check passes, so it
    can never make a valid answer worse.
    """
    m, n = A.shape
    if n > 20000 or A.nnz > 400000:
        return None
    try:
        s0 = (b - A @ x).copy()
        s0[cones.zero] = 0.0
        if cones.nn.size:
            s0[cones.nn] = np.maximum(s0[cones.nn], 0.0)
        rd = P @ x + q + A.T @ z
        M = (A.T @ A).tocsc()
        reg = 1e-12 * max(1.0, float(np.max(np.abs(M.data))) if M.nnz else 1.0)
        try:
            lam = spla.spsolve((M + reg * sp.eye(n)).tocsc(), -rd)
            dz = A @ lam
        except Exception:
            dz = np.zeros(m)
        out = None
        best_score = np.inf
        for t in (1.0, 0.5, 0.2, 0.05, 0.0):
            zt = z + t * dz
            if not np.all(np.isfinite(zt)):
                continue
            ok, inf = check_solved(P, q, A, b, cones, x, s0, zt, tol=tol)
            score = max(inf["rp"], inf["rd"], inf["cp"], inf["cd"],
                        inf["gap"] / max(1.0, abs(float(q @ x))))
            if score < best_score:
                best_score, out = score, (x.copy(), s0.copy(), zt.copy())
            if ok:
                return x, s0, zt
        return None
    except Exception:
        return None


def solve_problem(P, q, A, b, cones, deadline, want_cert=True, allow_scale=True):
    """Return a result dict (status + vectors) in original coordinates."""
    m, n = A.shape
    used_scale = False
    if allow_scale:
        try:
            Pt, qt, At, bt, X, R, c = equilibrate(P, q, A, b, cones)
            used_scale = True
        except Exception:
            Pt, qt, At, bt = P, q, A, b
            X, R, c = np.ones(n), np.ones(m), 1.0
    else:
        Pt, qt, At, bt = P, q, A, b
        X, R, c = np.ones(n), np.ones(m), 1.0

    def unscale(x, s, z):
        return _scale_result(x, s, z, X, R, c)

    def good(u, tol=TOL, polish=False):
        x, s, z = u
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(s)) and
                np.all(np.isfinite(z))):
            return False
        if check_solved(P, q, A, b, cones, x, s, z, tol=tol)[0]:
            return True
        if polish:
            ap = algebraic_polish(P, q, A, b, cones, x, s, z, tol=tol)
            if ap is not None:
                u[0], u[1], u[2] = ap
                return True
        return False

    def accept_scaled(x, s, z):
        return good(unscale(x, s, z), 0.6 * TOL)

    fancy = any(k in ("soc", "exponential", "psd")
                for k, _, _, _ in cones.blocks)
    modes = [dict(nt=1, split=0, corr=1, sigma=-1.0, gamma=0.995)]
    if fancy:
        modes.insert(0, dict(nt=0, split=1, corr=0, sigma=-1.0, gamma=0.995))
        modes.append(dict(nt=1, split=1, corr=0, sigma=-1.0, gamma=0.995))
    best_sol = None
    best_score = np.inf
    nmodes = len(modes)

    for mi, opts in enumerate(modes):
        if time.perf_counter() >= deadline - 0.12:
            break
        sub_deadline = min(deadline, time.perf_counter() +
                           max(0.12, (deadline - time.perf_counter()) * 0.62))
        try:
            res = solve_ipm(Pt, qt, At, bt, cones, sub_deadline,
                            max_iter=100, accept=accept_scaled, opts=opts)
        except Exception:
            res = None
        if res is None:
            continue
        u = list(unscale(res[0], res[1], res[2]))
        if good(u, 0.8 * TOL, polish=True):
            return {"status": "solved", "x": u[0], "s": u[1], "z": u[2]}
        # polish in the original coordinates from this warm start
        if time.perf_counter() < deadline - 0.1:
            p_dl = min(deadline, time.perf_counter() + 0.22)
            for popts in (opts, dict(nt=0, split=1, corr=0, sigma=-1.0,
                                     gamma=0.995)):
                if time.perf_counter() >= p_dl:
                    break
                try:
                    pr = solve_ipm(P, q, A, b, cones, p_dl, max_iter=45,
                                   accept=lambda xx, ss, zz: good((xx, ss, zz),
                                                                   0.6 * TOL),
                                   opts=popts, init=u)
                except Exception:
                    pr = None
                if pr is None:
                    continue
                pu = [pr[0], pr[1], pr[2]]
                if good(pu, 0.8 * TOL, polish=True):
                    return {"status": "solved", "x": pu[0], "s": pu[1],
                            "z": pu[2]}
                _, inf = check_solved(P, q, A, b, cones, *pu)
                score = max(inf["rp"], inf["rd"], inf["gap"] / max(1.0, abs(float(q @ pu[0]))))
                if score < best_score:
                    best_score, best_sol = score, pu
        _, inf = check_solved(P, q, A, b, cones, *u)
        score = max(inf["rp"], inf["rd"], inf["gap"] / max(1.0, abs(float(q @ u[0]))))
        if score < best_score:
            best_score, best_sol = score, u

    # unscaled attempt (useful when scaling is badly conditioned)
    if used_scale and time.perf_counter() < deadline - 0.2:
        try:
            res = solve_ipm(P, q, A, b, cones, deadline - 0.12, max_iter=70,
                            accept=lambda xx, ss, zz: good((xx, ss, zz),
                                                           0.6 * TOL),
                            opts=modes[0])
        except Exception:
            res = None
        if res is not None and good([res[0], res[1], res[2]], 0.8 * TOL,
                                    polish=True):
            return {"status": "solved", "x": res[0], "s": res[1], "z": res[2]}

    if want_cert and time.perf_counter() < deadline - 0.1:
        cert = try_certificates(P, q, A, b, cones, deadline)
        if cert is not None:
            return cert
    if best_sol is not None and good(list(best_sol), polish=True):
        return {"status": "solved", "x": best_sol[0], "s": best_sol[1],
                "z": best_sol[2]}
    return {"status": "unresolved"}



def try_certificates(P, q, A, b, cones, deadline):
    """Solve regularized auxiliary problems to look for Farkas certificates."""
    m, n = A.shape
    # ---------------- primal infeasibility ----------------
    # variables w = z (length m).  Equality rows: A' w = 0 (n rows, zero cone).
    # For each proper original cone block:  s_b = T_b(w_b) must be in K_b,
    # encoded as A_b w + s_b = 0 with A_b = -T_b.
    rows, cols, vals = [], [], []
    aux_spec = []
    # equality part: A.T @ w = 0
    At = A.T.tocsr()
    rows.append(At.nonzero()[0])
    cols.append(At.nonzero()[1])
    vals.append(At.data)
    aux_spec.append({"type": "zero", "dim": n})
    pos = n
    for kind, start, width, dim in cones.blocks:
        if kind == "zero":
            continue  # w_free has no cone restriction
        aux_spec.append({"type": kind, "dim": dim})
        if kind == "nonnegative":
            idx = np.arange(start, start + width)
            rows.append(pos + np.arange(width))
            cols.append(idx)
            vals.append(-np.ones(width))
        elif kind == "soc":
            idx = np.arange(start, start + width)
            rows.append(pos + np.arange(width))
            cols.append(idx)
            vals.append(-np.ones(width))
        elif kind == "exponential":
            # T(w) = (-w2, -w1, e*w3); need -T(w) + s = 0
            rr = np.array([pos, pos + 1, pos + 2])
            cc = np.array([start + 1, start, start + 2])
            vv = np.array([1.0, 1.0, -math.e])
            rows.append(rr)
            cols.append(cc)
            vals.append(vv)
        else:  # psd, self dual
            idx = np.arange(start, start + width)
            rows.append(pos + np.arange(width))
            cols.append(idx)
            vals.append(-np.ones(width))
        pos += width
    if n > 0:
        Aaux = sp.coo_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(pos, m)).tocsc()
        bq = np.zeros(pos)
        for rho in (1e-3, 1e-6):
            if time.perf_counter() > deadline - 0.05:
                break
            Paux = (rho * sp.eye(m)).tocsc()
            try:
                cones_aux = Cones(aux_spec, pos)
                res = solve_ipm(Paux, b, Aaux, bq, cones_aux,
                                min(deadline - 0.02, time.perf_counter() + 0.35),
                                max_iter=60)
            except Exception:
                res = None
            if res is not None:
                w, s_aux, z_aux, _ = res[:4]
                zz = w.copy()
                if check_primal_cert(A, b, cones, zz, tol=0.6 * TOL):
                    return {"status": "primal_infeasible", "z": zz}
    # ---------------- dual infeasibility ----------------
    # min q'x + 0.5 x'(rho I + lam P)x  s.t. A x + s = 0, s in K
    for lam, rho in ((1e6, 1e-7), (1e9, 1e-10)):
        if time.perf_counter() > deadline - 0.05:
            break
        Paux = (lam * P + rho * sp.eye(n)).tocsc()
        bq = np.zeros(m)
        try:
            res = solve_ipm(Paux, q, A, bq, cones,
                            min(deadline - 0.02, time.perf_counter() + 0.35),
                            max_iter=60)
        except Exception:
            res = None
        if res is not None:
            xx, ss, zz, _ = res[:4]
            if check_dual_cert(P, q, A, cones, xx, ss, tol=0.6 * TOL):
                return {"status": "dual_infeasible", "x": xx, "s": ss}
    return None


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main(argv):
    inp = out = None
    for i, a in enumerate(argv):
        if a == "--input" and i + 1 < len(argv):
            inp = argv[i + 1]
        elif a == "--output" and i + 1 < len(argv):
            out = argv[i + 1]
    if inp is None or out is None:
        sys.stderr.write("usage: solver.py --input PATH --output PATH\n")
        return 2
    n = m = 0
    result = {"status": "unresolved"}
    try:
        P, q, A, b, cone_spec, c0 = load_problem(inp)
        m, n = A.shape
        cones = Cones(cone_spec, m)
        dim = n + m
        if dim < 3000:
            budget = 0.60
        elif dim < 30000:
            budget = 2.0
        elif dim < 200000:
            budget = 7.0
        else:
            budget = 13.0
        deadline = T0 + budget
        # Fast LP path (delegated module), only for pure LPs.
        if P.nnz == 0 and all(c["type"] in ("zero", "nonnegative")
                              for c in cone_spec):
            try:
                import lp_fast
                lp = lp_fast.solve_lp(P, q, A, b, cone_spec, c0,
                                      time_left=8.0)
                if lp is not None:
                    result = lp
            except Exception:
                pass
        if result["status"] == "unresolved":
            result = solve_problem(P, q, A, b, cones, deadline)
        result = _clip_status(result)
    except Exception as exc:
        sys.stderr.write("solver error: %s\n" % exc)
        result = {"status": "unresolved"}
    try:
        save_result(out, result["status"], result.get("x"), result.get("s"),
                    result.get("z"), n=n, m=m)
    except Exception:
        try:
            save_result(out, "unresolved", n=n, m=m)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
