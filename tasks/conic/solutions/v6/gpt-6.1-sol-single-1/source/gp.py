"""Exact epigraph elimination and a dense Newton solver for log-sum-exp models."""

import os
import sys
import time
import numpy as np
from scipy import sparse, linalg
from common import norm, assess


def maxstep(v, dv):
    neg = dv < 0
    return min(1.0, float(np.min(-v[neg] / dv[neg]))) if np.any(neg) else 1.0


class LogSumModel:
    def __init__(self, p):
        if p.P.nnz or any(k not in ("zero", "nonnegative", "exponential") for k, _, _, _ in p.blocks):
            raise ValueError("not an epigraph model")
        self.p = p
        self.exp = np.asarray([i for k, _, i, _ in p.blocks if k == "exponential"], int)
        if not len(self.exp):
            raise ValueError("no exponential cones")
        A = p.A.tocsr()
        W, Y = A[self.exp + 2].tocsr(), A[self.exp + 1]
        if Y.nnz or np.any(p.b[self.exp + 1] <= 0) or np.any(W.getnnz(axis=1) != 1):
            raise ValueError("nonconstant perspective")
        self.wc, self.aw = W.indices.copy(), W.data.copy()
        if len(np.unique(self.wc)) != len(self.wc) or np.any(p.q[self.wc] != 0):
            raise ValueError("shared or costed epigraph")
        self.base = np.setdiff1d(np.arange(p.n), self.wc)
        n = len(self.base)
        if n > 600:
            raise ValueError("dense Newton dimension limit")
        self.nn = np.concatenate([np.arange(i, j) for k, _, i, j in p.blocks if k == "nonnegative"])
        self.eq = np.flatnonzero(p.eq)
        if A[self.exp][:, self.wc].nnz or A[self.eq][:, self.wc].nnz:
            raise ValueError("epigraph used outside its sum")
        N = A[self.nn].tocsr()
        self.Cw = N[:, self.wc].multiply(-1 / self.aw).tocsr()
        self.Cw.eliminate_zeros()
        if np.any(self.Cw.data <= 0):
            raise ValueError("nonmonotone epigraph")
        has_terms = self.Cw.getnnz(axis=1) > 0
        self.lrows, self.plain = np.flatnonzero(has_terms), np.flatnonzero(~has_terms)
        if N[self.lrows][:, self.base].nnz:
            raise ValueError("affine term outside logarithm")
        h = p.b[self.nn] + self.Cw @ p.b[self.exp + 2]
        if np.any(h[self.lrows] <= 0):
            raise ValueError("nonpositive sum bound")
        C = self.Cw[self.lrows].tocsr()
        self.ptr = C.indptr
        self.term = C.indices
        self.group = np.repeat(np.arange(len(self.lrows)), np.diff(self.ptr))
        y = p.b[self.exp + 1]
        self.B = A[self.exp][:, self.base].multiply(-1 / y[:, None]).tocsr()
        self.offset = p.b[self.exp] / y + np.log(y)
        self.V = self.B[self.term].tocsr()
        self.values = np.log(C.data) + self.offset[self.term] - np.log(h[self.lrows])[self.group]
        self.h = h
        G = N[self.plain][:, self.base].toarray()
        gs = np.maximum(1, np.maximum(np.max(abs(G), axis=1, initial=0), abs(h[self.plain])))
        self.gs = gs
        self.G, self.gh = G / gs[:, None], h[self.plain] / gs
        CE = A[self.eq][:, self.base].toarray()
        cs = np.maximum(1, np.max(abs(CE), axis=1, initial=0))
        self.C, self.cb, self.cs = CE / cs[:, None], p.b[self.eq] / cs, cs
        self.c = 1 / max(1, norm(p.q[self.base]))
        self.q = p.q[self.base] * self.c
        self.n, self.k = n, len(self.lrows) + len(self.plain)
        if not self.k:
            raise ValueError("no bounded sums")

    def evaluate(self, x, lam=None):
        if len(self.lrows):
            v = self.values + self.V @ x
            vmax = np.maximum.reduceat(v, self.ptr[:-1])
            ex = np.exp(v - vmax[self.group])
            sums = np.add.reduceat(ex, self.ptr[:-1])
            pi = ex / sums[self.group]
            F = vmax + np.log(sums)
            S = sparse.csr_matrix((pi, np.arange(len(pi)), self.ptr),
                                  shape=(len(self.lrows), len(pi)))
            G = (S @ self.V).toarray()
            if lam is not None:
                ll = lam[:len(self.lrows)]
                weights = ll[self.group] * pi
                H = (self.V.T @ self.V.multiply(weights[:, None])).toarray()
                H -= G.T @ (ll[:, None] * G)
        else:
            F, G = np.zeros(0), np.zeros((0, self.n))
            if lam is not None:
                H = np.zeros((self.n, self.n))
        F = np.r_[F, self.G @ x - self.gh]
        G = np.vstack((G, self.G))
        return (F, G, H) if lam is not None else (F, G)

    def candidate(self, x, lam, eqdual):
        p = self.p
        logt = self.offset + self.B @ x
        if norm(logt) > 705:
            return None
        t = np.exp(logt)
        xx = np.zeros(p.n)
        xx[self.base] = x
        xx[self.wc] = (p.b[self.exp + 2] - t) / self.aw
        zz = np.zeros(p.m)
        nnz = np.zeros(len(self.nn))
        sums = self.Cw @ t
        nnz[self.lrows] = lam[:len(self.lrows)] / np.maximum(sums[self.lrows], 1e-300) / self.c
        nnz[self.plain] = lam[len(self.lrows):] / self.gs / self.c
        zz[self.nn] = nnz
        zz[self.eq] = eqdual / self.cs / self.c
        zw = self.Cw.T @ nnz
        r = (p.b[self.exp] - p.A[self.exp][:, self.base] @ x) / p.b[self.exp + 1]
        a = zw * t / p.b[self.exp + 1]
        zz[self.exp], zz[self.exp + 1], zz[self.exp + 2] = -a, a * (r - 1), zw
        slack = p.b - p.A @ xx
        slack[self.exp] = r * p.b[self.exp + 1]
        slack[self.exp + 1] = p.b[self.exp + 1]
        slack[self.exp + 2] = t
        return {"status": "solved", "x": xx, "s": slack, "z": zz}


def try_gp(p, recover, original, budget=45.0):
    try:
        model = LogSumModel(p)
    except (ValueError, IndexError):
        return None
    start = time.monotonic()
    x = np.zeros(model.n)
    if len(model.cb):
        x = linalg.lstsq(model.C, model.cb, check_finite=False)[0]
    F, J = model.evaluate(x)
    s, lam = np.maximum(1, -F), np.ones(model.k)
    y = np.zeros(len(model.cb))
    verbose = bool(os.environ.get("SOLVER_VERBOSE"))
    best, besterr = None, float("inf")
    for it in range(100):
        if time.monotonic() - start > budget:
            break
        cand = model.candidate(x, lam, y)
        if cand is not None:
            cand = recover(cand)
            ok, err = assess(original, cand["x"], cand["s"], cand["z"])
            if err < besterr:
                best, besterr = cand, err
            if ok:
                return cand
        F, J, H = model.evaluate(x, lam)
        rp, re = F + s, model.C @ x - model.cb
        rd = model.q + J.T @ lam + model.C.T @ y
        mu = float(s @ lam) / model.k
        if verbose:
            print("gp", it, "err", besterr, "mu", mu, file=sys.stderr)
        M = H + J.T @ ((lam / s)[:, None] * J)
        K = np.block([[M, model.C.T], [model.C, np.zeros((len(y), len(y)))]])
        scale = 1 / np.sqrt(np.maximum(np.max(abs(K), axis=1, initial=0), 1e-8))
        Ks = K * scale[:, None] * scale[None, :]
        reg = np.r_[np.full(model.n, 1e-12), np.full(len(y), -1e-12)]
        try:
            factor = linalg.lu_factor(Ks + np.diag(reg), check_finite=False)
            def direction(rc):
                h = (rc + lam * rp) / s
                rhs = np.r_[-rd - J.T @ h, -re]
                v = linalg.lu_solve(factor, rhs * scale, check_finite=False)
                for _ in range(2):
                    v += linalg.lu_solve(factor, rhs * scale - Ks @ v, check_finite=False)
                v *= scale
                dx, dy = v[:model.n], v[model.n:]
                ds = -rp - J @ dx
                dl = h + (lam / s) * (J @ dx)
                return dx, ds, dl, dy
            da = direction(-s * lam)
            aa = min(maxstep(s, da[1]), maxstep(lam, da[2]))
            mua = float((s + aa * da[1]) @ (lam + aa * da[2])) / model.k
            sigma = min(0.7, max(0.01, (max(0, mua) / mu) ** 3))
            dx, ds, dl, dy = direction(-s * lam + sigma * mu - da[1] * da[2])
            alpha = 0.99 * min(maxstep(s, ds), maxstep(lam, dl))
            merit = max(norm(rp), norm(re), norm(rd), norm(s * lam - sigma * mu))
            accepted = False
            for _ in range(35):
                xn, sn, ln, yn = x + alpha * dx, s + alpha * ds, lam + alpha * dl, y + alpha * dy
                Fn, Jn = model.evaluate(xn)
                dn = model.q + Jn.T @ ln + model.C.T @ yn
                newmerit = max(norm(Fn + sn), norm(model.C @ xn - model.cb),
                               norm(dn), norm(sn * ln - sigma * mu))
                if newmerit <= (1 - 1e-3 * alpha) * merit:
                    x, s, lam, y = xn, sn, ln, yn
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted or not np.isfinite(x).all():
                break
        except (ValueError, FloatingPointError, linalg.LinAlgError):
            break
    if time.monotonic() - start < budget:
        try:
            from scipy.optimize import minimize
            def fun(v):
                if time.monotonic() - start >= budget:
                    raise TimeoutError()
                return float(model.q @ v)
            constraints = [{"type": "ineq", "fun": lambda v: -model.evaluate(v)[0],
                            "jac": lambda v: -model.evaluate(v)[1]}]
            if len(model.cb):
                constraints.append({"type": "eq", "fun": lambda v: model.C @ v - model.cb,
                                    "jac": lambda v: model.C})
            result = minimize(fun, x, jac=lambda v: model.q, method="SLSQP",
                              constraints=constraints, options={"ftol": 1e-12, "maxiter": 200})
            if hasattr(result, "multipliers"):
                mult = result.multipliers
                me = len(model.cb)
                trial = model.candidate(result.x, mult[me:], -mult[:me])
                if trial is not None:
                    trial = recover(trial)
                    ok, err = assess(original, trial["x"], trial["s"], trial["z"])
                    if ok or err < besterr:
                        return trial
        except (ValueError, TimeoutError, FloatingPointError, linalg.LinAlgError):
            pass
    return best
