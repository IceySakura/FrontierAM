#!/usr/bin/env python3
"""Black-box cubature on [0,1]^d under a hard point budget (JSON Lines oracle).

Strategy
--------
* d == 2: tensor Gauss-Legendre 39x39 grid plus 512 scrambled-Sobol points.
  The Sobol points integrate the residual of the GL tensor interpolant; that
  correction is applied only when it is statistically significant (4 sub-block
  standard error), i.e. for kinks/discontinuities.  Smooth or peaky integrands
  keep the spectrally accurate Gauss rule.
* d != 2: scrambled Sobol (2^m <= budget points) and the QMC mean.  A nested
  least-squares Legendre fit (degree chosen by leave-one-out PRESS) replaces
  the mean only when it is near-exact (PRESS ratio < 1e-8).
* Any internal failure falls back to the mean of the values already obtained.
All function values come from the oracle; every point counts against budget.
"""
import json
import math
import sys

import numpy as np


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------
class Oracle:
    def __init__(self, d, budget, fin, fout):
        self.d = d
        self.budget = budget
        self.used = 0
        self.fin = fin
        self.fout = fout
        self.all_x = []
        self.all_y = []

    def remaining(self):
        return self.budget - self.used

    def __call__(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, self.d)
        pts = np.clip(np.nan_to_num(pts, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        n = len(pts)
        if n == 0:
            return np.zeros(0)
        if n > self.remaining():
            raise RuntimeError("budget")
        out = np.empty(n)
        for s in range(0, n, 512):
            chunk = pts[s:s + 512]
            self.fout.write(json.dumps({"type": "query", "points": chunk.tolist()}) + "\n")
            self.fout.flush()
            line = self.fin.readline()
            while line and not line.strip():
                line = self.fin.readline()
            if not line:
                raise EOFError("oracle closed")
            msg = json.loads(line)
            vals = np.asarray(msg.get("values", []), dtype=float).reshape(-1)
            if vals.size != len(chunk):
                raise RuntimeError("bad values")
            self.used += len(chunk)
            out[s:s + len(chunk)] = vals
        self.all_x.append(pts)
        self.all_y.append(out)
        return out


class LocalOracle:
    """Same interface, for development: wraps a vectorised python function."""

    def __init__(self, d, budget, fun):
        self.d = d
        self.budget = budget
        self.used = 0
        self.fun = fun
        self.all_x = []
        self.all_y = []

    def remaining(self):
        return self.budget - self.used

    def __call__(self, pts):
        pts = np.clip(np.asarray(pts, dtype=float).reshape(-1, self.d), 0.0, 1.0)
        if len(pts) > self.remaining():
            raise RuntimeError("budget")
        self.used += len(pts)
        y = np.asarray(self.fun(pts), dtype=float).reshape(-1)
        self.all_x.append(pts)
        self.all_y.append(y)
        return y


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def sobol(d, n, seed):
    from scipy.stats import qmc
    m = int(round(math.log2(n)))
    if 2 ** m == n:
        return qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)
    return qmc.Sobol(d, scramble=True, seed=seed).random(n)


def legendre_1d(x, pmax):
    """Orthonormal shifted Legendre polynomials on [0,1]; shape (len(x), pmax+1)."""
    t = 2.0 * x - 1.0
    P = np.empty((len(x), pmax + 1))
    P[:, 0] = 1.0
    if pmax >= 1:
        P[:, 1] = t
    for k in range(1, pmax):
        P[:, k + 1] = ((2 * k + 1) * t * P[:, k] - k * P[:, k - 1]) / (k + 1)
    P *= np.sqrt(2.0 * np.arange(pmax + 1) + 1.0)
    return P


def total_degree_indices(d, p):
    """Multi-indices with |alpha| <= p, sorted by total degree."""
    out = [[]]
    for _ in range(d):
        out = [a + [k] for a in out for k in range(p + 1) if sum(a) + k <= p]
    out.sort(key=lambda a: (sum(a), a[::-1]))
    return np.array(out, dtype=int).reshape(-1, d)


def design(X, idx, pmax):
    N, d = X.shape
    A = np.ones((N, len(idx)))
    for j in range(d):
        Pj = legendre_1d(X[:, j], pmax)
        A *= Pj[:, idx[:, j]]
    return A


def ls_cv_estimate(X, y, max_terms, w=None, max_degree=60):
    """Nested total-degree LS fits; pick degree by PRESS; return (est, info).

    w: optional sampling weights (1/density) for the residual mean; with the
    constant in the basis the in-sample LS estimate is the constant coefficient.
    """
    N, d = X.shape
    # largest degree with number of terms <= max_terms
    p = 0
    while p < max_degree and math.comb(d + p + 1, d) <= max_terms:
        p += 1
    idx = total_degree_indices(d, p)
    degs = idx.sum(axis=1)
    A = design(X, idx, p)
    Q, R = np.linalg.qr(A)
    qy = Q.T @ y
    Q2c = np.cumsum(Q * Q, axis=1)
    best = None
    results = []
    for q in range(0, p + 1):
        k = int(np.sum(degs <= q))
        fit = Q[:, :k] @ qy[:k]
        r = y - fit
        h = Q2c[:, k - 1]
        if np.max(h) >= 0.999:
            break
        loo = r / (1.0 - h)
        press = float(np.mean(loo * loo))
        c = np.linalg.solve(R[:k, :k], qy[:k]) if k > 1 else np.array([qy[0] / R[0, 0]])
        est = float(c[0])
        results.append((q, k, press, est))
        if best is None or press < best[2]:
            best = (q, k, press, est)
    return best[3], {"best": best, "all": results}


# --------------------------------------------------------------------------
# main integration logic
# --------------------------------------------------------------------------
def gl_nodes(n):
    z, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (z + 1.0)
    w = 0.5 * w
    v = np.sqrt((1.0 - z * z) * w) * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return x, w, v


def bary_matrix(t, x, v):
    """Barycentric Lagrange interpolation matrix L[i, j] (nodes x, weights v)."""
    diff = t[:, None] - x[None, :]
    exact = np.abs(diff) < 1e-14
    diff[exact] = 1.0
    C = v[None, :] / diff
    L = C / np.sum(C, axis=1, keepdims=True)
    rows = np.any(exact, axis=1)
    if np.any(rows):
        L[rows] = exact[rows].astype(float)
    return L


CORR_K = 2.0


def gl2_with_correction(F, seed, n=39, m=512):
    """Tensor Gauss-Legendre n x n + scrambled Sobol correction of the interpolant residual."""
    x, w, v = gl_nodes(n)
    G = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    Fg = F(G).reshape(n, n)
    I_gl = float(w @ Fg @ w)
    S = sobol(2, m, seed)
    fs = F(S)
    L1 = bary_matrix(S[:, 0], x, v)
    L2 = bary_matrix(S[:, 1], x, v)
    ps = np.sum((L1 @ Fg) * L2, axis=1)
    r = fs - ps
    corr = float(np.mean(r))
    blocks = r.reshape(4, -1).mean(axis=1)
    se = float(np.std(blocks, ddof=1) / 2.0)
    info = {"I_gl": I_gl, "corr": corr, "se": se, "rms_r": float(np.sqrt(np.mean(r * r))),
            "std_f": float(np.std(fs)), "mean_s": float(np.mean(fs))}
    # The interpolant residual is integrated by QMC only when that correction is
    # clearly signal (non-smooth integrands); for resolved/analytic integrands the
    # Gauss rule is far more accurate than the noisy correction.
    if math.isfinite(corr) and abs(corr) > CORR_K * se:
        return I_gl + corr, info
    return I_gl, info


def integrate(F, d, budget, seed):
    if d == 2 and budget >= 2048:
        return gl2_with_correction(F, seed)
    n = 2 ** int(math.floor(math.log2(budget)))
    X = sobol(d, n, seed)
    y = F(X)
    est = float(np.mean(y))
    info = {"plain": est}
    try:
        max_terms = max(1, n // 3)
        ls, li = ls_cv_estimate(X, y, max_terms)
        press0 = li["all"][0][2]
        info["ls"] = ls
        info["press_ratio"] = li["best"][2] / max(press0, 1e-300)
        if li["best"][2] < 1e-8 * press0 and math.isfinite(ls):
            est = ls
    except Exception:
        pass
    return est, info


def main():
    fin = sys.stdin
    fout = sys.stdout
    line = fin.readline()
    while line and not line.strip():
        line = fin.readline()
    init = json.loads(line)
    d = int(init["dimension"])
    budget = int(init["budget"])
    seed = int(init.get("seed", 0) or 0)
    F = Oracle(d, budget, fin, fout)
    est = None
    try:
        est, _ = integrate(F, d, budget, seed)
        est = float(est)
    except Exception as exc:  # pragma: no cover - safety net
        sys.stderr.write("integrate failed: %r\n" % (exc,))
        est = None
    if est is None or not math.isfinite(est):
        try:
            if F.used == 0:
                X = sobol(d, min(budget, 1024), seed)
                F(X)
            ys = np.concatenate(F.all_y)
            est = float(np.mean(ys[np.isfinite(ys)]))
        except Exception:
            est = None
    if est is None or not math.isfinite(est):
        est = 0.0
    fout.write(json.dumps({"type": "result", "estimate": est}) + "\n")
    fout.flush()


if __name__ == "__main__":
    main()
