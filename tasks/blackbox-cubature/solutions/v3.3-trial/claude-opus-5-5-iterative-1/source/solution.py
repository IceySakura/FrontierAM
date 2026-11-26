"""Adaptive preintegration cubature for E f(Z), Z ~ N(0, I_d), from point queries (see theory/THEORY.md).

1. Pilot: n0 scrambled Sobol' points (normal scores); Stein moments b = E[(f - m) Z], H = E[(f - m)(Z Z^T - I)].
2. Geometry: parallel lines z = y + s v (v = b/|b| or the top eigenvector of H), or rays z = rho(s) theta from the
   origin when the pilot shows a rare high-value region around the origin.
3. Each line integral int f(z(s)) phi(s) ds is computed by the batched adaptive LineIntegrator: coarse grid,
   ENO-type detection of kinks/jumps, bracketing (bisection / branch-intersection), Gauss rules for the normal weight
   on every smooth piece.
4. The outer integral over the line origins (d-1 dims) or directions is a scrambled Sobol' average.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc, chi

from cubature_protocol import Session
from linequad import LineIntegrator, WindowIntegrator
import quadric

T0 = time.monotonic()


def log(*a):
    print(f"[{time.monotonic() - T0:6.2f}s]", *a, file=sys.stderr, flush=True)


class Budgeted:
    """Wraps the session: hard budget, batching, coordinate clipping."""

    def __init__(self, sess):
        self.sess = sess
        self.used = 0
        self.budget = sess.budget

    @property
    def remaining(self):
        return self.budget - self.used

    def __call__(self, Z):
        Z = np.clip(np.asarray(Z, float), -39.9, 39.9)
        if len(Z) == 0:
            return np.zeros(0)
        if len(Z) > self.remaining:
            raise RuntimeError('budget')
        v = self.sess.query(Z)
        self.used += len(Z)
        v = np.where(np.isfinite(v), v, 0.0)
        return v


def sobol_normal(n, dim, seed):
    s = qmc.Sobol(dim, scramble=True, seed=seed)
    u = s.random(n)
    return ndtri(np.clip(u, 1e-15, 1 - 1e-15))


def complement_basis(v, M=None):
    """Orthonormal basis (d, d-1) of v-perp; columns ordered by |eigenvalues| of M restricted to v-perp."""
    d = len(v)
    A = np.eye(d) - np.outer(v, v)
    q, _ = np.linalg.qr(np.column_stack([v, np.eye(d)]))
    B = q[:, 1:d]
    if M is not None:
        Mp = B.T @ M @ B
        ev, U = np.linalg.eigh(0.5 * (Mp + Mp.T))
        order = np.argsort(-np.abs(ev))
        B = B @ U[:, order]
    return B


def main():
    sess = Session()
    f = Budgeted(sess)
    d, N, seed = sess.d, sess.budget, sess.seed
    rng = np.random.default_rng([seed, int(os.environ.get('CUBATURE_SEED_OFFSET', '0'))])
    est = 0.0
    try:
        # ------------------------------------------------------------ pilot
        n0 = min(1024, N // 16) if N >= 4096 else max(1, N // 4)
        Z0 = sobol_normal(n0, d, rng.integers(1 << 31))
        F0 = f(Z0)
        est = float(F0.mean())
        m0, sd0 = est, F0.std()
        scale = max(np.mean(np.abs(F0)), 1e-12)
        c = F0 - m0
        b = (Z0 * c[:, None]).mean(0)
        H = (Z0.T * c) @ Z0 / n0 - np.eye(d) * c.mean()
        nb = np.linalg.norm(b)
        log(f"d={d} pilot mean={m0:.6g} sd={sd0:.4g} |b|={nb:.4g}")
        mode = os.environ.get('CUBATURE_MODE', 'auto')
        nzv = F0[F0 != 0]
        zfrac = np.mean(F0 == 0)
        digital = zfrac >= 0.2 and len(nzv) >= 8 and np.all(nzv == nzv[0]) and nb > 0
        if mode == 'auto':
            mode = 'window' if digital else ('qcv' if zfrac < 0.05 else 'hxf')
        log(f"zero fraction {zfrac:.3f}, constant nonzero values: {digital} -> mode {mode}")
        stages = None
        if mode in ('window', 'lines', 'wgrad'):
            v = b / nb if nb > 0 else np.linalg.eigh(H)[1][:, -1]
            Mord = H
            if mode == 'wgrad' or (mode == 'window' and not digital):
                kg = 12 if d <= 32 else 10
                G, C = fd_gradients(f, Z0, F0, kg)
                ev, U = np.linalg.eigh(C)
                v1 = U[:, -1]
                if v1 @ b < 0:
                    v1 = -v1
                log(f"GPCA from {len(G)} FD gradients: top ev share {ev[-1] / max(ev.sum(), 1e-300):.3f}, "
                    f"cos(v1, b)={v1 @ v:.4f}")
                v = v1
                stages = ()
                Mord = C + 1e-3 * np.trace(C) / d * (H @ H) / max(np.trace(H @ H) / d, 1e-300)
                mode = 'window' 
            try:
                est_l, nlines = run_parallel(f, d, v, Mord, scale, rng, window=(mode == 'window'),
                                             stages=stages)
                log(f"lines={nlines} used={f.used} est={est_l:.10g}")
                if np.isfinite(est_l):
                    est = est_l
            except Exception as exc:  # never fail: finish with RQMC on what is left
                log(f"line stage failed: {type(exc).__name__}: {exc}")
                if f.remaining >= 64:
                    n_rest = f.remaining
                    est_r = run_rqmc(f, d, b, H, rng)
                    est = (n0 * m0 + n_rest * est_r) / (n0 + n_rest)
        elif mode == 'grqmc':
            est = run_grqmc(f, d, Z0, F0, H, rng)
        elif mode == 'hcv':
            est = run_hinge_cv(f, d, Z0, F0, b, H, rng)
        elif mode == 'hxf':
            est = hinge_cv_crossfit(f, d, Z0, F0, b, H, rng, mmax=int(os.environ.get('HMAX', '16' if d <= 32 else '12')))
        elif mode == 'qcv':
            status, r = run_quadric_cv(f, d, Z0, F0, b, H, rng, scale)
            if status == 'toobig':
                # cheap jump detection on a few lines (half through the most extreme pilot points)
                k = 8
                Y = rng.standard_normal((k, d))
                dev = np.abs(F0 - np.median(F0))
                Y[:k // 2] = Z0[np.argsort(-dev)[:k // 2]]
                V = rng.standard_normal((k, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
                Y -= (Y * V).sum(1, keepdims=True) * V
                P, Jv, li, _ = find_jumps(f, Y, V, np.linspace(-5.0, 5.0, 21), scale, steps=12)
                good = np.abs(Jv) > 0.02 * max(F0.std(), 1e-12) + 1e-9 * scale
                njump = int(good.sum())
                status = 'fail' if njump >= 3 else 'nojump'
                log(f"jump detection on {k} lines: {njump} jumps -> {status}")
                if status == 'fail' and d <= 40 and os.environ.get('NORAYS') != '1':
                    # interior point: midpoint of a chord (two crossings on one line)
                    mids = []
                    for l in np.unique(li[good]):
                        pts = P[good][li[good] == l]
                        if len(pts) == 2:
                            mids.append(0.5 * (pts[0] + pts[1]))
                    if mids:
                        r = run_quadric_rays(f, d, mids[0], float(np.median(np.abs(Jv[good]))), b, H, rng, scale)
            if r is None:
                dev = np.abs(F0 - np.median(F0))
                rare = dev.max() > 20 * (np.quantile(dev, 0.98) + 1e-12 * scale)
                log(f"quadric CV not applicable ({status}, rare region: {rare})")
                if status == 'fail':
                    r = run_mis(f, d, Z0, F0, b, H, rng, force=False)
                elif os.environ.get('NOHINGE') != '1':
                    r = hinge_cv_crossfit(f, d, Z0, F0, b, H, rng, mmax=16 if d <= 16 else 12)
                else:
                    r = run_grqmc(f, d, Z0, F0, H, rng)
            est = r
        elif mode == 'ce':
            est_m = run_ce(f, d, Z0, F0, b, H, rng)
            if np.isfinite(est_m):
                est = est_m
        else:
            est_m = run_mis(f, d, Z0, F0, b, H, rng, force={'rqmc': False, 'mis': True}.get(mode, False))
            if np.isfinite(est_m):
                est = est_m
    except Exception as exc:
        log(f"failure: {type(exc).__name__}: {exc}")
        try:   # spend what is left on plain scrambled Sobol' and pool it with the pilot
            n_rest = f.remaining
            if n_rest >= 64 and 'Z0' in locals():
                s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
                Zr = ndtri(np.clip(s.random(n_rest), 1e-15, 1 - 1e-15))
                Fr = f(Zr)
                est = float((F0.sum() + Fr.sum()) / (len(F0) + len(Fr)))
        except Exception as exc2:
            log(f"fallback failure: {type(exc2).__name__}: {exc2}")
    if not np.isfinite(est):
        est = 0.0
    log(f"final est={est:.12g} used={f.used} time={time.monotonic() - T0:.2f}s")
    sess.result(float(est))


def importance_rotation(b, H):
    """Orthonormal d x d matrix whose columns are ordered by importance (eigenvectors of b b^T + H H^T)."""
    C = np.outer(b, b) + H @ H.T
    ev, U = np.linalg.eigh(C)
    return U[:, np.argsort(-ev)]


def run_rqmc(f, d, b, H, rng, n=None):
    """Scrambled Sobol' on the rotated normal scores; returns the plain average."""
    R = importance_rotation(b, H)
    n = f.remaining if n is None else n
    s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
    u = s.random(n)
    Z = ndtri(np.clip(u, 1e-15, 1 - 1e-15)) @ R.T
    return float(f(Z).mean())


def mis_proposal(Z0, F0, d):
    """Gaussian proposal N(mu, S) fitted to the pilot points weighted by |f - median| (shrunk towards I)."""
    w = np.abs(F0 - np.median(F0))
    if not np.isfinite(w).all() or w.sum() <= 0:
        return None
    w = w / w.sum()
    ess = 1.0 / (w * w).sum()
    mu = w @ Z0
    D = Z0 - mu
    C = (D.T * w) @ D
    lam = min(1.0, (d + 2.0) / max(ess, 1.0))
    S = (1 - lam) * C + lam * np.eye(d)
    ev, V = np.linalg.eigh(0.5 * (S + S.T))
    ev = np.clip(ev, 0.15, 1.5)
    return mu, V, ev, ess


def log_ratio(Z, mu, V, ev):
    """log of N(mu, V diag(ev) V^T)(z) / N(0, I)(z)."""
    Y = (Z - mu) @ V
    return -0.5 * ((Y * Y) / ev).sum(1) - 0.5 * np.log(ev).sum() + 0.5 * (Z * Z).sum(1)


def run_mis(f, d, Z0, F0, b, H, rng, force=None):
    """Deterministic-mixture importance sampling with scrambled Sobol' in both components (balance heuristic):
    q = a N(0,I) + (1-a) N(mu, S); estimate = mean over all points of f phi / q. Falls back to plain RQMC when the
    pilot predicts no variance reduction."""
    n0 = len(Z0)
    prop = mis_proposal(Z0, F0, d)
    R = f.remaining
    use = False
    if prop is not None:
        mu, V, ev, ess = prop
        n_is = 1 << int(math.floor(math.log2(max(0.7 * R, 1))))
        a = (n0 + R - n_is) / (n0 + R)
        lr = log_ratio(Z0, mu, V, ev)
        wq = 1.0 / (a + (1 - a) * np.exp(np.clip(lr, -700, 700)))
        m = F0.mean()
        v_nom = np.mean((F0 - m) ** 2)
        v_mis = np.mean(F0 ** 2 * wq) - m * m
        log(f"MIS: |mu|={np.linalg.norm(mu):.3g} ess={ess:.1f} ev=[{ev.min():.2f},{ev.max():.2f}] a={a:.3f} "
            f"v_nom={v_nom:.4g} v_mis(pilot)={v_mis:.4g}")
        use = v_mis < 0.6 * v_nom if force is None else force
    if not use:
        n_main = f.remaining
        est_main = run_rqmc(f, d, b, H, rng)
        return (n0 * F0.mean() + n_main * est_main) / (n0 + n_main)
    Rot = importance_rotation(b, H)
    n_nom = R - n_is
    s1 = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
    Zn = ndtri(np.clip(s1.random(n_nom), 1e-15, 1 - 1e-15)) @ Rot.T if n_nom > 0 else np.zeros((0, d))
    order = np.argsort(-ev)
    s2 = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
    Ui = ndtri(np.clip(s2.random(n_is), 1e-15, 1 - 1e-15))
    Zi = mu + (Ui * np.sqrt(ev[order])) @ V[:, order].T
    Zall = np.vstack([Z0, Zn, Zi])
    Fall = np.concatenate([F0, f(np.vstack([Zn, Zi]))])
    lr = log_ratio(Zall, mu, V, ev)
    wq = 1.0 / (a + (1 - a) * np.exp(np.clip(lr, -700, 700)))
    est = float(np.mean(Fall * wq))
    nom = float(np.mean(Fall[:n0 + n_nom]))
    log(f"MIS est={est:.10g} nominal-only={nom:.10g}")
    return est


def log_q_gauss(Z, mu, V, ev):
    """log density of N(mu, V diag(ev) V^T)."""
    Y = (Z - mu) @ V
    d = Z.shape[1]
    return -0.5 * ((Y * Y) / ev).sum(1) - 0.5 * np.log(ev).sum() - 0.5 * d * math.log(2 * math.pi)


def log_phi(Z):
    d = Z.shape[1]
    return -0.5 * (Z * Z).sum(1) - 0.5 * d * math.log(2 * math.pi)


def fit_gauss(Z, w, d, ev_lo=0.05, ev_hi=2.0):
    """Weighted Gaussian fit with shrinkage of the covariance towards its mean eigenvalue."""
    w = w / w.sum()
    ess = 1.0 / (w * w).sum()
    mu = w @ Z
    D = Z - mu
    C = (D.T * w) @ D
    lam = min(1.0, (d + 2.0) / max(ess, 1.0))
    target = np.eye(d) if os.environ.get('SHRINK_I', '1') == '1' else np.trace(C) / d * np.eye(d)
    S = (1 - lam) * C + lam * target
    ev, V = np.linalg.eigh(0.5 * (S + S.T))
    return mu, V, np.clip(ev, ev_lo, ev_hi), ess


def run_ce(f, d, Z0, F0, b, H, rng, n_stage=1024, n_iter=None, frac_nom=None):
    n_iter = int(os.environ.get('CE_IT', '3')) if n_iter is None else n_iter
    frac_nom = float(os.environ.get('CE_NOM', '0.25')) if frac_nom is None else frac_nom
    """Adaptive (cross-entropy) importance sampling for rare high-value regions, deterministic-mixture estimator
    over all stages (balance heuristic) with the constant control c = median of the pilot:
    I = c + mean_i (f(z_i) - c) phi(z_i) / q_mix(z_i), q_mix = sum_k (n_k / n) q_k."""
    c = float(np.median(F0))
    comps = [('nom', None)]
    Zs, Fs, counts = [Z0], [F0], [len(Z0)]
    g = np.abs(F0 - c)
    Zcur, wcur = Z0, g.copy()
    for it in range(n_iter):
        if wcur.sum() <= 0:
            break
        mu, V, ev, ess = fit_gauss(Zcur, wcur, d)
        order = np.argsort(-ev)
        s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
        U = ndtri(np.clip(s.random(n_stage), 1e-15, 1 - 1e-15))
        Z = mu + (U * np.sqrt(ev[order])) @ V[:, order].T
        Fz = f(Z)
        comps.append(('g', (mu, V, ev)))
        Zs.append(Z); Fs.append(Fz); counts.append(n_stage)
        lw = log_phi(Z) - log_q_gauss(Z, mu, V, ev)
        wnew = np.abs(Fz - c) * np.exp(np.clip(lw, -700, 50))
        if os.environ.get('CE_ACC', '1') == '1':
            Zcur = np.vstack([Zcur, Z]); wcur = np.concatenate([wcur, wnew])
        else:
            Zcur, wcur = Z, wnew
        log(f"CE stage {it}: |mu|={np.linalg.norm(mu):.3g} ev=[{ev.min():.3g},{ev.max():.3g}] ess={ess:.1f} "
            f"hit-mean={np.mean(np.abs(Fz - c) > 10 * np.median(np.abs(F0 - c)) + 1e-300):.3f}")
    # final allocation: remaining budget split between nominal (RQMC, rotated) and the last proposal
    R = f.remaining
    n_nom = int(frac_nom * R)
    n_last = R - n_nom
    Rot = importance_rotation(b, H)
    if n_nom > 0:
        s1 = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
        Zn = ndtri(np.clip(s1.random(n_nom), 1e-15, 1 - 1e-15)) @ Rot.T
        Zs.append(Zn); Fs.append(f(Zn)); counts.append(n_nom); comps.append(('nom', None))
    if n_last > 0 and len(comps) > 1:
        mu, V, ev = [p for k, p in comps if k == 'g'][-1]
        order = np.argsort(-ev)
        s2 = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
        U = ndtri(np.clip(s2.random(n_last), 1e-15, 1 - 1e-15))
        Z = mu + (U * np.sqrt(ev[order])) @ V[:, order].T
        Zs.append(Z); Fs.append(f(Z)); counts.append(n_last); comps.append(('g', (mu, V, ev)))
    Zall = np.vstack(Zs)
    Fall = np.concatenate(Fs)
    n = len(Zall)
    lp = log_phi(Zall)
    # mixture density relative to phi
    mix = np.zeros(n)
    for (kind, par), cnt in zip(comps, counts):
        if kind == 'nom':
            mix += cnt / n
        else:
            mix += cnt / n * np.exp(np.clip(log_q_gauss(Zall, *par) - lp, -700, 700))
    est = c + float(np.mean((Fall - c) / mix))
    log(f"CE est={est:.10g} (c={c:.6g}, nominal-only={np.mean(np.concatenate([Fs[0]] + [F for (k, _), F in zip(comps, Fs) if k == 'nom'][1:])):.10g})")
    return est


def fd_gradients(f, Z0, F0, k, h=1e-5):
    """Forward-difference gradients at k pilot points with the largest |f - median| among the nonzero values
    (the payoff is smooth there); returns the (k, d) array and the d x d matrix mean g g^T."""
    d = Z0.shape[1]
    dev = np.abs(F0 - np.median(F0))
    cand = np.nonzero(F0 != 0)[0] if np.any(F0 != 0) else np.arange(len(F0))
    # spread over the candidates: take every j-th by deviation rank (avoid only the extreme tail)
    order = cand[np.argsort(-dev[cand])]
    pick = order[np.linspace(0, len(order) - 1, min(k, len(order))).astype(int)]
    P = []
    for i in pick:
        P.append(Z0[i][None, :] + h * np.eye(d))
    vals = f(np.vstack(P)).reshape(len(pick), d)
    G = (vals - F0[pick][:, None]) / h
    C = G.T @ G / len(pick)
    return G, C


def run_grqmc(f, d, Z0, F0, H, rng, kg=16):
    """GPCA-rotated scrambled Sobol': forward-difference gradients at kg pilot points give C = mean g g^T; the
    remaining budget is one scrambled Sobol' point set in the eigenbasis of C (null space ordered by H)."""
    kg = min(kg, max(2, (f.remaining // 8) // (d + 1)))
    G, C = fd_gradients(f, Z0, F0, kg)
    Rot = gpca_rotation(C, H)
    n_rest = f.remaining
    s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
    Z = ndtri(np.clip(s.random(n_rest), 1e-15, 1 - 1e-15)) @ Rot.T
    return float(f(Z).mean())


def gpca_rotation(C, H):
    """Columns: eigenvectors of C by decreasing eigenvalue; the null space of C ordered by |eig| of H."""
    d = C.shape[0]
    ev, U = np.linalg.eigh(0.5 * (C + C.T))
    order = np.argsort(-ev)
    U = U[:, order]
    ev = ev[order]
    r = int(np.sum(ev > 1e-10 * max(ev[0], 1e-300)))
    if r < d:
        Bn = U[:, r:]
        Hp = Bn.T @ H @ Bn
        e2, W = np.linalg.eigh(0.5 * (Hp + Hp.T))
        U = np.column_stack([U[:, :r], Bn @ W[:, np.argsort(-np.abs(e2))]])
    return U


def find_jumps(f, Y, V, grid, scale, steps=20, thr_factor=8.0):
    """Batched jump localisation along lines z = Y_l + s V_l: grid scan, then bisection of every grid interval whose
    increment is large compared with the typical increment on the line. Returns boundary points, oriented jumps
    (value after minus value before along the line), the line index and the bracket half-width."""
    L, n = len(Y), len(grid)
    Fv = f((Y[:, None, :] + grid[None, :, None] * V[:, None, :]).reshape(L * n, -1)).reshape(L, n)
    dF = np.diff(Fv, axis=1)
    typ = np.median(np.abs(dF), axis=1, keepdims=True)
    big = np.abs(dF) > np.maximum(thr_factor * typ, 1e-6 * scale)
    li, gi = np.nonzero(big)
    cap = max(0, (f.remaining - 2048) // max(steps, 1))       # never let the bisections eat the whole budget
    if len(li) > cap:
        li, gi = li[:cap], gi[:cap]
    if len(li) == 0:
        return np.zeros((0, Y.shape[1])), np.zeros(0), np.zeros(0, int), Fv
    a = grid[gi].copy(); b = grid[gi + 1].copy()
    fa = Fv[li, gi].copy(); fb = Fv[li, gi + 1].copy()
    for _ in range(steps):
        mid = 0.5 * (a + b)
        fm = f(Y[li] + mid[:, None] * V[li])
        left = np.abs(fm - fa) <= np.abs(fm - fb)
        a = np.where(left, mid, a); fa = np.where(left, fm, fa)
        b = np.where(left, b, mid); fb = np.where(left, fb, mid * 0 + fb)
        fb = np.where(left, fb, fm)
    s = 0.5 * (a + b)
    P = Y[li] + s[:, None] * V[li]
    return P, fb - fa, li, Fv


def find_kinks(f, Y, V, grid, scale):
    """Batched kink localisation along lines z = Y_l + s V_l: ENO-type detection on the grid, refinement of the
    isolated flagged intervals only (clean one-sided stencils) by bisection with the branch test, final location at
    the intersection of the two one-sided secants. Returns kink points and the one-sided slope jumps."""
    L, n = len(Y), len(grid)
    Fv = f((Y[:, None, :] + grid[None, :, None] * V[:, None, :]).reshape(L * n, -1)).reshape(L, n)

    def ev(li, s):
        return f(Y[li] + np.asarray(s)[:, None] * V[li])

    integ = LineIntegrator(ev, grid=grid, eps_line=1e-14 * scale, flag_factor=2.0, max_ref=24)
    flag = integ._flag(grid, Fv, scale)
    brs = []
    for l, j in np.argwhere(flag):
        if j < 2 or j > n - 4:
            continue
        if flag[l, max(0, j - 2):j].any() or flag[l, j + 1:j + 3].any():
            continue
        brs.append({'line': int(l), 'a': grid[j], 'b': grid[j + 1],
                    'left': [(grid[j - 1], Fv[l, j - 1]), (grid[j], Fv[l, j])],
                    'right': [(grid[j + 1], Fv[l, j + 1]), (grid[j + 2], Fv[l, j + 2])]})
    if not brs:
        return np.zeros((0, Y.shape[1])), np.zeros(0)
    locs = integ._refine(brs, scale)
    P, D = [], []
    for br, loc in zip(brs, locs):
        (ta1, fa1), (ta0, fa0) = br['left'][-2], br['left'][-1]
        (tb0, fb0), (tb1, fb1) = br['right'][0], br['right'][1]
        if br['b'] - br['a'] > 1e-5 or ta0 == ta1 or tb0 == tb1:
            continue
        sl = (fa0 - fa1) / (ta0 - ta1)
        sr = (fb1 - fb0) / (tb1 - tb0)
        if abs(fb0 - fa0) > 1e-3 * max(abs(sr - sl), 1e-12) + 1e-9 * scale:
            continue                      # a jump, not a kink
        P.append(Y[br['line']] + loc * V[br['line']])
        D.append(sr - sl)
    return (np.array(P) if P else np.zeros((0, Y.shape[1]))), np.array(D)


def hinge_fit(Z, F, iters=30, ridge=1e-8):
    """Least-squares fit of f ~ P2(z) + max(q(z), 0) (P2, q quadratic polynomials) by alternating least squares on
    the active set {q > 0} (initial active set: points above the quadratic regression). Returns (rms, coef of q)."""
    Phi = quadric.quad_features(Z)
    p = Phi.shape[1]
    c, *_ = np.linalg.lstsq(Phi, F, rcond=None)
    act = (F - Phi @ c) > 0
    best = None
    for it in range(iters):
        X = np.hstack([Phi, Phi * act[:, None]])
        coef = np.linalg.solve(X.T @ X + ridge * np.eye(2 * p), X.T @ F)
        new = Phi @ coef[p:] > 0
        rms = float(np.sqrt(np.mean((F - X @ coef) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, coef[p:].copy())
        if (new == act).all():
            break
        act = new
    return best


def run_hinge_cv(f, d, Z0, F0, b, H, rng, mmax=12):
    """Kink across a quadric: fit the hinge model f ~ P2(x) + max(q(x), 0) on a separate scrambled Sobol' sample
    (pilot + extra points), x = U^T z with U = I (d <= mmax) or the top mmax GPCA directions (forward-difference
    gradients); then one scrambled Sobol' set with regression control variates max(q, 0), z max(q, 0) and the
    quadratic monomials of x, whose means are known exactly (generalised chi-square moments)."""
    kg = min(16, max(2, (f.remaining // 8) // (d + 1)))
    G, C = fd_gradients(f, Z0, F0, kg)
    Rot = gpca_rotation(C, H)
    U = Rot[:, :min(d, mmax)]
    m_ = U.shape[1]
    p = (m_ + 1) * (m_ + 2) // 2
    nfit = max(len(Z0), min(10 * p, f.remaining // 4))
    Zf, Ff = Z0, F0
    if nfit > len(Z0):
        s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
        Ze = ndtri(np.clip(s.random(nfit - len(Z0)), 1e-15, 1 - 1e-15)) @ Rot.T
        Zf = np.vstack([Z0, Ze]); Ff = np.concatenate([F0, f(Ze)])
    rms, cq = hinge_fit(Zf @ U, Ff)
    Mx, mx, m0 = quadric.coef_to_quadric(cq, m_)
    M = U @ Mx @ U.T
    m = U @ mx
    e0, g0 = quadric.pos_part_moments(M, m, m0)
    log(f"hinge fit (dim {m_}) on {len(Zf)} points: rms={rms:.4g} (sd {Ff.std():.4g}); E max(q,0)={e0:.6g}")
    n = f.remaining
    s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
    Z = ndtri(np.clip(s.random(n), 1e-15, 1 - 1e-15)) @ Rot.T
    F = f(Z)
    h1 = np.maximum(quadric.q_eval(Z, M, m, m0), 0)
    X_ = Z @ U
    Phi = quadric.quad_features(X_)[:, :-1]
    iu = np.triu_indices(m_)
    EPhi = np.concatenate([(iu[0] == iu[1]).astype(float), np.zeros(m_)])
    X = np.column_stack([h1 - e0, Z * h1[:, None] - g0, Phi - EPhi])
    Xc = X - X.mean(0)
    beta, *_ = np.linalg.lstsq(Xc, F - F.mean(), rcond=None)
    est = float(F.mean() - X.mean(0) @ beta)
    rsd = float(np.std(F - F.mean() - Xc @ beta))
    log(f"hinge CV: plain={F.mean():.10g} cv={est:.10g} resid sd={rsd:.4g} vs sd={F.std():.4g}")
    if not (rms < 0.3 * Ff.std() and rsd < 0.3 * F.std()):
        log("hinge model does not fit: plain RQMC average of all points")
        return float((Ff.sum() + F.sum()) / (len(Ff) + len(F)))
    return est


def clean_quadric(M, m, m0, rel=1e-6):
    """Zero the eigenvalues of M below rel * max |eigenvalue| (fit noise in the null directions of a cylinder)."""
    lam, U = np.linalg.eigh(0.5 * (M + M.T))
    lam = np.where(np.abs(lam) < rel * np.abs(lam).max(), 0.0, lam)
    return (U * lam) @ U.T, m, m0


def checked_moments(M, m, m0, d, rng, tag):
    """P(q<0) and E[Z 1{q<0}] (Gil-Pelaez / conditional Sobol'), cross-checked against free Monte Carlo on the
    fitted surface (no queries); the Monte Carlo values are used if the two disagree."""
    lam = np.linalg.eigvalsh(M)
    kact = int(np.sum(np.abs(lam) > 0))
    if kact <= 3:
        p0, g0 = quadric.indicator_moments_cond(M, m, m0, n=1 << 20, seed=int(rng.integers(1 << 30)))
    else:
        p0, g0 = quadric.indicator_moments_fast(M, m, m0)
    nmc = 1 << 17
    Zc = rng.standard_normal((nmc, d))
    hc = (quadric.q_eval(Zc, M, m, m0) < 0).astype(float)
    pmc = hc.mean()
    gmc = (Zc * hc[:, None]).mean(0)
    se = math.sqrt(max(pmc * (1 - pmc), 1.0 / nmc) / nmc)
    bad = (not np.isfinite(p0)) or abs(p0 - pmc) > 6 * se + 1e-9 or not np.all(np.isfinite(g0)) \
        or np.abs(g0 - gmc).max() > 6 * math.sqrt(max(pmc, 1.0 / nmc) / nmc) + 1e-9
    log(f"{tag}: active dims={kact} P(q<0)={p0:.10g} (free MC {pmc:.6g} +- {se:.2g}){' MISMATCH' if bad else ''}")
    if bad:
        return None
    return p0, g0


def quadric_cv_estimate(f, d, M, m, m0, b, H, rng, tag="quadric", Jpts=None, Jvals=None, R=None):
    if R is None:
        R = int(os.environ.get('QCV_R', '4' if f.remaining >= 8000 else '1'))
    if R == 1:
        return quadric_cv_single(f, d, M, m, m0, b, H, rng, tag, Jpts, Jvals)
    if R == 0:
        return quadric_cv_rule(f, d, M, m, m0, b, H, rng, tag, Jpts, Jvals)
    return quadric_cv_select(f, d, M, m, m0, b, H, rng, tag, Jpts, Jvals, R)


def quadric_cv_single(f, d, M, m, m0, b, H, rng, tag="quadric", Jpts=None, Jvals=None):
    """One scrambled Sobol' set on the remaining budget. The jump model J^(z) = a + c.z (least squares on the jumps
    measured at the boundary points) is subtracted on the side of smaller Gaussian mass with the exact mean of the
    subtracted term; then, if enough points fall on that side, regression control variates (1{q<0} - p, z 1{q<0} - g)
    absorb what the jump model misses. Safeguard: a correction larger than 6 Monte Carlo standard errors of the plain
    average is rejected."""
    M, m, m0 = clean_quadric(M, m, m0)
    mom = checked_moments(M, m, m0, d, rng, tag) if time.monotonic() - T0 < 15.0 else None
    Rot = importance_rotation(b, H)
    n = f.remaining
    s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
    Z = ndtri(np.clip(s.random(n), 1e-15, 1 - 1e-15)) @ Rot.T
    Fz = f(Z)
    plain = float(Fz.mean())
    if mom is None:
        return plain
    p0, g0 = mom
    if Jvals is not None:
        Jvals = np.asarray(Jvals, float)
    if p0 > 0.5:     # work on the smaller side
        M, m, m0, p0, g0 = -M, -m, -m0, 1.0 - p0, -g0
        if Jvals is not None:
            Jvals = -Jvals
    h0 = (quadric.q_eval(Z, M, m, m0) < 0).astype(float)
    base = 0.0
    Rr = Fz.copy()
    if Jpts is not None and len(Jpts) >= d + 2:
        A1 = np.column_stack([np.ones(len(Jpts)), Jpts])
        coefJ, *_ = np.linalg.lstsq(A1, Jvals, rcond=None)
        Rr = Fz - h0 * (coefJ[0] + Z @ coefJ[1:])
        base = float(coefJ[0] * p0 + coefJ[1:] @ g0)
    est = base + float(Rr.mean())
    if n * p0 >= 200:
        X = np.column_stack([h0 - p0, Z * h0[:, None] - g0])
        Xc = X - X.mean(0)
        beta, *_ = np.linalg.lstsq(Xc, Rr - Rr.mean(), rcond=None)
        est = base + float(Rr.mean() - X.mean(0) @ beta)
    log(f"{tag} CV: n={n} plain={plain:.10g} cv={est:.10g}")
    if not np.isfinite(est) or abs(est - plain) > 6 * Fz.std() / math.sqrt(n) + 1e-12 * abs(plain):
        log(f"{tag}: implausible correction, plain average used")
        return plain
    return est


def quadric_cv_rule(f, d, M, m, m0, b, H, rng, tag="quadric", Jpts=None, Jvals=None):
    """One scrambled Sobol' set; the jump model J^ = a + c.z is used (inside the surface, fixed coefficients) when it
    fits the boundary jumps to 5%, otherwise regression control variates (1{q<0} - p, z 1{q<0} - g)."""
    M, m, m0 = clean_quadric(M, m, m0)
    mom = checked_moments(M, m, m0, d, rng, tag)
    Rot = importance_rotation(b, H)
    n = f.remaining
    s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
    Z = ndtri(np.clip(s.random(n), 1e-15, 1 - 1e-15)) @ Rot.T
    Fz = f(Z)
    plain = float(Fz.mean())
    if mom is None:
        return plain
    p0, g0 = mom
    h0 = (quadric.q_eval(Z, M, m, m0) < 0).astype(float)
    use_j = False
    if Jpts is not None and len(Jpts) >= d + 2:
        A1 = np.column_stack([np.ones(len(Jpts)), Jpts])
        coefJ, *_ = np.linalg.lstsq(A1, Jvals, rcond=None)
        rmsJ = float(np.sqrt(np.mean((A1 @ coefJ - Jvals) ** 2)))
        use_j = rmsJ < 0.05 * np.median(np.abs(Jvals)) or n * min(p0, 1 - p0) < 200
    if use_j:
        est = float(np.mean(Fz - h0 * (coefJ[0] + Z @ coefJ[1:]))) + float(coefJ[0] * p0 + coefJ[1:] @ g0)
        how = 'jump model'
    elif n * min(p0, 1 - p0) >= 200:
        X = np.column_stack([h0 - p0, Z * h0[:, None] - g0])
        Xc = X - X.mean(0)
        beta, *_ = np.linalg.lstsq(Xc, Fz - Fz.mean(), rcond=None)
        est = float(Fz.mean() - X.mean(0) @ beta)
        how = 'regression'
    else:
        est, how = plain, 'plain'
    log(f"{tag} CV ({how}): n={n} plain={plain:.10g} cv={est:.10g}")
    if not np.isfinite(est) or abs(est - plain) > 6 * Fz.std() / math.sqrt(n) + 1e-12 * abs(plain):
        log(f"{tag}: implausible correction, plain average used")
        return plain
    return est


def quadric_cv_select(f, d, M, m, m0, b, H, rng, tag="quadric", Jpts=None, Jvals=None, R=4):
    """Scrambled Sobol' on the remaining budget as R independent scrambles; several unbiased estimators built from the
    learned surface are evaluated on every scramble and the one with the smallest spread across the scrambles is
    returned (mean over the scrambles):
      plain      the plain average;
      Jin, Jout  f - h J^ with h the indicator of the inside {q<0} (resp. the outside) and J^(z) = a + c.z the jump
                 model fitted to the jumps measured at the boundary points; exact mean a p + c.g of the subtracted term;
      reg        regression control variates (1{q<0} - p, z 1{q<0} - g) with coefficients pooled over all scrambles;
      Jsmall+reg the jump model on the smaller side followed by the regression.
    The exact means p = P(q<0), g = E[Z 1{q<0}] come from generalised chi-square inversion (cross-checked by free Monte
    Carlo on the fitted surface)."""
    M, m, m0 = clean_quadric(M, m, m0)
    mom = checked_moments(M, m, m0, d, rng, tag) if time.monotonic() - T0 < 15.0 else None
    Rot = importance_rotation(b, H)
    n = f.remaining
    sizes = [n // R] * (R - 1) + [n - (n // R) * (R - 1)]
    Zs = []
    if os.environ.get('QCV_BLOCKS', '0') == '1':
        # one scrambled sequence; its R consecutive blocks only serve to compare the candidates
        s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
        Zs.append(ndtri(np.clip(s.random(n), 1e-15, 1 - 1e-15)) @ Rot.T)
    else:
        for nn in sizes:
            s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
            Zs.append(ndtri(np.clip(s.random(nn), 1e-15, 1 - 1e-15)) @ Rot.T)
    Z = np.vstack(Zs)
    Fz = f(Z)
    idx = np.repeat(np.arange(R), sizes)

    def rep_means(v):
        return np.array([v[idx == r].mean() for r in range(R)])
    cands = {'plain': rep_means(Fz)}
    if mom is not None:
        p0, g0 = mom
        hin = (quadric.q_eval(Z, M, m, m0) < 0).astype(float)
        if Jpts is not None and len(Jpts) >= d + 2:
            A1 = np.column_stack([np.ones(len(Jpts)), Jpts])
            coefJ, *_ = np.linalg.lstsq(A1, Jvals, rcond=None)
            Jz = coefJ[0] + Z @ coefJ[1:]
            EJin = float(coefJ[0] * p0 + coefJ[1:] @ g0)
            EJout = float(coefJ[0] * (1 - p0) - coefJ[1:] @ g0)
            cands['Jin'] = rep_means(Fz - hin * Jz) + EJin
            cands['Jout'] = rep_means(Fz + (1 - hin) * Jz) - EJout
        if n * min(p0, 1 - p0) >= 200:
            X = np.column_stack([hin - p0, Z * hin[:, None] - g0])
            Xc = X - X.mean(0)
            beta, *_ = np.linalg.lstsq(Xc, Fz - Fz.mean(), rcond=None)
            cands['reg'] = rep_means(Fz - X @ beta)
            if 'Jin' in cands:
                small_in = p0 <= 0.5
                Rres = Fz - hin * Jz if small_in else Fz + (1 - hin) * Jz
                base = EJin if small_in else -EJout
                beta2, *_ = np.linalg.lstsq(Xc, Rres - Rres.mean(), rcond=None)
                cands['Jsmall+reg'] = rep_means(Rres - X @ beta2) + base
    pref = {'plain': 1.0, 'Jin': 1.05, 'Jout': 1.05, 'reg': 1.1, 'Jsmall+reg': 1.15}
    score = {k: (np.std(v, ddof=1) if R > 1 else 0.0) * pref[k] for k, v in cands.items()}
    best = min(score, key=score.get)
    w = np.array(sizes, float) / n
    est = float(w @ cands[best])          # = the average over all points
    plain = float(w @ cands['plain'])
    se_plain = float(np.std(cands['plain'], ddof=1) / math.sqrt(R))
    log(f"{tag} CV: n={n} R={R} " + " ".join(f"{k}={cands[k].mean():.10g}(sd {np.std(cands[k], ddof=1):.2g})"
                                             for k in cands) + f" -> {best}")
    if not np.isfinite(est) or abs(est - plain) > 8 * se_plain + 6 * Fz.std() / math.sqrt(n):
        log(f"{tag}: implausible correction, plain average used")
        return plain
    return est


def robust_quadric_fit(P, npar, tag="quadric", tol=3e-5):
    for it in range(4):
        if len(P) < npar + 4:
            return None
        M, m, m0, gap = quadric.fit_quadric(P)
        resid = np.abs(quadric.q_eval(P, M, m, m0)) / np.maximum(np.linalg.norm(2 * P @ M + m, axis=1), 1e-300)
        log(f"{tag} fit {it}: n={len(P)} gap={gap:.3g} median/max geometric residual={np.median(resid):.3g}/"
            f"{resid.max():.3g}")
        if resid.max() < tol:
            return M, m, m0
        P = P[resid < max(tol, 3 * np.median(resid))]
    return None


def run_quadric_rays(f, d, c, jump_typ, b, H, rng, scale, R=7.0, steps=None, reserve=1500):
    """Jump across a single convex quadric: boundary points by pure bisection along rays from an interior point c
    (one crossing per ray), exact quadric fit, control variates."""
    npar = d * (d + 1) // 2 + d + 1
    small = d <= 16
    steps = (22 if small else 16) if steps is None else steps
    need = int((1.25 if small else 1.08) * npar) + 12
    fc = float(f(c[None, :])[0])
    if need * (steps + 1) * 1.05 + reserve > f.remaining:
        log("rays: not enough budget")
        return None
    nr = int(min(f.remaining - reserve, need * 1.12 * (steps + 1)) // (steps + 1))
    Th = rng.standard_normal((nr, d)); Th /= np.linalg.norm(Th, axis=1, keepdims=True)
    fo = f(c + R * Th)
    jump_min = 0.3 * jump_typ
    ok = np.abs(fo - fc) > jump_min
    log(f"rays: {nr} from interior point (f={fc:.4g}); crossing within R: {ok.mean():.2f}")
    if ok.sum() < need:
        return None
    Th = Th[ok]; fo = fo[ok]
    a = np.zeros(len(Th)); bb = np.full(len(Th), R)
    fa = np.full(len(Th), fc); fb = fo.copy()
    for _ in range(steps):
        mid = 0.5 * (a + bb)
        fm = f(c + mid[:, None] * Th)
        left = np.abs(fm - fa) <= np.abs(fm - fb)
        a = np.where(left, mid, a); fa = np.where(left, fm, fa)
        bb = np.where(left, bb, mid); fb = np.where(left, fb, fm)
    keep = np.abs(fb - fa) > jump_min
    P = c + (0.5 * (a + bb))[keep, None] * Th[keep]
    Jv = (fa - fb)[keep]                      # inside (towards c) minus outside
    fit = robust_quadric_fit(P, npar, tag="ray quadric", tol=2.5 * R / 2 ** steps)
    if fit is None:
        return None
    M, m, m0 = fit
    if quadric.q_eval(c[None, :], M, m, m0)[0] > 0:
        M, m, m0 = -M, -m, -m0
    return quadric_cv_estimate(f, d, M, m, m0, b, H, rng, tag="ray quadric", Jpts=P, Jvals=Jv)


def hinge_cv_crossfit(f, d, Z0, F0, b, H, rng, mmax=16):
    """Cross-fitted hinge control variates: GPCA rotation; the remaining budget as two independently scrambled
    Sobol' sets A and B; the hinge model f ~ P2(x) + max(q(x), 0) (x = U^T z, U the top mmax GPCA directions) is fitted
    on A and used as control variate on B and vice versa (exact means of max(q,0), z max(q,0) and the quadratic
    monomials of x); each half is unbiased because its control variate does not depend on it."""
    kg = min(16, max(2, (f.remaining // 8) // (d + 1)))
    G, C = fd_gradients(f, Z0, F0, kg)
    Rot = gpca_rotation(C, H)
    U = Rot[:, :min(d, mmax)]
    m_ = U.shape[1]
    n = f.remaining
    n1 = n // 2
    Zs, Fs = [], []
    for nn in (n1, n - n1):
        s = qmc.Sobol(d, scramble=True, seed=rng.integers(1 << 31))
        Zs.append(ndtri(np.clip(s.random(nn), 1e-15, 1 - 1e-15)) @ Rot.T)
    Fall = f(np.vstack(Zs))
    Fs = [Fall[:n1], Fall[n1:]]
    plain = float(Fall.mean())
    e_plain = abs(Fs[0].mean() - Fs[1].mean()) / 2
    iu = np.triu_indices(m_)
    EPhi = np.concatenate([(iu[0] == iu[1]).astype(float), np.zeros(m_)])
    ests = []
    for a, bq in ((0, 1), (1, 0)):
        if time.monotonic() - T0 > 15.0:
            log("crossfit hinge: time guard -> plain average")
            return plain
        rms, cq = hinge_fit(Zs[a] @ U, Fs[a])
        if not rms < 0.3 * Fs[a].std():
            log(f"crossfit hinge: no fit (rms {rms:.3g} vs sd {Fs[a].std():.3g}) -> plain average")
            return plain
        Mx, mx, m0 = quadric.coef_to_quadric(cq, m_)
        M = U @ Mx @ U.T
        mm = U @ mx
        e0, g0 = quadric.pos_part_moments(M, mm, m0)
        Z, F = Zs[bq], Fs[bq]
        h1 = np.maximum(quadric.q_eval(Z, M, mm, m0), 0)
        Phi = quadric.quad_features(Z @ U)[:, :-1]
        X = np.column_stack([h1 - e0, Z * h1[:, None] - g0, Phi - EPhi])
        # coefficients from the fitting half (independent of the evaluation half)
        Za, Fa = Zs[a], Fs[a]
        h1a = np.maximum(quadric.q_eval(Za, M, mm, m0), 0)
        Xa = np.column_stack([h1a - e0, Za * h1a[:, None] - g0, quadric.quad_features(Za @ U)[:, :-1] - EPhi])
        Xac = Xa - Xa.mean(0)
        beta, *_ = np.linalg.lstsq(Xac, Fa - Fa.mean(), rcond=None)
        ests.append(float(F.mean() - X.mean(0) @ beta))
        log(f"crossfit hinge half {bq}: plain={F.mean():.10g} cv={ests[-1]:.10g} fit rms={rms:.3g}")
    est = 0.5 * (ests[0] + ests[1])
    if not np.isfinite(est) or abs(est - plain) > 8 * e_plain + 6 * Fall.std() / math.sqrt(n):
        log("crossfit hinge: implausible correction -> plain average")
        return plain
    return est


def run_quadric_cv(f, d, Z0, F0, b, H, rng, scale, max_frac=0.6):
    """Jump across a single quadric surface: locate boundary points along lines, fit the quadric exactly, then
    scrambled Sobol' with regression control variates h0 = 1{q < 0}, h_i = z_i 1{q < 0} whose expectations are
    computed exactly (generalised chi-square inversion). Returns None when the structure is not detected."""
    npar = d * (d + 1) // 2 + d + 1
    need = int(1.15 * npar) + 10
    jump_min = 0.02 * max(F0.std(), 1e-12) + 1e-9 * scale
    grid = np.linspace(-5.0, 5.0, 21)
    steps = 17
    if need * (len(grid) / 2 + steps) > max_frac * f.budget:
        log(f"quadric: {npar} parameters is too many for the budget")
        return 'toobig', None
    dev = np.abs(F0 - np.median(F0))
    order = np.argsort(-dev)
    # rare extreme values separated from the bulk (a rare region of a jump family): keep looking for jumps
    heavy = dev.max() > 20 * (np.quantile(dev, 0.98) + 1e-12 * scale)
    tried_rays = False
    P_all, J_all, V_all = [], [], []
    used0 = f.used
    nl = 0
    batch = 10
    inner = [Z0[i] for i in order[:8]]          # candidate interior points (largest deviation)
    rate_rand, rate_in = 1.0, 1.0
    per_line = len(grid) + 2 * steps
    while True:
        k = batch
        # origins: interior candidates (chord midpoints found so far, pilot extremes) and fresh normal points;
        # when random lines rarely cross (rare region), almost all lines go through interior candidates
        n_in = k // 2 if rate_rand >= 0.5 * rate_in else (3 * k) // 4
        Y = rng.standard_normal((k, d))
        if inner:
            pick = rng.integers(0, len(inner), n_in)
            Y[:n_in] = np.array(inner)[pick]
        else:
            n_in = 0
        V = rng.standard_normal((k, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        Y -= (Y * V).sum(1, keepdims=True) * V
        if f.used - used0 + k * per_line * 1.2 > max_frac * f.budget:
            break
        before = f.used
        P, Jv, li, _ = find_jumps(f, Y, V, grid, scale, steps=steps)
        per_line = max(per_line * 0.5, (f.used - before) / k)
        nl += k
        keep = np.abs(Jv) > jump_min            # a genuine jump survives the bisection
        P_all.append(P[keep]); J_all.append(Jv[keep]); V_all.append(V[li[keep]])
        lk = li[keep]
        cnt = np.bincount(lk, minlength=k)
        if n_in > 0:
            rate_in = 0.7 * rate_in + 0.3 * cnt[:n_in].mean()
        if n_in < k:
            rate_rand = 0.7 * rate_rand + 0.3 * cnt[n_in:].mean()
        for l in np.unique(lk):
            pts = P[keep][lk == l]
            if len(pts) >= 2:
                inner.append(0.5 * (pts[0] + pts[1]))
        npts = sum(len(p) for p in P_all)
        log(f"quadric search: lines={nl} boundary points={npts} (need {need}) used={f.used} "
            f"rates in/rand={rate_in:.2f}/{rate_rand:.2f}")
        if nl == 10 and npts < 3 and not heavy:
            return 'nojump', None      # no jumps: not a jump family
        if npts >= 3 and not tried_rays and os.environ.get('NORAYS') != '1':
            # a chord midpoint is an interior point of a convex region: rays from it cross exactly once
            Pc = np.vstack(P_all); Jc = np.concatenate(J_all)
            mid = None
            for P_, J_, li_ in [(P[keep], Jv[keep], li[keep])]:
                for l in np.unique(li_):
                    pts = P_[li_ == l]
                    if len(pts) == 2:
                        mid = 0.5 * (pts[0] + pts[1])
                        break
            if mid is not None:
                tried_rays = True
                r = run_quadric_rays(f, d, mid, float(np.median(np.abs(Jc))), b, H, rng, scale)
                if r is not None:
                    return 'ok', r
        if nl >= 40 and npts < 3:
            return 'nojump', None
        if npts >= need:
            break
        rate = max(npts / nl, 0.05)
        batch = int(min(96, max(8, (need - npts) / rate * 1.1)))
    P = np.vstack(P_all) if P_all else np.zeros((0, d))
    if len(P) < need:
        log("quadric: not enough boundary points")
        return 'fail', None
    Jall = np.concatenate(J_all)
    Vall = np.vstack(V_all)
    strong = np.abs(Jall) > 0.2 * np.median(np.abs(Jall))
    P, Jall, Vall = P[strong], Jall[strong], Vall[strong]
    fit = robust_quadric_fit(P, npar, tag="quadric")
    if fit is None:
        return 'fail', None
    M, m, m0 = fit
    mids = inner[8:]
    if mids and np.median(quadric.q_eval(np.array(mids), M, m, m0)) > 0:
        M, m, m0 = -M, -m, -m0          # orient: chord midpoints (interior of the closed surface) have q < 0
    # inside = {q < 0}; a crossing in direction v goes from inside to outside iff grad q . v > 0
    gv = np.einsum('ij,ij->i', 2 * P @ M + m, Vall)
    Jin = np.where(gv > 0, -Jall, Jall)
    resid = np.abs(quadric.q_eval(P, M, m, m0)) / np.maximum(np.linalg.norm(2 * P @ M + m, axis=1), 1e-300)
    good = resid < 3e-5
    est = quadric_cv_estimate(f, d, M, m, m0, b, H, rng, tag="quadric", Jpts=P[good], Jvals=Jin[good])
    return 'ok', est


def update_direction(v, B, Y, r, integ, d, bpts):
    """New preintegration direction. Crossing points of lines with a single crossing are accumulated in bpts (all
    stages); with at least d + 8 of them the crossing surface is fitted by total least squares (normal = eigenvector
    of the smallest eigenvalue of their covariance; exact for hyperplanes). With fewer, the crossing location s is
    regressed on u = B^T y (ridge) and the normal v - B beta is used; without crossings, the line-based Stein
    estimate b = mean_j [y_j (L_j - mean L) + v M_j]."""
    one = np.array([len(c) == 1 for c in integ.cuts])
    if isinstance(integ, WindowIntegrator):
        one = np.array([len(c) >= 1 and c[0] > -7 for c in integ.cuts])
    for j in np.nonzero(one)[0]:
        bpts.append(Y[j] + integ.cuts[j][0] * v)
    if len(bpts) >= d + 8 and one.mean() > 0.6:
        P = np.array(bpts)
        C = np.cov(P.T)
        ev, U = np.linalg.eigh(C)
        n = U[:, 0]
        how = f"TLS on {len(bpts)} crossing points, ev0={ev[0]:.3g} ev1={ev[1]:.3g}"
    elif one.sum() >= min(d - 1, 24) and one.mean() > 0.6:
        s = np.array([c[0] for c, o in zip(integ.cuts, one) if o])
        U = Y[one] @ B
        Uc = U - U.mean(0)
        sc = s - s.mean()
        lam = 1e-3 * np.trace(Uc.T @ Uc) / U.shape[1] + 1e-12
        beta = np.linalg.solve(Uc.T @ Uc + lam * np.eye(U.shape[1]), Uc.T @ sc)
        n = v - B @ beta
        how = f"crossing fit on {one.sum()} lines"
    else:
        n = (Y * (r - r.mean())[:, None]).mean(0) + v * integ.moment.mean()
        how = "line Stein"
    nn = np.linalg.norm(n)
    if not np.isfinite(nn) or nn == 0:
        return v
    n = n / nn
    if n @ v < 0:
        n = -n
    log(f"  direction update ({how}): cos(old,new)={n @ v:.6f}")
    return n


def run_parallel(f, d, v, H, scale, rng, stages=None, grid=None, eps_rel=1e-8, window=False, wgrid=None):
    """Preintegration along v with outer scrambled Sobol' over v-perp. The first stages refine v by the
    line-based Stein estimate b = E[Z f(Z)] = mean_j [y_j (L_j - mean L) + v M_j] (L_j, M_j: line integrals of f and
    s f); every stage's mean is kept and the stages are combined with weights n_s / var_s."""
    grid = np.linspace(-6.5, 6.5, 41) if grid is None else grid
    stages = (max(32, d // 2 + 16), d + 16) if stages is None else stages
    wgrid = np.linspace(-6.0, 6.0, 25) if wgrid is None else wgrid
    results = []
    bpts = []
    per_line = None
    st = 0
    while True:
        B = complement_basis(v, H)
        sob = qmc.Sobol(d - 1, scramble=True, seed=rng.integers(1 << 31))
        rem = f.remaining
        if st < len(stages):
            L = stages[st]
        else:
            L = int(rem / (per_line * 1.1)) if per_line else 64
        if per_line is not None:
            L = min(L, int(rem / (per_line * 1.1)))
        if L < 8:
            break
        U = sob.random(L)
        Y = ndtri(np.clip(U, 1e-15, 1 - 1e-15)) @ B.T

        def evaluate(li, s, Y=Y, v=v):
            return f(Y[li] + np.asarray(s)[:, None] * v[None, :])

        before = f.used
        try:
            if window:
                integ = WindowIntegrator(evaluate, grid=wgrid)
                r = integ.run(L, scale)
                irr = integ.irregular
                if irr:
                    sub = np.array(irr)
                    g2 = LineIntegrator(lambda li, s, sub=sub: evaluate(sub[li], s), grid=grid,
                                        eps_line=eps_rel * scale, flag_factor=2.0)
                    r2 = g2.run(len(sub), scale)
                    r[sub] = r2
                    integ.moment[sub] = g2.moment
                    for q, l in enumerate(sub):
                        integ.cuts[l] = g2.cuts[q]
            else:
                integ = LineIntegrator(evaluate, grid=grid, eps_line=eps_rel * scale, flag_factor=2.0)
                r = integ.run(L, scale)
        except RuntimeError:
            break
        cost = (f.used - before) / L
        per_line = cost if per_line is None else max(per_line, cost)
        var = r.var(ddof=1) if L > 1 else np.inf
        results.append((L, r.mean(), var))
        log(f"stage {st} L={L} cost/line={cost:.1f} cuts/line={integ.ncuts / L:.2f} mean={r.mean():.10g} "
            f"irregular={len(integ.irregular) if window else 0} "
            f"sd={np.sqrt(var):.3g} used={f.used}")
        if st < len(stages):
            v = update_direction(v, B, Y, r, integ, d, bpts)
        st += 1
        if f.remaining < 64:
            break
    w = np.array([n / max(vv, 1e-300) for n, m, vv in results])
    mu = np.array([m for n, m, vv in results])
    est = float((w * mu).sum() / w.sum())
    return est, sum(n for n, m, vv in results)


if __name__ == '__main__':
    main()
