"""Long-run covariance estimator for a single MCMC / time-series trajectory.

Method (see README.md):
  1. centre and whiten the trajectory with its sample covariance (affine equivariance,
     numerical stability);
  2. for a few block-aggregation levels m, fit a VAR(p) to the m-block means
     (order p by AIC) and draw from the Jeffreys-prior posterior of the VAR
     parameters, mapping each stationary draw to its long-run covariance;
  3. return the Bayes estimator under the whitened Frobenius loss
     ||S^-1/2 (H - S) S^-1/2||_F over the pooled draws;
  4. map back to the input coordinates.
"""
import os
import sys

import numpy as np


def lagmat(y, p):
    n, d = y.shape
    Z = np.empty((n - p, p * d))
    for j in range(1, p + 1):
        Z[:, (j - 1) * d: j * d] = y[p - j: n - j]
    return Z


def select_order(y, pmax):
    """AIC order selection for a VAR on the common sample t = pmax..n-1."""
    n, d = y.shape
    if pmax <= 0:
        return 0
    ics = order_ics(y, pmax)
    p2 = int(np.argmin(ics[2.0]))
    if P_RULE == "pen" or p2 < P_SWITCH:
        return int(np.argmin(ics[AIC_PEN])) if P_RULE == "pen" else p2
    if P_RULE == "low":
        return max(p2, int(np.argmin(ics[LOW_PEN])))
    if P_RULE == "mult":
        return int(min(pmax, np.ceil(P_MULT * p2)))
    return p2


def order_ics(y, pmax):
    n, d = y.shape
    Z = lagmat(y, pmax)
    Yt = y[pmax:]
    N = n - pmax
    G = Z.T @ Z
    c = Z.T @ Yt
    YY = Yt.T @ Yt
    pens = sorted(set([2.0, AIC_PEN, LOW_PEN]))
    lds = []
    for p in range(pmax + 1):
        k = p * d
        if p == 0:
            Su = YY / N
        else:
            try:
                B = np.linalg.solve(G[:k, :k], c[:k])
            except np.linalg.LinAlgError:
                break
            Su = (YY - c[:k].T @ B) / N
        Su = (Su + Su.T) / 2
        sign, ld = np.linalg.slogdet(Su)
        if sign <= 0 or not np.isfinite(ld):
            break
        lds.append(ld)
    lds = np.array(lds)
    ps = np.arange(len(lds))
    return {pen: lds + pen * ps * d * d / N for pen in pens}


def var1_bias(A, Su, G0, N):
    """Pope (1990) first-order bias of mean-adjusted OLS VAR(1); returns E[A_hat - A]
    evaluated at A (shrunk so that A - bias stays stationary)."""
    d = A.shape[0]
    I = np.eye(d)
    At = A.T
    lam = np.linalg.eigvals(A)
    S = np.linalg.inv(I - At) + At @ np.linalg.inv(I - At @ At)
    for l in lam:
        S = S + l * np.linalg.inv(I - l * At)
    bias = -(Su @ np.real(S) @ np.linalg.inv(G0)) / N
    # keep the corrected matrix stationary
    for _ in range(20):
        r = np.max(np.abs(np.linalg.eigvals(A - bias)))
        if r < 0.999:
            break
        bias = bias * 0.8
    else:
        bias = np.zeros_like(bias)
    return bias


def var_posterior_draws(y, p, ndraw, rng, rmax=0.9995):
    """Long-run covariances of draws from the Jeffreys posterior of a VAR(p) fitted to y."""
    n, d = y.shape
    k = p * d
    if k > 0:
        Z = lagmat(y, p)
        Yt = y[p:]
        G = Z.T @ Z
        Bh = np.linalg.solve(G, Z.T @ Yt)
        U = Yt - Z @ Bh
        Gi = np.linalg.inv(G)
        LG = np.linalg.cholesky((Gi + Gi.T) / 2)
    else:
        Yt = y
        Bh = np.zeros((0, d))
        U = Yt
    SSu = U.T @ U
    df = len(Yt) - k
    if p == 1 and BIAS_CORR > 0:
        Bh = Bh - BIAS_CORR * var1_bias(Bh.T, SSu / df, Z.T @ Z / len(Yt), len(Yt)).T
    SSi = np.linalg.inv(SSu)
    Linv = np.linalg.cholesky((SSi + SSi.T) / 2)
    out = []
    got = 0
    for _ in range(4):
        m = ndraw
        A = np.zeros((m, d, d))
        ii = np.arange(d)
        A[:, ii, ii] = np.sqrt(rng.chisquare(df - ii, size=(m, d)))
        tri = np.tril_indices(d, -1)
        A[:, tri[0], tri[1]] = rng.standard_normal((m, len(tri[0])))
        LW = Linv @ A
        Wd = LW @ np.transpose(LW, (0, 2, 1))
        Sud = np.linalg.inv(Wd)
        Sud = (Sud + np.transpose(Sud, (0, 2, 1))) / 2
        if k > 0:
            LS = np.linalg.cholesky(Sud)
            E = rng.standard_normal((m, k, d))
            Bd = Bh + np.einsum("ij,sjk,slk->sil", LG, E, LS)
            if p == 1:
                Cm = np.transpose(Bd, (0, 2, 1))
            else:
                Cm = np.zeros((m, k, k))
                Cm[:, :d, :] = np.transpose(Bd, (0, 2, 1))
                Cm[:, d:, :-d] = np.eye(k - d)
            r = np.max(np.abs(np.linalg.eigvals(Cm)), axis=1)
            ok = r < rmax
            if not ok.any():
                continue
            Asum = np.transpose(Bd[ok].reshape(-1, p, d, d).sum(1), (0, 2, 1))
            M = np.linalg.inv(np.eye(d) - Asum)
            Sd = M @ Sud[ok] @ np.transpose(M, (0, 2, 1))
        else:
            Sd = Sud
        out.append(Sd)
        got += len(Sd)
        if got >= ndraw:
            break
    if not out:
        return None
    Ss = np.concatenate(out)[:ndraw]
    return (Ss + np.transpose(Ss, (0, 2, 1))) / 2


def bayes_combine(Ss, weights=None, iters=20):
    """argmin_H sum_s w_s ||S_s^-1/2 H S_s^-1/2 - I||_F via IRLS."""
    d = Ss.shape[1]
    P = np.linalg.inv(Ss)
    P = (P + np.transpose(P, (0, 2, 1))) / 2
    base = np.ones(len(Ss)) if weights is None else weights
    w = base.copy()
    H = None
    for _ in range(iters):
        K = np.einsum("s,sij,skl->ikjl", w, P, P).reshape(d * d, d * d)
        rhs = np.einsum("s,sij->ij", w, P).reshape(-1)
        Hn = np.linalg.solve(K, rhs).reshape(d, d)
        Hn = (Hn + Hn.T) / 2
        if H is not None and np.abs(Hn - H).max() < 1e-9 * max(1.0, np.abs(H).max()):
            H = Hn
            break
        H = Hn
        PH = np.einsum("sij,jk->sik", P, H)
        l2 = np.einsum("sij,sji->s", PH, PH) - 2 * np.einsum("sii->s", PH) + d
        w = base / np.sqrt(np.maximum(l2, 1e-12))
    return H


def block_means(y, m):
    n, d = y.shape
    a = n // m
    return y[: a * m].reshape(a, m, d).mean(1)


def post_risk(Ss, H):
    """posterior mean of ||S_s^-1/2 H S_s^-1/2 - I||_F"""
    d = H.shape[0]
    P = np.linalg.inv(Ss)
    PH = np.einsum("sij,jk->sik", P, H)
    l2 = np.einsum("sij,sji->s", PH, PH) - 2 * np.einsum("sii->s", PH) + d
    return float(np.mean(np.sqrt(np.maximum(l2, 0.0))))


def wdist(Ha, Hb):
    """||Hb^-1/2 (Ha - Hb) Hb^-1/2||_F"""
    w, V = np.linalg.eigh(Hb)
    w = np.maximum(w, 1e-300)
    R = V / np.sqrt(w)
    D = R.T @ (Ha - Hb) @ R
    return float(np.sqrt((D * D).sum()))


AIC_PEN = 2.0
USE_FS = True
BIAS_CORR = 0.5
P_RULE = "pen"
P_SWITCH = 3
LOW_PEN = 1.0
P_MULT = 1.5
LEVELS = (1, 2, 4, 8, 16)
MODE = "single"
KAPPA = 1.5
MIN_BLOCKS = 300


def estimate_whitened(z, rng, ndraw=1000, pmax_cap=40):
    """z: centred, whitened trajectory. Returns H for z."""
    n, d = z.shape
    res = []
    for m in LEVELS:
        if m > 1 and n // m < MIN_BLOCKS:
            break
        za = block_means(z, m) if m > 1 else z
        na = len(za)
        za = za - za.mean(0)
        pmax = int(min(pmax_cap, max(1, na // (10 * d))))
        p = select_order(za, pmax)
        k = p * d
        nd = ndraw if k <= 60 else max(150, int(ndraw * (60.0 / k) ** 2))
        Ss = var_posterior_draws(za, p, nd, rng)
        if Ss is None:
            continue
        Ss = m * Ss
        H = bayes_combine(Ss)
        res.append((m, Ss, H, post_risk(Ss, H)))
        if MODE == "single":
            break
    if MODE == "pool":
        Ss = np.concatenate([r[1] for r in res])
        w = np.concatenate([np.full(len(r[1]), 1.0 / len(r[1])) for r in res])
        return bayes_combine(Ss, w)
    if MODE == "lepski":
        for i, (m, Ss, H, R) in enumerate(res):
            ok = True
            for (m2, Ss2, H2, R2) in res[i + 1:]:
                if wdist(H, H2) > KAPPA * R2:
                    ok = False
                    break
            if ok:
                return H
        return res[-1][2]
    return res[0][2]


FS_MAX_STATES = 200
FS_MIN_PER_STATE = 20
FS_ALPHA = 0.0
FS_MAX_CTX = 300
FS_NBOOT = 20
FS_ZCRIT = 4.0
FS_MAX_ORDER = 1
FS_LAST_R = 0


def markov_ll(sym, K, r, t0=1):
    """log-likelihood and free-parameter count of an order-r chain fitted to sym[t0+1:]."""
    n = len(sym)
    code = np.zeros(n, dtype=np.int64)
    for i in range(r):
        code[r - 1:] = code[r - 1:] * K + sym[r - 1 - i: n - i]
    ctx = code[t0: n - 1]
    nxt = sym[t0 + 1:]
    key = ctx * K + nxt
    uk, cnt = np.unique(key, return_counts=True)
    uc, cinv = np.unique(uk // K, return_inverse=True)
    tot = np.bincount(np.asarray(cinv).reshape(-1), weights=cnt)
    ll = float(np.sum(cnt * np.log(cnt / tot[np.asarray(cinv).reshape(-1)])))
    return ll, len(uk) - len(uc)


def markov_gstat(sym, K):
    l1, _ = markov_ll(sym, K, 1)
    l2, _ = markov_ll(sym, K, 2)
    return 2.0 * (l2 - l1)


def markov_order1_ok(sym, K, rng, nboot=FS_NBOOT):
    """Parametric-bootstrap G test of first-order Markov vs second order."""
    n = len(sym)
    N = np.zeros((K, K))
    np.add.at(N, (sym[:-1], sym[1:]), 1.0)
    N[sym[-1], sym[0]] += 1.0
    P = N / N.sum(1, keepdims=True)
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 2.0
    g_obs = markov_gstat(sym, K)
    st = np.full(nboot, sym[0])
    seq = np.empty((n, nboot), dtype=np.int64)
    U = rng.random((n, nboot))
    for t in range(n):
        seq[t] = st
        st = (U[t][:, None] > cum[st]).sum(1)
    gb = np.array([markov_gstat(seq[:, b], K) for b in range(nboot)])
    mu, sd = gb.mean(), gb.std(ddof=1)
    return g_obs <= mu + FS_ZCRIT * max(sd, 1e-9) + 1e-9 * abs(mu)


def finite_state_estimate(z, rng, ndraw=1000):
    """If the trajectory visits only a few distinct points and passes a first-order Markov
    check, treat it as a finite-state Markov chain: Dirichlet posterior on the observed
    transition support, long-run covariance of each draw via the fundamental matrix, Bayes
    estimate under the whitened Frobenius loss."""
    n, d = z.shape
    # group rows that are equal up to round-off (z is whitened, so O(1) scale)
    zr = np.round(z * 1e6)
    _, first, inv = np.unique(zr, axis=0, return_index=True, return_inverse=True)
    uniq = z[first]
    sym = np.asarray(inv).reshape(-1).astype(np.int64)
    K = len(uniq)
    if K < 2 or K > FS_MAX_STATES or n < FS_MIN_PER_STATE * K:
        return None
    # a finite chain re-enters its states many times; a continuous chain with rejections
    # (repeated rows) enters each value once
    entries = np.bincount(sym[1:][sym[1:] != sym[:-1]], minlength=K)
    entries[sym[0]] += 1
    if entries.mean() < 10 or np.mean(entries >= 2) < 0.9:
        return None
    global FS_LAST_R
    FS_LAST_R = 1
    if not markov_order1_ok(sym, K, rng):
        FS_LAST_R = 2
        return None
    r = 1
    code = np.zeros(n, dtype=np.int64)
    for i in range(r):
        code[r - 1:] = code[r - 1:] * K + sym[r - 1 - i: n - i]
    cs = code[r - 1:]
    last = sym[r - 1:]
    uctx, cidx = np.unique(cs, return_inverse=True)
    cidx = np.asarray(cidx).reshape(-1)
    C = len(uctx)
    lastsym = np.zeros(C, dtype=np.int64)
    lastsym[cidx] = last
    N = np.zeros((C, C))
    np.add.at(N, (cidx[:-1], cidx[1:]), 1.0)
    N[cidx[-1], cidx[0]] += 1.0        # close the walk so every context has an exit
    sup = N > 0
    F = uniq[lastsym]
    nd = ndraw if C <= 60 else max(200, int(ndraw * (60.0 / C) ** 2))
    out = []
    ones = np.ones(C)
    shape = np.where(sup, N + FS_ALPHA, 1.0)
    for _ in range(nd):
        G = np.where(sup, rng.standard_gamma(shape), 0.0)
        rs = G.sum(1, keepdims=True)
        if np.any(rs <= 0):
            continue
        P = G / rs
        A = np.eye(C) - P.T
        A[-1] = 1.0
        b = np.zeros(C); b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            Zm = np.linalg.inv(np.eye(C) - P + np.outer(ones, pi))
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(pi)) or pi.min() < -1e-9:
            continue
        pi = np.maximum(pi, 0.0); pi /= pi.sum()
        Fc = F - pi @ F
        Mz = pi[:, None] * Zm
        Mm = Mz + Mz.T - np.diag(pi)
        Sg = Fc.T @ Mm @ Fc
        out.append((Sg + Sg.T) / 2)
    if len(out) < nd // 2:
        return None
    Ss = np.array(out)
    ev = np.linalg.eigvalsh(Ss)
    Ss = Ss[ev[:, 0] > 1e-10 * np.maximum(ev[:, -1], 1e-300)]
    if len(Ss) < nd // 2:
        return None
    return bayes_combine(Ss)


def estimate(x, seed=1):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    rng = np.random.default_rng(seed)
    mu = x.mean(0)
    y = x - mu
    C = (y.T @ y) / n
    C = (C + C.T) / 2
    ev, V = np.linalg.eigh(C)
    tol = ev.max() * 1e-12 if ev.max() > 0 else 0.0
    keep = ev > tol
    if not keep.any():
        return np.zeros((d, d))
    T = V[:, keep] / np.sqrt(ev[keep])          # whitening map (d x r)
    Tinv = V[:, keep] * np.sqrt(ev[keep])       # back map (d x r)
    z = y @ T
    Hz = finite_state_estimate(z, rng) if USE_FS else None
    if Hz is None:
        Hz = estimate_whitened(z, rng)
    H = Tinv @ Hz @ Tinv.T
    return H


def finalize(H, d):
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (d, d) or not np.isfinite(H).all():
        raise ValueError("bad estimate")
    H = (H + H.T) / 2
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, 0.0)
    H = (V * w) @ V.T
    H = (H + H.T) / 2
    return np.ascontiguousarray(H, dtype="<f8")


def fallback(x):
    """Plain batch means (always PSD) if anything goes wrong."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    b = max(1, int(np.sqrt(n)))
    a = n // b
    if a < 2:
        y = x - x.mean(0)
        return y.T @ y / max(1, n - 1)
    Y = x[: a * b].reshape(a, b, d).mean(1) - x.mean(0)
    return b * Y.T @ Y / (a - 1)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    x = np.load(inp, allow_pickle=False)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    d = x.shape[1]
    try:
        seed = int(os.environ.get("SEED", "1") or 1)
    except ValueError:
        seed = 1
    try:
        if not np.isfinite(x).all():
            bad = ~np.isfinite(x)
            colmean = np.nanmean(np.where(bad, np.nan, x), axis=0)
            colmean = np.where(np.isfinite(colmean), colmean, 0.0)
            x = np.where(bad, colmean, x)
        H = finalize(estimate(x, seed), d)
    except Exception:
        try:
            H = finalize(fallback(x), d)
        except Exception:
            H = np.zeros((d, d))
    with open(outp, "wb") as f:
        np.save(f, np.ascontiguousarray(H, dtype="<f8"), allow_pickle=False)


if __name__ == "__main__":
    main()
