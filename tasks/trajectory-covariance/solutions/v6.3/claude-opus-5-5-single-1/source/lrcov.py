"""Long-run covariance estimator for a single (MCMC) trajectory.

Pipeline (all steps are affine equivariant, so a fixed invertible linear map
and translation of the input coordinates is handled exactly):
  1. centre and whiten the trajectory with its sample covariance;
  2. if the trajectory only takes a few distinct values that are revisited
     (finite-state chain), use the plug-in Markov-chain estimator
     Sigma = F'(D Z + Z' D - D)F with the fundamental matrix Z of the
     empirical transition matrix;
  3. otherwise fit VAR(p) models by least squares, select p by AIC, and use
     the VAR long-run covariance A(1)^{-1} Sigma_u A(1)^{-T};
  4. draw parameters from the (Jeffreys-prior / Dirichlet) posterior of the
     fitted model, compute the long-run covariance of every draw, and return
     the matrix minimising the mean whitened Frobenius error over the draws
     (IRLS); this shrinks noisy, right-skewed estimates in a calibrated way;
  5. map back to the input coordinates and symmetrise / clip to PSD.
"""
import os

import numpy as np


def _seed():
    try:
        return int(os.environ.get("SEED", "1"))
    except ValueError:
        return 1


def _sym_psd(H):
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, 0.0)
    H = (V * w) @ V.T
    return 0.5 * (H + H.T)


def whiten(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    n = X.shape[0]
    C0 = Xc.T @ Xc / n
    w, V = np.linalg.eigh(C0)
    keep = w > max(w.max(), 1e-300) * 1e-12
    Wm = V[:, keep] / np.sqrt(w[keep])
    back = V[:, keep] * np.sqrt(w[keep])
    return Xc @ Wm, back


# ---------------------------------------------------------------- finite state
def discrete_states(X, max_states=1500):
    n, d = X.shape
    sc = np.abs(X).max(axis=0)
    sc[sc == 0] = 1.0
    Q = np.round(X / sc * 1e8)
    uniq, inv, counts = np.unique(Q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    K = uniq.shape[0]
    if K < 2 or K > max_states or K > n / 10:
        return None
    # number of separate entries (runs) into each state
    starts = np.ones(n, dtype=bool)
    starts[1:] = inv[1:] != inv[:-1]
    entries = np.bincount(inv[starts], minlength=K)
    if np.median(entries) < 3:
        return None
    return inv, K


def markov_lrv(Z, inv, K):
    n, r = Z.shape
    s = inv
    N = np.zeros((K, K))
    np.add.at(N, (s[:-1], s[1:]), 1.0)
    N[s[-1], s[0]] += 1.0  # wrap-around: balanced counts, irreducible walk
    c = N.sum(axis=1)
    P = N / c[:, None]
    pi = c / c.sum()
    # state values (mean of whitened rows in each state)
    F = np.zeros((K, r))
    np.add.at(F, s, Z)
    F /= np.bincount(s, minlength=K)[:, None]
    F = F - pi @ F
    A = np.eye(K) - P + np.outer(np.ones(K), pi)
    ZF = np.linalg.solve(A, F)
    DF = F * pi[:, None]
    S = DF.T @ ZF
    S = S + S.T - DF.T @ F
    return S


# ------------------------------------------------------------------------ VAR
def _lagmat(Z, pmax):
    n, r = Z.shape
    T = n - pmax
    Xd = np.empty((T, 1 + pmax * r))
    Xd[:, 0] = 1.0
    for k in range(1, pmax + 1):
        Xd[:, 1 + (k - 1) * r: 1 + k * r] = Z[pmax - k: n - k]
    return Xd, Z[pmax:]


def var_select(Z, pmax, crit="aic"):
    n, r = Z.shape
    Xd, Y = _lagmat(Z, pmax)
    T = Y.shape[0]
    G = Xd.T @ Xd
    b = Xd.T @ Y
    yy = Y.T @ Y
    best = None
    ics = []
    for p in range(0, pmax + 1):
        m = 1 + p * r
        try:
            B = np.linalg.solve(G[:m, :m], b[:m])
        except np.linalg.LinAlgError:
            break
        Su = (yy - b[:m].T @ B) / T
        sign, logdet = np.linalg.slogdet(Su)
        if sign <= 0:
            break
        k = p * r * r
        if crit == "aic":
            ic = logdet + 2.0 * k / T
        elif crit == "aicc":
            ic = logdet + 2.0 * k / max(T - p * r - r - 1, 1)
        elif crit == "hq":
            ic = logdet + 2.0 * np.log(np.log(T)) * k / T
        else:
            ic = logdet + np.log(T) * k / T
        ics.append(ic)
        if best is None or ic < best[0]:
            best = (ic, p)
    return best[1], ics


def var_fit(Z, p):
    n, r = Z.shape
    if p == 0:
        U = Z - Z.mean(axis=0)
        return np.zeros((0, r, r)), U
    Xd, Y = _lagmat(Z, p)
    B, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    U = Y - Xd @ B
    Phi = B[1:].reshape(p, r, r).transpose(0, 2, 1)  # Phi[k] maps Z_{t-k-1} -> Z_t
    return Phi, U


def kernel_lrv(U, bw, kind="bartlett"):
    T, r = U.shape
    U = U - U.mean(axis=0)
    S = U.T @ U / T
    L = int(np.ceil(bw))
    for k in range(1, min(L, T - 1) + 1):
        x = k / bw
        if kind == "bartlett":
            w = 1.0 - x
        elif kind == "parzen":
            w = 1 - 6 * x * x + 6 * x ** 3 if x <= 0.5 else 2 * (1 - x) ** 3
        else:
            w = 1.0
        if w <= 0:
            continue
        G = U[k:].T @ U[:-k] / T
        S += w * (G + G.T)
    return S


def andrews_bw(U, kind="bartlett"):
    """Andrews (1991) AR(1) plug-in bandwidth, averaged over components."""
    T, r = U.shape
    U = U - U.mean(axis=0)
    num = 0.0
    den = 0.0
    for j in range(r):
        u = U[:, j]
        rho = np.dot(u[1:], u[:-1]) / max(np.dot(u[:-1], u[:-1]), 1e-300)
        rho = float(np.clip(rho, -0.97, 0.97))
        s2 = np.mean((u[1:] - rho * u[:-1]) ** 2)
        if kind == "bartlett":
            num += 4 * rho * rho * s2 * s2 / ((1 - rho) ** 6 * (1 + rho) ** 2)
        else:
            num += 4 * rho * rho * s2 * s2 / (1 - rho) ** 8
        den += s2 * s2 / (1 - rho) ** 4
    a = num / max(den, 1e-300)
    if kind == "bartlett":
        return 1.1447 * (a * T) ** (1.0 / 3.0)
    return 2.6614 * (a * T) ** 0.2


def var_lrv(Z, p, resid_kernel=True):
    n, r = Z.shape
    Phi, U = var_fit(Z, p)
    A1 = np.eye(r) - Phi.sum(axis=0)
    if resid_kernel:
        bw = andrews_bw(U)
        Om = kernel_lrv(U, max(bw, 1.0))
    else:
        Uc = U - U.mean(axis=0)
        Om = Uc.T @ Uc / U.shape[0]
    Ai = np.linalg.inv(A1)
    return Ai @ Om @ Ai.T


def default_pmax(n, r):
    p = int(max(1, min(np.floor(24 * (n / 10000.0) ** 0.25), (n / 10) // r - 1, 60)))
    while p > 1 and n * (1 + p * r) ** 2 > 5e9:  # keep the Gram matrix cheap
        p -= 1
    return p


def estimate(X, crit="aic", resid_kernel=True, use_markov=True, pmax=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    Z, back = whiten(X)
    r = Z.shape[1]
    Hz = None
    if use_markov:
        ds = discrete_states(X)
        if ds is not None:
            try:
                Hz = markov_lrv(Z, *ds)
                if not np.all(np.isfinite(Hz)):
                    Hz = None
            except Exception:
                Hz = None
    if Hz is None:
        if pmax is None:
            pmax = default_pmax(n, r)
        p, _ = var_select(Z, pmax, crit)
        Hz = var_lrv(Z, p, resid_kernel)
    H = back @ Hz @ back.T
    return _sym_psd(H)


# ------------------------------------------------- posterior-risk adjustment
def _risk_min(Sb, H0, iters=15, power=1.0):
    """Minimise mean_b ||S_b^{-1/2}(H - S_b)S_b^{-1/2}||_F^power over symmetric H.
    power=2: closed form; power=1: Weiszfeld / IRLS iterations."""
    r = H0.shape[0]
    Si = np.array([np.linalg.inv(S) for S in Sb])
    H = H0.copy()
    w = np.ones(len(Sb))
    nit = 1 if power == 2 else iters
    for _ in range(nit):
        M = np.einsum('b,bij,bkl->ikjl', w, Si, Si).reshape(r * r, r * r)
        rhs = np.einsum('b,bij->ij', w, Si).reshape(-1)
        try:
            Hn = np.linalg.solve(M, rhs).reshape(r, r)
        except np.linalg.LinAlgError:
            break
        Hn = 0.5 * (Hn + Hn.T)
        H = Hn
        if power == 2:
            break
        # weights 1/e_b
        e = np.empty(len(Sb))
        for b in range(len(Sb)):
            L = np.linalg.cholesky(Sb[b])
            A = np.linalg.solve(L, H - Sb[b])
            e[b] = np.linalg.norm(np.linalg.solve(L, A.T))
        w = 1.0 / np.maximum(e, 1e-3)
    return H


def var_draws(Z, p, B=200, seed=1):
    rng = np.random.default_rng(seed)
    n, r = Z.shape
    Xd, Y = _lagmat(Z, max(p, 0)) if p > 0 else (np.ones((n, 1)), Z)
    T, m = Xd.shape
    G = Xd.T @ Xd
    Bh = np.linalg.solve(G, Xd.T @ Y)
    U = Y - Xd @ Bh
    SSE = U.T @ U
    Gi = np.linalg.inv(G)
    Lx = np.linalg.cholesky(0.5 * (Gi + Gi.T))
    dof = T - m
    Lsse = np.linalg.cholesky(SSE)
    I = np.eye(r)
    A0 = I - Bh[1:].reshape(p, r, r).sum(axis=0).T if p > 0 else I
    s0 = np.sign(np.linalg.det(A0))
    out = []
    for _ in range(B):
        # Sigma_u ~ IW(SSE, dof): Sigma = Lsse (W)^{-1} Lsse', W ~ Wishart(dof, I)
        Lw = np.zeros((r, r))
        for i in range(r):
            Lw[i, i] = np.sqrt(rng.chisquare(dof - i))
            Lw[i, :i] = rng.standard_normal(i)
        Wi = np.linalg.inv(Lw @ Lw.T)
        Su = Lsse @ Wi @ Lsse.T
        Lu = np.linalg.cholesky(0.5 * (Su + Su.T))
        Bb = Bh + Lx @ rng.standard_normal((m, r)) @ Lu.T
        A = I - Bb[1:].reshape(p, r, r).sum(axis=0).T if p > 0 else I
        if np.sign(np.linalg.det(A)) != s0:
            continue
        try:
            Ai = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        S = Ai @ Su @ Ai.T
        S = 0.5 * (S + S.T)
        if np.all(np.isfinite(S)) and np.linalg.eigvalsh(S)[0] > 0:
            out.append(S)
    return out


def markov_draws(Z, inv, K, B=200, seed=1, prior=0.0):
    rng = np.random.default_rng(seed)
    n, r = Z.shape
    s = inv
    N = np.zeros((K, K))
    np.add.at(N, (s[:-1], s[1:]), 1.0)
    N[s[-1], s[0]] += 1.0
    F = np.zeros((K, r))
    np.add.at(F, s, Z)
    F /= np.bincount(s, minlength=K)[:, None]
    mask = N > 0
    out = []
    ones = np.ones(K)
    for _ in range(B):
        G = np.where(mask, rng.gamma(np.where(mask, N + prior, 1.0)), 0.0)
        P = G / G.sum(axis=1, keepdims=True)
        # stationary distribution
        A = np.eye(K) - P + np.outer(ones, ones) / K
        try:
            pi = np.linalg.solve(A.T, ones / K)
            pi = np.maximum(pi, 0); pi /= pi.sum()
            Fb = F - pi @ F
            ZF = np.linalg.solve(np.eye(K) - P + np.outer(ones, pi), Fb)
        except np.linalg.LinAlgError:
            continue
        DF = Fb * pi[:, None]
        S = DF.T @ ZF
        S = S + S.T - DF.T @ Fb
        S = 0.5 * (S + S.T)
        if np.all(np.isfinite(S)) and np.linalg.eigvalsh(S)[0] > 0:
            out.append(S)
    return out


def estimate2(X, crit="aic", power=1.0, B=200, use_markov=True, pmax=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    Z, back = whiten(X)
    r = Z.shape[1]
    Hz = None
    if use_markov:
        ds = discrete_states(X)
        if ds is not None:
            try:
                H0 = markov_lrv(Z, *ds)
                Sb = markov_draws(Z, *ds, B=B, seed=_seed()) if ds[1] <= 150 else []
                Hz = _risk_min(Sb, H0, power=power) if len(Sb) >= 20 else H0
                if not np.all(np.isfinite(Hz)):
                    Hz = None
            except Exception:
                Hz = None
    if Hz is None:
        if pmax is None:
            pmax = default_pmax(n, r)
        p, _ = var_select(Z, pmax, crit)
        H0 = var_lrv(Z, p, False)
        try:
            Sb = var_draws(Z, p, B=B, seed=_seed())
            Hz = _risk_min(Sb, H0, power=power) if len(Sb) >= 20 else H0
            if not np.all(np.isfinite(Hz)):
                Hz = H0
        except Exception:
            Hz = H0
    H = back @ Hz @ back.T
    return _sym_psd(H)
