"""Long-run covariance estimator for a single MCMC / time-series trajectory.

Usage: python3 lrc.py INPUT.npy OUTPUT.npy

Target: Sigma = lim n Cov(mean of n steps).  The returned matrix H is a Bayes decision under
the evaluation loss  ||S^{-1/2}(H - S)S^{-1/2}||_F  computed from posterior draws S_j of one or
more fitted models:

  * VAR(p) models (orders weighted by AIC), normal / inverse-Wishart posteriors; every draw's
    eigenvalues, whitened by the marginal covariance, are shrunk toward isotropy by an
    empirical-Bayes factor.
  * Scalar-coefficient AR(q) model (shared AR polynomial for all coordinates), used for chains
    with many exact repeats (Metropolis rejections) unless the VAR fits clearly better; its
    estimate is averaged with the VAR estimate.
  * Trajectories with finitely many, repeatedly revisited values are additionally modelled as a
    finite Markov chain (plug-in transition matrix, exact long-run covariance via the
    fundamental matrix); the Markov, VAR and scalar-AR estimates are averaged.

Everything is affine equivariant.  Randomness uses a fixed seed (SEED, default 1).
"""
import os
import sys

import numpy as np

RNG = np.random.default_rng(int(os.environ.get('SEED', '1') or 1))

EB_C = 0.5          # EB noise-inflation constant, multiplied by d
W_SAR_REJ = 0.7     # weight of the scalar-AR estimate for chains with many repeats
REPEAT_FRAC = 0.3   # fraction of exactly repeated consecutive rows that marks a rejection sampler


# ------------------------------------------------------------------ decision step
def bayes_action(Ss, iters=40, prior=None):
    """H minimising mean_j ||S_j^{-1/2}(H - S_j)S_j^{-1/2}||_F (IRLS in whitened coordinates)."""
    J, d, _ = Ss.shape
    pw = np.ones(J) if prior is None else np.asarray(prior, float) * J / np.sum(prior)
    w0, V0 = np.linalg.eigh(np.einsum('j,jab->ab', pw / J, Ss))
    w0 = np.maximum(w0, 1e-300)
    Wh = (V0 / np.sqrt(w0)) @ V0.T
    Wb = (V0 * np.sqrt(w0)) @ V0.T
    T = np.linalg.inv(Wh[None] @ Ss @ Wh[None])
    T = 0.5 * (T + np.swapaxes(T, 1, 2))
    Lc = np.linalg.cholesky(T)
    w = pw.copy()
    H = None
    eye = np.eye(d)
    for _ in range(iters):
        Kop = np.einsum('j,jab,jcd->acbd', w, T, T).reshape(d * d, d * d)
        rhs = np.einsum('j,jab->ab', w, T).reshape(d * d)
        Hn = np.linalg.lstsq(Kop, rhs, rcond=None)[0].reshape(d, d)
        Hn = 0.5 * (Hn + Hn.T)
        Wm = np.einsum('jba,bc,jcd->jad', Lc, Hn, Lc) - eye
        loss = np.sqrt((Wm ** 2).sum((1, 2)))
        if H is not None and np.abs(Hn - H).max() <= 1e-10 * np.abs(Hn).max():
            H = Hn
            break
        H = Hn
        w = pw / np.maximum(loss, 1e-4)
    return Wb @ H @ Wb


def valid_draws(Ss):
    Ss = np.asarray(Ss)
    if Ss.ndim != 3 or len(Ss) == 0:
        return np.zeros((0, 1, 1))
    ok = np.all(np.isfinite(Ss.reshape(len(Ss), -1)), axis=1)
    Ss = Ss[ok]
    if len(Ss) == 0:
        return Ss
    Ss = 0.5 * (Ss + np.swapaxes(Ss, 1, 2))
    ev = np.linalg.eigvalsh(Ss)
    ok = ev[:, 0] > 1e-12 * np.maximum(ev[:, -1], 1e-300)
    return Ss[ok]


def eb_draws(Ss, H0, G0, c):
    """Empirical-Bayes shrinkage of every draw's log-eigenvalues (whitened by G0) toward their
    mean.  The factor compares the eigenvalue dispersion of the point estimate H0 with the
    posterior noise of those directions (c inflates the noise for eigenvalue spreading)."""
    J, d, _ = Ss.shape
    if d < 2:
        return Ss
    w, V = np.linalg.eigh(G0)
    Gi = (V / np.sqrt(w)) @ V.T
    Gh = (V * np.sqrt(w)) @ V.T
    Wz = Gi[None] @ Ss @ Gi[None]
    Wz = 0.5 * (Wz + np.swapaxes(Wz, 1, 2))
    Hz = Gi @ H0 @ Gi
    lm, U = np.linalg.eigh(0.5 * (Hz + Hz.T))
    if lm[0] <= 0:
        return Ss
    r = np.maximum(np.einsum('ai,jab,bi->ji', U, Wz, U), 1e-300)
    s2 = np.var(np.log(r), axis=0).mean()
    ll = np.log(lm)
    disp = ((ll - ll.mean()) ** 2).sum() / (d - 1)
    tau2 = max(0.0, disp - c * s2)
    k = tau2 / (tau2 + s2) if tau2 + s2 > 0 else 1.0
    lj, Vj = np.linalg.eigh(Wz)
    lj = np.log(np.maximum(lj, 1e-300))
    mj = lj.mean(1, keepdims=True)
    lt = np.exp(mj + k * (lj - mj))
    Wt = (Vj * lt[:, None, :]) @ np.swapaxes(Vj, 1, 2)
    return Gh[None] @ Wt @ Gh[None]


# ------------------------------------------------------------------ finite-state chains
def lrc_from_P(P, F, pi):
    """Long-run covariance of F[X_t] for a chain with transition matrix P, stationary law pi."""
    K = P.shape[0]
    Fc = F - pi @ F
    G = np.linalg.solve(np.eye(K) - P + pi[None, :], Fc)
    DF = pi[:, None] * Fc
    S = DF.T @ G
    return S + S.T - DF.T @ Fc


def markov_plugin(rows, inv):
    K = rows.shape[0]
    N = np.zeros((K, K))
    np.add.at(N, (inv[:-1], inv[1:]), 1.0)
    N[inv[-1], inv[0]] += 1.0  # circular closure keeps the empirical chain irreducible
    cnt = N.sum(1)
    return lrc_from_P(N / cnt[:, None], rows, cnt / cnt.sum())


# ------------------------------------------------------------------ VAR(p)
def design(xc, p):
    M = xc.shape[0]
    Y = xc[p:]
    if p == 0:
        return Y, np.zeros((M, 0))
    return Y, np.hstack([xc[p - i:M - i] for i in range(1, p + 1)])


def var_order(xc):
    """AIC for VAR orders 0..pmax on a common sample; returns (best p, its AIC, all AICs, n)."""
    M, d = xc.shape
    pmax = int(min(40, max(1, np.floor(12 * (M / 100.0) ** 0.25))))
    pmax = max(1, min(pmax, (M - 1) // (4 * d) - 1))
    n = M - pmax
    Y, Z = design(xc, pmax)
    G = Z.T @ Z
    g = Z.T @ Y
    yy = Y.T @ Y
    aics = np.full(pmax + 1, np.inf)
    for p in range(0, pmax + 1):
        if p == 0:
            Se = yy / n
        else:
            k = p * d
            B = np.linalg.solve(G[:k, :k], g[:k])
            Se = (yy - g[:k].T @ B) / n
        sign, ld = np.linalg.slogdet(Se)
        if sign > 0:
            aics[p] = ld + 2.0 * p * d * d / n
    best_p = int(np.argmin(aics))
    return best_p, aics[best_p], aics, n


def lrc_from_B(B, Se, d, p):
    Asum = np.zeros(B.shape[:-2] + (d, d))
    for i in range(p):
        Asum = Asum + np.swapaxes(B[..., i * d:(i + 1) * d, :], -1, -2)
    Ai = np.linalg.inv(np.eye(d) - Asum)
    return Ai @ Se @ np.swapaxes(Ai, -1, -2)


def wishart(Lw, nu, J, d):
    A = np.zeros((J, d, d))
    for i in range(d):
        A[:, i, i] = np.sqrt(RNG.chisquare(nu - i, J))
        A[:, i, :i] = RNG.standard_normal((J, i))
    LA = Lw @ A
    return LA @ np.swapaxes(LA, -1, -2)


def var_point(xc, p):
    """Plain least-squares VAR(p) spectral density at frequency zero (no posterior step)."""
    M, d = xc.shape
    if p == 0:
        return xc.T @ xc / M
    Y, Z = design(xc, p)
    B = np.linalg.lstsq(Z, Y, rcond=None)[0]
    E = Y - Z @ B
    return lrc_from_B(B, E.T @ E / Y.shape[0], d, p)


def var_draws(xc, p, J):
    """Posterior draws (normal / inverse-Wishart, flat prior) of the VAR(p) long-run covariance."""
    M, d = xc.shape
    Y, Z = design(xc, p)
    n = Y.shape[0]
    ZZ = Z.T @ Z
    B = np.linalg.solve(ZZ, Z.T @ Y)
    E = Y - Z @ B
    SE = E.T @ E
    point = lrc_from_B(B, SE / n, d, p)
    k = p * d
    nu = max(n - k, d + 2)
    Se = np.linalg.inv(wishart(np.linalg.cholesky(np.linalg.inv(SE)), nu, J, d))
    Se = 0.5 * (Se + np.swapaxes(Se, 1, 2))
    LZ = np.linalg.cholesky(np.linalg.inv(ZZ))
    Ls = np.linalg.cholesky(Se)
    Bd = B[None] + LZ[None] @ RNG.standard_normal((J, k, d)) @ np.swapaxes(Ls, 1, 2)
    if k <= 60:  # keep stationary draws
        if p == 1:
            C = np.swapaxes(Bd, 1, 2)
        else:
            C = np.zeros((len(Bd), k, k))
            C[:, :d, :] = np.swapaxes(Bd, 1, 2)
            C[:, d:, :-d] = np.eye(k - d)
        ok = np.abs(np.linalg.eigvals(C)).max(1) < 1.0
        if ok.sum() >= max(5, J // 10):
            Bd, Se = Bd[ok], Se[ok]
    return valid_draws(lrc_from_B(Bd, Se, d, p)), point


def var_estimate(xc, order, J=400):
    """Bayes decision from draws pooled over VAR orders (Akaike weights > 1%), each draw shrunk
    toward isotropy by the empirical-Bayes step."""
    M, d = xc.shape
    p0, _, aics, n = order
    G0 = xc.T @ xc / M
    if p0 == 0:
        return G0
    w = np.exp(-0.5 * n * (aics - aics[p0]))
    w[~np.isfinite(w)] = 0.0
    w[0] = 0.0
    w /= w.sum()
    ps = sorted(set([p for p in range(1, len(w)) if w[p] > 0.01] + [p0]))
    ws = np.array([w[p] for p in ps])
    ws /= ws.sum()
    draws, prior, point = [], [], None
    for p, wp in zip(ps, ws):
        Sd, pt = var_draws(xc, p, max(40, int(round(J * wp))))
        if p == p0:
            point = pt
        if len(Sd):
            draws.append(Sd)
            prior.append(np.full(len(Sd), wp / len(Sd)))
    if not draws or sum(len(D) for D in draws) < 20:
        return point
    Ss = np.concatenate(draws)
    pr = np.concatenate(prior)
    try:
        St = eb_draws(Ss, point, G0, EB_C * d)
        if np.all(np.isfinite(St)):
            Ss = St
    except Exception:
        pass
    return bayes_action(Ss, prior=pr)


# ------------------------------------------------------------------ scalar-coefficient AR(q)
def lag_blocks(z, qmax):
    """C[i, j] = sum_{t=qmax}^{M-1} z_{t-i} z_{t-j}^T for 0 <= i, j <= qmax (windowed cumsums)."""
    M, d = z.shape
    C = np.empty((qmax + 1, qmax + 1, d, d))
    for h in range(qmax + 1):
        c = np.einsum('ua,ub->uab', z[:M - h], z[h:]).reshape(M - h, d * d)
        cs = np.vstack([np.zeros((1, d * d)), np.cumsum(c, axis=0)])
        for i in range(qmax + 1 - h):
            j = i + h
            blk = (cs[M - j] - cs[qmax - j]).reshape(d, d)
            C[j, i] = blk
            C[i, j] = blk.T
    return C


def sar_fit(xc):
    """x_t = sum_i a_i x_{t-i} + e_t with scalar a_i, order by AIC (same scale as var_order)."""
    M, d = xc.shape
    qmax = int(min(40, max(2, np.floor(20 * (M / 1000.0) ** 0.3))))
    qmax = max(1, min(qmax, M // 10))
    n = M - qmax
    G0 = xc.T @ xc / M
    w, V = np.linalg.eigh(G0)
    Wh = (V / np.sqrt(w)) @ V.T
    C = lag_blocks(xc @ Wh, qmax)
    R = np.trace(C, axis1=2, axis2=3)
    ldW = np.linalg.slogdet(Wh)[1]
    best = None
    for q in range(0, qmax + 1):
        a = np.linalg.solve(R[1:q + 1, 1:q + 1], R[1:q + 1, 0]) if q > 0 else np.zeros(0)
        at = np.concatenate([[1.0], -a])
        Sz = np.einsum('i,j,ijab->ab', at, at, C[:q + 1, :q + 1]) / n
        Sz = 0.5 * (Sz + Sz.T)
        aic = np.linalg.slogdet(Sz)[1] - 2.0 * ldW + 2.0 * q / n
        if best is None or aic < best[0]:
            best = (aic, q, a, Sz)
    aic, q, a, Sz = best
    Wi = np.linalg.inv(Wh)
    Se = Wi @ Sz @ Wi
    s2 = np.trace(Sz) / d
    return dict(q=q, a=a, Se=0.5 * (Se + Se.T), Rq=R[1:q + 1, 1:q + 1] / s2, n=n, aic=aic, d=d)


def sar_draws(f, J=400):
    d, n, q = f['d'], f['n'], f['q']
    nu = max(n - q, d + 2)
    Sed = np.linalg.inv(wishart(np.linalg.cholesky(np.linalg.inv(f['Se'] * n)), nu, J, d))
    Sed = 0.5 * (Sed + np.swapaxes(Sed, 1, 2))
    if q == 0:
        return Sed
    L = np.linalg.cholesky(np.linalg.inv(f['Rq']))
    A = f['a'][None] + RNG.standard_normal((J, q)) @ L.T
    s = 1.0 - A.sum(1)
    ok = s > 1e-6
    return Sed[ok] / (s[ok] ** 2)[:, None, None]


def sar_candidate(xc, p, aic_var):
    """Scalar-AR estimate unless the VAR fits clearly better (log-likelihood margin)."""
    M = xc.shape[0]
    f = sar_fit(xc)
    margin = 0.5 * (f['aic'] - aic_var) * (M - p)   # > 0: VAR better
    if margin > 30.0 + 0.005 * M:
        return None
    Ss = valid_draws(sar_draws(f))
    if len(Ss) < 20:
        return None
    H = bayes_action(Ss)
    return H if np.all(np.isfinite(H)) else None


# ------------------------------------------------------------------ driver
def estimate(x):
    M, d = x.shape
    xc = x - x.mean(0)
    rows, inv = np.unique(x, axis=0, return_inverse=True)
    inv = inv.ravel()
    K = rows.shape[0]
    jumps = np.count_nonzero(inv[1:] != inv[:-1])
    repeats = (M - 1 - jumps) / max(M - 1, 1)
    finite = K <= 2000 and K <= M / 4 and jumps >= 3 * K

    order = var_order(xc)
    p, aic_var = order[0], order[1]
    try:
        V = var_estimate(xc, order)
        if not np.all(np.isfinite(V)):
            raise ValueError
    except Exception:
        V = var_point(xc, p)
    H = None
    if finite or repeats > REPEAT_FRAC:
        try:
            H = sar_candidate(xc, p, aic_var)
        except Exception:
            H = None
    if finite:
        parts = [V, V if H is None else H]
        try:
            S = markov_plugin(rows, inv)
            if np.all(np.isfinite(S)):
                parts.append(S)
        except Exception:
            pass
        return sum(parts) / len(parts)
    if H is not None:
        return W_SAR_REJ * H + (1.0 - W_SAR_REJ) * V
    return V


def finalize(S, x):
    d = x.shape[1]
    if S is None or not np.all(np.isfinite(S)):
        xc = x - x.mean(0)
        S = xc.T @ xc / max(len(x), 1)
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    S = (V * np.maximum(w, 0.0)) @ V.T
    S = 0.5 * (S + S.T)
    if not np.all(np.isfinite(S)):
        S = np.zeros((d, d))
    return np.ascontiguousarray(S, dtype='<f8')


def main():
    inp, out = sys.argv[1], sys.argv[2]
    x = np.load(inp, allow_pickle=False).astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    S = None
    try:
        S = estimate(x)
    except Exception:
        try:
            xc = x - x.mean(0)
            S = var_point(xc, var_order(xc)[0])
        except Exception:
            S = None
    np.save(out, finalize(S, x))


if __name__ == '__main__':
    main()
