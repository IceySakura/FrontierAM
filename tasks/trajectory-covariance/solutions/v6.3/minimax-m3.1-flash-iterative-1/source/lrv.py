"""Estimate the long-run covariance H = sum_{k=-inf}^{+inf} Gamma(k) of a
stationary vector time series from a single trajectory.

The estimator combines two ingredients, both estimated from the given
trajectory (no model of the sampling scheme is assumed, no table, no state
carried between calls):

* a non-parametric estimate: a flat-top weighted sum of the empirical
  autocovariance series, with a truncation length chosen by a signal-to-noise
  criterion;
* a model based estimate: a VAR fitted by least squares (order selected by
  BIC) whose implied long-run covariance is computed from the companion
  system.  Because the long-run covariance is a functional of the
  autocovariance function only, a model that reproduces the empirical
  autocovariances well also gives a good long-run covariance - with a much
  smaller sampling error than the raw autocovariance sum.

The two estimates are mixed with a weight that depends on how well the fitted
model reproduces the empirical autocovariance function.
"""
import numpy as np

__all__ = ["estimate_long_run_covariance"]

# The autocovariance is considered resolved while its size exceeds
# NOISE_FACTOR times the estimated sampling noise of the estimates.
NOISE_FACTOR = 2.0
# A lag is only dropped when it stays below the noise level for several
# consecutive lags (guards against a single unlucky fluctuation).
CONSEC = 2
# The lag window is flat up to the truncation length and then tapered to zero
# over TAPER_FACTOR times that length.  The taper averages over the remaining
# lags, which removes the phase sensitivity of a hard cut (important when the
# autocovariance oscillates, i.e. when the dynamics have mixed sign
# eigenvalues) at the price of a small extra bias.
TAPER_FACTOR = 2.0
# Largest VAR order considered by the BIC selection.
MAX_ORDER = 3
# The companion system of the fitted VAR must be stable; eigenvalues closer to
# one than this make the implied long-run covariance numerically unreliable.
MAX_EIG = 0.999
# Model based and non-parametric estimate are mixed with weights (1 - w, w);
# w = 1 when the model reproduces the autocovariances well.
MODEL_TOL = 3.0
# The residuals of the fitted VAR should be white; the mean squared residual
# autocorrelation is about one under that null hypothesis.
RESID_TOL = 3.0


# --------------------------------------------------------------------- helpers
def _sym(a):
    return 0.5 * (a + a.T)


def _psd(a):
    """Project a symmetric matrix onto the cone of positive semi-definite matrices."""
    a = _sym(a)
    try:
        w, v = np.linalg.eigh(a)
    except np.linalg.LinAlgError:
        return a
    if w[0] < 0.0:
        w = np.maximum(w, 0.0)
        a = (v * w) @ v.T
    return _sym(a)


def _sym_sqrt(a):
    """Symmetric positive semi-definite square root."""
    w, v = np.linalg.eigh(_sym(a))
    w = np.maximum(w, 0.0)
    return (v * np.sqrt(w)) @ v.T


def _lyapunov(a, q):
    """Solve a X a^T - X + q = 0 (numpy only, via the Kronecker form)."""
    n = a.shape[0]
    try:
        m = np.eye(n * n) - np.kron(a, a)
        x = np.linalg.solve(m, q.reshape(-1, order='F'))
        return x.reshape((n, n), order='F')
    except np.linalg.LinAlgError:
        return None


def _acf(x, kmax):
    """Biased autocovariance matrices r[k] = (1/M) sum_t z_t z_{t+k}^T.

    z is the column-centred data.  r[0] estimates the marginal covariance and
    r[k] estimates E[(x_t-mu)(x_{t+k}-mu)^T].
    """
    M, d = x.shape
    z = x - x.mean(axis=0)
    nfft = 1 << int(np.ceil(np.log2(2.0 * M)))
    Z = np.fft.rfft(z, n=nfft, axis=0)
    out = np.empty((kmax + 1, d, d))
    for i in range(d):
        Zi = Z[:, i]
        for j in range(i, d):
            c = np.fft.irfft(Zi * np.conj(Z[:, j]), n=nfft)[:kmax + 1] / M
            out[:, i, j] = c
            if i != j:
                out[:, j, i] = np.fft.irfft(Z[:, j] * np.conj(Zi), n=nfft)[:kmax + 1] / M
    return out


def _lam(r):
    """Symmetrised lag matrices: lam[k] = (r[k] + r[k].T)/2."""
    return 0.5 * (r + np.transpose(r, (0, 2, 1)))


def _whiten(r):
    """Divide the symmetrised lag matrices by the marginal covariance r[0].

    In that metric the sampling noise of the autocovariance estimates is
    (to first order) the same for every lag and every coordinate, so a single
    scalar noise level can be used for the whole matrix valued series.
    """
    d = r.shape[1]
    G = _sym_sqrt(r[0])
    Gi = np.linalg.pinv(G, rcond=1e-12)
    if not np.all(np.isfinite(Gi)):
        Gi = np.eye(d)
    return np.einsum('ab,kbc,cd->kad', Gi, _lam(r), Gi)


def _noise_level(R, M, d):
    """Sampling noise level of the whitened autocovariance matrices.

    For a stationary series the sampling covariance of the lag k estimate is
    (1/M) [ G0 G0^T + 2 sum_j G(j) G(j)^T ], hence the mean squared norm of
    the whitened noise is (d^2 + 2 A)/M with A = sum_j ||R(j)||_F^2.
    """
    sig = np.sqrt(np.einsum('kij,kij->k', R, R))
    prov = d / np.sqrt(float(M))
    k = 1
    while k < len(sig) and sig[k] > prov:
        k += 1
    k = min(max(k, 1), len(sig) - 1)
    a = float(np.sum(sig[1:k + 1] ** 2))
    return float(np.sqrt((d * d + 2.0 * a) / float(M)))


def _truncation(R, M, d):
    """Largest lag whose autocovariance is still resolved above the noise."""
    sig = np.sqrt(np.einsum('kij,kij->k', R, R))
    nsd = _noise_level(R, M, d)
    thr = NOISE_FACTOR * nsd
    n = len(sig) - 1
    k = 1
    while k < n:
        if np.all(sig[k + 1:k + 1 + CONSEC] < thr):
            break
        k += 1
    return max(k, 1)


def _flat_sum(r, K1, K2):
    """Flat-top weighted sum  r[0] + sum_k w(k) (r[k] + r[k]^T).

    w(k) = 1 for k <= K1 and a quadratic taper to 0 at k = K2.
    """
    S = r[0].copy()
    K2 = int(min(max(K2, K1), r.shape[0] - 1))
    for k in range(1, K2 + 1):
        if k <= K1:
            w = 1.0
        else:
            u = (k - K1) / float(max(K2 - K1, 1))
            w = (1.0 - u) ** 2
        S += w * (r[k] + r[k].T)
    return S


# --------------------------------------------------------------------- VAR model
def _var_fit(x, p):
    """Least squares fit of a VAR(p); returns (F, Q, coef, residuals)."""
    M, d = x.shape
    if M < 8 * (p * d + d):
        return None
    z = x - x.mean(axis=0)
    y = z[p:]
    X = np.hstack([z[p - i - 1:M - i - 1] for i in range(p)])
    XtX = X.T @ X
    tr = np.trace(XtX)
    if not np.isfinite(tr) or tr <= 0:
        return None
    XtX = XtX + 1e-8 * tr / (p * d) * np.eye(p * d)
    try:
        coef = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return None
    res = y - X @ coef
    Q = res.T @ res / res.shape[0]
    n = p * d
    F = np.zeros((n, n))
    for i in range(p):
        F[:d, i * d:(i + 1) * d] = coef[i * d:(i + 1) * d].T
    if p > 1:
        F[d:, : (p - 1) * d] = np.eye((p - 1) * d)
    Qs = np.zeros((n, n))
    Qs[:d, :d] = Q
    try:
        ev = np.abs(np.linalg.eigvals(F))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(ev)) or ev.max() > MAX_EIG:
        return None
    return F, Qs, coef, res


def _var_long_run(F, Qs):
    """Long-run covariance and stationary covariance of the companion system."""
    n = F.shape[0]
    Sig = _lyapunov(F, Qs)
    if Sig is None:
        return None
    I = np.eye(n)
    try:
        Slr = (Sig + F @ np.linalg.solve(I - F, Sig)
               + Sig @ np.linalg.solve(I - F.T, F.T))
    except np.linalg.LinAlgError:
        return None
    return Slr, Sig


def _var_model(x, pmax=MAX_ORDER):
    """VAR(p) model with BIC order selection; returns (S, F, Sig, resid stat)."""
    M, d = x.shape
    best = None
    for p in range(1, pmax + 1):
        out = _var_fit(x, p)
        if out is None:
            continue
        F, Qs, coef, res = out
        got = _var_long_run(F, Qs)
        if got is None:
            continue
        Slr, Sig = got
        s2 = max(np.trace(res.T @ res) / (res.shape[0] * d), 1e-300)
        bic = res.shape[0] * d * np.log(s2) + np.log(res.shape[0]) * (p * d * d)
        if best is None or bic < best[0]:
            best = (bic, Slr[:d, :d], F, Sig, res)
    if best is None:
        return None
    stat = _residual_stat(best[4], min(best[4].shape[0] - 1, 12))
    return best[1], best[2], best[3], stat


def _residual_stat(res, q):
    """Whiteness statistic of the VAR residuals.

    Returns the mean squared lag `1..q` residual autocorrelation divided by
    the value expected under the null hypothesis of white noise (about one).
    A value clearly above one means that the VAR does not capture the
    dependence structure of the series.
    """
    n, d = res.shape
    if n < 50 or d == 0:
        return None
    scale = np.sqrt(np.mean(res * res, axis=0))
    scale[scale <= 0] = 1.0
    e = res / scale
    kmax = int(min(q, n // 4))
    if kmax < 1:
        return None
    r = _acf(e, kmax)
    stat = 0.0
    for k in range(1, kmax + 1):
        stat += float(np.sum(r[k] * r[k]))
    noise = d * (d + 1.0) / n
    return stat / (kmax * noise)


def _model_weight(r, R, F, Sig, d, M, K):
    """How well does the fitted model reproduce the empirical autocovariances?

    Returns a weight in [0, 1]: one if the model fits, smaller otherwise.
    """
    kmax = int(min(max(K, 2), 200, r.shape[0] - 1))
    n = F.shape[0]
    G = _sym_sqrt(r[0])
    Gi = np.linalg.pinv(G, rcond=1e-12)
    if not np.all(np.isfinite(Gi)):
        return 0.0
    P = np.eye(n)
    emp = _lam(r[:kmax + 1])
    mod = np.empty_like(emp)
    for k in range(kmax + 1):
        g = (P @ Sig)[:d, :d]
        mod[k] = 0.5 * (g + g.T)
        P = P @ F
    E = np.einsum('ab,kbc,cd->kad', Gi, emp, Gi)
    Mm = np.einsum('ab,kbc,cd->kad', Gi, mod, Gi)
    sig = np.sqrt(np.einsum('kij,kij->k', E, E))
    nsd = _noise_level(R, M, d)
    use = sig[1:] > 1.5 * nsd
    if not np.any(use):
        return 0.0
    dif = (E[1:] - Mm[1:])[use]
    noise2 = d * (d + 1.0) / M
    q = float(np.sum(dif * dif) / (noise2 * float(np.sum(use))))
    if not np.isfinite(q):
        return 0.0
    return 1.0 if q <= MODEL_TOL else min(1.0, MODEL_TOL / q)


# --------------------------------------------------------------------- estimator
def estimate_long_run_covariance(x):
    """Estimate the long-run covariance of the columns of x (M x d)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    M, d = x.shape
    if M < 4 or d == 0:
        return np.zeros((d, d))
    x = np.where(np.isfinite(x), x, 0.0)
    kmax = int(min(M // 2, 4000))
    r = _acf(x, kmax)
    R = _whiten(r)
    K = _truncation(R, M, d)
    K = int(min(K, kmax))
    H = _psd(_flat_sum(r, K, int(round(TAPER_FACTOR * K))))
    out = _var_model(x)
    if out is not None:
        Sm, F, Sig, stat = out
        if np.all(np.isfinite(Sm)):
            Hm = _psd(Sm)
            w = _model_weight(r, R, F, Sig, d, M, K)
            if stat is not None and stat > RESID_TOL:
                # the VAR does not reproduce the dependence structure: trust
                # the non-parametric estimate more
                w *= min(1.0, RESID_TOL / stat)
            if w > 0.0:
                H = _psd((1.0 - w) * H + w * Hm)
    return H
