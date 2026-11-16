"""Long-run covariance estimation from a single trajectory.

Given an (M, d) time series x, estimate the d x d long-run covariance
    Sigma = lim_{n->infty} n Cov(mean(x_1..x_n))
          = Gamma_0 + sum_{k>=1} (Gamma_k + Gamma_k^T),
    Gamma_k = (1/M) sum_t x_t x_{t+k}^T   (x demeaned).

Components
----------
autocov(x, maxlag)   FFT-based autocovariance sequence (Gamma_k^T).
hac(G, L, kernel)    kernel HAC with truncation lag L.
var_fit / var_lrcov  VAR(p) fit and its implied long-run covariance.
prewhitened(y, L)    VAR(1) prewhitening + kernel HAC on the residuals.
estimate(x, method)  public entry point; 'auto' is the recommended choice.
finalize(H)          symmetrise and project onto the PSD cone.
"""

import numpy as np

# ---------------------------------------------------------------- kernels


def _k_bartlett(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    return np.where(a <= 1.0, 1.0 - a, 0.0)


def _k_parzen(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    out = np.zeros_like(x)
    m = a <= 0.5
    out[m] = 1.0 - 6.0 * a[m] ** 2 + 6.0 * a[m] ** 3
    m = (a > 0.5) & (a <= 1.0)
    out[m] = 2.0 * (1.0 - a[m]) ** 3
    return out


def _k_qs(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    out = np.zeros_like(x)
    m = a > 0.0
    z = 6.0 * np.pi * a[m] / 5.0
    out[m] = (25.0 / (12.0 * np.pi ** 2 * a[m] ** 2)) * (np.sin(z) / z - np.cos(z))
    out[~m] = 1.0
    return out


KERNELS = dict(bartlett=_k_bartlett, parzen=_k_parzen, qs=_k_qs)

# ------------------------------------------------------------- autocov


def autocov(x, maxlag=None):
    """G[k] = (1/M) sum_t x_t x_{t+k}^T  (= Gamma_k^T), k = 0..maxlag.

    Computed from one zero-padded FFT of the whole series, so the cost is
    O(d^2 M log M) and does not grow with the number of lags requested.
    """
    x = np.asarray(x, dtype=np.float64)
    M, d = x.shape
    if M <= 1:
        return np.zeros((1, d, d), dtype=np.float64)
    y = x - x.mean(0, keepdims=True)
    n = 1
    while n < 2 * M - 1:
        n <<= 1
    Z = np.fft.fft(y, n=n, axis=0)
    # outer product per frequency, frequency on axis 0 -> IFFT over it
    P = Z.conj()[:, :, None] * Z[:, None, :]
    # ifft(P)[j] = sum_{t-s = j (mod n)} y_t y_s^T.  For j < M the wrap-around
    # terms (t-s = j-n) do not contribute, so this is exactly M * Gamma_j^T.
    g = np.fft.ifft(P, axis=0).real[:M] / M
    if maxlag is not None:
        g = g[:min(maxlag + 1, g.shape[0])]
    return g


# ---------------------------------------------------------------- HAC


def hac(G, L, kernel='parzen'):
    """Kernel HAC from an autocovariance sequence."""
    L = int(L)
    if L < 1:
        return G[0].copy()
    K = min(L, G.shape[0] - 1)
    if K < 1:
        return G[0].copy()
    w = KERNELS[kernel](np.arange(1, K + 1) / L)
    W = (G[1:K + 1] * w[:, None, None]).sum(0)
    return G[0] + W + W.T


# ---------------------------------------------------------------- VAR


def var_fit(y, p):
    """Fit VAR(p) by OLS. Returns (A, Sigma) with A shape (d, d*p)."""
    M, d = y.shape
    if p <= 0:
        return None, (y.T @ y) / max(M, 1)
    if M <= p * d + 3:
        return None, None
    X = np.hstack([y[p - 1 - i:M - 1 - i] for i in range(p)])
    Y = y[p:]
    XtX = X.T @ X
    try:
        A_T = np.linalg.solve(XtX, X.T @ Y)
    except np.linalg.LinAlgError:
        A_T = np.linalg.lstsq(X, Y, rcond=None)[0]
    E = Y - X @ A_T
    Sigma = (E.T @ E) / max(M - p * d - 1, 1)
    return A_T.T, Sigma


def var_lrcov(A, Sigma):
    """Long-run covariance implied by a VAR(p): (I-sum A_i)^-1 Sigma (.)^-T."""
    if A is None or Sigma is None or not np.isfinite(Sigma).all():
        return None
    d = Sigma.shape[0]
    I = np.eye(d)
    B = I.copy()
    p = A.shape[1] // d
    for i in range(p):
        B = B - A[:, i * d:(i + 1) * d]
    try:
        return np.linalg.solve(B, Sigma @ np.linalg.solve(B.T, I))
    except np.linalg.LinAlgError:
        return None


def var_aic(y, pmax):
    """VAR order by AIC (log|Sigma| + 2 d^2 p / M). Returns (A, Sigma)."""
    M, d = y.shape
    best = None
    for p in range(0, pmax + 1):
        A, Sigma = var_fit(y, p)
        if Sigma is None or not np.isfinite(Sigma).all():
            continue
        try:
            det = np.linalg.det(Sigma)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(det) or det <= 0.0:
            continue
        aic = np.log(det) + 2.0 * d * d * p / M
        if best is None or aic < best[0]:
            best = (aic, p, A, Sigma)
    return (best[2], best[3]) if best else (None, None)


# -------------------------------------------------------- prewhitening


def prewhitened(y, L, kernel='parzen'):
    """VAR(1) prewhitened kernel HAC.

    Fit x_t = A x_{t-1} + e_t, apply the kernel HAC to the residuals e and
    map the result back with (I-A)^-1.  For white-ish residuals a small L
    already removes almost all of the remaining bias, while the variance
    stays close to that of the plain sample covariance.
    """
    M, d = y.shape
    if M < d + 5:
        return hac(autocov(y, L), L, kernel)
    A, Sigma = var_fit(y, 1)
    if A is None:
        return hac(autocov(y, L), L, kernel)
    e = y[1:] - y[:-1] @ A.T
    I = np.eye(d)
    H_e = hac(autocov(e, L), L, kernel)
    B = I - A
    try:
        return np.linalg.solve(B, H_e @ np.linalg.solve(B.T, I))
    except np.linalg.LinAlgError:
        return H_e


# ------------------------------------------------------------ helpers


def finalize(H):
    """Symmetrise and project onto the PSD cone (tolerance-level clipping)."""
    H = np.asarray(H, dtype=np.float64)
    H = 0.5 * (H + H.T)
    if H.size == 0:
        return H
    if H.shape[0] == 1:
        if H[0, 0] < 0.0:
            H[0, 0] = 0.0
        return H
    try:
        eig, vec = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return H
    if eig[0] < 0.0:
        eig = np.maximum(eig, 0.0)
        H = vec @ (vec * eig).T
    return H


def _L_fixed(M, kind):
    if kind == 'm2':
        return max(1, int(round(4.0 * M ** 0.2)))
    if kind == 'm3':
        return max(1, int(round(M ** (1.0 / 3.0))))
    if kind == 'nw':
        return max(1, int(round(4.0 * (M / 100.0) ** 0.2)))
    raise ValueError(kind)


def estimate(x, method='auto'):
    """Estimate the long-run covariance of x (shape (M, d)).

    Methods:
      'sample'        sample covariance (Gamma_0 only)
      'hac_<L>'       plain Parzen HAC with truncation lag L
      'hac_m2' / 'hac_m3'   Parzen HAC, L ~ 4 M^{1/5} / M^{1/3}
      'pw_<L>'       VAR(1) prewhitened Parzen HAC, lag L
      'pw_m2' / 'pw_m3'     prewhitened, L ~ 4 M^{1/5} / M^{1/3}
      'var'          AIC-selected VAR(p), implied long-run covariance
      'auto'         prewhitened Parzen HAC with a data-adaptive lag
    """
    x = np.asarray(x, dtype=np.float64)
    M, d = x.shape
    y = x - x.mean(0, keepdims=True)

    if method == 'sample':
        H = y.T @ y / M
    elif method.startswith('hac'):
        H = hac(autocov(y, _L_fixed(M, method[4:])), _L_fixed(M, method[4:]), 'parzen')
    elif method.startswith('pw'):
        H = prewhitened(y, _L_fixed(M, method[3:]), 'parzen')
    elif method == 'var':
        H = var_lrcov(*var_aic(y, min(15, max(1, M // (d * 4)))))
    elif method == 'auto':
        H = estimate_auto(y, M, d)
    else:
        raise ValueError('unknown method: ' + repr(method))

    if H is None or not np.isfinite(H).all():
        H = y.T @ y / M
    return finalize(H)


def estimate_auto(y, M, d):
    """Data-adaptive prewhitened estimator (the default).

    A VAR(1) is fitted and used as a pilot model.  The pilot tells us both
    how much smoothing is still needed (its spectral curvature) and whether
    prewhitening is safe at all.  If the pilot is unstable, near-singular or
    otherwise ill-conditioned we fall back to a plain kernel HAC.
    """
    A, Sigma = var_fit(y, 1)
    if A is None:
        return finalize(hac(autocov(y, 1), 1, 'parzen'))

    I = np.eye(d)
    B = I - A

    # --- is the pilot trustworthy? ---------------------------------------
    # The pilot is useless when the residual covariance is degenerate
    # (observations live in a low-dimensional subspace, e.g. a finite-state
    # chain with few states) or when the fitted VAR is unstable, because
    # (I-A)^-1 then amplifies noise without bound.
    try:
        eig_sigma = np.linalg.eigvalsh(Sigma)
        cond_sigma = eig_sigma.max() / max(eig_sigma.min(), 1e-300)
        cond_B = np.linalg.cond(B)
        rho_A = max(abs(np.linalg.eigvals(A)))
    except (np.linalg.LinAlgError, ValueError):
        return finalize(hac(autocov(y, 1), 1, 'parzen'))

    if (not np.isfinite(cond_sigma) or not np.isfinite(cond_B)
            or cond_sigma > 1e12 or cond_B > 1e6 or rho_A >= 0.98):
        return _plain_adaptive(y, M)

    # --- lag choice ------------------------------------------------------
    # Andrews (1991) optimal bandwidth for the Parzen kernel is
    #   b = 2.6614 * (alpha * M)^{1/5},
    # where alpha is the normalised curvature of the pilot spectrum at
    # frequency zero.  The truncation lag is the bandwidth rounded up and
    # capped by a slow-growing ceiling so that the variance stays bounded.
    alpha = _curvature(A, Sigma)
    if not np.isfinite(alpha) or alpha <= 0.0:
        return _plain_adaptive(y, M)
    cap = max(2, int(round(8.0 * M ** 0.25)))
    L = max(1, min(cap, int(round(2.6614 * (alpha * M) ** 0.2))))
    return finalize(prewhitened(y, L, 'parzen'))


def _plain_adaptive(y, M):
    """Plain Parzen HAC with a lag chosen from residual autocorrelation."""
    G = autocov(y, min(200, M))
    nrm = np.linalg.norm(G[0], 'fro')
    if nrm <= 0.0:
        return finalize(hac(G, 1, 'parzen'))
    rho = min(1.0, np.linalg.norm(G[1], 'fro') / nrm)
    cap = max(2, int(round(8.0 * M ** 0.25)))
    L = max(1, min(cap, int(round((1.0 + 4.0 * rho) * M ** 0.2))))
    return finalize(hac(G, L, 'parzen'))


def _curvature(A, Sigma):
    """Normalised spectral curvature at frequency zero for a VAR(1).

    For f(w) = (I - A e^{-iw})^{-1} Sigma (I - A' e^{iw})^{-1} this is
    -tr(f''(0)) / tr(f(0)), the quantity appearing in Andrews (1991)
    optimal-bandwidth formulas.
    """
    I = np.eye(A.shape[0])
    B = I - A
    try:
        C0 = np.linalg.solve(B, I)
    except np.linalg.LinAlgError:
        return float('nan')
    C1 = -1j * C0 @ A @ C0
    C2 = -2.0 * C0 @ A @ C0 @ A @ C0 + C0 @ A @ C0
    f0 = C0 @ Sigma @ C0.T
    f2 = C2 @ Sigma @ C0.T + 2.0 * C1 @ Sigma @ C1.T + C0 @ Sigma @ C2.T
    t0 = np.trace(f0).real
    if abs(t0) <= 0.0:
        return float('nan')
    return -np.trace(f2).real / t0
