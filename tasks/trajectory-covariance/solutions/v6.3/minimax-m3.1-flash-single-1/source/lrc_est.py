"""Long-run covariance estimation from a single stationary trajectory.

Target
------
For a stationary vector process x_1, x_2, ... the quantity that is estimated is

    S = lim_{n->inf}  n * Cov( (1/n) * sum_{t=1..n} x_t )

which, for any stationary process with absolutely summable autocovariances,
equals

    S = Gamma(0) + 2 * sum_{k>=1} Gamma(k),      Gamma(k) = E[(x_t-mu)(x_{t+k}-mu)']

The estimator is completely generic: it only uses the observed trajectory, makes
no assumption about the generating model, and is equivariant with respect to
invertible linear changes of the coordinates.

Method
------
1.  Empirical autocovariances Gamma_hat(h), h = 0..K, computed with BLAS matrix
    products (exact, no FFT needed).
2.  The asymptotic standard error of every autocovariance entry is estimated
    (Bartlett-type formula) and used to detect which lags carry signal
    ("significant lags", contiguous block starting at lag 1).  If no lag is
    significant the trajectory looks like white noise and the sample covariance
    is returned.
3.  A family of multivariate Yule-Walker VAR(p) models, p = 1..P, is fitted.
    The Yule-Walker system makes the *implied* autocovariance sequence of the
    model match the empirical one at lags 0..p, so the model can extrapolate
    the tail of the autocovariance function.  The order is chosen by an
    out-of-sample mismatch criterion with a complexity penalty.
4.  Two candidates are compared:
      * the model based long-run covariance  (I-B)^{-1} Gamma_eps (I-B)^{-T},
      * the plain truncated sum  Gamma_hat(0) + 2 sum_{h<=m} (Gamma_hat(h)+' ).
    The comparison uses the out-of-sample mismatch of the implied
    autocovariances together with the variance each candidate carries.  The
    truncated sum wins for processes whose autocorrelation dies out abruptly
    (e.g. moving average type), the model wins for geometrically decaying
    (AR / Markov chain type) autocorrelation, where it is much more accurate.
5.  A parametric model can extrapolate a much larger tail than the data
    support (this happens for slowly mixing chains whose fitted root sits
    close to one).  The model estimate is therefore damped towards the
    truncated sum whenever it is much larger than it, which bounds the damage
    such an extrapolation can do.
6.  The result is symmetrised and projected onto the positive semi-definite
    cone.
"""

import numpy as np

__all__ = ["estimate_lrc"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _acovs(Xc, K):
    """Empirical autocovariances.

    G[h] = R[h] / (M-h) with R[h] = sum_t x_{t+h} x_t'  (later time first).
    """
    M, d = Xc.shape
    K = int(min(K, M - 1))
    G = np.empty((K + 1, d, d))
    for h in range(K + 1):
        G[h] = Xc[h:].T @ Xc[:M - h]
    cnt = (M - np.arange(K + 1)).astype(float)
    G /= cnt[:, None, None]
    return G


def _yw(G, p):
    """Multivariate Yule-Walker fit.

    Solves, for the unknown stack vec_F([B_1;...;B_p]), the block system

        Gamma(k) = sum_{j<=k} B_j Gamma(k-j) + sum_{j>k} B_j Gamma(j-k)'

    which is the identity satisfied by the autocovariance sequence of a
    VAR(p) model.  The solution makes the implied autocovariances equal to
    G[0..p].
    """
    d = G.shape[1]
    ps = d * d
    I = np.eye(d)
    A = np.zeros((p * ps, p * ps))
    rhs = np.zeros(p * ps)
    for k in range(1, p + 1):
        r0 = (k - 1) * ps
        rhs[r0:r0 + ps] = G[k].ravel(order="F")
        for j in range(1, p + 1):
            c0 = (j - 1) * ps
            if j <= k:
                A[r0:r0 + ps, c0:c0 + ps] = np.kron(G[k - j].T, I)
            else:
                A[r0:r0 + ps, c0:c0 + ps] = np.kron(G[j - k], I)
    sol = np.linalg.solve(A, rhs)
    return [sol[(j - 1) * ps:j * ps].reshape(d, d, order="F") for j in range(1, p + 1)]


def _implied(G, B, p, H):
    """Autocovariance sequence implied by the fitted VAR(p) model."""
    d = G.shape[1]
    out = np.empty((H + 1, d, d))
    out[:p + 1] = G[:p + 1]
    for h in range(p + 1, H + 1):
        acc = np.zeros((d, d))
        for j in range(1, p + 1):
            acc += B[j - 1] @ out[h - j]
        out[h] = acc
    return out


def _innov(G, B, p):
    """Innovation covariance of the fitted model."""
    Geps = G[0].copy()
    for j in range(1, p + 1):
        Geps -= B[j - 1] @ G[j].T
    return 0.5 * (Geps + Geps.T)


def _stable(B, d, tol=0.999):
    """(ok, A) with A = (I - sum B_j)^{-1} when the model is stable enough."""
    if len(B) == 0:
        return True, np.eye(d)
    Bsum = sum(B)
    ev = np.linalg.eigvals(Bsum)
    r = float(np.max(np.abs(ev)))
    if not np.isfinite(r) or r >= tol:
        return False, None
    try:
        A = np.linalg.inv(np.eye(d) - Bsum)
    except np.linalg.LinAlgError:
        return False, None
    if not np.all(np.isfinite(A)):
        return False, None
    return True, A


def _psd(H, d):
    """Symmetrise and clip negative eigenvalues."""
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    if w.min() < 0:
        w = np.maximum(w, 0.0)
        H = (V * w) @ V.T
        H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        H = np.zeros((d, d))
    return H


# --------------------------------------------------------------------------
# main estimator
# --------------------------------------------------------------------------
def estimate_lrc(X, KMAX=48, PMAX=8, HH=10, CT=1.5, PEN=0.25, CAP=2.0):
    """Estimate the long-run covariance matrix of a trajectory X (M x d)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("expected a 2-d array")
    M, d = X.shape
    if M < 8 or d < 1:
        return np.zeros((d, d))

    Xc = X - X.mean(axis=0)
    K = int(min(KMAX, M - 1))
    G = _acovs(Xc, K)

    # ---- noise level of every autocovariance entry -----------------------
    G0 = G[0]
    dg = np.diag(G0)
    nu2 = (np.outer(dg, dg) + G0 * G0 + 2.0 * (G[1:] * G[1:]).sum(axis=0)) / M
    nu2 = np.maximum(nu2, 1e-300)

    # ---- which lags carry signal? ---------------------------------------
    chi = (G * G / nu2).sum(axis=(1, 2)) / float(d * d)
    sig = chi > CT * CT
    m_sig = 0
    for h in range(1, K + 1):
        if sig[h]:
            m_sig = h
        else:
            break

    def flat(m):
        H = G[0].copy()
        for h in range(1, min(m, K) + 1):
            H += G[h] + G[h].T
        return H

    if m_sig == 0:
        # no detectable autocorrelation: the sample covariance is the best
        # estimator available
        return _psd(G[0], d)

    # ---- VAR(p) candidates ----------------------------------------------
    # a fitted root very close to one makes the long-run variance essentially
    # unestimable; such models are discarded.
    root_tol = 1.0 - 0.5 / np.sqrt(M)
    cands = []
    for p in range(1, PMAX + 1):
        try:
            B = _yw(G, p)
            ok, A = _stable(B, d, tol=min(0.999, max(root_tol, 0.5)))
            if not ok:
                continue
            Gtil = _implied(G, B, p, K)
            Geps = _innov(G, B, p)
            Hm = A @ Geps @ A.T
            if not np.all(np.isfinite(Hm)):
                continue
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        hi = min(K, p + HH)
        mis = 0.0
        for h in range(p + 1, hi + 1):
            mis += float(np.sum((G[h] - Gtil[h]) ** 2 / nu2))
        cands.append((p, Gtil, Hm, mis))

    if not cands:
        return _psd(flat(m_sig), d)

    # ---- order selection -------------------------------------------------
    p, Gtil, Hm, _ = min(cands, key=lambda c: c[3] + PEN * c[0] * d * d * np.log(max(M, 3)))

    # ---- model versus truncated sum -------------------------------------
    W0 = max(m_sig, p)
    W1 = min(K, W0 + HH)
    mis_model = 0.0
    mis_flat = 0.0
    for h in range(W0 + 1, W1 + 1):
        mis_model += float(np.sum((G[h] - Gtil[h]) ** 2 / nu2))
        mis_flat += float(np.sum(G[h] ** 2 / nu2))
    # the truncated sum carries the variance of m_sig empirical
    # autocovariances, the model only p of them
    v_flat = (1.0 + 4.0 * m_sig) * d * d
    v_model = (1.0 + 2.0 * p) * d * d
    if mis_model + v_model > mis_flat + v_flat:
        H = flat(m_sig)
    else:
        H = Hm
        # Safety net: a parametric model can extrapolate a much larger tail
        # than the data support (slowly mixing chains whose fitted root sits
        # close to one).  In that case the model is damped towards the
        # truncated sum instead of being trusted blindly.
        Hf = flat(m_sig)
        tf = float(np.trace(Hf))
        tm = float(np.trace(Hm))
        if tm > CAP * tf > 0.0:
            lam = CAP * tf / tm
            H = (1.0 - lam) * Hf + lam * Hm
    return _psd(H, d)
