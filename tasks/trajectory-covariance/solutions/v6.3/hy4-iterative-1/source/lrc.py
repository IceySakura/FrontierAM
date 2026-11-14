"""Long-run covariance estimation from a single trajectory.

Self-contained implementation (NumPy only).

Method
------
1. Center and whiten the trajectory by the symmetric square root of its
   lag-zero covariance.  Every later step is therefore equivariant under a
   fixed invertible linear change of coordinates.
2. Fit vector autoregressions of orders 0..P by least squares (one shared Gram
   matrix) and select an order with AIC.  For any invertible VAR filter the
   exact identity  Sigma_x = B^{-T} Sigma_e B^{-1}  holds with
   B = I - sum(A_i) and e the filtered series; the VAR is used only as a
   prewhitening filter, not as a model assumption.
3. Estimate the long-run covariance of the (nearly white) residuals with a
   flat-top lag window whose bandwidth comes from a Politis-style adaptive
   rule, and shrink that correction toward white noise with a James--Stein
   factor derived from its estimated variance.
4. Recolor, project onto the PSD cone, and transform back.
5. Apply a mode-dependent Stein shrinkage: in lag-zero-whitened coordinates
   the eigenvalues tau_i of W = G^{-1/2} H G^{-1/2} are the integrated
   autocorrelation times of the modes, with nu_i = M/tau_i effectively
   independent samples each.  Under the whitened Frobenius loss the best
   multiple of a noisy covariance estimate is nu/(nu + d + 1), so each mode
   is shrunk by nu_i/(nu_i + (d+1)).  This regularizes slowly mixing modes
   strongly and leaves nearly independent ones untouched, and is equivariant.
6. Apply a further James--Stein shrinkage toward a common-autocorrelation
   structure.  If every coordinate has the same autocorrelation shape then
   S = tau * G with G the lag-zero covariance; the target tau*G has far less
   variance than the unrestricted estimate.  The weight of that shrinkage is
   computed from the H-whitened discrepancy between H and tau*G and from an
   estimate of the sampling variance, so it is near one when the common-shape
   model is supported and near zero otherwise.  Both shrinkages are affine
   equivariant.

The output is always a finite, symmetric, positive semidefinite C-order
float64 matrix.
"""
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PMAX = 8            # maximum VAR order
CRITERION = 'aic'   # order-selection criterion
BW_C = 2.5          # threshold multiplier for the residual bandwidth rule
SHRINK_C = 1.0      # James--Stein multiplier for the residual correction
STEIN_F = 1.0       # shrinkage multiplier (0 disables it)
STEIN_G = 0.0       # extra shrinkage for very slowly mixing modes
SHRINK_MODE = 'nl'  # 'nl' = mode-dependent, 'scalar' = single factor
COMMON_F = 2.0      # shrinkage toward common-autocorrelation structure (0 = off)
ALPHA_POWER = 1.0   # exponent on the common-ACF shrinkage weight (>1 = sharper)
F_LO = 0.3          # Stein factor when a common autocorrelation shape is evident
F_HI = 1.75         # Stein factor otherwise
ADAPT_F = False     # interpolate the Stein factor with the common-ACF evidence
TAU_MODE = 'current'   # scale used for the common-ACF target: current/unshrunk/mix
KMAX = 20000        # largest autocovariance lag computed
CAP_FRAC = 0.10     # cap on residual bandwidth as a fraction of the sample

# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------


def clip_psd(H):
    H = (H + H.T) / 2
    w, V = np.linalg.eigh(H)
    return (V * np.maximum(w, 0.0)) @ V.T


def whiten(X):
    M, d = X.shape
    Xc = X - X.mean(axis=0)
    C0 = Xc.T @ Xc / M
    w, V = np.linalg.eigh(C0)
    w = np.maximum(w, np.finfo(float).eps * max(1.0, float(np.max(w))))
    return Xc @ ((V / np.sqrt(w)) @ V.T), (V * np.sqrt(w)) @ V.T


def sym_pair(C):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, np.finfo(float).eps * max(1.0, float(np.max(w))))
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def recolor(B, Se):
    Bi = np.linalg.inv(B)
    H = Bi.T @ Se @ Bi
    return (H + H.T) / 2


def acvf(Y, K):
    """Biased sample autocovariances (only symmetric sums are used)."""
    M, d = Y.shape
    K = min(K, M - 1)
    L = 1 << (2 * M - 1).bit_length()
    F = np.fft.rfft(Y, n=L, axis=0)
    cF = np.conj(F)
    C = np.empty((K + 1, d, d), dtype=np.float64)
    for i in range(d):
        C[:, i, :] = np.fft.irfft(F * cF[:, i:i + 1], n=L, axis=0)[:K + 1] / M
    return C


def unbiased_C(C, M):
    k = np.arange(len(C))
    return C * (M / np.maximum(M - k, 1))[:, None, None]


def choose_b(C, M, d, c=BW_C, max_b=None):
    """First lag after which L consecutive whitened ACF matrices are all
    below a noise-level threshold estimated from far lags."""
    K = len(C) - 1
    U = unbiased_C(C, M)
    nrm = np.linalg.norm(U, axis=(1, 2)) / d
    kk = np.arange(K + 1)
    eff = np.maximum(M - kk, 1)
    if K > 100:
        far = (kk >= max(1, K // 2)) & (kk < K)
        sf = float(np.median(nrm[far] * np.sqrt(eff[far])))
        if not np.isfinite(sf) or sf <= 0:
            sf = 1.0
    else:
        sf = 1.0
    thr = c * np.sqrt(np.log10(max(M, 10))) * sf / np.sqrt(eff)
    L = max(5, int(np.ceil(np.sqrt(np.log10(max(M, 10))))) + 2)
    top = min(K - L, len(nrm) - 1 - L)
    if max_b is None:
        max_b = K + 1
    if top < L:
        return 1
    ok = nrm[1:top + 1] < thr[1:top + 1]
    good = np.lib.stride_tricks.sliding_window_view(ok, L).all(axis=1)
    idx = np.flatnonzero(good)
    b = int(idx[0]) + 1 if len(idx) else top + 1
    return min(b, int(max_b))


def flat_weights(k, b):
    taper = max(2, b // 2)
    return np.clip(1.0 - (np.asarray(k, dtype=float) - (b - 1)) / (taper + 1),
                   0.0, 1.0)


# ---------------------------------------------------------------------------
# VAR bank
# ---------------------------------------------------------------------------


def var_fits(Y, pmax):
    M, d = Y.shape
    Z = np.column_stack([Y[pmax - i - 1:M - i - 1, :] for i in range(pmax)])
    T = Y[pmax:, :]
    n = len(T)
    G = Z.T @ Z
    R = Z.T @ T
    fits = []
    for p in range(pmax + 1):
        if p == 0:
            E = T.copy()
            B = np.eye(d)
        else:
            pd = p * d
            coef = np.linalg.solve(G[:pd, :pd], R[:pd])
            E = T - Z[:, :pd] @ coef
            B = np.eye(d) - coef.reshape(p, d, d).sum(axis=0)
        Ec = E - E.mean(axis=0)
        Sg = Ec.T @ Ec / n
        sign, ld = np.linalg.slogdet(Sg)
        if sign <= 0:
            ld = 2 * np.sum(np.log(np.maximum(np.diag(Sg), 1e-300)))
        fits.append(dict(p=p, E=E, B=B, Ec=Ec,
                         aic=n * ld + 2 * p * d * d,
                         bic=n * ld + p * d * d * np.log(n)))
    return fits


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


def estimate_one(X):
    M, d = X.shape
    Y, W = whiten(X)
    pmax = int(min(PMAX, max(1, (M - 2) // (4 * d))))
    K = min(M - 1, KMAX)
    fits = var_fits(Y, pmax)
    f = min(fits, key=lambda z: z[CRITERION])
    p, Ec, B = f['p'], f['Ec'], f['B']
    n = len(Ec)
    Se = Ec.T @ Ec / n
    Ses, Seis = sym_pair(Se)
    if n > 20:
        Ce = acvf(Ec, min(K, n - 1))
        Re = np.einsum('ab,kbc,cd->kad', Seis, Ce, Seis, optimize=True)
        b = choose_b(Re, n, d, max_b=max(8, int(CAP_FRAC * n)))
        k = np.arange(1, len(Re))
        w = flat_weights(k, b)
        pairs = Re[1:] + Re[1:].transpose(0, 2, 1)
        He = clip_psd(np.eye(d) + np.tensordot(w, pairs, axes=(0, 0)))
        delta = He - np.eye(d)
        sq = float(np.sum(delta**2))
        wsum = 1.0 + 2.0 * float(np.sum(w**2))
        q2 = max(float(np.trace(He)**2 + np.trace(He @ He)), d * (d + 1))
        V = wsum * q2 / n
        lam = max(0.0, 1.0 - SHRINK_C * V / sq) if sq > 0 else 0.0
        lam = min(1.0, lam)
        Se_lrc = Ses @ (np.eye(d) + lam * delta) @ Ses
    else:
        Se_lrc = Se
    if p > 0:
        Se_lrc = Se_lrc * (n / (n - p * d - 1.0))
    Hy = clip_psd(recolor(B, Se_lrc))
    return clip_psd(W @ Hy @ W)


def stein_shrink(H, X, f=None):
    """Stein shrinkage with effective sample sizes nu = M/tau.

    In lag-zero-whitened coordinates the eigenvalues of W = G^{-1/2} H G^{-1/2}
    are the integrated autocorrelation times tau_i of the modes.  Each mode
    has about M/tau_i independent samples, so its optimal shrinkage factor is
    nu_i/(nu_i + STEIN_F*(d+1)).  Modes with slow mixing are shrunk more.
    The scalar version uses one common tau = mean(tau_i).
    """
    sf = STEIN_F if f is None else f
    if sf <= 0:
        return H
    M, d = X.shape
    Xc = X - X.mean(axis=0)
    G = Xc.T @ Xc / M
    try:
        if SHRINK_MODE == 'nl':
            Gs, Gis = sym_pair(G)
            W = Gis @ H @ Gis
            W = (W + W.T) / 2
            eig, V = np.linalg.eigh(W)
            eig = np.maximum(eig, 1e-10)
            nu = M / eig
            den = sf * (d + 1) + STEIN_G * (d + 1)**2 / nu
            c = nu / (nu + den)
            Ws = (V * (c * eig)) @ V.T
            return clip_psd(Gs @ clip_psd(Ws) @ Gs)
        tau = float(np.trace(np.linalg.solve(G, H))) / d
        nu = M / max(tau, 1e-6)
        c = nu / (nu + sf * (d + 1))
        return c * H
    except np.linalg.LinAlgError:
        return H


def common_evidence(H, X):
    """James--Stein weight for the hypothesis S proportional to lag-0 Cov."""
    M, d = X.shape
    Xc = X - X.mean(axis=0)
    G = Xc.T @ Xc / M
    try:
        tau = float(np.trace(np.linalg.solve(G, H))) / d
        if not np.isfinite(tau) or tau <= 0:
            return 0.0
        T = tau * G
        _, Hi = sym_pair(H)
        delta = Hi @ (T - H) @ Hi
        nd2 = float(np.sum(delta * delta))
        Vd = COMMON_F * d * (d + 1) * tau / M
        a = float(np.clip(Vd / max(nd2, 1e-300), 0.0, 1.0))
        return a ** ALPHA_POWER
    except np.linalg.LinAlgError:
        return 0.0


def pooled_ar1_tau(X):
    """Integrated autocorrelation time from a pooled scalar AR(1) fit."""
    M, d = X.shape
    Xc = X - X.mean(axis=0)
    num = float(np.sum(Xc[1:] * Xc[:-1]))
    den = float(np.sum(Xc[:-1] * Xc[:-1]))
    if den <= 0:
        return 1.0
    rho = min(max(num / den, -0.999), 0.99999)
    return (1.0 + rho) / (1.0 - rho)


def common_acf_shrink(H, X, Hbase=None):
    """Shrink H toward tau*G (G = lag-zero covariance), in the H-whitened
    metric, with a James--Stein weight.  If all coordinates share the same
    autocorrelation shape, S is proportional to G and the target has much
    lower variance; otherwise the weight is near zero and H is unchanged."""
    if COMMON_F <= 0:
        return H
    M, d = X.shape
    Xc = X - X.mean(axis=0)
    G = Xc.T @ Xc / M
    try:
        tau = float(np.trace(np.linalg.solve(G, H))) / d
        if TAU_MODE == 'unshrunk' and Hbase is not None:
            tau = float(np.trace(np.linalg.solve(G, Hbase))) / d
        elif TAU_MODE == 'mix' and Hbase is not None:
            t1 = float(np.trace(np.linalg.solve(G, Hbase))) / d
            tau = 0.5 * (tau + t1)
        elif TAU_MODE == 'ar1':
            tau = pooled_ar1_tau(X)
        elif TAU_MODE == 'mixar1':
            tau = 0.5 * (tau + pooled_ar1_tau(X))
        if not np.isfinite(tau) or tau <= 0:
            return H
        T = tau * G
        _, Hi = sym_pair(H)
        delta = Hi @ (T - H) @ Hi
        nd2 = float(np.sum(delta * delta))
        Vd = COMMON_F * d * (d + 1) * tau / M
        alpha = float(np.clip(Vd / max(nd2, 1e-300), 0.0, 1.0))
        if ALPHA_POWER != 1.0:
            alpha = alpha ** ALPHA_POWER
        return clip_psd((1.0 - alpha) * H + alpha * T)
    except np.linalg.LinAlgError:
        return H


def fallback(X):
    """Cheap, always valid Bartlett-kernel estimate."""
    M, d = X.shape
    Xc = X - X.mean(0)
    K = min(M - 1, 2000)
    C = acvf(Xc, K)
    B = max(1.0, M ** (1.0 / 3.0))
    k = np.arange(1, K + 1)
    w = np.maximum(1 - k / B, 0.0)
    pairs = C[1:] + C[1:].transpose(0, 2, 1)
    return clip_psd(C[0] + np.tensordot(w, pairs, axes=(0, 0)))


def estimate(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        X = np.atleast_2d(X)
    if not np.isfinite(X).all():
        # Defensive only: valid inputs are finite.  Replace bad entries by the
        # mean of the finite entries of their column.
        fin = np.isfinite(X)
        col = np.where(fin.any(axis=0),
                       np.divide(np.nansum(np.where(fin, X, 0.0), axis=0),
                                 np.maximum(fin.sum(axis=0), 1)), 0.0)
        X = np.where(fin, X, col[None, :])
    try:
        Hbase = estimate_one(X)
        if ADAPT_F and COMMON_F > 0:
            a0 = common_evidence(Hbase, X)
            sf = F_LO + (F_HI - F_LO) * (1.0 - a0)
        else:
            sf = None
        H = stein_shrink(Hbase, X, sf)
        H = common_acf_shrink(H, X, Hbase)
        H = clip_psd(H)
    except Exception:
        try:
            H = fallback(X)
        except Exception:
            Xc = X - X.mean(0)
            H = clip_psd(Xc.T @ Xc / max(1, len(X)))
    if not np.isfinite(H).all():
        Xc = X - X.mean(0)
        H = clip_psd(Xc.T @ Xc / max(1, len(X)))
    H = (H + H.T) / 2.0
    return np.ascontiguousarray(H, dtype='<f8')


def main():
    if len(sys.argv) != 3:
        print('usage: lrc.py INPUT.npy OUTPUT.npy', file=sys.stderr)
        return 1
    X = np.load(sys.argv[1], allow_pickle=False)
    np.save(sys.argv[2], estimate(X), allow_pickle=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
