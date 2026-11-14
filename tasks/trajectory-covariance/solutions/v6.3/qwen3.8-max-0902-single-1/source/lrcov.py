"""Long-run covariance estimation from a single MCMC trajectory.

Affine-equivariant lag-window (spectral) estimator whose bandwidth is chosen by
minimising an estimate of the scored loss  E ||S^{-1/2}(H-S)S^{-1/2}||_F^2 ,
decomposed into a squared-bias part and a variance part, both estimated from the
trajectory itself.

For a kernel w with w(u) = 1 - cq |u|^q + o(|u|^q) near zero and bandwidth b,
    H(b) = sum_k w(k/b) G_k ,      G_k = autocovariance at lag k (1/T scaled)
    bias(H(b)) ~ -cq b^-q D_q ,    D_q = sum_k |k|^q G_k
    Var-total  ~ (b/T) * Iw2 * c_v, c_v = sum_m [ (tr G_m)^2 + ||G_m||_F^2 ]
with all norms measured after whitening by S (proxied by a pilot estimate),
because the score is a whitened Frobenius norm.  Minimising gives
    b* = [ 2 q cq^2 ||D_q||^2 T / (Iw2 c_v) ]^(1/(2q+1)).
"""
import numpy as np

__all__ = ["estimate", "KERNELS", "kernel_w"]

# ------------------------------------------------------------------ kernels --
_KERNEL_DEFS = {           # kind: (q, cq, Iw2)
    "bart": (1, 1.0, 2.0 / 3.0),
    "th":   (2, np.pi ** 2 / 4.0, 0.75),
    "parz": (2, 6.0, 151.0 / 320.0),   # placeholder, recomputed below
}
KERNELS = dict(_KERNEL_DEFS)


def kernel_w(u, kind):
    a = np.abs(np.asarray(u, dtype=float))
    if kind == "bart":
        return np.maximum(0.0, 1.0 - a)
    if kind == "th":
        return np.where(a < 1.0, 0.5 * (1.0 + np.cos(np.pi * np.minimum(a, 1.0))), 0.0)
    if kind == "parz":
        out = np.zeros_like(a)
        m = a <= 0.5
        out[m] = 1.0 - 6.0 * a[m] ** 2 + 6.0 * a[m] ** 3
        m2 = (~m) & (a < 1.0)
        out[m2] = 2.0 * (1.0 - a[m2]) ** 3
        return out
    if kind == "rect":
        return (a < 1.0).astype(float)
    raise ValueError(kind)


def _kernel_consts():
    u = np.linspace(0.0, 1.0, 200001)
    for k in ("bart", "th", "parz"):
        Iw2 = 2.0 * np.trapz(kernel_w(u, k) ** 2, u)
        q, cq, _ = KERNELS[k]
        KERNELS[k] = (q, cq, Iw2)


_kernel_consts()


def _psd_inv_sqrt(H, floor=1e-10):
    Hs = 0.5 * (H + H.T)
    ev, U = np.linalg.eigh(Hs)
    mx = max(ev[-1], 0.0)
    ev = np.maximum(ev, floor * mx if mx > 0 else floor)
    Minv = (U / ev) @ U.T                   # H^{-1}
    return (U * ev) @ U.T, Minv, ev


def _clip_psd(H):
    H = 0.5 * (H + H.T)
    ev, U = np.linalg.eigh(H)
    if ev[0] < 0.0:
        H = (U * np.maximum(ev, 0.0)) @ U.T
    return 0.5 * (H + H.T)


# -------------------------------------------------------- autocovariances ---
def acov_lags(x, K):
    """G[k,i,j] = (1/T) sum_t y[t,i] y[t+k,j], k = 0..K, y = x - mean(x)."""
    T, d = x.shape
    L = 1
    while L < 2 * T:
        L *= 2
    xc = x - x.mean(axis=0)
    F = np.fft.rfft(xc, n=L, axis=0)
    G = np.empty((K + 1, d, d))
    idx = np.arange(K + 1)
    for i in range(d):
        Fi = F[:, i]
        for j in range(i, d):
            c = np.fft.irfft(Fi * np.conj(F[:, j]), n=L) / T
            G[idx, i, j] = c[idx]
            if j != i:
                G[idx, j, i] = c[(L - idx) % L]
    return G


def windowed(G, b, kind):
    """sum_k w(k/b) G_k over all integer k (G_-k = G_k^T)."""
    K = G.shape[0] - 1
    if K < 1:
        return G[0].copy()
    k = np.arange(1, K + 1, dtype=float)
    w = kernel_w(k / b, kind)
    nz = np.nonzero(w)[0]
    if nz.size == 0:
        return G[0].copy()
    Gs = G[1:][nz]
    Gs = Gs + np.swapaxes(Gs, 1, 2)
    return G[0] + np.einsum("k,kij->ij", w[nz], Gs)


# ------------------------------------------------------- MSE-optimal rule ---
def _bandwidth(G, T, d, kind, M, Lc, bmul, bmin, bmax):
    """M = metric (proxy for S^-1). Returns (b, diag)."""
    q, cq, Iw2 = KERNELS[kind]
    Gc = G[: Lc + 1]
    # tr(M G_k) and tr(M G_k M G_k)
    B = np.matmul(M, Gc)                       # (Lc+1,d,d)
    tMG = np.einsum("ij,kji->k", M, Gc)         # tr(M G_k)
    MGM = np.einsum("kij,kji->k", B, B)         # ||G_k||^2 in M-metric
    frac = 1.0 - np.arange(Lc + 1) / T
    tr0 = tMG[0]
    dg0 = float(np.einsum("i,i->", np.diag(B[0]), np.diag(B[0])))
    # variance constant (debiased).  Per-lag noise of sum_ij Var(Ghat_k[i,j])
    # is (1/T)[c_tr + (tr G_k)^2] with c_tr = sum_u (tr G_u)^2 (correlated case).
    ctr = float(tr0 ** 2 + 2.0 * np.sum(tMG[1:] ** 2))
    nk = (frac / T) * (ctr + tMG ** 2)          # per-lag noise, all entries
    cv_raw = float(MGM[0] + tr0 ** 2 + 2.0 * np.sum(MGM[1:] + tMG[1:] ** 2))
    cv_corr = float(2.0 * np.sum(nk[1:]) + nk[0])
    cv = max(cv_raw - cv_corr, 1e-8 * max(cv_raw, 1e-30))
    # D_q and its noise
    kk = np.arange(1, Lc + 1, dtype=float)
    kq = kk ** q
    Gs = Gc[1:] + np.swapaxes(Gc[1:], 1, 2)
    Dq = np.einsum("k,kij->ij", kq, Gs)
    BD = np.matmul(M, Dq)
    D2 = float(np.einsum("ij,ji->", BD, BD))
    nd = float(np.sum(kq ** 2 * 2.0 * nk[1:]))
    D2 = max(D2 - 0.75 * nd, 0.05 * D2)
    num = 2.0 * q * (cq ** 2) * D2 * T
    den = Iw2 * cv
    b = (num / den) ** (1.0 / (2 * q + 1)) if num > 0 and den > 0 else bmin
    b = float(np.clip(b * bmul, bmin, bmax))
    return b, {"b": b, "cv": cv, "D2": D2, "Lc": Lc}


def _ips_lag(G, T, d, M):
    """Initial-positive-sequence style cutoff on the M-averaged autocorrelation."""
    K = G.shape[0] - 1
    tMG = np.einsum("ij,kji->k", M, G)
    t0 = tMG[0]
    if not np.isfinite(t0) or t0 <= 0:
        return 1, 1.0
    r = tMG / t0
    L = 0
    k = 1
    while k + 1 <= K:
        s = r[k] + r[k + 1]
        if s <= 0:
            break
        L = k + 1
        k += 2
    tau = 1.0 + 2.0 * float(np.sum(r[1: L + 1])) if L >= 1 else 1.0
    return max(L, 1), max(tau, 1.0)


def estimate(x, kind="bart", bmul=1.0, bmin=4.0, bcap=0.25, ret_diag=False,
             pilot="bart"):
    x = np.asarray(x, dtype=np.float64)
    T, d = x.shape
    if T < 4 or d < 1:
        H = np.atleast_2d(np.cov(x.T, bias=True)) * max(T, 1)
        return (H, {}) if ret_diag else H
    K = T - 1
    G = acov_lags(x, K)
    bmax = max(bmin + 1.0, bcap * T)
    diag = {}

    # ---- pilot: metric from lag-0 covariance, IPS cutoff, Bartlett window ---
    G0 = 0.5 * (G[0] + G[0].T)
    try:
        ev, U = np.linalg.eigh(G0)
        ev = np.maximum(ev, 1e-10 * max(ev[-1], 1e-30))
        M0 = (U / ev) @ U.T
    except np.linalg.LinAlgError:
        M0 = np.eye(d) / max(np.trace(G0) / d, 1e-30)
    L0, tau0 = _ips_lag(G, T, d, M0)
    b0 = float(np.clip(max(2.0 * tau0, 4.0), bmin, bmax))
    H0 = windowed(G, b0, pilot)
    H0 = _clip_psd(H0)
    _, M1, _ = _psd_inv_sqrt(H0)   # M1 = H0^{-1}  (metric proxy for S^{-1})
    L1, tau1 = _ips_lag(G, T, d, M1)
    diag["pilot"] = (b0, tau0, tau1)

    # ---- metric-proxied MSE-optimal bandwidth -----------------------------
    Lc = int(min(K, max(20, int(np.ceil(3.0 * max(tau1, 1.0))), 2 * L1)))
    b, dg = _bandwidth(G, T, d, kind, M1, Lc, bmul, bmin, bmax)
    tau = max(tau1, 1.0)
    lo = max(4.0, bmin, 0.5 * tau)
    hi = min(bmax, max(10.0, 6.0 * tau))
    b = float(np.clip(b, lo, hi)) if hi > lo else float(min(bmax, max(bmin, tau)))
    dg["b"] = b; dg["tau"] = tau; dg["lo"] = lo; dg["hi"] = hi
    diag.update(dg)
    H = windowed(G, b, kind)
    H = _clip_psd(H)
    if not np.all(np.isfinite(H)):
        H = H0
    return (H, diag) if ret_diag else H


# ---------------------------------------------------------- comparators -----
def est_fixed(x, kind="bart", c=1.0):
    x = np.asarray(x, float)
    T, d = x.shape
    G = acov_lags(x, min(T - 1, 20000))
    b = max(2.0, c * np.sqrt(T))
    return _clip_psd(windowed(G, b, kind))


def est_mise(x):
    """Multivariate initial monotone sequence estimator (Vats-Flegal-Jones 2019)."""
    x = np.asarray(x, float)
    T, d = x.shape
    b = int(np.floor(np.sqrt(T)))
    K = min(T - 1, b + 2)
    G = acov_lags(x, K)
    Lam = G[0].copy()
    best = Lam.copy()
    prev = np.linalg.eigvalsh(Lam)
    k = 1
    while k + 1 <= K:
        pair = (G[k] + G[k].T) + (G[k + 1] + G[k + 1].T)
        if np.linalg.eigvalsh(pair).min() <= 0:
            break
        Lam = Lam + pair
        ev = np.linalg.eigvalsh(Lam)
        if np.any(ev > prev + 1e-12):
            break
        prev = ev
        best = Lam
        k += 2
    return _clip_psd(best)


def est_ips(x, kind="bart"):
    """Initial positive sequence, then smooth kernel at that truncation."""
    x = np.asarray(x, float)
    T, d = x.shape
    K = min(T - 1, 20000)
    G = acov_lags(x, K)
    G0 = 0.5 * (G[0] + G[0].T)
    ev, U = np.linalg.eigh(G0)
    M = (U / np.maximum(ev, 1e-12 * max(ev[-1], 1e-30))) @ U.T
    L, tau = _ips_lag(G, T, d, M)
    return _clip_psd(windowed(G, max(L, 2.0) * 1.0, kind)), tau


def est_var1(x):
    """Parametric VAR(1) (Yule-Walker) plug-in."""
    x = np.asarray(x, float)
    T, d = x.shape
    G = acov_lags(x, min(T - 1, 4000))
    A = np.linalg.solve(0.5 * (G[0] + G[0].T), G[1]).T     # A = G1 G0^-1
    S = G[0].copy()
    Ak = np.eye(d)
    tot = G[0].copy()
    for k in range(1, 3000):
        Ak = Ak @ A
        tot = tot + (Ak @ S) + (Ak @ S).T
        if np.max(np.abs(Ak)) < 1e-12:
            break
    return _clip_psd(tot)
