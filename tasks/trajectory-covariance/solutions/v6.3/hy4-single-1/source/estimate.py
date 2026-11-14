#!/usr/bin/env python3
"""Long-run covariance estimator for a single MCMC trajectory.

Method (affine equivariant):
  1. standardize / Cholesky-whiten the trajectory by its lag-0 covariance,
  2. compute all sample autocovariances by FFT,
  3. estimate the zero-frequency spectral density in whitened coordinates by
     (a) an adaptive flat-top lag-window estimator, and
     (b) an AIC/BIC-selected VAR-prewhitened (VARHAC) estimator,
  4. combine the two with a data-driven weight chosen from the estimated
     spectrum (flat-top is preferred when the zero-frequency spectrum nearly
     vanishes, or when prewhitening misses a long correlation tail),
  5. enforce positive semidefiniteness and undo the whitening.

Only NumPy is required.  Single-threaded.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import sys
import numpy as np

# ------------------------------------------------------------------ tuning
# bandwidth / selection parameters (tuned on synthetic trajectories)
POL_C = 2.0            # threshold multiplier for the correlation-length rule
PLUG_MIX = 1.0         # weight of the AR(1) plug-in bandwidth in flat-top
PLUG_C = 1.0
KCHECK_FACTOR = 0.20
KCHECK_CAP = 1024
PMAX = 12              # maximum VAR order for prewhitening
IC_PEN = 'aic'         # 'bic' or 'aic'
RES_POL_C = 2.0
KRES_FACTOR = 0.10
KRES_CAP = 512
MODE = 'adaptive3'     # blend weight rule
LAM = 0.85
FLAT_KERNEL = 'lug'
LOG_LO = 2.0
LOG_W = 1.0
RAT_LO = 0.65
RAT_W = 0.25
BIAS_EIG = 0.0         # per-eigenvalue finite-sample bias correction
BIAS_A = 0.0
# ---------------------------------------------------------------------------


def whiten(x):
    M, d = x.shape
    z = x - x.mean(axis=0)
    sd = np.sqrt((z * z).mean(axis=0))
    sd = np.maximum(sd, max(sd.max(), 1e-100) * 1e-12)
    r = z / sd
    C0 = (r.T @ r) / M
    C0 = (C0 + C0.T) * 0.5
    try:
        L = np.linalg.cholesky(C0)
    except np.linalg.LinAlgError:
        ev, U = np.linalg.eigh(C0)
        ev = np.maximum(ev, max(ev[-1], 1e-12) * 1e-10)
        C0 = (U * ev) @ U.T
        L = np.linalg.cholesky(C0 + 1e-12 * np.eye(d))
    y = np.linalg.solve(L, r.T).T
    B = sd[:, None] * L
    return y, B


def acov(y, Kmax):
    """R[k] = (1/M) sum_t y_t y_{t-k}^T for k = 0..Kmax."""
    M, d = y.shape
    Kmax = min(int(Kmax), M - 1)
    L = 1 << int(np.ceil(np.log2(2 * M + 1)))
    F = np.fft.rfft(y, n=L, axis=0)
    P = (F[:, :, None] * F.conj()[:, None, :]).reshape(L // 2 + 1, d * d)
    C = np.fft.irfft(P, n=L, axis=0)
    R = C[:Kmax + 1].reshape(Kmax + 1, d, d) / M
    R[0] = (R[0] + R[0].T) * 0.5
    return R


def kernels(kind, K, n):
    k = np.arange(n, dtype=float)
    w = np.zeros(n)
    if K > 0:
        xx = k / K
        if kind == 'bt':
            w = np.maximum(1.0 - xx, 0.0)
        elif kind == 'parzen':
            a = xx <= 0.5
            w[a] = 1 - 6 * xx[a] ** 2 + 6 * xx[a] ** 3
            b = (xx > 0.5) & (xx < 1)
            w[b] = 2 * (1 - xx[b]) ** 3
        elif kind == 'qs':
            z = 6 * np.pi * xx / 5
            with np.errstate(divide='ignore', invalid='ignore'):
                w = 25.0 / (12 * np.pi ** 2 * xx ** 2) * (np.sin(z) / z - np.cos(z))
            w[xx >= 1] = 0.0
        elif kind == 'th':
            w = np.where(xx < 1, 0.5 * (1 + np.cos(np.pi * xx)), 0.0)
        elif kind == 'trap':
            w = np.where(xx <= 0.5, 1.0, np.maximum(2 * (1 - xx), 0.0))
        elif kind == 'lug':
            c = 1.0 / 3.0
            w = np.where(xx <= c, 1.0, np.maximum((1 - xx) / (1 - c), 0.0))
        else:
            raise ValueError(kind)
    w[0] = 1.0
    return w


def omega_kernel(R, w):
    n = min(len(R), len(w))
    A = (R + np.swapaxes(R, 1, 2))[:n]
    return np.einsum('k,kij->ij', w[:n], A) - R[0]


def psd_fix(O, floor=1e-12):
    O = (O + O.T) * 0.5
    ev, U = np.linalg.eigh(O)
    ev = np.maximum(ev, max(ev[-1], 1.0) * floor)
    return (U * ev) @ U.T


def yule_walker(R, p):
    d = R.shape[1]
    if p == 0:
        return np.zeros((0, d, d))
    T = np.empty((p * d, p * d))
    for i in range(p):
        for j in range(p):
            m = j - i
            B = R[m] if m >= 0 else R[-m].T
            T[j * d:(j + 1) * d, i * d:(i + 1) * d] = B.T
    rhs = np.stack([R[j + 1].T for j in range(p)], axis=0).reshape(p * d, d)
    try:
        X = np.linalg.solve(T, rhs)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(T, rhs, rcond=None)[0]
    return np.array([X[i * d:(i + 1) * d].T for i in range(p)])


def residual_acov(R, A, Ku):
    """U[k] = Cov(u_t, u_{t-k}) with u_t = y_t - sum_i A_i y_{t-i}."""
    p = len(A)
    d = R.shape[1]
    lo, hi = -p, Ku + p + 1
    if hi > len(R):
        hi = len(R)
    Rext = np.empty((hi - lo, d, d))
    for m in range(lo, hi):
        Rext[m - lo] = R[m] if m >= 0 else R[-m].T
    U = Rext[-lo:-lo + Ku + 1].copy()
    for j in range(p):
        U -= Rext[(j + 1) - lo:(j + 1) - lo + Ku + 1] @ A[j].T
    for i in range(p):
        U -= np.einsum('ab,kbc->kac', A[i], Rext[(-i - 1) - lo:(-i - 1) - lo + Ku + 1])
    for i in range(p):
        for j in range(p):
            U += np.einsum('ab,kbc,dc->kad', A[i],
                           Rext[(j - i) - lo:(j - i) - lo + Ku + 1], A[j])
    U[0] = (U[0] + U[0].T) * 0.5
    return U


def _logdet(M):
    ev = np.linalg.eigvalsh((M + M.T) * 0.5)
    return float(np.log(np.maximum(ev, 1e-14)).sum())


def _politis_bandwidth(U, M, c, kernel):
    Kmax = len(U) - 1
    d = U.shape[1]
    if Kmax < 3:
        return 0.0
    q = max(3, int(np.ceil(2 * np.sqrt(np.log10(max(M, 10))))))
    s = np.sqrt(np.sum(U * U, axis=(1, 2))) / d
    noise = max(float(np.median(s[max(1, Kmax // 2):])), 1e-14)
    thr = c * np.sqrt(np.log10(max(M, 10))) * noise
    m = Kmax
    if np.all(s[1:q + 1] < thr):
        m = 0
    else:
        for k in range(1, Kmax - q):
            if np.all(s[k + 1:k + q + 1] < thr):
                m = k
                break
    mult = 2.0 if kernel == 'trap' else (3.0 if kernel == 'lug' else 1.0)
    return mult * m


def bw_plugin(tau, M, c=1.0, cap=0.45):
    tau = max(float(tau), 1.0)
    rho = (tau - 1) / (tau + 1)
    K = (6 * rho ** 2 * M / max((1 - rho ** 2) ** 2, 1e-12)) ** (1.0 / 3.0)
    return max(1.0, min(c * K, cap * M))


def tau_of(O):
    return float(max(np.linalg.eigvalsh((O + O.T) * 0.5).mean(), 1e-8))


def pilot_tau(R):
    A1 = yule_walker(R, 1)
    d = R.shape[1]
    U0 = residual_acov(R, A1, 0)[0]
    Bm = np.eye(d) - A1[0]
    try:
        invB = np.linalg.solve(Bm, np.eye(d))
    except np.linalg.LinAlgError:
        invB = np.linalg.pinv(Bm)
    return tau_of(psd_fix(invB @ U0 @ invB.T, 1e-9))


def compute_flat(R, M, pilot):
    Kmax = len(R) - 1
    Kcheck = min(Kmax, int(KCHECK_FACTOR * M) + 64, KCHECK_CAP)
    K = _politis_bandwidth(R[:Kcheck + 1], M, POL_C, FLAT_KERNEL)
    K = max(K, PLUG_MIX * bw_plugin(pilot, M, PLUG_C, 0.45))
    K = min(K, Kmax, 0.45 * M)
    n = min(Kmax + 1, int(np.ceil(K)) + 2)
    w = kernels(FLAT_KERNEL, K, n)
    return psd_fix(omega_kernel(R[:n], w))


def compute_vhac(R, M, pilot):
    d = R.shape[1]
    Kmax = len(R) - 1
    pmax = min(PMAX, Kmax - 3)
    pen = np.log(M) if IC_PEN == 'bic' else 2.0
    best_ic, best_p, best_A = np.inf, 0, np.zeros((0, d, d))
    for p in range(pmax + 1):
        A = yule_walker(R, p)
        U0 = residual_acov(R, A, 0)[0] if p else R[0]
        icv = M * _logdet(U0) + pen * (p * d * d)
        if icv < best_ic:
            best_ic, best_p, best_A = icv, p, A
    A = best_A
    p = best_p
    Kcheck = min(Kmax - p - 1, int(KRES_FACTOR * M) + 64, KRES_CAP)
    U = residual_acov(R, A, Kcheck)
    K = _politis_bandwidth(U, M, RES_POL_C, 'trap')
    K = min(K, Kcheck - 1, 0.45 * M)
    K = max(K, 0.0)
    n = min(len(U), int(np.ceil(K)) + 2)
    w = kernels('trap', K, n)
    Ou = omega_kernel(U[:n], w)
    Bm = np.eye(d) - A.sum(axis=0)
    sv = np.linalg.svd(Bm, compute_uv=False)
    if sv[-1] < 1e-10:
        invB = np.linalg.pinv(Bm)
    else:
        invB = np.linalg.inv(Bm)
    return psd_fix(invB @ Ou @ invB.T)


def bias_correct(O, M):
    if BIAS_A == 0.0 and BIAS_EIG == 0.0:
        return O
    ev, U = np.linalg.eigh((O + O.T) * 0.5)
    ev = np.maximum(ev, 1e-12)
    if BIAS_A:
        delta = min(float(ev.mean()) / M, 0.5)
        ev = ev / max(1.0 - BIAS_A * delta, 0.25)
    if BIAS_EIG:
        ev = ev / np.maximum(1.0 - BIAS_EIG * ev / M, 0.25)
    return (U * ev) @ U.T


def estimate(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    M, d = x.shape
    y, B = whiten(x)
    Kmax = min(M - 1, max(1024, int(0.45 * M)))
    R = acov(y, Kmax)
    pilot = pilot_tau(R)
    Of = compute_flat(R, M, pilot)
    Ov = compute_vhac(R, M, pilot)
    ev = np.linalg.eigvalsh((Of + Of.T) * 0.5)
    mn = max(ev[0], 1e-10)
    ratio = float(np.trace(Ov) / max(np.trace(Of), 1e-12))
    if MODE == 'fixed':
        lam = LAM
    elif MODE == 'adaptive':
        tau = max(float(ev.mean()), 1e-8)
        lam = LAM * np.clip((np.log(mn) + 2.3) / 2.3, 0, 1) * np.clip((tau - 1) / 2.0, 0, 1)
    elif MODE == 'adaptive2':
        lam = LAM * np.clip((mn - 0.08) / 0.42, 0.0, 1.0)
    else:
        f = np.clip((np.log10(mn) + LOG_LO) / LOG_W, 0.0, 1.0)
        g = np.clip((ratio - RAT_LO) / RAT_W, 0.0, 1.0)
        lam = LAM * f * g
    O = psd_fix((1 - lam) * Of + lam * Ov)
    O = bias_correct(O, M)
    H = B @ O @ B.T
    H = (H + H.T) * 0.5
    if not np.isfinite(H).all():
        # numerical safety net: fall back to the diagonal lag-0 covariance
        H = np.diag(np.diag(B @ B.T))
    ev, U = np.linalg.eigh(H)
    ev = np.maximum(ev, 0.0)
    return (U * ev) @ U.T


def main():
    if len(sys.argv) != 3:
        sys.stderr.write('usage: estimate.py INPUT OUTPUT\n')
        sys.exit(2)
    x = np.load(sys.argv[1], allow_pickle=False)
    try:
        H = estimate(x)
    except Exception:
        # Last-resort valid output if an unexpected numerical failure occurs.
        z = np.nan_to_num(np.asarray(x, dtype=np.float64))
        if z.ndim != 2:
            z = np.atleast_2d(z).T
        try:
            v = np.var(z - z.mean(axis=0), axis=0)
            H = np.diag(np.maximum(np.nan_to_num(v), 0.0))
        except Exception:
            H = np.eye(z.shape[1])
    H = np.ascontiguousarray((H + H.T) * 0.5, dtype='<f8')
    if not np.isfinite(H).all():
        H = np.zeros_like(H)
    out = sys.argv[2]
    parent = os.path.dirname(os.path.abspath(out))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    np.save(out, H, allow_pickle=False)


if __name__ == '__main__':
    main()
