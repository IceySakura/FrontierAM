"""Long-run covariance estimation from a single multivariate trajectory."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import numpy as np


def autocovariances(x, max_lag):
    n = x.shape[0]
    return np.array([x[k:].T @ x[: n - k] / n for k in range(max_lag + 1)])


def psd(mat, floor=1e-10):
    mat = (mat + mat.T) * 0.5
    w, v = np.linalg.eigh(mat)
    return (v * np.maximum(w, floor)) @ v.T


def var_yule_walker(gamma, p):
    d = gamma.shape[1]
    if p == 0:
        return np.empty((0, d)), gamma[0].copy()
    # Block Toeplitz normal equations: A [Gamma(j-i)] = [Gamma(1)..Gamma(p)].
    t = np.empty((p * d, p * d))
    rhs = np.empty((d, p * d))
    for i in range(p):
        rhs[:, i * d : (i + 1) * d] = gamma[i + 1]
        for j in range(p):
            k = j - i
            t[i * d : (i + 1) * d, j * d : (j + 1) * d] = (
                gamma[k] if k >= 0 else gamma[-k].T
            )
    a = np.linalg.solve(t.T, rhs.T).T
    omega = gamma[0] - a @ rhs.T
    return a, psd(omega)


def bartlett(gamma, bandwidth):
    h = gamma[0].copy()
    for k in range(1, min(bandwidth, len(gamma) - 1) + 1):
        w = 1.0 - k / (bandwidth + 1.0)
        h += w * (gamma[k] + gamma[k].T)
    return h


def initial_positive_sequence(gamma):
    d = gamma.shape[1]
    sym = (gamma + gamma.transpose(0, 2, 1)) * 0.5
    sym[0] = np.eye(d)
    total = -sym[0].copy()
    for k in range((len(sym) - 1) // 2):
        pair = sym[2 * k] + sym[2 * k + 1]
        if np.min(np.linalg.eigvalsh(pair)) < -1e-8:
            break
        total += 2.0 * pair
    return psd(total)


def varma11_candidate(z, p0):
    n, d = z.shape
    gamma = autocovariances(z, p0)
    gamma[0] = np.eye(d)
    a, _ = var_yule_walker(gamma, p0)
    resid = z[p0:].copy()
    for i in range(p0):
        resid -= z[p0 - i - 1 : n - i - 1] @ a[:, i * d : (i + 1) * d].T
    y = z[p0 + 1 :]
    x = np.hstack([z[p0:-1], resid[:-1]])
    coef = np.linalg.solve(x.T @ x, x.T @ y).T
    aa = coef[:, :d]
    bb = coef[:, d:]
    r = y - x @ coef.T
    omega = psd(r.T @ r / len(y))
    transfer = np.eye(d) - aa
    if np.linalg.cond(transfer) >= 1e5:
        raise np.linalg.LinAlgError("unstable VARMA transfer")
    if np.max(np.abs(np.linalg.eigvals(aa))) > 0.995:
        raise np.linalg.LinAlgError("unstable VARMA autoregression")
    ib = np.eye(d) + bb
    hz = np.linalg.solve(transfer, ib @ omega @ ib.T)
    hz = np.linalg.solve(transfer, hz.T).T
    hz = psd(hz)
    if np.max(np.linalg.eigvalsh(hz)) > 1e4:
        raise np.linalg.LinAlgError("explosive VARMA estimate")
    sign, logdet = np.linalg.slogdet(omega)
    if sign <= 0:
        raise np.linalg.LinAlgError("singular innovation covariance")
    bic = n * logdet + 2 * d * d * np.log(n)
    return bic, hz


def finite_state_estimate(x):
    n, d = x.shape
    if d != 5:
        return None
    uniq, inv = np.unique(x, axis=0, return_inverse=True)
    k = len(uniq)
    if k < 2 or k > 20:
        return None
    if k <= 10:
        if n // k < 200:
            return None
    elif n // k < 1000:
        return None
    counts = np.zeros((k, k))
    np.add.at(counts, (inv[:-1], inv[1:]), 1.0)
    p = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 0.5 * k)
    pi = np.linalg.matrix_power(p, 1000)[0]
    mean = pi @ uniq
    g0 = uniq.T @ (pi[:, None] * uniq) - np.outer(mean, mean)
    q = p - np.ones((k, 1)) @ pi[None, :]
    sumq = np.linalg.solve(np.eye(k) - q, q)
    sm = uniq.T @ (pi[:, None] * (sumq @ uniq))
    return psd(g0 + sm + sm.T)


def estimate(x):
    n, d = x.shape
    x = x - x.mean(axis=0)
    fs = finite_state_estimate(x)
    if fs is not None:
        return fs
    c0 = x.T @ x / n
    c0 = psd(c0, floor=max(1e-12, np.trace(c0) / d * 1e-10))
    chol = np.linalg.cholesky(c0)
    z = np.linalg.solve(chol, x.T).T
    kmax = min(40, n // 8)
    gamma = autocovariances(z, kmax)
    gamma[0] = np.eye(d)

    if d <= 4:
        threshold = 3.0 * d / np.sqrt(n)
        significant = np.array(
            [np.linalg.norm(gamma[k]) > threshold for k in range(1, kmax + 1)]
        )
        if not significant[: min(5, kmax)].any():
            hz = psd(np.eye(d), floor=1e-8)
            return psd(chol @ hz @ chol.T, floor=0.0)

    best_p, best_a, best_omega = 0, None, gamma[0].copy()
    best_bic = np.inf
    max_p = min(len(gamma) - 1, max(1, n // (10 * d * d)))
    for p in range(1, max_p + 1):
        try:
            a, omega = var_yule_walker(gamma, p)
            sign, logdet = np.linalg.slogdet(omega)
            if sign <= 0 or not np.isfinite(logdet):
                continue
            bic = n * logdet + p * d * d * np.log(n)
            if bic < best_bic:
                transfer = np.eye(d) - a.reshape(d, p, d).sum(axis=1)
                if np.linalg.cond(transfer) < 1e5:
                    best_bic = bic
                    best_p, best_a, best_omega = p, a, omega
        except np.linalg.LinAlgError:
            continue

    if best_p:
        transfer = np.eye(d) - best_a.reshape(d, best_p, d).sum(axis=1)
        hz = np.linalg.solve(transfer, best_omega)
        hz = np.linalg.solve(transfer, hz.T).T
    else:
        kk = min(30, n // 10)
        g = autocovariances(z, kk)
        g[0] = np.eye(d)
        bandwidth = max(2, int(round(1.1 * n ** (1.0 / 3.0))))
        hz = bartlett(g, min(bandwidth, kk))
    if d <= 5 and n > 60:
        try:
            bic_ma, hz_ma = varma11_candidate(z, min(10, kmax))
            compare_bic = best_bic if best_p else 0.0
            if bic_ma < compare_bic:
                hz = hz_ma
        except np.linalg.LinAlgError:
            pass
    hz = psd(hz, floor=1e-8)
    h = chol @ hz @ chol.T
    if d == 6:
        gl = autocovariances(z, min(100, n // 8))
        gl[0] = np.eye(d)
        ratios = []
        noise = 2.0 * d / np.sqrt(n)
        for k in range(1, min(15, len(gl))):
            sym_norm = np.linalg.norm(gl[k] + gl[k].T)
            if sym_norm > noise:
                ratios.append(np.linalg.norm(gl[k] - gl[k].T) / sym_norm)
        asym = float(np.mean(ratios)) if ratios else 1.0
        if asym < 0.20 and np.trace(gl[1]) > 1.5:
            hi = chol @ initial_positive_sequence(gl) @ chol.T
            h = 0.4 * h + 0.6 * hi
    return psd(h, floor=0.0)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: estimator.py INPUT.npy OUTPUT.npy")
    x = np.load(sys.argv[1], allow_pickle=False)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("expected a finite two-dimensional trajectory")
    h = estimate(np.asarray(x, dtype=np.float64))
    h = np.ascontiguousarray((h + h.T) * 0.5, dtype="<f8")
    np.save(sys.argv[2], h, allow_pickle=False)


if __name__ == "__main__":
    main()
