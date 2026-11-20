"""An optional reduced transition model for reversible finite chains."""

import numpy as np


def pooled_model(counts, diagonals, freq, states, n):
    """Fit a common proposal rate on the observed graph if supported."""
    m, d = states.shape
    i, j = np.nonzero(counts + counts.T)
    if m < 3 or m > 128 or len(i) < 2*m-2:
        return None
    if np.min(freq) * n < 3:
        return None
    edge_obs = counts[i, j]
    edge_total = edge_obs + counts[j, i]
    if np.quantile(edge_total, 0.15) < 4:
        return None
    deg = np.bincount(i, minlength=m).astype(float)
    if np.min(deg) < 1:
        return None

    flow = 0.5 * edge_total / (n-1)
    # Composite likelihood of the unrestricted reversible fit.
    ll = np.sum(edge_obs * np.log(flow / freq[i]))
    dpos = diagonals > 0
    ll += np.sum(diagonals[dpos] * np.log(diagonals[dpos] / ((n-1)*freq[dpos])))
    df = max(1, len(i)//2 - 1)
    log_freq = np.log(freq)
    log_freq -= log_freq.mean()
    from scipy.optimize import minimize
    from scipy.special import expit, softmax

    best = None
    temperature = 0.023
    for expo in (0.0, 1.0) if deg.max() > 1.2*deg.min() else (0.0,):
        ld = expo * np.log(deg)
        qmax = 0.999999 / np.max(deg * np.exp(-ld))

        def likelihood(theta, grad=True):
            rate = expit(theta[-1])
            logq = np.log(max(qmax * rate, 1e-100))
            rat = (theta[j] - theta[i] + ld[i] - ld[j]) / temperature
            lp = logq - ld[i] - temperature * np.logaddexp(0, -rat)
            t = np.exp(lp)
            out = np.bincount(i, weights=t, minlength=m)
            stay = np.maximum(1 - out, 1e-12)
            loss = -(np.dot(edge_obs, lp) + np.dot(diagonals, np.log(stay)))
            if not grad:
                return loss, t
            v = edge_obs - t * (diagonals / stay)[i]
            r = v * expit(-rat)
            g = np.bincount(i, weights=r, minlength=m) - np.bincount(j, weights=r, minlength=m)
            return loss / n, np.r_[g, -v.sum()*(1-rate)] / n

        x0 = np.r_[log_freq, 0.0]
        # The common mechanism must predict the observed traffic even before
        # we allow it to refine the stationary frequencies.
        for _ in range(8):
            f, g = likelihood(x0)
            eps = 0.002
            lead = x0.copy()
            lead[-1] += eps
            curvature = (likelihood(lead)[1][-1] - g[-1]) / eps
            x0[-1] -= np.clip(g[-1] / max(curvature, 0.01), -1.5, 1.5)
        f0, _ = likelihood(x0)
        if 2*(ll + f0*n) > 3.25*df + 12 + 2*m:
            continue
        fit = minimize(likelihood, x0, jac=True, method="L-BFGS-B",
                       bounds=[(-24, 24)]*m + [(-18, 15)],
                       options={"maxiter": 80, "ftol": 2e-11, "gtol": 1e-7})
        f, p = likelihood(fit.x, grad=False)
        deviance = 2*(ll + f)
        if best is None or deviance < best[0]:
            best = deviance, softmax(fit.x[:-1]), p
    if best is None:
        return None

    dev, pi, p = best
    excess = max(0, dev - df)
    weight = 0.83 * np.exp(-(excess/max(0.50*df, 5.0))**2)
    if weight < 0.02:
        return None
    mat = np.zeros((m, m))
    mat[i, j] = p * pi[i]
    mat = (mat + mat.T) * 0.5
    degree = mat.sum(axis=1)
    mat = -mat
    mat[np.diag_indices(m)] = degree
    mat += np.outer(pi, pi)
    x = states - pi @ states
    f = pi[:, None] * x
    h = f.T @ (2 * np.linalg.solve(mat, f) - x)
    return weight, (h + h.T) * 0.5
