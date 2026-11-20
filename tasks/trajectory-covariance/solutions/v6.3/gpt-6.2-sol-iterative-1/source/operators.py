"""Reversible transition estimates based on the observed state."""

import numpy as np

from tails import residual_correction
from rates import clustered_time


def sym(a):
    return (a.T + a) * 0.5


def positive(a):
    e, u = np.linalg.eigh(sym(a))
    return (u * np.maximum(e, 0.0)) @ u.T


def solve_chol(l, b):
    return np.linalg.solve(l.T, np.linalg.solve(l, b))


def reversible_evidence(y, l, values, vectors):
    """Approximate area-statistic test for a skew part of the lag matrix."""
    n, d = y.shape
    q = np.linalg.solve(l, np.eye(d)).T @ vectors
    a = y[:-1].T @ y[1:] / (n - 1)
    k = q.T @ (a - a.T) @ q
    i, j = np.triu_indices(d, 1)
    r = np.clip(1.0 - values, -0.99999, 0.99999)
    v = 2 * (1 - r[i]**2) * (1 - r[j]**2) / (1 - r[i] * r[j])
    return np.sum(k[i, j]**2 / np.maximum(v, 1e-10)) * (n - 1)


def extra_features(y):
    """Linear-space-invariant degree-two features and radial cubic terms."""
    d = y.shape[1]
    radius = np.einsum("ij,ij->i", y, y)
    cutoff = max(d * 2.0, np.quantile(radius[::4], 0.98))
    bounded = y * np.minimum(1.0, np.sqrt(cutoff / np.maximum(radius, 1e-40)))[:, None]
    i, j = np.triu_indices(d)
    return (bounded[:, i] * bounded[:, j],
            bounded * (np.minimum(radius, cutoff) - d)[:, None])


def pooled_time(y, w, a, t, repeats):
    """Estimate a common rate under a fitted Gaussian random-walk model."""
    center = t.mean()
    n, d = y.shape
    if d < 2 or not 0.13 < repeats < 0.95:
        return center, None
    accept = 1 - repeats
    if (n - 1) * accept < 40:
        return center, None
    from scipy.special import stdtr, stdtrit

    jump_scale = -2 * stdtrit(d, accept / 2) / np.sqrt(d)
    a2 = 2 * stdtr(d + 2, -jump_scale * np.sqrt(d + 2) / 2)
    a4 = 2 * stdtr(d + 4, -jump_scale * np.sqrt(d + 4) / 2)
    model = 2 / (jump_scale**2 * a2)
    delta = w[np.any(w != 0, axis=1)]
    jump = np.einsum("ij,ij->i", delta, np.linalg.solve(a, delta.T).T)
    jump_radial = np.mean(jump**2) / np.mean(jump)**2
    radial_model = (d + 2) / d * a4 * accept / a2**2
    radius = np.einsum("ij,ij->i", y, y)
    target_radial = np.mean(radius**2) / np.mean(radius)**2
    tolerance = 0.07 + np.sqrt(3 * center / (d * n))
    z = (np.log(model / center) / tolerance)**2
    z += ((jump_radial - radial_model) / (0.05 + 1.7 / np.sqrt(len(jump))))**2
    z += ((target_radial - 1 - 2 / d) / 0.15)**2
    mix = 0.90 * np.exp(-0.5 * z)
    return center * np.exp(mix * np.log(model / center)), (model, jump_scale, mix)


def poisson_fit(y, repeats, nonlinear=True):
    """Return an estimate and its Poisson filter, or None for directed data."""
    n, d = y.shape
    nn = n - 1
    y = y - (y[:-1].sum(axis=0) + y[1:].sum(axis=0)) / (2 * nn)
    w = np.diff(y, axis=0)
    c = sym((y.T @ y - 0.5 * np.outer(y[0], y[0]) - 0.5 * np.outer(y[-1], y[-1])) / nn)
    a = sym(w.T @ w / (2 * nn))
    l = np.linalg.cholesky(c)
    il = np.linalg.solve(l, np.eye(d))
    r, v = np.linalg.eigh(sym(il @ a @ il.T))
    r = np.clip(r, 2.0 / n, 2 - 1e-10)
    dof = d * (d - 1) / 2
    area_test = reversible_evidence(y, l, r, v)
    if area_test > (dof + 4.0*np.sqrt(dof * 2) + 3.0) * (1 + 0.6 * repeats):
        return None

    tau = (2 - r) / r
    # Pool the relaxation rates when their spread is consistent with a
    # common rate. Their sampling spread increases for slow dynamics.
    center = np.mean(r)
    shape_dof = d * (d + 1) / 2 - 1
    iso_noise = 1 + 1.5 * repeats
    iso_test = nn * np.sum((r - center)**2) / (center * (2 - center) * iso_noise)
    iso = np.clip((1.7 * shape_dof - iso_test) / max(0.9 * shape_dof, 1), 0.0, 1.0)
    t = 1 / r
    pooled, model = pooled_time(y, w, a, t, repeats) if iso > 0 else (t.mean(), None)
    if iso < 0.65:
        clustered = clustered_time(r, nn, iso_noise)
        t = t + (0.78 * (1-iso)) * (clustered - t)
    rfit = 1 / ((1 - iso) * t + iso * pooled)
    # For nearly white data, an uninformative transition adds pure noise.
    iid_test = nn * np.sum((1 - r)**2)
    iid = max(0.0, 1 - max(shape_dof - 2, 0) / max(iid_test, 1e-20))
    if iid_test < 5 * (shape_dof + 1):
        rfit = 1 + iid * (rfit - 1)

    lv = l @ v
    ac = np.linalg.solve(a, c)
    ef = 1 + (2 * (1 - rfit) / rfit**2) * r
    mr = (2 - rfit) / rfit
    shape_weight = 1 - 0.17 * repeats
    phi = shape_weight * ef + (1 - shape_weight) * mr
    if nonlinear:
        # Temper the uncertain amplification from slow fitted directions.
        u = tau / n
        shrink = 1 / (1 + (1-iso) * (0.30 * (d+2) * u + 0.17 * np.sum(u)))
        phi *= shrink
    h0 = (lv * phi) @ lv.T
    b0 = (il.T @ v / rfit) @ lv.T
    g = y @ b0
    if not nonlinear:
        return sym(h0), g, tau
    base = h0.copy()
    # Build the extra features successively and use the Schur complement of
    # the Dirichlet matrix. Variance across blocks guards against overfitting.
    phi = y
    diff = w
    mat = a
    coef = ac
    nblocks = min(14, max(6, int(n / (5 * tau.max()))))
    for block in extra_features(y):
        block -= (block[:-1].mean(0) + block[1:].mean(0)) * 0.5
        dw = np.diff(block, axis=0)
        b = diff.T @ dw / (2 * nn)
        e = dw.T @ dw / (2 * nn)
        lmat = np.linalg.cholesky(mat + 1e-10 * np.eye(len(mat)))
        beta = solve_chol(lmat, b)
        schur = sym(e - b.T @ beta)
        ridge = max(np.trace(e) * 1e-8 / len(e), 1e-13)
        le = np.linalg.cholesky(schur + ridge * np.eye(len(e)))
        f = (block.T @ y - 0.5 * (np.outer(block[0], y[0]) +
                                  np.outer(block[-1], y[-1]))) / nn
        q = f - b.T @ coef
        u = np.linalg.solve(le, q)
        tsq = 2 * u.T @ u
        rphi = block - phi @ beta
        rd = dw - diff @ beta
        dg = diff @ coef
        qblocks = []
        for i in range(nblocks):
            j, k = (i * nn) // nblocks, ((i + 1) * nn) // nblocks
            cb = (rphi[j:k].T @ y[j:k] + rphi[j + 1:k + 1].T @ y[j + 1:k + 1]) / 2
            db = rd[j:k].T @ dg[j:k] / 2
            qblocks.append(np.linalg.solve(le, (cb - db) / (k - j)) - u)
        qu = np.concatenate(qblocks)
        noise = 2 * (qu.T @ qu) / (nblocks * (nblocks - 1))
        noise *= min(2.0, 1.04 + 0.9 * tau.max() * nblocks / n)
        lb = np.linalg.cholesky(base + 1e-10*np.eye(d))
        ib = np.linalg.solve(lb, np.eye(d))
        total = np.trace(ib @ tsq @ ib.T)
        stochastic = np.trace(ib @ noise @ ib.T)
        keep = np.clip(1 - stochastic / max(total, 1e-20), 0.0, 1.0)
        if keep > 0.0:
            cor = lb @ positive(ib @ (tsq - noise) @ ib.T) @ lb.T
            h0 += (0.4 + 0.6 * keep) * cor
            g += (rphi @ np.linalg.solve(le.T, u)) * np.sqrt(keep)
        coef = np.r_[coef - beta @ solve_chol(le, q), solve_chol(le, q)]
        phi = np.column_stack((phi, block))
        diff = np.column_stack((diff, dw))
        mat = np.block([[mat, b], [b.T, e]])
    if iso < 0.45:
        h0 = residual_correction(y, g, h0, np.max(tau))
    if model is not None and iso > 0.25 and model[2] > 0.34:
        from rwgauss import conditional_gaussian

        try:
            fit = conditional_gaussian(y, a, model[0], model[1])
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            fit = None
        if fit is not None:
            alternative, fit_quality = fit
            weight = 0.90 * iso**1.4 * model[2] * fit_quality
            h0 = (1 - weight) * h0 + weight * alternative
    return sym(h0), g, tau
