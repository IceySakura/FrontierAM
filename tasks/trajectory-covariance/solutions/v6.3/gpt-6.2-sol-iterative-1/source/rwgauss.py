"""Gaussian matched random-walk approximation fitted from move statistics."""

import numpy as np


def radial_tau(d, s):
    """Project the fitted Gaussian kernel onto radial coordinate polynomials."""
    from scipy.special import eval_genlaguerre, gamma, roots_genlaguerre, roots_legendre

    ar, wr = roots_genlaguerre(23, d/2-1)
    br, wb = roots_genlaguerre(17, (d-1)/2-1)
    mr, wm = roots_legendre(25)
    jump = s * np.sqrt(ar/2)[:, None, None]
    perp = br[None, :, None] * 2
    mid = 4.25 * (mr + 1)[None, None, :]
    x1 = mid - jump
    x2 = mid + jump
    q1 = (x1*x1 + perp)*0.5
    q2 = (x2*x2 + perp)*0.5
    pw = wr[:, None, None] * wb[None, :, None] * wm[None, None, :]
    pw *= (8.5 / (np.sqrt(2*np.pi)*gamma(d/2)*gamma((d-1)/2))) * np.exp(-0.5*x2*x2)
    pw = pw * (1 / (2*d))
    ap, bp = [], []
    for k in range(5):
        l = eval_genlaguerre(k, d/2, q1)
        r = eval_genlaguerre(k, d/2, q2)
        ap.append(((x2*r - x1*l) * np.sqrt(pw)).ravel())
        bp.append(((r - l) * np.sqrt(pw*perp)).ravel())
    ap, bp = np.asarray(ap), np.asarray(bp)
    mat = ap @ ap.T + bp @ bp.T
    c = np.eye(5)[0]
    return 2 * np.linalg.solve(mat, c)[0] - 1


def log_hold_spline(d, s, extent):
    from scipy.interpolate import CubicSpline
    from scipy.stats import ncx2

    grid = np.linspace(0, extent, 180)
    ss = s*s
    low = ncx2.cdf(grid/ss, d, grid/ss)
    high = ncx2.sf(grid*(1+ss)/ss, d, grid/(ss*(1+ss)))
    high *= np.exp((grid*ss/(1+ss) - d*np.log1p(ss)) / 2)
    aa = np.minimum(low + high, 1 - 1e-8)
    return CubicSpline(grid, np.log1p(-aa))


def conditional_gaussian(y, a, model_time, s):
    """Fit stationary shape from the likelihood of the observed transitions."""
    from scipy.optimize import minimize

    n, d = y.shape
    loc = np.r_[0, np.flatnonzero(np.any(y[1:] != y[:-1], axis=1)) + 1]
    stays = np.diff(np.r_[loc, n]).astype(float) - 1
    moves = len(loc) - 1
    if moves < 30:
        return None
    root = np.linalg.cholesky(a * model_time)
    z = np.ascontiguousarray(np.linalg.solve(root, y[loc].T).T)
    q = np.einsum("ij,ij->i", z, z)
    extent = min(380., max(50., 1.55*q.max()+12))
    spline = log_hold_spline(d, s, extent)
    dz = np.diff(z, axis=0)
    dp = dz.T @ dz / (2*s*s)
    step = np.r_[np.ones(moves), 0]
    exposure = stays + step
    groups = np.searchsorted(np.quantile(q, np.arange(1, 8)/8), q)
    pp = -np.expm1(spline(np.minimum(extent, q)))
    dd = np.bincount(groups, weights=step-exposure*pp, minlength=8)
    vv = np.bincount(groups, weights=exposure*pp*(1-pp), minlength=8)
    test = np.sum(dd*dd / np.maximum(vv, 1))
    w = np.exp(-max(0.0, test-8.) / 21.)
    if w < 0.13:
        return None
    i, j = np.tril_indices(d)
    jj = np.flatnonzero(i == j)
    eye = np.eye(d)
    ridge = 0.013*n
    eps = 0.036

    def likelihood(theta):
        r = np.zeros((d, d))
        vals = theta[d:].copy()
        vals[jj] = np.exp(vals[jj])
        r[i, j] = vals
        k = r @ r.T
        cen = z - theta[:d]
        tmp = cen @ r
        xx = np.einsum("ij,ij->i", tmp, tmp)
        dx = np.diff(xx)
        rad = np.sqrt(dx*dx + eps*eps)
        wx = np.minimum(extent, xx)
        logh = spline(wx)
        likelihood = moves * theta[d+jj].sum() - np.sum(k * dp)
        likelihood += stays @ logh - 0.25 * rad.sum()
        lack = k - eye
        likelihood -= 0.5*ridge*np.sum(lack*lack) + 0.025*ridge*np.sum(theta[:d]**2)

        hder = stays * spline(wx, 1) * (xx <= extent)
        ss = 0.25*dx/rad
        hder[:-1] += ss
        hder[1:] -= ss
        mgrad = -2 * (hder @ cen) @ k - 0.05 * ridge * theta[:d]
        grad = 0.5*moves*np.linalg.solve(k, eye) - dp + cen.T @ (hder[:, None] * cen)
        grad = grad - ridge*lack
        out = (2*grad @ r)[i, j]
        out[jj] *= vals[jj]
        return -likelihood/n, -np.r_[mgrad, out]/n

    start = np.zeros(d+len(i))
    opt = minimize(likelihood, start, jac=True, method="L-BFGS-B",
                   bounds=[(-1.4, 1.4)] * len(start),
                   options={"maxiter": 42, "ftol": 2e-10, "gtol": 8e-7})
    r = np.zeros((d,d))
    vals = opt.x[d:].copy()
    vals[jj] = np.exp(vals[jj])
    r[i,j] = vals
    rb = np.linalg.solve(r, root.T)
    model = rb.T @ rb * radial_tau(d, s)
    return model, w
