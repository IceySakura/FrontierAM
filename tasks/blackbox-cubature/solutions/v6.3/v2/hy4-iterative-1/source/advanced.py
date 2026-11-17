"""Structure-aware cubature: line samples + exactly integrable surrogate features.

Development module.  solve(oracle) uses only oracle.values(points) and returns
the integral estimate over the unit cube (bounds scaling lives in the wrapper).
"""
from __future__ import annotations
import math, itertools
import numpy as np
from scipy.stats import qmc
from scipy.optimize import least_squares, minimize_scalar
from numpy.polynomial.legendre import leggauss

EPS = 1e-300


def sobol(d, n, seed):
    return qmc.Sobol(d=d, scramble=True, seed=int(seed) & 0x7fffffff).random(n)


# --------------------------------------------------------------------------- 1D shape machinery
def _gl(n=24):
    z, w = leggauss(n)
    return (z + 1) / 2, w / 2


GLZ, GLW = _gl(28)


def graded_integral(logg, u, scale, lo=0.0, hi=1.0):
    """Integral of exp(logg(t)) over [lo,hi] with geometric grading around u."""
    u = min(max(float(u), lo), hi)
    delta = min(max(float(scale), 1e-12), 0.25)
    pts = {lo, hi, u}
    for side in (-1, 1):
        for k in range(0, 96):
            t = u + side * delta * (2.0 ** (k / 2.0))
            if t < lo or t > hi:
                break
            pts.add(t)
    pts = sorted(pts)
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= a:
            continue
        t = a + (b - a) * GLZ
        vals = np.exp(np.clip(logg(t), -700, 700))
        total += (b - a) * float(GLW @ vals)
    return total


class ShapeFit:
    """1D positive profile fit by variable projection.

    log g(t) is one of four shapes: Gaussian, Laplace, generalized Cauchy
    (product peaks) or cusp.  Linear parameters are profiled out, leaving at
    most two nonlinear parameters, which makes the fit fast and reliable.
    """
    NAMES = ('gauss', 'laplace', 'cauchy', 'cusp')

    def __init__(self, t, g):
        t = np.asarray(t, float)
        g = np.asarray(g, float)
        mx = float(np.max(g)) if g.size else 0.0
        keep = g > max(mx * 1e-11, 1e-300)
        self.t, self.g = t[keep], g[keep]
        self.lg = np.log(self.g)
        self.kind = None
        self.err = float('inf')
        self.scale = 1e-3
        self.u = 0.5
        self.integral = 0.0
        self.ok = len(self.t) >= 6
        self.logg = None

    def fit(self):
        if not self.ok:
            return self
        t, lg = self.t, self.lg
        j = int(np.argmax(lg))
        u0 = float(np.clip(t[j], 0.0, 1.0))
        b0 = float(lg[j])
        # rough width: where log profile has dropped by one unit
        drop = np.abs(t - u0)[lg < b0 - 1.0]
        w = float(np.min(drop)) if drop.size else 0.05
        w = min(max(w, 1e-6), 0.5)
        best = None

        def consider(kind, logg, u, scale, err):
            nonlocal best
            if not np.isfinite(err) or err >= self.err:
                return
            self.err = float(err)
            best = (kind, logg, float(u), float(scale))

        # Gaussian: quadratic in t (closed form)
        V = np.column_stack([np.ones_like(t), t, t * t])
        beta, *_ = np.linalg.lstsq(V, lg, rcond=None)
        lam = -beta[2]
        if lam > 0:
            u = beta[1] / (2 * lam)
            b = beta[0] + lam * u * u
            logg = lambda x, b=b, lam=lam, u=u: b - lam * (np.asarray(x, float) - u) ** 2
            err = float(np.sqrt(np.mean((V @ beta - lg) ** 2)))
            consider('gauss', logg, u, 1.0 / math.sqrt(lam) / 20, err)
        # Laplace: profile (b, lam) for candidate kink positions
        def lap_err(u):
            A = np.column_stack([np.ones_like(t), -np.abs(t - u)])
            bt, *_ = np.linalg.lstsq(A, lg, rcond=None)
            if bt[1] <= 0:
                return 1e6, None, None
            return float(np.sqrt(np.mean((A @ bt - lg) ** 2))), bt, A
        us = np.linspace(max(-0.25, float(t.min()) - 0.1), min(1.25, float(t.max()) + 0.1), 41)
        vals = [lap_err(u)[0] for u in us]
        k0 = int(np.argmin(vals))
        lo = us[max(0, k0 - 1)]; hi = us[min(len(us) - 1, k0 + 1)]
        try:
            rr = minimize_scalar(lambda u: lap_err(u)[0], bounds=(lo, hi), method='bounded',
                                 options={'xatol': 1e-13})
            uu, ee = float(rr.x), float(rr.fun)
        except Exception:
            uu, ee = float(us[k0]), float(vals[k0])
        e2, bt, A = lap_err(uu)
        if bt is not None and bt[1] > 0:
            b, lam = float(bt[0]), float(bt[1])
            logg = lambda x, b=b, lam=lam, u=uu: b - lam * np.abs(np.asarray(x, float) - u)
            consider('laplace', logg, uu, 1.0 / lam / 20, min(ee, e2))
        # generalized Cauchy / cusp: profile (b, p), optimize (u, log eps)
        for kind in ('cauchy', 'cusp'):
            def resid(q, kind=kind):
                u, le = q
                eps = math.exp(le)
                s = (t - u) ** 2 + eps if kind == 'cauchy' else np.abs(t - u) + eps
                B = np.column_stack([np.ones_like(t), -np.log(s)])
                bt, *_ = np.linalg.lstsq(B, lg, rcond=None)
                r = B @ bt - lg
                if bt[1] <= 0:
                    r = r + 10.0 * (1.0 - bt[1])
                return r
            eps0 = w * w if kind == 'cauchy' else w
            q0 = [u0, math.log(max(eps0, 1e-12))]
            try:
                r = least_squares(resid, q0, bounds=([-0.5, -32.0], [1.5, 5.0]),
                                  max_nfev=70, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                u, le = float(r.x[0]), float(r.x[1])
                eps = math.exp(le)
                s = (t - u) ** 2 + eps if kind == 'cauchy' else np.abs(t - u) + eps
                B = np.column_stack([np.ones_like(t), -np.log(s)])
                bt, *_ = np.linalg.lstsq(B, lg, rcond=None)
                err = float(np.sqrt(np.mean((B @ bt - lg) ** 2)))
                if bt[1] > 0:
                    b, pw = float(bt[0]), float(bt[1])
                    if kind == 'cauchy':
                        logg = lambda x, b=b, pw=pw, eps=eps, u=u: b - pw * np.log((np.asarray(x, float) - u) ** 2 + eps)
                        sc = math.sqrt(eps) / 20
                    else:
                        logg = lambda x, b=b, pw=pw, eps=eps, u=u: b - pw * np.log(np.abs(np.asarray(x, float) - u) + eps)
                        sc = eps / 20
                    consider(kind, logg, u, sc, err)
            except Exception:
                continue
        if best is None:
            return self
        self.kind, self.logg, self.u, self.scale = best
        self.integral = graded_integral(self.logg, self.u, self.scale)
        return self

    def eval(self, t):
        return np.exp(np.clip(self.logg(np.asarray(t, float)), -700, 700))

    def logeval(self, t):
        return self.logg(np.asarray(t, float))


# --------------------------------------------------------------------------- plane wave machinery
def wave_integral(w):
    """Integral of exp(i w.x) over [0,1]^d."""
    w = np.asarray(w, float)
    z = np.exp(0.5j * w) * np.sinc(w / (2 * math.pi))
    return np.prod(z)


def sinc_fit_1d(t, y, wmax=260.0, ngrid=5200):
    """Variable projection single-sinusoid fit. Returns (w, c, A, B, sse)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = len(t)
    wg = np.linspace(0.01, wmax, ngrid)
    # chunked to limit memory
    best = (-1.0, None)
    yy = float(y @ y)
    ysum = float(y.sum())
    for s in range(0, ngrid, 650):
        w = wg[s:s + 650]
        C = np.cos(np.outer(w, t)); S = np.sin(np.outer(w, t))
        sc = C.sum(1); ss = S.sum(1)
        scc = (C * C).sum(1); sss = (S * S).sum(1); scs = (C * S).sum(1)
        syc = C @ y; sys_ = S @ y
        G = np.empty((len(w), 3, 3))
        G[:, 0, 0] = n; G[:, 0, 1] = G[:, 1, 0] = sc
        G[:, 0, 2] = G[:, 2, 0] = ss
        G[:, 1, 1] = scc; G[:, 2, 2] = sss; G[:, 1, 2] = G[:, 2, 1] = scs
        R = np.stack([np.full(len(w), ysum), syc, sys_], axis=1)
        G[:, 0, 0] += 1e-9; G[:, 1, 1] += 1e-9; G[:, 2, 2] += 1e-9
        try:
            beta = np.linalg.solve(G, R[..., None])[..., 0]
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(G, R[..., None], rcond=None)[0][..., 0]
        sse = yy - np.einsum('ij,ij->i', beta, R)
        k = int(np.argmin(sse))
        if best[1] is None or sse[k] < best[0]:
            best = (float(sse[k]), (float(w[k]), beta[k].copy(), wg, sse, k))
    sse0, (w0, beta0, wg, sse_all, k0) = best

    def profiled(w):
        c = np.cos(w * t); s = np.sin(w * t)
        A = np.array([[n, c.sum(), s.sum()],
                      [c.sum(), c @ c, c @ s],
                      [s.sum(), c @ s, s @ s]])
        A[0, 0] += 1e-10; A[1, 1] += 1e-10; A[2, 2] += 1e-10
        b = np.array([ysum, c @ y, s @ y])
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e30
        return yy - float(beta @ b)
    step = wg[1] - wg[0]
    lo = max(0.0001, w0 - 2 * step); hi = w0 + 2 * step
    try:
        r = minimize_scalar(profiled, bounds=(lo, hi), method='bounded',
                            options={'xatol': 1e-11})
        wr = float(r.x)
        if r.fun <= sse0:
            w0 = wr
    except Exception:
        pass
    try:
        def rr(p):
            wv = p[0]
            cc = np.cos(wv * t); ss = np.sin(wv * t)
            Mx = np.column_stack([np.ones(n), cc, ss])
            bt, *_ = np.linalg.lstsq(Mx, y, rcond=None)
            return y - Mx @ bt
        out = least_squares(rr, [w0], bounds=([1e-5], [wmax * 1.5]),
                            max_nfev=35, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        if np.linalg.norm(rr(out.x)) <= np.linalg.norm(rr([w0])):
            w0 = float(out.x[0])
    except Exception:
        pass
    c = np.cos(w0 * t); s = np.sin(w0 * t)
    A = np.array([[n, c.sum(), s.sum()],
                  [c.sum(), c @ c, c @ s],
                  [s.sum(), c @ s, s @ s]])
    A[0, 0] += 1e-10; A[1, 1] += 1e-10; A[2, 2] += 1e-10
    b = np.array([ysum, c @ y, s @ y])
    beta = np.linalg.solve(A, b)
    sse = yy - float(beta @ b)
    return float(w0), float(beta[0]), float(beta[1]), float(beta[2]), max(0.0, float(sse))


def line_model(t, freqs, y):
    """Variable-projection linear fit of y to a DC term and K sinusoids."""
    cols = [np.ones_like(t)]
    for w in freqs:
        z = w * t
        cols.append(np.cos(z)); cols.append(np.sin(z))
    M = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    return M, beta, y - M @ beta


def fit_line_spectrum(t, y, K=4, wmax=260.0):
    """Fit up to K sinusoids on a 1D line, jointly refining all frequencies."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    sy = float(np.std(y)) + 1e-300
    freqs = []
    resid = y - y.mean()
    for k in range(K):
        w, c, A, B, sse = sinc_fit_1d(t, resid, wmax=wmax)
        amp = math.hypot(A, B)
        if k > 0 and amp < 0.003 * sy:
            break
        freqs.append(w)
        if len(freqs) > 1:
            def rr(p):
                M, beta, r = line_model(t, p, y)
                return r
            try:
                out = least_squares(rr, np.asarray(freqs), bounds=(1e-4, wmax * 1.5),
                                    max_nfev=60, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                freqs = list(np.sort(np.abs(out.x)))
            except Exception:
                pass
        M, beta, r = line_model(t, freqs, y)
        resid = r
        if np.sqrt(np.mean(r ** 2)) < 1e-11 * sy:
            break
    M, beta, r = line_model(t, freqs, y)
    out = []
    for k, w in enumerate(freqs):
        amp = math.hypot(beta[1 + 2 * k], beta[2 + 2 * k])
        out.append((float(w), float(amp), float(np.sqrt(np.mean(r ** 2)) / sy)))
    out.sort(key=lambda z: -z[1])
    return out


def _wave_gain(X, residual, W):
    """Variance reduction from fitting residual with cos/sin of X@W for each row."""
    Z = X @ W.T
    C = np.cos(Z); S = np.sin(Z)
    rc = residual @ C; rs = residual @ S
    sc = (C * C).sum(0); ss = (S * S).sum(0); cs = (C * S).sum(0)
    det = sc * ss - cs * cs
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    A = (rc * ss - rs * cs) / det
    B = (rs * sc - rc * cs) / det
    return A * rc + B * rs


def discover_waves(lines, X, y, d, max_waves=3, verbose=False, spectra=None):
    """Assemble multi-dimensional plane-wave features from 1D line spectra."""
    if spectra is None:
        spectra = [fit_line_spectrum(L['t'], L['y'], K=max_waves) for L in lines]
    K = min(max_waves, max(1, max(len(s) for s in spectra)))
    omega = np.zeros((d, K))
    for i in range(d):
        for k in range(K):
            omega[i, k] = spectra[i][k][0] if k < len(spectra[i]) else 0.0
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=d)), float)
    if len(signs) > 512:
        signs = signs[:512]
    yc = y - y.mean()
    yvar = float(yc @ yc) + 1e-300
    residual = yc.copy()
    wlist = []
    sy = float(np.std(y)) + 1e-300
    for k in range(K):
        wabs = omega[:, k]
        if np.max(np.abs(wabs)) < 1e-6:
            continue
        W = signs * wabs[None, :]
        gain = _wave_gain(X, residual, W)
        j = int(np.argmax(gain))
        if gain[j] < 1e-8 * yvar:
            continue
        w = W[j].copy()
        # per-axis discrete alternatives (other detected line frequencies, zero)
        trials = [w.copy()]
        for i in range(d):
            opts = [w[i], 0.0]
            for m in range(min(len(spectra[i]), max_waves)):
                f = spectra[i][m][0]
                opts += [f, -f]
            for v in opts:
                tw = w.copy(); tw[i] = v
                trials.append(tw)
        TW = np.asarray(trials)
        tg = _wave_gain(X, residual, TW)
        jj = int(np.argmax(tg))
        if tg[jj] > gain[j]:
            w = TW[jj].copy()
        wlist.append(w)

        def make_matrix(p):
            Kc = len(p) // d
            cols = [np.ones(len(X))]
            for kk in range(Kc):
                z = X @ p[kk * d:(kk + 1) * d]
                cols.append(np.cos(z)); cols.append(np.sin(z))
            return np.column_stack(cols)

        def jres(p):
            M = make_matrix(p)
            beta, *_ = np.linalg.lstsq(M, y, rcond=None)
            return y - M @ beta
        p0 = np.concatenate(wlist)
        try:
            out = least_squares(jres, p0, max_nfev=90, ftol=1e-14, xtol=1e-14,
                                gtol=1e-14, x_scale='jac')
            if np.linalg.norm(jres(out.x)) <= np.linalg.norm(jres(p0)):
                p0 = out.x
        except Exception:
            pass
        M = make_matrix(p0)
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        residual = y - M @ beta
        wlist = [p0[kk * d:(kk + 1) * d].copy() for kk in range(len(p0) // d)]
        if np.sqrt(np.mean(residual ** 2)) < 1e-11 * sy:
            break
    waves = []
    for w in wlist:
        zi = wave_integral(w)
        waves.append((w, float(zi.real), float(zi.imag),
                      float(1 - (residual @ residual) / yvar)))
    # harmonic augmentation: Fourier ridges often are Fourier series in one direction
    return waves, spectra


# --------------------------------------------------------------------------- features
class Feature:
    __slots__ = ('name', 'fn', 'integral', 'group')

    def __init__(self, name, fn, integral, group):
        self.name = name; self.fn = fn; self.integral = float(integral); self.group = group


def poly_features(d, degree=2):
    out = []
    C = None
    def mk1(i):
        return lambda X: X[:, i] - 0.5
    for i in range(d):
        out.append(Feature(f'x{i}', mk1(i), 0.0, 'poly'))
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    out.append(Feature(f'x{i}x{j}', lambda X, i=i: (X[:, i] - 0.5) ** 2 - 1 / 12, 0.0, 'poly'))
                else:
                    out.append(Feature(f'x{i}x{j}', lambda X, i=i, j=j: (X[:, i] - 0.5) * (X[:, j] - 0.5), 0.0, 'poly'))
    return out


def wave_features(waves):
    out = []
    for k, (w, ic, is_, q) in enumerate(waves):
        out.append(Feature(f'cos{k}', lambda X, w=w: np.cos(X @ w), ic, 'wave'))
        out.append(Feature(f'sin{k}', lambda X, w=w: np.sin(X @ w), is_, 'wave'))
    return out


def product_features(lines, x0, f0, X_qmc, y_qmc, d, c_candidates, max_evals=7):
    """Rank-1 separable product features from axis lines.

    For each candidate offset c, fit 1D positive profiles on every axis line,
    construct the rank-1 product surrogate with its exact integral, and score it
    by the residual of an intercept+scale fit on the QMC sample.  The best c is
    refined by a short golden-section search.
    """
    scale = float(np.max(np.abs(y_qmc))) + 1e-12
    yc = y_qmc - np.mean(y_qmc)

    def build(c):
        denom = f0 - c
        if not np.isfinite(denom) or denom <= 0 or abs(denom) < 1e-280:
            return None
        fits = []
        for L in lines:
            g = L['y'] - c
            if not np.all(np.isfinite(g)) or np.min(g) < -1e-10 * scale:
                return None
            sf = ShapeFit(L['t'], g).fit()
            if not sf.ok or sf.kind is None or not np.isfinite(sf.integral) or sf.integral <= 0:
                return None
            fits.append(sf)
        logden = (d - 1) * math.log(denom)
        logI = sum(math.log(max(sf.integral, EPS)) for sf in fits) - logden
        if not np.isfinite(logI) or logI > 700 or logI < -700:
            return None

        def fn(X, fits=fits, logden=logden):
            lg = np.zeros(len(X))
            for i, sf in enumerate(fits):
                lg = lg + sf.logeval(X[:, i])
            return np.exp(np.clip(lg - logden, -700, 700))
        v = fn(X_qmc)
        A = np.column_stack([np.ones(len(v)), v])
        try:
            beta = np.linalg.lstsq(A, y_qmc, rcond=None)[0]
            resid = y_qmc - A @ beta
        except Exception:
            return None
        quality = float(np.sqrt(np.mean(resid ** 2)) / scale)
        return (quality, Feature('prod', fn, math.exp(logI), 'prod'), c, fits)

    best = None
    tried = 0
    for c in c_candidates:
        if not np.isfinite(c) or c >= f0 - 1e-14:
            continue
        r = build(c)
        tried += 1
        if r is not None and (best is None or r[0] < best[0]):
            best = r
        if tried >= max_evals:
            break
    if best is not None:
        c0 = best[2]
        step = max(1e-9 * scale, 1e-7 * max(1.0, abs(c0)))
        for c in (c0 - step, c0 + step, c0 - step / 3, c0 + step / 3):
            if c >= f0 - 1e-14:
                continue
            r = build(c)
            if r is not None and r[0] < best[0]:
                best = r
    if best is None:
        return []
    return [best[1]]


# --------------------------------------------------------------------------- regression CV
def build_matrix(features, X):
    if not features:
        return np.ones((len(X), 1))
    return np.column_stack([np.ones(len(X))] + [f.fn(X) for f in features])


def cv_estimate(blocks, features, lam=1e-9, gate=0.5):
    """Cross-fitted control-variate estimate.

    A fold only uses its control-variate correction when the out-of-block
    residual is substantially smaller than the raw spread of y; otherwise it
    falls back to the plain QMC mean.  This keeps exact surrogate fits while
    preventing a poor surrogate from biasing the estimate.
    """
    nrep = len(blocks)
    Xall = np.vstack([b[0] for b in blocks])
    yall = np.concatenate([b[1] for b in blocks])
    I = np.array([1.0] + [f.integral for f in features])
    syall = float(np.std(yall)) + 1e-12
    if nrep < 2:
        A = build_matrix(features, Xall)
        sc = np.sqrt((A ** 2).mean(0)) + 1e-10
        sc[0] = 1.0
        B = A / sc
        beta = np.linalg.lstsq(B, yall, rcond=1e-9)[0] / sc
        resid = yall - A @ beta
        corr = float(I @ beta - np.mean(A @ beta))
        ok = np.std(resid) < gate * syall
        est = float(np.mean(yall) + corr) if ok and np.isfinite(corr) else float(np.mean(yall))
        return est, float(np.std(resid)), float('inf')
    ests, rstd = [], []
    for r in range(nrep):
        Xt = np.vstack([blocks[j][0] for j in range(nrep) if j != r])
        yt = np.concatenate([blocks[j][1] for j in range(nrep) if j != r])
        Xo, yo = blocks[r]
        At = build_matrix(features, Xt)
        Ao = build_matrix(features, Xo)
        sc = np.sqrt((At ** 2).mean(0)) + 1e-10
        sc[0] = 1.0
        Bt = At / sc
        Bo = Ao / sc
        G = np.vstack([Bt, math.sqrt(lam * len(yt)) * np.eye(Bt.shape[1])])
        z = np.concatenate([yt, np.zeros(Bt.shape[1])])
        beta = np.linalg.lstsq(G, z, rcond=1e-9)[0] / sc
        resid = yo - Ao @ beta
        corr = float(I @ beta - np.mean(Ao @ beta))
        syt = float(np.std(yt)) + 1e-12
        ok = (np.std(resid) < gate * syt) and np.isfinite(corr)
        est = float(np.mean(yo) + corr) if ok else float(np.mean(yo))
        ests.append(est)
        rstd.append(float(np.std(resid)))
    return float(np.mean(ests)), float(np.mean(rstd)), float(np.std(ests)) / math.sqrt(nrep)


def harmonics_matrix(X, p, M):
    cols = [np.ones(len(X))]
    for m in range(1, M + 1):
        z = X @ (m * p)
        cols.append(np.cos(z)); cols.append(np.sin(z))
    return np.column_stack(cols)


def harmonic_waves(spectra, w0_signed, X, y, d, M=10):
    """Fourier-series ridge with a refined fundamental frequency vector."""
    if w0_signed is None:
        return []
    dom = np.abs(np.asarray(w0_signed, float))
    sgn = np.sign(np.asarray(w0_signed, float))
    sgn[sgn == 0] = 1.0
    cands = []
    for denom in (1, 2, 3, 4, 5):
        cands.append(sgn * dom / denom)
    mins = []
    for sp in spectra:
        mins.append(min([z[0] for z in sp]) if sp else 0.0)
    mins = np.asarray(mins)
    if np.min(mins) > 1e-8 and np.max(mins) > 0:
        cands.append(sgn * mins)
    def fit(p, refine=False):
        def resid(q):
            Mx = harmonics_matrix(X, q, M)
            beta, *_ = np.linalg.lstsq(Mx, y, rcond=1e-10)
            return y - Mx @ beta
        if refine:
            try:
                out = least_squares(resid, p, max_nfev=70, ftol=1e-15,
                                    xtol=1e-15, gtol=1e-15, x_scale='jac')
                if np.linalg.norm(resid(out.x)) <= np.linalg.norm(resid(p)):
                    p = out.x
            except Exception:
                pass
        Mx = harmonics_matrix(X, p, M)
        beta, *_ = np.linalg.lstsq(Mx, y, rcond=1e-10)
        r = y - Mx @ beta
        return float(np.sqrt(np.mean(r ** 2))), np.asarray(p, float)
    scored = [(fit(p)[0], p) for p in cands]
    scored.sort(key=lambda z: z[0])
    _, pbest = fit(scored[0][1], refine=True)
    out = []
    for m in range(1, M + 1):
        w = m * pbest
        zi = wave_integral(w)
        out.append((w, float(zi.real), float(zi.imag), 0.0))
    return out


def signcombo_waves(spectra, X, y, d, cap=160):
    """Signed frequency vectors from per-axis line-spectrum magnitudes."""
    mag = np.array([sp[0][0] if sp else 0.0 for sp in spectra], float)
    if np.max(mag) < 1e-6:
        return []
    combos = np.array([z for z in itertools.product([-1.0, 1.0], repeat=d - 1)], float)
    if len(combos) > cap:
        idx = np.linspace(0, len(combos) - 1, cap).astype(int)
        combos = combos[idx]
    W = np.column_stack([np.ones(len(combos)), combos]) * mag[None, :]
    out = []
    for w in W:
        zi = wave_integral(w)
        out.append((w, float(zi.real), float(zi.imag), 0.0))
    return out
