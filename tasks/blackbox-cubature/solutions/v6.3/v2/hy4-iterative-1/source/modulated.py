"""Fourier x product-envelope features with exact integrals.

Models f(x) = c + H(x) * sum_m [A_m cos(m w.x) + B_m sin(m w.x)],
H(x) = prod_i h_i(x_i) with 1D positive profiles fitted from axis lines.
"""
from __future__ import annotations
import math, itertools, time
import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.legendre import leggauss
import advanced as A

GLZ, GLW = leggauss(28)
GLZ = (GLZ + 1) / 2
GLW = GLW / 2


def _breakpoints(u, scale, wmag, M, nmin=24):
    delta = min(max(float(scale), 1e-11), 0.25)
    pts = {0.0, 1.0, float(np.clip(u, 0.0, 1.0))}
    for side in (-1, 1):
        for k in range(0, 80):
            t = u + side * delta * (2.0 ** (k / 2.0))
            if t < 0.0 or t > 1.0:
                break
            pts.add(float(t))
    n_osc = int(max(nmin, math.ceil(abs(wmag) * M / math.pi * 3) + 1))
    n_osc = min(n_osc, 400)
    for t in np.linspace(0.0, 1.0, n_osc + 1):
        pts.add(float(t))
    return sorted(pts)


def osc_integrals(h, w, u, scale, M):
    """(Zc[m], Zs[m]) = integral h(t) exp(i m w t), m=1..M."""
    bp = _breakpoints(u, scale, w, M)
    Z = np.zeros(M, complex)
    lo = np.asarray(bp[:-1]); hi = np.asarray(bp[1:])
    mid = lo[:, None] + (hi - lo)[:, None] * GLZ[None, :]
    hv = np.clip(h(mid.ravel()).reshape(mid.shape), 0.0, 1e100)
    wt = (hi - lo)[:, None] * GLW[None, :]
    for m in range(1, M + 1):
        ph = m * w * mid
        integ = hv * np.exp(1j * ph)
        Z[m - 1] = float(np.sum(wt * integ.real)) + 1j * float(np.sum(wt * integ.imag))
    return Z


class ModLine:
    """1D fit g(t) = sum_m h(t) [A_m cos(m w t) + B_m sin(m w t)] + dc."""
    KINDS = ('flat', 'gauss', 'laplace', 'cauchy', 'cusp')

    def __init__(self, t, g, w0, M=1, w_inits=None):
        self.t = np.asarray(t, float)
        self.g = np.asarray(g, float)
        self.w0 = abs(float(w0)) if abs(float(w0)) > 1e-6 else 1.0
        self.w_inits = list(w_inits) if w_inits else []
        self.M = int(M)
        self.ok = len(self.t) >= 16
        self.best = None

    def _h(self, kind, p, t):
        if kind == 'flat':
            return np.ones_like(t), 0.5, 0.2
        w = p[0]
        if kind in ('gauss', 'laplace'):
            lam, u = p[1], p[2]
            v = (t - u) ** 2 if kind == 'gauss' else np.abs(t - u)
            return np.exp(-np.clip(lam * v, 0, 700)), u, 1.0 / math.sqrt(max(lam, 1e-12)) / 20
        u, le, pw = p[1], p[2], p[3]
        eps = math.exp(le)
        s = (t - u) ** 2 + eps if kind == 'cauchy' else np.abs(t - u) + eps
        return np.exp(-np.clip(pw * np.log(s), -700, 700)), u, math.sqrt(eps) / 20 if kind == 'cauchy' else eps / 20

    def _design(self, kind, p, t):
        h, u, sc = self._h(kind, p, t)
        w = p[0]
        cols = [np.ones_like(t)]
        for m in range(1, self.M + 1):
            z = m * w * t
            cols.append(h * np.cos(z)); cols.append(h * np.sin(z))
        return np.column_stack(cols), h, u, sc

    def fit(self):
        if not self.ok:
            return self
        t, g = self.t, self.g
        j = int(np.argmax(np.abs(g)))
        u0 = float(np.clip(t[j], 0.0, 1.0))
        w_est = self.w0
        # rough width from envelope of |g|
        amp = np.abs(g)
        half = np.abs(t - u0)[amp < 0.5 * amp.max()]
        wd = float(np.min(half)) if half.size else 0.1
        wd = min(max(wd, 1e-5), 0.5)
        inits = {
            'flat': ([w_est], ([1e-5], [400.0])),
            'gauss': ([w_est, 1.0 / wd ** 2, u0], ([1e-5, 1e-6, -0.5], [400.0, 1e7, 1.5])),
            'laplace': ([w_est, 1.0 / wd, u0], ([1e-5, 1e-6, -0.5], [400.0, 1e7, 1.5])),
            'cauchy': ([w_est, u0, math.log(wd ** 2), 1.0],
                       ([1e-5, -0.5, -30.0, 0.02], [400.0, 1.5, 5.0, 12.0])),
            'cusp': ([w_est, u0, math.log(wd), 0.5],
                     ([1e-5, -0.5, -30.0, 0.02], [400.0, 1.5, 5.0, 12.0])),
        }
        w_inits = list(self.w_inits) + [w_est, w_est / 2, w_est / 3, w_est * 2,
                   max(0.2, w_est / 5), min(300.0, w_est * 1.5)]
        seen = set()
        w_inits = [w for w in w_inits if not (round(w, 6) in seen or seen.add(round(w, 6)))]
        for kind in self.KINDS:
            for wi in w_inits:
                p0, bounds = inits[kind]
                p0 = list(p0); p0[0] = float(np.clip(wi, 1e-4, 399.0))
                def resid(p, kind=kind):
                    Mx, _, _, _ = self._design(kind, p, t)
                    beta, *_ = np.linalg.lstsq(Mx, g, rcond=1e-9)
                    return g - Mx @ beta
                try:
                    out = least_squares(resid, p0, bounds=bounds, max_nfev=60,
                                        ftol=1e-13, xtol=1e-13, gtol=1e-13)
                    p = out.x
                except Exception:
                    p = np.asarray(p0)
                Mx, h, u, sc = self._design(kind, p, t)
                beta, *_ = np.linalg.lstsq(Mx, g, rcond=1e-9)
                r = g - Mx @ beta
                err = float(np.sqrt(np.mean(r ** 2)))
                if self.best is None or err < self.best[0]:
                    self.best = (err, kind, np.asarray(p).copy(), beta.copy(), u, sc)
                if err < 1e-11 * (np.std(g) + 1e-12):
                    break
            if self.best is not None and self.best[0] < 1e-11 * (np.std(g) + 1e-12):
                break
        return self

    def result(self):
        if self.best is None:
            return None
        err, kind, p, beta, u, sc = self.best
        w = float(p[0])
        def hfun(t, kind=kind, p=p):
            return self._h(kind, p, np.asarray(t, float))[0]
        # normalize envelope to max 1 for numerical stability
        grid = np.linspace(0, 1, 2001)
        hmax = float(np.max(hfun(grid)))
        if not np.isfinite(hmax) or hmax <= 0:
            return None
        def hnorm(t):
            return hfun(t) / hmax
        def logh(t):
            return np.log(np.clip(hfun(t), 1e-300, 1e300)) - math.log(hmax)
        Z = osc_integrals(hnorm, w, u, sc, self.M)
        return dict(w=w, h=hnorm, logh=logh, Z=Z, err=err, kind=kind)


def modulated_features(lines, spectra, w0_signed, X, y, d, c_candidates, M=1,
                       time_budget=2.2):
    """Best envelope x plane-wave feature pair, with a hard time budget."""
    if w0_signed is None:
        return []
    t_start = time.perf_counter()
    mags = np.array([sp[0][0] if sp else 1.0 for sp in spectra], float)
    signs0 = np.sign(np.asarray(w0_signed, float))
    signs0[signs0 == 0] = 1.0
    combos = np.array([z for z in itertools.product([-1.0, 1.0], repeat=d - 1)], float)
    if len(combos) > 64:
        combos = combos[np.linspace(0, len(combos) - 1, 64).astype(int)]
    best = None
    # one offset is enough: the 1D design contains an intercept
    c = 0.0
    fits = []
    for i, L in enumerate(lines):
        if time.perf_counter() - t_start > time_budget:
            return []
        freqs = [sp[0] for sp in [spectra[i]] for k in range(min(3, len(sp)))]
        freqs = [spectra[i][k][0] for k in range(min(3, len(spectra[i])))]
        freqs += [f / 2 for f in freqs]
        ml = ModLine(L['t'], L['y'] - c, mags[i], M=M, w_inits=freqs).fit().result()
        if ml is None or not np.isfinite(ml['w']):
            return []
        fits.append(ml)
    W = np.column_stack([np.ones(len(combos)), combos]) * np.array([f['w'] for f in fits])[None, :]
    logH = np.zeros(len(X))
    for i, f in enumerate(fits):
        logH = logH + f['logh'](X[:, i])
    H = np.exp(np.clip(logH, -700, 700))
    for w in W:
        if time.perf_counter() - t_start > time_budget + 0.5:
            break
        cols = [np.ones(len(X))]
        ints = [1.0]
        for m in range(1, M + 1):
            z = m * (X @ w)
            cols.append(H * np.cos(z)); cols.append(H * np.sin(z))
            Zi = [f['Z'][m - 1] if wi >= 0 else np.conj(f['Z'][m - 1])
                  for wi, f in zip(w, fits)]
            Z = np.prod(Zi)
            ints.append(float(Z.real)); ints.append(float(Z.imag))
        Mx = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(Mx, y, rcond=1e-9)
        r = y - Mx @ beta
        quality = float(np.sqrt(np.mean(r ** 2))) / (float(np.std(y)) + 1e-12)
        if best is None or quality < best[0]:
            feats = []
            for j in range(1, len(cols)):
                feats.append(A.Feature(f'mod{j}',
                                       lambda XX, j=j, fits=fits, w=w, M=M: _eval_mod(XX, j, fits, w, M),
                                       ints[j], 'mod'))
            best = (quality, feats)
    if best is None or best[0] > 0.15:
        return []
    return best[1]


def _eval_mod(X, j, fits, w, M):
    m = (j + 1) // 2
    is_cos = (j % 2 == 1)
    logH = np.zeros(len(X))
    for i, f in enumerate(fits):
        logH = logH + f['logh'](X[:, i])
    H = np.exp(np.clip(logH, -700, 700))
    z = m * (np.asarray(X, float) @ w)
    return H * (np.cos(z) if is_cos else np.sin(z))
