#!/usr/bin/env python3
import json
import math
import sys
import warnings
import itertools
import time

import numpy as np
from scipy.stats import qmc
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf
from scipy.linalg import hankel, svd, eig
from scipy.signal import fftconvolve
from scipy.interpolate import UnivariateSpline, BSpline

warnings.filterwarnings("ignore", message="The balance properties of Sobol")
START = 0.0


def qmc_points(d, n, seed):
    if n <= 0:
        return np.empty((0, d))
    return qmc.Sobol(d, scramble=True, seed=int(seed)).random(n)


def send(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


class Oracle:
    def __init__(self, dimension, budget):
        self.d = int(dimension)
        self.budget = int(budget)
        self.used = 0

    def query(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        n = len(points)
        if n == 0:
            return np.empty(0)
        if self.used + n > self.budget:
            raise RuntimeError("budget exceeded")
        vals = []
        for start in range(0, n, 512):
            chunk = points[start:start + 512]
            send({"type": "query", "points": chunk.tolist()})
            ans = json.loads(sys.stdin.readline())
            got = np.asarray(ans.get("values", []), dtype=float)
            if got.size != len(chunk) or not np.all(np.isfinite(got)):
                raise RuntimeError("invalid oracle values")
            vals.append(got)
            self.used += len(chunk)
        return np.concatenate(vals)


def pilot_points(d, h=0.02):
    pts = [np.full(d, 0.5)]
    for j in range(d):
        for s in (-1.0, 1.0):
            p = np.full(d, 0.5)
            p[j] += s * h
            pts.append(p)
    for j in range(d):
        for k in range(j + 1, d):
            for sj in (-1.0, 1.0):
                for sk in (-1.0, 1.0):
                    p = np.full(d, 0.5)
                    p[j] += sj * h
                    p[k] += sk * h
                    pts.append(p)
    return np.asarray(pts)


def pilot_derivatives(pts, vals, d, h=0.02):
    f0 = vals[0]
    grad = np.zeros(d)
    H = np.zeros((d, d))
    idx = 1
    for j in range(d):
        fm, fp = vals[idx], vals[idx + 1]
        idx += 2
        grad[j] = (fp - fm) / (2 * h)
        H[j, j] = (fp + fm - 2 * f0) / (h * h)
    for j in range(d):
        for k in range(j + 1, d):
            v = vals[idx:idx + 4]
            idx += 4
            H[j, k] = H[k, j] = (v[0] - v[1] - v[2] + v[3]) / (4 * h * h)
    M = H @ H + np.outer(grad, grad)
    evals, evecs = np.linalg.eigh(M)
    w = evecs[:, -1].copy()
    if w[np.argmax(np.abs(w))] < 0:
        w = -w
    ratio = float(evals[-1] / max(evals.sum(), 1e-300))
    return grad, H, w, ratio


def one_exp(a, u, lo, hi):
    if a < 1e-14:
        return hi - lo
    if hi <= u:
        return (math.exp(-a * (u - hi)) - math.exp(-a * (u - lo))) / a
    if lo >= u:
        return (math.exp(-a * (lo - u)) - math.exp(-a * (hi - u))) / a
    return (2.0 - math.exp(-a * (u - lo)) - math.exp(-a * (hi - u))) / a


def gauss_factor(a, u):
    if a < 1e-14:
        return 1.0
    return math.sqrt(math.pi) / (2 * a) * (erf(a * u) + erf(a * (1 - u)))


def safe_exp(logval):
    if logval > 700:
        return float("inf")
    if logval < -700:
        return 0.0
    return math.exp(logval)


class SeparableModel:
    def __init__(self, kind, d):
        self.kind = kind
        self.d = d

    def initial(self, x, logy):
        d = self.d
        y = np.exp(logy)
        w = np.maximum(y, 1e-300)
        u0 = np.clip((x * w[:, None]).sum(axis=0) / w.sum(), 0.05, 0.95)
        if self.kind == "gaussian":
            a0 = np.full(d, 5.0)
            base = -np.sum((a0 * (x - u0)) ** 2, axis=1)
            p0 = np.concatenate([np.log(a0), u0, [0.0]])
            lo = np.concatenate([np.full(d, -10.0), np.zeros(d), [-300.0]])
            hi = np.concatenate([np.full(d, 10.0), np.ones(d), [300.0]])
        elif self.kind == "continuous":
            a0 = np.full(d, 5.0)
            base = -np.sum(a0 * np.abs(x - u0), axis=1)
            p0 = np.concatenate([np.log(a0), u0, [0.0]])
            lo = np.concatenate([np.full(d, -10.0), np.zeros(d), [-300.0]])
            hi = np.concatenate([np.full(d, 10.0), np.ones(d), [300.0]])
        elif self.kind == "peak":
            z0 = np.full(d, 5.0)
            base = -np.sum(np.log(z0 ** -2 + (x - u0) ** 2), axis=1)
            p0 = np.concatenate([np.log(z0), u0, [0.0]])
            lo = np.concatenate([np.full(d, -10.0), np.zeros(d), [-300.0]])
            hi = np.concatenate([np.full(d, 10.0), np.ones(d), [300.0]])
        else:  # cusp
            eps0 = np.full(d, 0.01)
            pp0 = np.full(d, 0.5)
            base = -np.sum(pp0 * np.log(eps0 + np.abs(x - u0)), axis=1)
            p0 = np.concatenate([pp0, np.log(eps0), u0, [0.0]])
            lo = np.concatenate([np.full(d, 0.01), np.full(d, -12.0), np.zeros(d), [-300.0]])
            hi = np.concatenate([np.full(d, 3.0), np.full(d, 0.0), np.ones(d), [300.0]])
        p0[-1] = float(np.mean(logy - base))
        return p0, lo, hi

    def base(self, p, x):
        d = self.d
        if self.kind == "gaussian":
            a = np.exp(p[:d]); u = p[d:2 * d]
            return -np.sum((a * (x - u)) ** 2, axis=1)
        if self.kind == "continuous":
            a = np.exp(p[:d]); u = p[d:2 * d]
            return -np.sum(a * np.abs(x - u), axis=1)
        if self.kind == "peak":
            z = np.exp(p[:d]); u = p[d:2 * d]
            return -np.sum(np.log(z ** -2 + (x - u) ** 2), axis=1)
        pp = p[:d]; eps = np.exp(p[d:2 * d]); u = p[2 * d:3 * d]
        return -np.sum(pp * np.log(eps + np.abs(x - u)), axis=1)

    def fit(self, x, y):
        logy = np.log(y)
        p0, lo, hi = self.initial(x, logy)
        cidx = len(p0) - 1

        def resid(p):
            return p[cidx] + self.base(p, x) - logy

        res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=250)
        return res.x

    def predict(self, p, x):
        return np.exp(p[-1] + self.base(p, x))

    def integral(self, p, step=None):
        d = self.d
        c = p[-1]
        if self.kind == "gaussian":
            a = np.exp(p[:d]); u = p[d:2 * d]
            factors = [gauss_factor(z, uu) for z, uu in zip(a, u)]
        elif self.kind == "continuous":
            a = np.exp(p[:d]); u = p[d:2 * d]
            factors = [one_exp(z, uu, 0.0, 1.0) for z, uu in zip(a, u)]
        elif self.kind == "peak":
            z = np.exp(p[:d]); u = p[d:2 * d]
            factors = [zz * (math.atan(zz * uu) + math.atan(zz * (1 - uu))) for zz, uu in zip(z, u)]
        else:
            pp = p[:d]; eps = np.exp(p[d:2 * d]); u = p[2 * d:3 * d]
            factors = []
            for e, uu, power in zip(eps, u, pp):
                if abs(1 - power) < 1e-12:
                    factors.append(math.log((e + uu) / e) + math.log((e + 1 - uu) / e))
                else:
                    factors.append(((e + uu) ** (1 - power) + (e + 1 - uu) ** (1 - power) - 2 * e ** (1 - power)) / (1 - power))
        if step is not None:
            j, lo, hi = step
            if self.kind == "continuous":
                a = math.exp(p[j]); u = p[d + j]
                factors[j] = one_exp(a, u, lo, hi)
            elif self.kind == "gaussian":
                a = math.exp(p[j]); u = p[d + j]
                if a < 1e-14:
                    factors[j] = hi - lo
                else:
                    factors[j] = math.sqrt(math.pi) / (2 * a) * (erf(a * (hi - u)) - erf(a * (lo - u)))
            else:
                nodes, weights = np.polynomial.legendre.leggauss(40)
                nodes = lo + (hi - lo) * (nodes + 1) / 2
                weights = weights * (hi - lo) / 2
                if self.kind == "peak":
                    z = math.exp(p[j]); u = p[d + j]
                    fv = 1.0 / (z ** -2 + (nodes - u) ** 2)
                else:
                    pp = p[j]; e = math.exp(p[d + j]); u = p[2 * d + j]
                    fv = (e + np.abs(nodes - u)) ** (-pp)
                factors[j] = float(weights @ fv)
        logint = c + sum(math.log(max(f, 1e-300)) for f in factors)
        return safe_exp(logint)


def fit_oscillatory(x, y, w):
    n, d = x.shape
    t = x @ w

    def linear(omega):
        A = np.stack([np.ones(n), np.cos(omega * t), np.sin(omega * t)], axis=1)
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        rms = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
        return rms, coef

    grid = np.linspace(0.0, 100.0, 201)
    best = min(grid, key=lambda o: linear(o)[0])
    rms, coef = linear(best)
    alpha0 = best * w
    A0 = math.hypot(coef[1], coef[2])
    phi0 = math.atan2(-coef[2], coef[1])
    p0 = np.concatenate([alpha0, [phi0, A0, coef[0]]])

    def resid(p):
        alpha = p[:d]; phi = p[d]; A = p[d + 1]; C = p[d + 2]
        return C + A * np.cos(x @ alpha + phi) - y

    res = least_squares(resid, p0, method="lm", max_nfev=200)
    return res.x


def oscillatory_predict(p, x):
    d = x.shape[1]
    return p[d + 2] + p[d + 1] * np.cos(x @ p[:d] + p[d])


def oscillatory_integral(p, d):
    alpha = p[:d]; phi = p[d]; A = p[d + 1]; C = p[d + 2]
    fac = 1.0 + 0j
    for aj in alpha:
        if abs(aj) < 1e-14:
            fac *= 1.0
        else:
            fac *= (np.exp(1j * aj) - 1) / (1j * aj)
    return float(C + A * (np.exp(1j * phi) * fac).real)


def active_rank(grad, H):
    M = H @ H + np.outer(grad, grad)
    evals, evecs = np.linalg.eigh(M)
    top = float(evals[-1])
    if top <= 1e-24:
        return 0, evals, evecs
    k = int(np.sum(evals > 1e-6 * top))
    return min(k, len(evals)), evals, evecs


def line_fourier_terms(t, y, R):
    N = len(y)
    dt = t[1] - t[0]
    L = N // 2
    Y = hankel(y[:N - L], y[N - L - 1:])
    U, s, Vt = svd(Y, full_matrices=False)
    rank = min(2 * R + 1, len(s))
    Us = U[:, :rank]
    Z = np.linalg.pinv(Us[:-1]) @ Us[1:]
    ev = eig(Z, left=False, right=False)
    freqs = np.angle(ev) / dt
    freqs = np.sort(freqs[freqs > 1e-6])
    cand = []
    for f in freqs:
        if not cand or abs(f - cand[-1]) > 0.02:
            cand.append(float(f))
    if not cand:
        cand = [1.0]
    cols = [np.ones(N)]
    for f in cand:
        cols.append(np.cos(f * t))
        cols.append(np.sin(f * t))
    A = np.stack(cols, axis=1)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    amps = [math.hypot(coef[1 + 2 * i], coef[2 + 2 * i]) for i in range(len(cand))]
    order = np.argsort(-np.asarray(amps))[:R]
    freqs = np.asarray([cand[i] for i in order])
    if len(freqs) < R:
        freqs = np.pad(freqs, (0, R - len(freqs)), constant_values=1.0)
    cols = [np.ones(N)]
    for f in freqs:
        cols.append(np.cos(f * t))
        cols.append(np.sin(f * t))
    A = np.stack(cols, axis=1)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    p0 = [coef[0]]
    for r in range(R):
        B = coef[1 + 2 * r]
        D = coef[2 + 2 * r]
        p0 += [float(freqs[r]), math.atan2(-D, B), math.hypot(B, D)]
    p0 = np.asarray(p0)

    def model(p):
        val = np.full(N, p[0])
        for r in range(R):
            w, ph, am = p[1 + 3 * r:4 + 3 * r]
            val += am * np.cos(w * t + ph)
        return val

    res = least_squares(lambda p: model(p) - y, p0, method="lm", max_nfev=150)
    p = res.x
    out = []
    for r in range(R):
        w, ph, am = p[1 + 3 * r:4 + 3 * r]
        if w < 0:
            w = -w
            ph = -ph
        if am < 0:
            am = -am
            ph += math.pi
        out.append((abs(float(w)), float(ph), float(am)))
    out.sort(key=lambda z: -z[2])
    return out


def fourier_integral(C, alphas, phis, amps):
    est = float(C)
    R, d = alphas.shape
    for r in range(R):
        fac = 1.0 + 0j
        for a in alphas[r]:
            if abs(a) < 1e-14:
                fac *= 1.0
            else:
                fac *= (np.exp(1j * a) - 1) / (1j * a)
        est += float(amps[r] * (np.exp(1j * phis[r]) * fac).real)
    return est


def solve_fourier(oracle, d, k, seed, x_init, y_init, N=128):
    if oracle.used + d * N + 120 > oracle.budget:
        return None, None, None
    t = np.linspace(-0.5, 0.5, N)
    line_pts = np.empty((d * N, d))
    for j in range(d):
        pp = np.full((N, d), 0.5)
        pp[:, j] += t
        line_pts[j * N:(j + 1) * N] = pp
    line_vals = oracle.query(line_pts)
    line_y = line_vals.reshape(d, N)

    nrem = oracle.budget - oracle.used
    if nrem < 120:
        return None, None, None
    xq = qmc_points(d, nrem, seed + 2)
    yq = oracle.query(xq)
    x_all = np.vstack([xq, line_pts, x_init])
    y_all = np.concatenate([yq, line_vals, y_init])
    perm = np.random.default_rng(seed + 3).permutation(len(x_all))
    x_all = x_all[perm]
    y_all = y_all[perm]
    ntrain = max(80, int(0.85 * len(x_all)))
    xt, yt = x_all[:ntrain], y_all[:ntrain]
    xv, yv = x_all[ntrain:], y_all[ntrain:]
    yscale = max(1e-12, float(np.sqrt(np.mean(y_all ** 2))))
    ymean_all = float(y_all.mean())
    ystd_all = float(y_all.std())
    if 1 <= k <= 5:
        candidates = sorted(set([max(1, k - 1), k, min(5, k + 1)]))
    else:
        candidates = [1, 2, 3, 4, 5]
    candidates = [R for R in candidates if 1 <= R <= 5]
    best = None
    for R in candidates:
        if time.monotonic() - START > 24:
            break
        try:
            alphas = np.zeros((R, d))
            amps = np.zeros(R)
            psis = np.zeros((R, d))
            for j in range(d):
                terms = line_fourier_terms(t, line_y[j], R)
                for r in range(R):
                    w, ph, am = terms[r]
                    alphas[r, j] = w
                    amps[r] += am
                    psis[r, j] = ph
            amps /= d
            phis = np.zeros(R)
            perms = list(itertools.permutations(range(R))) if R <= 4 else [tuple(range(R)), tuple(range(R - 1, -1, -1))]
            best_perm = None
            for perm in perms:
                al_p = alphas[list(perm)].copy()
                ps_p = psis[list(perm)]
                am_p = amps[list(perm)]
                ph_p = np.zeros(R)
                score = 0.0
                for r in range(R):
                    best_sign = None
                    if d <= 8:
                        for signs in itertools.product((-1.0, 1.0), repeat=d):
                            s = np.asarray(signs)
                            theta = s * ps_p[r]
                            resultant = abs(np.mean(np.exp(1j * theta)))
                            if best_sign is None or resultant > best_sign[0]:
                                best_sign = (resultant, s, theta)
                    else:
                        best_sign = (1.0, np.ones(d), ps_p[r])
                    res_len, s, theta = best_sign
                    score += res_len
                    al_p[r] *= s
                    beta = float(np.angle(np.mean(np.exp(1j * theta))))
                    ph_p[r] = beta - 0.5 * float(al_p[r].sum())
                if best_perm is None or score > best_perm[0]:
                    best_perm = (score, al_p, ph_p, am_p)
            _, alphas, phis, amps = best_perm

            p0 = np.concatenate([[0.0], alphas.ravel(), phis, amps])

            def model(p):
                C = p[0]
                al = p[1:1 + R * d].reshape(R, d)
                ph = p[1 + R * d:1 + R * d + R]
                am = p[1 + R * d + R:]
                return C + np.sum(am[None, :] * np.cos(xt @ al.T + ph[None, :]), axis=1)

            res = least_squares(lambda p: model(p) - yt, p0, method="lm", max_nfev=300)
            p = res.x
            C = p[0]
            al = p[1:1 + R * d].reshape(R, d)
            ph = p[1 + R * d:1 + R * d + R]
            am = p[1 + R * d + R:]
            predv = C + np.sum(am[None, :] * np.cos(xv @ al.T + ph[None, :]), axis=1)
            rms = float(np.sqrt(np.mean((predv - yv) ** 2)))
            est = fourier_integral(C, al, ph, am)
            bound = abs(C) + float(np.sum(np.abs(am)))
            plausible = math.isfinite(est) and abs(est) <= 1.01 * bound + 1e-12
            thr = 1e-10 if R >= 4 else 1e-7
            if plausible and rms < thr * yscale and (best is None or rms < best[0]):
                best = (rms, est)
        except Exception:
            pass
    # Rank-one periodic ridge with several harmonics.
    for K in (3, 4, 5, 6, 7, 8):
        if time.monotonic() - START > 24:
            break
        try:
            f0 = np.zeros(d)
            psis = np.zeros(d)
            ok = True
            for j in range(d):
                yy = line_y[j]

                def fit_omega(omega):
                    cols = [np.ones(N)]
                    for k in range(1, K + 1):
                        cols.append(np.cos(k * omega * t))
                        cols.append(np.sin(k * omega * t))
                    A = np.stack(cols, axis=1)
                    coef = np.linalg.lstsq(A, yy, rcond=None)[0]
                    return float(np.sqrt(np.mean((A @ coef - yy) ** 2))), coef

                grid = np.linspace(0.0, 80.0, 321)
                vals = [fit_omega(o)[0] for o in grid]
                i = int(np.argmin(vals))
                lo = grid[max(0, i - 1)]
                hi = grid[min(len(grid) - 1, i + 1)]
                opt = minimize_scalar(lambda o: fit_omega(o)[0], bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
                fj = float(opt.x)
                rmsj, coefj = fit_omega(fj)
                if fj <= 1e-6 or rmsj > 1e-6 * max(1.0, float(np.sqrt(np.mean(yy ** 2)))):
                    ok = False
                    break
                f0[j] = fj
                psis[j] = math.atan2(-coefj[2], coefj[1])
            if not ok:
                continue
            best_sign = None
            if d <= 8:
                for signs in itertools.product((-1.0, 1.0), repeat=d):
                    s = np.asarray(signs)
                    theta = s * psis
                    resultant = abs(np.mean(np.exp(1j * theta)))
                    if best_sign is None or resultant > best_sign[0]:
                        best_sign = (resultant, s, theta)
            else:
                best_sign = (1.0, np.ones(d), psis)
            _, s, theta = best_sign
            alpha0 = s * f0

            def design(alpha, xx):
                ph = xx @ alpha
                cols = [np.ones(len(xx))]
                for k in range(1, K + 1):
                    cols.append(np.cos(k * ph))
                    cols.append(np.sin(k * ph))
                return np.stack(cols, axis=1)

            def resid_alpha(alpha):
                A = design(alpha, xt)
                coef = np.linalg.lstsq(A, yt, rcond=None)[0]
                return A @ coef - yt

            res = least_squares(resid_alpha, alpha0, method="lm", max_nfev=120)
            alpha = res.x
            A = design(alpha, xt)
            coef = np.linalg.lstsq(A, yt, rcond=None)[0]
            Av = design(alpha, xv)
            predv = Av @ coef
            rms = float(np.sqrt(np.mean((predv - yv) ** 2)))
            C = coef[0]
            est = float(C)
            bound = abs(C)
            for k in range(1, K + 1):
                a = coef[2 * k - 1]
                b = coef[2 * k]
                fac = 1.0 + 0j
                for aj in alpha:
                    z = k * aj
                    fac *= 1.0 if abs(z) < 1e-14 else (np.exp(1j * z) - 1) / (1j * z)
                est += float(a * fac.real + b * fac.imag)
                bound += math.hypot(a, b)
            if math.isfinite(est) and abs(est) <= 1.01 * bound + 1e-12 and rms < 1e-9 * yscale:
                if best is None or rms < best[0]:
                    best = (rms, est)
        except Exception:
            pass
    if best is not None and best[0] < 1e-7 * yscale:
        return best[1], x_all, y_all
    return None, x_all, y_all


def residual_M(vals, d, h=0.02):
    f0 = vals[0]
    grad = np.zeros(d)
    H = np.zeros((d, d))
    idx = 1
    for j in range(d):
        fm, fp = vals[idx], vals[idx + 1]
        idx += 2
        grad[j] = (fp - fm) / (2 * h)
        H[j, j] = (fp + fm - 2 * f0) / (h * h)
    for j in range(d):
        for k in range(j + 1, d):
            v = vals[idx:idx + 4]
            idx += 4
            H[j, k] = H[k, j] = (v[0] - v[1] - v[2] + v[3]) / (4 * h * h)
    M = H @ H + np.outer(grad, grad)
    evals, evecs = np.linalg.eigh(M)
    return evals, evecs


def abs_exp_integral(v, a, u):
    lengths = np.abs(v)
    lengths = lengths[lengths > 1e-14]
    lo = sum(min(0.0, x) for x in v)
    u0 = u - lo
    d = len(lengths)
    if d == 0:
        return math.exp(-a * abs(u0))
    L = lengths.sum()
    m = d - 1
    prod = math.prod(lengths)

    def Rm(T, aa, mm):
        if T <= 0:
            return 0.0
        if aa < 1e-14:
            return T ** (mm + 1) / (mm + 1)
        e = math.exp(-aa * T)
        vals = [(1 - e) / aa]
        for j in range(1, mm + 1):
            vals.append(-T ** j * e / aa + j / aa * vals[-1])
        return vals[mm]

    def Lm(T, aa, mm):
        if T <= 0:
            return 0.0
        if aa < 1e-14:
            return T ** (mm + 1) / (mm + 1)
        e = math.exp(-aa * T)
        vals = [(1 - e) / aa]
        for j in range(1, mm + 1):
            vals.append(T ** j / aa - j / aa * vals[-1])
        return vals[mm]

    total = 0.0
    for mask in range(1 << d):
        z = 0.0
        bits = 0
        for j in range(d):
            if mask >> j & 1:
                z += lengths[j]
                bits += 1
        sign = -1.0 if bits % 2 else 1.0
        if z >= u0:
            J = math.exp(-a * (z - u0)) * Rm(L - z, a, m)
        else:
            T = u0 - z
            J = Lm(T, a, m)
            T2 = L - z
            J += sum(math.comb(m, j) * T ** (m - j) * Rm(T2 - T, a, j) for j in range(m + 1))
        total += sign * J
    return float(total / (prod * math.factorial(m)))


def relu_integral(v, u):
    lengths = np.abs(v)
    lengths = lengths[lengths > 1e-14]
    lo = sum(min(0.0, x) for x in v)
    u0 = u - lo
    d = len(lengths)
    if d == 0:
        return max(0.0, -u0)
    L = lengths.sum()
    m = d - 1
    prod = math.prod(lengths)
    if u0 <= 0:
        return L / 2 - u0
    if u0 >= L:
        return 0.0
    total = 0.0
    for mask in range(1 << d):
        z = 0.0
        bits = 0
        for j in range(d):
            if mask >> j & 1:
                z += lengths[j]
                bits += 1
        sign = -1.0 if bits % 2 else 1.0
        T = L - z
        lower = max(0.0, u0 - z)
        if T <= 0:
            continue
        I1 = T ** (m + 2) / (m + 2) - lower ** (m + 2) / (m + 2)
        I0 = T ** (m + 1) / (m + 1) - lower ** (m + 1) / (m + 1)
        total += sign * (I1 + (z - u0) * I0)
    return float(total / (prod * math.factorial(m)))


def abs_integral(v, u):
    lengths = np.abs(v)
    L = lengths.sum()
    lo = sum(min(0.0, x) for x in v)
    u0 = u - lo
    return 2 * relu_integral(v, u) - (L / 2 - u0)


def ridge_spline_estimate(x, y, w):
    t = x @ w
    order = np.argsort(t)
    ts = t[order]
    ys = y[order]
    # average duplicate projections
    ut, inv = np.unique(ts, return_inverse=True)
    if len(ut) < 10:
        return None, 1e9, None
    uy = np.bincount(inv, weights=ys) / np.bincount(inv)
    spl = UnivariateSpline(ut, uy, s=0, k=3)
    lo = sum(min(0.0, v) for v in w)
    hi = sum(max(0.0, v) for v in w)
    lengths = np.abs(w)
    L = hi - lo
    if L < 1e-14:
        return float(spl(0.0)), 0.0, spl
    N = 8192
    ds = L / N
    grid = np.arange(N + 1) * ds
    p = np.zeros(N + 1)
    p[0] = 1.0 / ds
    for l in lengths:
        if l < 1e-14:
            continue
        q = np.zeros(N + 1)
        q[grid <= l] = 1.0 / l
        p = fftconvolve(p, q)[:N + 1] * ds
    est = float(np.trapezoid(spl(lo + grid) * p, grid))
    rms = float(np.sqrt(np.mean((spl(t) - y) ** 2)))
    return est, rms, spl


def additive_spline_estimate(x, y, n_basis=30, xv=None):
    n, d = x.shape
    k = 3
    cols = [np.ones(n)]
    integrals = [1.0]
    ts = []
    interior = np.linspace(0.0, 1.0, n_basis - k + 1)[1:-1]
    t0 = np.concatenate([np.zeros(k + 1), interior, np.ones(k + 1)])
    base_int = []
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        base_int.append(float(BSpline(t0, c, k).integrate(0.0, 1.0)))
    for j in range(d):
        t = t0
        ts.append(t)
        B = BSpline.design_matrix(x[:, j], t, k).toarray()
        # drop one basis function per coordinate to avoid intercept collinearity
        for i in range(1, n_basis):
            cols.append(B[:, i])
            integrals.append(base_int[i])
    A = np.stack(cols, axis=1)
    coef = np.linalg.lstsq(A, y, rcond=1e-10)[0]
    est = float(coef @ np.asarray(integrals))
    rms = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    predv = None
    if xv is not None:
        vcols = [np.ones(len(xv))]
        for j in range(d):
            Bv = BSpline.design_matrix(xv[:, j], ts[j], k).toarray()
            for i in range(1, n_basis):
                vcols.append(Bv[:, i])
        Av = np.stack(vcols, axis=1)
        predv = Av @ coef
    return est, rms, predv


def fit_one_exp(t, y):
    best = None
    for u0 in np.quantile(t, [0.1, 0.3, 0.5, 0.7, 0.9]):
        p0 = [math.log(3.0), float(u0), math.log(max(float(y.max()), 1e-6)), 0.0]
        lo = [-5.0, float(t.min()), -30.0, -100.0]
        hi = [5.0, float(t.max()), 30.0, 100.0]

        def resid(p):
            a = math.exp(float(np.clip(p[0], -5, 5)))
            A = math.exp(float(np.clip(p[2], -30, 30)))
            return p[3] + A * np.exp(-a * np.abs(t - p[1])) - y

        res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=150)
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, res.x)
    return best[1]


def fit_one_abs(t, y):
    best = None
    for u0 in np.quantile(t, [0.1, 0.3, 0.5, 0.7, 0.9]):
        p0 = [float(u0), 1.0, 0.0, 0.0]
        lo = [float(t.min()), -100.0, -100.0, -100.0]
        hi = [float(t.max()), 100.0, 100.0, 100.0]

        def resid(p):
            return p[3] + p[2] * t + p[1] * np.abs(t - p[0]) - y

        res = least_squares(resid, p0, bounds=(lo, hi), max_nfev=150)
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, res.x)
    return best[1]


def fit_kinked_sum(x, y, pp, pv, R, kind):
    n, d = x.shape
    resy = y.copy()
    respv = pv.copy()
    Vs = []
    aa = []
    uu = []
    AA = []
    bb = []
    C = 0.0
    for r in range(R):
        evals, evecs = residual_M(respv, d)
        v = evecs[:, -1]
        t = x @ v
        if kind == "exp":
            p = fit_one_exp(t, resy)
            a = math.exp(float(np.clip(p[0], -5, 5)))
            u = p[1]
            A = math.exp(float(np.clip(p[2], -30, 30)))
            C = p[3]
            Vs.append(v); aa.append(a); uu.append(u); AA.append(A)
            term = A * np.exp(-a * np.abs(x @ v - u))
            pterm = A * np.exp(-a * np.abs(pp @ v - u))
        else:
            p = fit_one_abs(t, resy)
            u, A, b, C = p
            Vs.append(v); uu.append(u); AA.append(A); bb.append(b)
            term = A * np.abs(x @ v - u) + b * (x @ v)
            pterm = A * np.abs(pp @ v - u) + b * (pp @ v)
        resy -= term
        respv -= pterm
    if kind == "exp":
        p0 = np.concatenate([[C], np.asarray(Vs).ravel(), np.log(aa), uu, np.log(AA)])

        def model(p):
            C = p[0]
            V = p[1:1 + R * d].reshape(R, d)
            la = p[1 + R * d:1 + R * d + R]
            u = p[1 + R * d + R:1 + R * d + 2 * R]
            lA = p[1 + R * d + 2 * R:]
            a = np.exp(np.clip(la, -5, 5))
            A = np.exp(np.clip(lA, -30, 30))
            t = x @ V.T
            return C + np.sum(A[None, :] * np.exp(-a[None, :] * np.abs(t - u[None, :])), axis=1)

        lo = np.concatenate([[-100.0], np.full(R * d, -10.0), np.full(R, -5.0), np.full(R, -10.0), np.full(R, -30.0)])
        hi = np.concatenate([[100.0], np.full(R * d, 10.0), np.full(R, 5.0), np.full(R, 10.0), np.full(R, 30.0)])
        res = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi), max_nfev=250)
        p = res.x
        C = p[0]
        V = p[1:1 + R * d].reshape(R, d)
        la = p[1 + R * d:1 + R * d + R]
        u = p[1 + R * d + R:1 + R * d + 2 * R]
        lA = p[1 + R * d + 2 * R:]
        a = np.exp(np.clip(la, -5, 5))
        A = np.exp(np.clip(lA, -30, 30))
        est = C + sum(A[r] * abs_exp_integral(V[r], a[r], u[r]) for r in range(R))
    else:
        p0 = np.concatenate([[C], np.asarray(Vs).ravel(), uu, AA, bb])

        def model(p):
            C = p[0]
            V = p[1:1 + R * d].reshape(R, d)
            u = p[1 + R * d:1 + R * d + R]
            A = p[1 + R * d + R:1 + R * d + 2 * R]
            b = p[1 + R * d + 2 * R:]
            t = x @ V.T
            return C + np.sum(A[None, :] * np.abs(t - u[None, :]) + b[None, :] * t, axis=1)

        lo = np.concatenate([[-100.0], np.full(R * d, -10.0), np.full(R, -10.0), np.full(R, -100.0), np.full(R, -100.0)])
        hi = np.concatenate([[100.0], np.full(R * d, 10.0), np.full(R, 10.0), np.full(R, 100.0), np.full(R, 100.0)])
        res = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi), max_nfev=250)
        p = res.x
        C = p[0]
        V = p[1:1 + R * d].reshape(R, d)
        u = p[1 + R * d:1 + R * d + R]
        A = p[1 + R * d + R:1 + R * d + 2 * R]
        b = p[1 + R * d + 2 * R:]
        est = C + sum(A[r] * abs_integral(V[r], u[r]) + b[r] * (V[r].sum() / 2) for r in range(R))
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    return float(est), rms, p


def solve_kinked(x, y, pp, pv, d, k):
    n = len(x)
    ntrain = max(80, int(0.85 * n))
    xt, yt = x[:ntrain], y[:ntrain]
    xv, yv = x[ntrain:], y[ntrain:]
    yscale = max(1e-12, float(np.sqrt(np.mean(y ** 2))))
    ymean_k = float(y.mean())
    candidates = [1, 2, 3, 4, 5, 6]
    if 1 <= k <= 6:
        candidates = [k] + [r for r in candidates if r != k]
    best = None
    for R in candidates:
        for kind in ("abs", "exp"):
            if time.monotonic() - START > 24:
                break
            try:
                est, rms, p = fit_kinked_sum(xt, yt, pp, pv, R, kind)
                if kind == "exp":
                    C = p[0]; V = p[1:1 + R * d].reshape(R, d)
                    la = p[1 + R * d:1 + R * d + R]
                    u = p[1 + R * d + R:1 + R * d + 2 * R]
                    lA = p[1 + R * d + 2 * R:]
                    a = np.exp(np.clip(la, -5, 5)); A = np.exp(np.clip(lA, -30, 30))
                    pred = C + np.sum(A[None, :] * np.exp(-a[None, :] * np.abs(xv @ V.T - u[None, :])), axis=1)
                else:
                    C = p[0]; V = p[1:1 + R * d].reshape(R, d)
                    u = p[1 + R * d:1 + R * d + R]
                    A = p[1 + R * d + R:1 + R * d + 2 * R]
                    b = p[1 + R * d + 2 * R:]
                    pred = C + np.sum(A[None, :] * np.abs(xv @ V.T - u[None, :]) + b[None, :] * (xv @ V.T), axis=1)
                vrms = float(np.sqrt(np.mean((pred - yv) ** 2)))
                plausible = math.isfinite(est) and abs(est) < 100 * max(1.0, abs(ymean_k), yscale)
                if ymean_k > 0:
                    plausible = plausible and 0.1 * ymean_k < est < 10 * ymean_k
                kthr = 1e-3
                if plausible and rms < kthr * yscale and vrms < kthr * yscale:
                    if best is None or rms < best[0]:
                        best = (rms, est)
                    if rms < 1e-10 * yscale:
                        return est
            except Exception:
                pass
    if best is not None:
        return best[1]
    return None


def weighted_kmeans(x, w, R, seed=0, iters=30):
    rng = np.random.default_rng(seed)
    w = np.maximum(w, 1e-300)
    idx = rng.choice(len(x), size=R, replace=False, p=w / w.sum())
    C = x[idx].copy()
    for _ in range(iters):
        dist = ((x[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = dist.argmin(axis=1)
        for r in range(R):
            m = lab == r
            if m.any():
                C[r] = (x[m] * w[m, None]).sum(axis=0) / w[m].sum()
    return C


def fit_mixture(x, y, R, seed=0, max_nfev=250):
    n, d = x.shape
    C = weighted_kmeans(x, y, R, seed)
    p0 = np.zeros((R, 1 + 2 * d))
    p0[:, 0] = math.log(max(float(y.max()), 1e-6) / R)
    p0[:, 1:1 + d] = math.log(10.0)
    p0[:, 1 + d:] = C
    lo = np.zeros((R, 1 + 2 * d)); hi = np.zeros((R, 1 + 2 * d))
    lo[:, 0] = -30; hi[:, 0] = 30
    lo[:, 1:1 + d] = -10; hi[:, 1:1 + d] = 10
    lo[:, 1 + d:] = 0; hi[:, 1 + d:] = 1

    def model(p):
        p = p.reshape(R, 1 + 2 * d)
        lw = p[:, 0]; la = p[:, 1:1 + d]; u = p[:, 1 + d:]
        a = np.exp(la)
        val = np.zeros(n)
        for r in range(R):
            val += np.exp(lw[r] - np.sum((a[r] * (x - u[r])) ** 2, axis=1))
        return val

    res = least_squares(lambda p: model(p) - y, p0.ravel(),
                        bounds=(lo.ravel(), hi.ravel()), max_nfev=max_nfev)
    return res.x


def mixture_predict(p, x, R):
    d = x.shape[1]
    p = p.reshape(R, 1 + 2 * d)
    lw = p[:, 0]; la = p[:, 1:1 + d]; u = p[:, 1 + d:]
    a = np.exp(la)
    val = np.zeros(len(x))
    for r in range(R):
        val += np.exp(lw[r] - np.sum((a[r] * (x - u[r])) ** 2, axis=1))
    return val


def mixture_integral(p, R, d):
    p = p.reshape(R, 1 + 2 * d)
    lw = p[:, 0]; la = p[:, 1:1 + d]; u = p[:, 1 + d:]
    a = np.exp(la); w = np.exp(lw)
    total = 0.0
    for r in range(R):
        total += w[r] * math.prod(gauss_factor(z, uu) for z, uu in zip(a[r], u[r]))
    return float(total)


def legendre_design(x, degree):
    n, d = x.shape
    z = 2 * x - 1
    maxk = degree
    uni = np.empty((d, maxk + 1, n))
    from scipy.special import eval_legendre
    for j in range(d):
        for k in range(maxk + 1):
            uni[j, k] = eval_legendre(k, z[:, j]) * math.sqrt(2 * k + 1)
    idx = []

    def rec(pos, rem, cur):
        if pos == d:
            if sum(cur) > 0:
                idx.append(tuple(cur))
            return
        for k in range(rem + 1):
            rec(pos + 1, rem - k, cur + [k])

    rec(0, degree, [])
    cols = []
    for t in idx:
        col = np.ones(n)
        for j, k in enumerate(t):
            if k:
                col *= uni[j, k]
        cols.append(col)
    return np.stack(cols, axis=1) if cols else np.empty((n, 0))


def polynomial_estimate(x, y, degree):
    A = legendre_design(x, degree)
    if A.shape[1] == 0:
        return float(y.mean()), 0.0
    am = A.mean(axis=0)
    Ac = A - am
    yc = y - y.mean()
    coef = np.linalg.lstsq(Ac, yc, rcond=1e-10)[0]
    est = float(y.mean() - coef @ am)
    rms = float(np.sqrt(np.mean((Ac @ coef - yc) ** 2)))
    return est, rms


def solve_checkerboard(oracle, x, y):
    lo = float(y.min()); hi = float(y.max())
    center = (lo + hi) / 2
    amp = (hi - lo) / 2
    if amp < 1e-14:
        return center
    d = x.shape[1]
    thresholds = []
    for j in range(d):
        def at(v):
            p = np.full(d, 0.5)
            p[j] = v
            return float(oracle.query(p)[0])
        v0 = at(0.0); v1 = at(1.0)
        if abs(v0 - v1) < 1e-12 * max(1.0, abs(center), abs(amp)):
            thresholds.append(0.5)
            continue
        a, b = 0.0, 1.0
        fa = v0
        for _ in range(30):
            m = (a + b) / 2
            fm = at(m)
            if abs(fm - fa) < 1e-12 * max(1.0, abs(center), abs(amp)):
                a = m
            else:
                b = m
        thresholds.append((a + b) / 2)
    t = np.asarray(thresholds)
    signs = np.where(x >= t, 1.0, -1.0)
    prod = signs.prod(axis=1)
    A = float(np.mean((y - center) * prod))
    return float(center + A * np.prod(1 - 2 * t))


def find_step_coordinate(x, y):
    zero = (y == 0.0)
    fz = zero.mean()
    if fz < 0.05 or fz > 0.95:
        return None
    best = None
    n, d = x.shape
    for j in range(d):
        order = np.argsort(x[:, j])
        z = zero[order]
        cs = np.concatenate([[0], np.cumsum(z)])
        total = cs[-1]
        k = np.arange(n + 1)
        # orient=1: zeros are a suffix (positive side is low).
        err_suffix = 2 * cs + n - k - total
        # orient=-1: zeros are a prefix (positive side is high).
        err_prefix = k + total - 2 * cs
        for orient, err in ((1, err_suffix), (-1, err_prefix)):
            kk = int(np.argmin(err))
            acc = 1 - err[kk] / n
            if best is None or acc > best[0]:
                best = (acc, j, orient)
    if best is not None and best[0] > 0.95:
        return best[1], best[2]
    return None


def solve_discontinuous(oracle, x_init, y_init, remaining_points):
    step = find_step_coordinate(x_init, y_init)
    if step is None:
        return None
    j, orient = step
    d = x_init.shape[1]

    def at(v):
        p = np.full(d, 0.5)
        p[j] = v
        return float(oracle.query(p)[0])

    v0 = at(0.0); v1 = at(1.0)
    z0 = (v0 == 0.0); z1 = (v1 == 0.0)
    if z0 == z1:
        return None
    a, b = 0.0, 1.0
    za = z0
    for _ in range(30):
        m = (a + b) / 2
        zm = (at(m) == 0.0)
        if zm == za:
            a = m
        else:
            b = m
    t = (a + b) / 2
    positive_low = not z0
    lo, hi = (0.0, t) if positive_low else (t, 1.0)
    # Sample the positive side adaptively with the remaining budget.
    nrem = oracle.budget - oracle.used
    if nrem > 0:
        u = qmc_points(d, nrem, 12345)
        u[:, j] = lo + (hi - lo) * u[:, j]
        yrem = oracle.query(u)
        x = np.vstack([x_init, u])
        y = np.concatenate([y_init, yrem])
    else:
        x, y = x_init, y_init
    mask = y > 0
    if mask.sum() < 40:
        return None
    xp, yp = x[mask], y[mask]
    best = None
    for kind in ("continuous", "gaussian", "peak", "cusp"):
        model = SeparableModel(kind, d)
        try:
            p = model.fit(xp, yp)
            pred = model.predict(p, xp)
            rms = float(np.sqrt(np.mean((pred - yp) ** 2)))
            est = model.integral(p, step=(j, lo, hi))
            if math.isfinite(est) and (best is None or rms < best[0]):
                best = (rms, est)
        except Exception:
            pass
    if best is not None and best[0] < 1e-7 * max(1.0, float(np.sqrt(np.mean(yp ** 2)))):
        return best[1]
    return None


def fit_models(x, y, w, ratio, pp=None, pv=None, k=0, diag_ratio=0.0):
    if time.monotonic() - START > 24:
        return float(y.mean())
    n, d = x.shape
    ntrain = max(60, int(0.75 * n))
    xt, yt = x[:ntrain], y[:ntrain]
    xv, yv = x[ntrain:], y[ntrain:]
    yscale = max(1e-12, float(np.sqrt(np.mean(y ** 2))))
    candidates = []

    def add(name, est, rms):
        if math.isfinite(est):
            candidates.append((rms, name, est))

    # Low-degree polynomial control variate.
    maxdeg = {2: 12, 3: 8, 5: 5, 8: 4}.get(d, 4)
    for deg in range(2, maxdeg + 1):
        try:
            A = legendre_design(xt, deg)
            if A.shape[1] >= ntrain // 2:
                continue
            am = A.mean(axis=0); Ac = A - am; yc = yt - yt.mean()
            coef = np.linalg.lstsq(Ac, yc, rcond=1e-10)[0]
            Av = legendre_design(xv, deg)
            pred = yt.mean() + (Av - am) @ coef
            rms = float(np.sqrt(np.mean((pred - yv) ** 2)))
            est, _ = polynomial_estimate(x, y, deg)
            add("poly%d" % deg, est, rms)
        except Exception:
            pass

    poly = [c for c in candidates if c[1].startswith("poly")]
    if poly:
        poly.sort(key=lambda c: c[0])
        if poly[0][0] < 1e-8 * yscale:
            return poly[0][2]

    try:
        nb = 40 if d <= 3 else 30
        est, _, predv = additive_spline_estimate(xt, yt, nb, xv)
        if predv is not None:
            vrms = float(np.sqrt(np.mean((predv - yv) ** 2)))
            sign_ch = bool(np.any(y < 0))
            additive_like = diag_ratio > 0.9 or k >= d - 1
            mean_ok = abs(est - float(y.mean())) < 0.1 * max(float(y.std()), 1e-12)
            thr = 1e-2 if (sign_ch or (diag_ratio > 0.9 and mean_ok)) else 1e-8
            if vrms < thr * yscale and (sign_ch or mean_ok):
                add("additive", est, vrms)
                return est
    except Exception:
        pass

    if np.all(y > 0):
        for kind in ("gaussian", "continuous", "peak", "cusp"):
            try:
                model = SeparableModel(kind, d)
                p = model.fit(xt, yt)
                pred = model.predict(p, xv)
                rms = float(np.sqrt(np.mean((pred - yv) ** 2)))
                est = model.integral(p)
                add(kind, est, rms)
            except Exception:
                pass
        # Mixture of separable Gaussian bumps.
        best_rank1 = min([c[0] for c in candidates if c[1] in ("gaussian", "continuous", "peak", "cusp")], default=1e9)
        if best_rank1 > 1e-6 * yscale:
            ranks = () if d >= 5 else (3, 2, 4)
            for R in ranks:
                try:
                    p = fit_mixture(xt, yt, R, max_nfev=80 if d >= 5 else 250)
                    pred = mixture_predict(p, xv, R)
                    rms = float(np.sqrt(np.mean((pred - yv) ** 2)))
                    est = mixture_integral(p, R, d)
                    if rms < 1e-7 * yscale:
                        add("mix%d" % R, est, rms)
                    if rms < 1e-8 * yscale:
                        break
                except Exception:
                    pass

    exact_already = any(c[0] < 1e-8 * yscale for c in candidates)
    if (ratio > 0.9 or np.any(y < 0)) and not exact_already:
        try:
            p = fit_oscillatory(xt, yt, w)
            pred = oscillatory_predict(p, xv)
            rms = float(np.sqrt(np.mean((pred - yv) ** 2)))
            est = oscillatory_integral(p, d)
            add("osc", est, rms)
        except Exception:
            pass

    if ratio > 0.8 or k == 1:
        try:
            est, _, spl = ridge_spline_estimate(xt, yt, w)
            if spl is not None:
                vrms = float(np.sqrt(np.mean((spl(xv @ w) - yv) ** 2)))
                if vrms < 1e-8 * yscale and abs(est - float(y.mean())) < 0.1 * max(float(y.std()), 1e-12):
                    add("ridge_spline", est, vrms)
        except Exception:
            pass

    additive_good = any(c[1] == "additive" and c[0] < 1e-5 * yscale for c in candidates)
    if pp is not None and k <= 6 and not additive_good and not any(c[0] < 1e-8 * yscale for c in candidates):
        kest = solve_kinked(x, y, pp, pv, d, k)
        if kest is not None:
            return kest

    if not candidates:
        return float(y.mean())
    candidates.sort(key=lambda c: c[0])
    rms, name, est = candidates[0]
    if rms < 1e-3 * yscale:
        return est
    return float(y.mean())


def main():
    global START
    START = time.monotonic()
    init = json.loads(sys.stdin.readline())
    if init.get("type") != "init":
        raise SystemExit("missing init")
    d = int(init["dimension"])
    budget = int(init["budget"])
    seed = int(init.get("seed", 0))
    oracle = Oracle(d, budget)

    pp = pilot_points(d)
    pv = oracle.query(pp)
    grad, H, w, ratio = pilot_derivatives(pp, pv, d)

    nq = budget - oracle.used
    if nq <= 0:
        send({"type": "result", "estimate": float(pv[0])})
        return
    qpts = qmc_points(d, nq, seed)
    ninit = min(256, nq)
    x_init = qpts[:ninit]
    y_init = oracle.query(x_init)
    fallback_mean = float(y_init.mean())

    ymin, ymax = float(y_init.min()), float(y_init.max())
    ymean = float(y_init.mean())
    ystd = float(y_init.std())
    two_level = False
    if ystd > 1e-14:
        center = (ymin + ymax) / 2
        amp = (ymax - ymin) / 2
        s = (y_init - center) / amp
        lower_frac = float(np.mean(y_init < center))
        two_level = bool(np.max(np.abs(np.abs(s) - 1)) < 1e-8 and 0.02 < lower_frac < 0.98)

    estimate = None
    try:
        if two_level:
            estimate = solve_checkerboard(oracle, x_init, y_init)
        else:
            step = find_step_coordinate(x_init, y_init)
            if step is not None:
                estimate = solve_discontinuous(oracle, x_init, y_init, qpts[ninit:])
            if estimate is None:
                k, _, _ = active_rank(grad, H)
                ymean0 = float(y_init.mean()); ystd0 = float(y_init.std())
                sign_changing = bool(np.any(y_init < 0))
                low_mean = abs(ymean0) < 0.2 * max(ystd0, 1e-12)
                fourier_like = sign_changing or (low_mean and k <= 2)
                diag_ratio = float(np.sum(np.diag(H) ** 2) / max(float(np.sum(H * H)), 1e-300))
                if fourier_like and diag_ratio <= 0.9:
                    est, xf, yf = solve_fourier(oracle, d, k, seed, x_init, y_init, N=128)
                elif False:
                    est, xf, yf = solve_fourier(oracle, d, k, seed, x_init, y_init, N=64)
                else:
                    est, xf, yf = None, None, None
                if est is not None or xf is not None:
                    if yf is not None:
                        fallback_mean = float(yf.mean())
                    if est is not None:
                        estimate = est
                    elif xf is not None:
                        if fourier_like:
                            estimate = fit_models(xf, yf, w, ratio, pp, pv, k, diag_ratio)
                        else:
                            estimate = fit_models(xf, yf, w, ratio, pp, pv, k, diag_ratio)
                if estimate is None:
                    rem = budget - oracle.used
                    if rem > 0:
                        if step is None and nq > ninit:
                            extra = qpts[ninit:ninit + rem]
                        else:
                            extra = qmc_points(d, rem, seed + 1)
                        y_rest = oracle.query(extra)
                        x = np.vstack([x_init, extra])
                        y = np.concatenate([y_init, y_rest])
                        fallback_mean = float(y.mean())
                    else:
                        x, y = x_init, y_init
                    estimate = fit_models(x, y, w, ratio, pp, pv, k, diag_ratio)
    except Exception:
        estimate = fallback_mean

    if not math.isfinite(estimate):
        estimate = fallback_mean
    send({"type": "result", "estimate": float(estimate)})


if __name__ == "__main__":
    main()
