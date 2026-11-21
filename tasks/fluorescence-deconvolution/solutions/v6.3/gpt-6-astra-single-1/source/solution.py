#!/usr/bin/env python3
"""Resource-bounded, nonnegative fluorescence reconstruction."""
import os
import sys
import time

START = time.monotonic()
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import ctypes as ct
import importlib.util
import sysconfig
import numpy as np
import scipy

_extension = os.path.join(os.path.dirname(scipy.__file__), "optimize",
                          "_lbfgsb" + sysconfig.get_config_var("EXT_SUFFIX"))
_spec = importlib.util.spec_from_file_location("_lbfgsb", _extension)
_lbfgsb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lbfgsb)

LBFGS_FACTR = 1e-8 / np.finfo(np.float64).eps
np.finfo(np.float32)
GRID = np.indices((32, 32, 32), dtype=np.float32).reshape(3, -1).T
F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


class BudgetExpired(Exception):
    pass


def bounded_optimize(objective, initial, bounds, maxiter):
    """Reverse-communication interface to the installed SciPy L-BFGS-B engine."""
    n, history = len(initial), 12
    x = initial.copy()
    limits = np.asarray(bounds, np.float64)
    lower, upper = limits[:, 0].copy(), limits[:, 1].copy()
    nbd = np.full(n, 2, np.int32)
    f, g = np.array(0., np.float64), np.zeros(n, np.float64)
    wa = np.zeros(2 * history * n + 5 * n + 11 * history**2 + 8 * history)
    iwa = np.zeros(3 * n, np.int32)
    task, csave = np.zeros(1, "S60"), np.zeros(1, "S60")
    lsave, isave, dsave = np.zeros(4, np.int32), np.zeros(44, np.int32), np.zeros(29)
    task[:] = b"START"
    iterations = 0
    for _ in range(maxiter * 12 + 30):
        _lbfgsb.setulb(history, x, lower, upper, nbd, f, g,
                       LBFGS_FACTR, 1e-3, wa, iwa, task, -1,
                       csave, lsave, isave, dsave, 8)
        request = task.tobytes()
        if request.startswith(b"FG"):
            value, g = objective(x)
            f = np.array(value, np.float64)
        elif request.startswith(b"NEW_X"):
            iterations += 1
            if iterations >= maxiter:
                break
        else:
            break


def load_backend():
    lib = ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "native.so"))
    lib.convolve.argtypes = [F32, F32, F32]
    lib.smooth.argtypes = [F32, F32, F32]
    lib.maximum5.argtypes = [F32, F32]
    lib.render.argtypes = [F64, I32, ct.c_int, F32]
    lib.evaluate.argtypes = [
        F64, I32, ct.c_int, F32, F32, F32,
        ct.c_double, ct.c_double, ct.c_double, F64, F32,
    ]
    lib.evaluate.restype = ct.c_double
    lib.prepare_likelihood.argtypes = [F32, ct.c_double, ct.c_double, ct.c_int]
    lib.configure.argtypes = [ct.c_double]
    return lib


class Solver:
    def __init__(self, y, kernels, lib, stop):
        self.y = y
        self.k = np.ascontiguousarray(kernels, dtype=np.float32).ravel()
        self.lib = lib
        self.stop = stop
        self.fallback = y
        self.smoothing_kernels = {}
        sm = self.smooth(y, 1.8)
        bg = y[sm < np.quantile(sm, .25)]
        level = np.sqrt(2 * np.mean(bg * bg)) if bg.size else 0.
        self.sigma, self.gain = (.008, 80.) if level < .015 else (.025, 25.)
        self.weights = np.ascontiguousarray(
            1 / (np.maximum(self.smooth(y, .8) - .3 * self.sigma, .001)
                 / self.gain + self.sigma**2), dtype=np.float32)
        self.lib.prepare_likelihood(y, self.sigma, self.gain, 0)

    def left(self):
        return self.stop - time.monotonic()

    def check(self, reserve=0.):
        if self.left() < reserve:
            raise BudgetExpired

    def conv(self, x, kernels=None):
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty_like(x)
        self.lib.convolve(x, out, self.k if kernels is None else kernels)
        return out

    def smooth(self, x, sigma):
        if sigma not in self.smoothing_kernels:
            d = np.arange(-4, 5)
            k = np.exp(-.5 * (d / sigma)**2)
            k[np.abs(d) > int(4 * sigma + .5)] = 0
            self.smoothing_kernels[sigma] = (k / k.sum()).astype(np.float32)
        out = np.empty_like(x)
        self.lib.smooth(np.ascontiguousarray(x, dtype=np.float32), out,
                        self.smoothing_kernels[sigma])
        return out

    def maximum5(self, x):
        out = np.empty_like(x)
        self.lib.maximum5(np.ascontiguousarray(x, dtype=np.float32), out)
        return out

    def render(self, p, types):
        out = np.empty_like(self.y)
        pp = np.ascontiguousarray(p, dtype=np.float64)
        tt = np.ascontiguousarray(types, dtype=np.int32)
        self.lib.render(pp, tt, len(types), out)
        return out

    def baseline(self):
        yo = np.maximum(self.y - .8 * self.sigma, 0)
        x = yo.copy() + .0001
        sens = self.conv(np.ones_like(x))
        offset = .25 * self.sigma
        for _ in range(30):
            if self.left() < .015:
                break
            x *= self.conv((yo + offset) / (self.conv(x) + offset + 1e-8)) / sens
            x = self.smooth(x, .4)
        self.fallback = x
        return x

    def fit(self, p, types, maxiter=60, tie=False, prior=8.):
        if not len(types):
            return np.zeros_like(self.y), np.empty((0, 8), np.float64)
        types = np.ascontiguousarray(types, dtype=np.int32)
        full = np.ascontiguousarray(p, dtype=np.float64).reshape(-1, 8)
        if np.any(types >= 2):
            self.lib.configure(.5)
        elif np.any(types == 1):
            self.lib.configure(.85 * np.exp(np.median(full[types == 1, 0])))
        else:
            self.lib.configure(.85)
        bounds = []
        for typ in types:
            if typ == 0:
                bounds += [(-3.51, .588), (.5, 30.5), (.5, 30.5), (.5, 30.5),
                           (0., 0.), (0., 0.), (0., 0.), (-.799, .588)]
            elif typ == 1:
                bounds += [(-2.996, .588), (-2., 33.), (-2., 33.), (-2., 33.),
                           (-30., 30.), (-30., 30.), (-30., 30.), (-.799, .917)]
            else:
                bounds += [(-4.606, .694), (-3., 34.), (0., 0.), (0., 0.),
                           (-.9, .9), (-.9, .9), (0., 0.), (-1.204, 1.099)]
        mapping = np.arange(full.size)
        tube_ix = np.flatnonzero(types == 1) * 8
        if tie and len(tube_ix) > 1:
            mapping[tube_ix] = tube_ix[0]
        active, mapping = np.unique(mapping, return_inverse=True)
        initial = full.ravel()[active]
        bounds = [bounds[j] for j in active]
        initial = np.clip(initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        image = np.empty_like(self.y)
        best = initial.copy()
        best_loss = float("inf")

        def objective(v):
            nonlocal best, best_loss
            self.check(.012)
            pp = np.ascontiguousarray(v[mapping])
            grad = np.empty_like(pp)
            loss = self.lib.evaluate(
                pp, types, len(types), self.y, self.weights, self.k,
                self.sigma, self.gain, prior, grad, image)
            if not np.isfinite(loss) or not np.isfinite(grad).all():
                raise BudgetExpired
            if loss < best_loss:
                best_loss = loss
                best = v.copy()
            return loss, np.bincount(mapping, weights=grad, minlength=len(v))

        try:
            bounded_optimize(objective, initial, bounds, maxiter)
        except BudgetExpired:
            pass
        pp = best[mapping].reshape(-1, 8)
        return self.render(pp, types), pp

    def sheet_seeds(self, x):
        params, types = [], []
        for axis in range(3):
            profile = x.mean(axis=tuple(a for a in range(3) if a != axis))
            if profile.max() < .09 or profile.max() < profile.mean() * 2.5:
                continue
            j = int(np.argmax(profile))
            q = np.arange(max(0, j - 3), min(32, j + 4))
            w = np.maximum(profile[q] - np.median(profile), 0)
            p = np.zeros(8)
            p[0] = np.log(profile.max())
            p[1] = np.sum(q * w) / (w.sum() + 1e-10)
            p[7] = np.log(.7)
            params.append(p)
            types.append(2 + axis)
        return params, types

    def tube_seed(self, x, rng):
        self.check(.015)
        mask = x.ravel() > max(.06, x.max() * .16)
        pos, weights = GRID[mask], x.ravel()[mask]
        if len(pos) < 60:
            return None
        sample = rng.choice(len(pos), min(len(pos), 400), replace=False,
                            p=weights / weights.sum())
        pp, ww = pos[sample], weights[sample]
        pairs = rng.choice(len(pos), (200, 2), p=weights / weights.sum())
        origins = pos[pairs[:, 0]]
        directions = pos[pairs[:, 1]] - origins
        lengths = np.linalg.norm(directions, axis=1)
        directions /= np.maximum(lengths[:, None], 1e-5)
        delta = pp[None, :, :] - origins[:, None, :]
        axial = np.sum(delta * directions[:, None, :], axis=2)
        d2 = np.sum(delta * delta, axis=2) - axial * axial
        scores = np.sum(ww[None, :] * np.exp(-d2 / (2 * 1.3**2)), axis=1)
        scores[lengths < 6] = 0
        for best in np.argsort(-scores)[:16]:
            self.check(.005)
            center = origins[best].copy()
            direction = directions[best].copy()
            for _ in range(5):
                delta = pos - center
                t = delta @ direction
                d2 = np.sum(delta * delta, axis=1) - t * t
                w = weights * np.exp(-d2 / 2)
                center = np.average(pos, axis=0, weights=w)
                delta = pos - center
                cov = (delta.T * w) @ delta / w.sum()
                eig, vec = np.linalg.eigh(cov)
                direction = vec[:, -1]
            width = np.sqrt(max(0, np.mean(eig[:2])))
            half = np.sqrt(max(0, 3 * (eig[-1] - width * width)))
            if half < 3.5:
                continue
            t = (pos - center) @ direction
            d2 = np.sum((pos - center)**2, axis=1) - t * t
            near = d2 < 2.2**2
            bins = np.arange(-half, half + 1, 2.)
            hist = np.histogram(t[near], bins, weights=weights[near])[0]
            if len(hist) < 3 or np.quantile(hist, .2) < np.median(hist) * .4:
                continue
            if np.sum(weights[near & (np.abs(t) < half)]) / (2 * half) < .8:
                continue
            amplitude = np.quantile(weights[near], .95)
            if amplitude < .15:
                continue
            p = np.zeros(8)
            p[0] = np.log(amplitude)
            p[1:4] = center
            p[4:7] = direction * max(2., half - 1)
            p[7] = np.log(np.clip(width, .7, 1.4))
            return p
        return None

    def residual_score(self, model):
        m = self.conv(model)
        expected = m + self.sigma * .39894228 * np.exp(-self.gain * m)
        filt = np.concatenate([
            np.convolve(q, np.array([.05, .24, .42, .24, .05], np.float32))[2:-2]
            for q in self.k.reshape(3, 9)])
        filt = filt.reshape(3, 9)
        filt /= filt.sum(axis=1, keepdims=True)
        filt = filt.ravel()
        variance = m / self.gain + self.sigma**2 * .5
        return self.conv(self.y - expected, filt) / np.sqrt(
            self.conv(variance, filt * filt) + 1e-12)

    def point_seeds(self, model, base):
        score = self.residual_score(model)
        residual = np.maximum(base - model, 0)
        ix = np.argwhere((residual == self.maximum5(residual)) &
                         (residual > .025) & (score > 3.4))
        ix = ix[np.argsort(-score[tuple(ix.T)])]
        params = []
        for center in ix[:16]:
            p = np.zeros(8)
            p[0] = np.log(.7)
            p[1:4] = center
            p[7] = np.log(.8)
            params.append(p)
        return params

    def prune(self, p, types):
        tt = np.ascontiguousarray(types, dtype=np.int32)
        p = np.ascontiguousarray(p, dtype=np.float64)
        image, grad = np.empty_like(self.y), np.empty(p.size)

        def loss(pp):
            self.check(.003)
            return self.lib.evaluate(
                pp.ravel(), tt, len(tt), self.y, self.weights, self.k,
                self.sigma, self.gain, 0., grad, image)

        base = loss(p)
        keep = np.ones(len(tt), bool)
        for j in np.flatnonzero(tt == 0):
            q = p.copy()
            q[j, 0] = -25
            if loss(q) - base < 7.:
                keep[j] = False
        return p[keep], tt[keep].tolist()

    def posterior_points(self, p, types):
        out = self.render(p, types)
        m = self.conv(out)
        psfvar = np.sum(self.k.reshape(3, 9) * np.arange(-4, 5, dtype=np.float32)**2,
                        axis=1)
        for j in np.flatnonzero(np.asarray(types) == 0):
            if self.left() < .006:
                break
            pp = p[j]
            amp, s2 = np.exp(pp[0]), np.exp(2 * pp[7])
            mass = amp * (2 * np.pi * s2)**1.5
            effvar = s2 + psfvar
            peak = mass / np.sqrt((2 * np.pi)**3 * np.prod(effvar))
            pos = np.clip(np.round(pp[1:4]).astype(int), 0, 31)
            bg = max(0., float(m[tuple(pos)]) - peak) + self.sigma**2 * self.gain
            neff = mass * self.gain / (1 + 4 * bg / peak)
            v = np.minimum(.5 * effvar / max(neff, 1), 1.5)
            delta = GRID - pp[1:4]
            old = amp * np.exp(-.5 * np.sum(delta**2, axis=1) / s2)
            new = amp * np.sqrt(s2**3 / np.prod(s2 + v)) * np.exp(
                -.5 * np.sum(delta**2 / (s2 + v), axis=1))
            out += (new - old).reshape(out.shape).astype(np.float32)
        return np.maximum(out, 0)

    def check_model(self, out, base):
        score = self.residual_score(out)
        mask = (self.conv(out) > .015) | (self.conv(base) > .02)
        if np.count_nonzero(mask) < 64:
            return out
        quality = float(np.mean(score[mask]**2))
        if quality <= 1.6:
            return out
        # Coherent forward residuals indicate a shape-model mismatch.
        local = np.clip((self.smooth(score * score, .8) - 3.) / 10., 0., 1.)
        global_weight = np.clip((quality - 3.) / 6., 0., 1.)
        blend = np.maximum(local, global_weight)
        return (1 - blend) * out + blend * base

    def run(self):
        base = self.baseline()
        self.check(.035)
        p, types = self.sheet_seeds(base)
        model = np.zeros_like(base)
        if p:
            model, pp = self.fit(p, types, maxiter=60)
            p = list(pp)
            self.fallback = np.maximum(base, model)
        else:
            rng = np.random.default_rng(1234)
            residual = base.copy()
            for _ in range(5):
                if self.left() < .105:
                    break
                tp = self.tube_seed(residual, rng)
                if tp is None:
                    break
                candidate, pp = self.fit(p + [tp], types + [1], maxiter=25)
                last = pp[-1]
                if (np.exp(last[7]) < .62 or np.linalg.norm(last[4:7]) < 3.0 or
                        np.exp(last[0] + 2 * last[7]) * 2 * np.pi < .85):
                    break
                p, types, model = list(pp), types + [1], candidate
                residual = np.maximum(base - model, 0)
                self.fallback = np.maximum(base, model)
        self.check(.03)
        new = self.point_seeds(model, base)
        p += new
        types += [0] * len(new)
        if not len(types):
            return np.zeros_like(base)
        self.check(.025)
        self.lib.prepare_likelihood(self.y, self.sigma, self.gain, 1)
        out, p = self.fit(p, types, maxiter=60, tie=True)
        self.fallback = out
        if self.left() > .035 and 0 in types:
            try:
                pp, tt = self.prune(p, types)
                if len(tt) != len(types):
                    out, p = self.fit(pp, tt, maxiter=25, tie=True)
                    types = tt
                    self.fallback = out
            except BudgetExpired:
                pass
        if self.left() > .012 and 0 in types:
            out = self.posterior_points(p, types)
        if self.left() > .008:
            out = self.check_model(out, base)
        return out


def generic_reconstruction(y, h, stop):
    from scipy.fft import rfftn, irfftn, next_fast_len
    from scipy.ndimage import gaussian_filter
    shape = y.shape
    padded = tuple(next_fast_len(a + b - 1) for a, b in zip(shape, h.shape))
    center = tuple(v // 2 for v in h.shape)
    crop = tuple(slice(c, c + n) for c, n in zip(center, shape))
    spectrum = rfftn(h, padded)
    reverse = rfftn(h[::-1, ::-1, ::-1], padded)

    def conv(x, kernel):
        return irfftn(rfftn(x, padded) * kernel, padded)[crop].astype(np.float32)

    scale = .025 if np.quantile(y, .75) > .014 else .008
    data = np.maximum(y - .7 * scale, 0)
    x = data + .0001
    sensitivity = np.maximum(conv(np.ones_like(y), reverse), 1e-6)
    for _ in range(35):
        if time.monotonic() > stop - .008:
            break
        x *= conv((data + .2 * scale) / (conv(x, spectrum) + .2 * scale + 1e-8),
                  reverse) / sensitivity
        x = gaussian_filter(x, .4)
    return x


def main():
    stop = START + 1.07
    absolute = os.environ.get("DECONV_DEADLINE_MONOTONIC")
    if absolute:
        try:
            stop = min(stop, float(absolute) - .065)
        except ValueError:
            pass
    with np.load(sys.argv[1], allow_pickle=False) as data:
        y = np.ascontiguousarray(data["observed"], dtype=np.float32)
        h = np.ascontiguousarray(data["psf"], dtype=np.float32)
    result = y.copy()
    solver = None
    try:
        if np.any(y) and time.monotonic() < stop - .025:
            if y.shape == (32, 32, 32) and h.shape == (9, 9, 9):
                kernels = [h.sum(axis=tuple(j for j in range(3) if j != axis))
                           for axis in range(3)]
                product = (kernels[0][:, None, None] * kernels[1][None, :, None] *
                           kernels[2][None, None, :])
                symmetric = max(float(np.max(np.abs(k - k[::-1]))) for k in kernels)
                if np.max(np.abs(product - h)) < 1e-6 and symmetric < 1e-6:
                    solver = Solver(y, np.concatenate(kernels), load_backend(), stop)
                    result = solver.run()
                else:
                    result = generic_reconstruction(y, h, stop)
            else:
                result = generic_reconstruction(y, h, stop)
    except Exception:
        if solver is not None:
            result = solver.fallback
    result = np.asarray(result, dtype=np.float32)
    np.nan_to_num(result, copy=False, nan=0., posinf=0., neginf=0.)
    np.maximum(result, 0., out=result)
    with open(sys.argv[2], "wb") as output:
        np.save(output, result, allow_pickle=False)
        output.flush()


if __name__ == "__main__":
    main()
