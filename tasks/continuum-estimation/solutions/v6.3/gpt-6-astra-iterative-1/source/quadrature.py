"""Stable integration over interpolating power-series models."""
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def gauss_grid(n, low, high):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    half = (high - low) / 2.0
    return low + (nodes + 1.0) * half, weights * half


class PowerModel:
    def __init__(self, x, y, offset=1.5):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n = len(x)
        self.lx = np.log(x / x[-1])
        self.log_max_x = np.log(x[-1])
        self.scale = np.ptp(y)
        self.yn = (y - y[0]) / self.scale
        self.offset = offset
        self.deadline = time.monotonic() + 30.0

    def check_time(self):
        if time.monotonic() > self.deadline:
            raise ValueError("Numerical integration budget exceeded")

    def design(self, p, d):
        p, d = np.broadcast_arrays(p, d)
        exponents = np.stack((p, p+d, p+d+self.offset), axis=-1)
        return np.concatenate((
            np.ones(p.shape + (self.n, 1)),
            np.exp(self.lx[:, None] * exponents[..., None, :]),
        ), axis=-1)

    def convert(self, p, d, coefficients, log_weights):
        p, d = np.broadcast_arrays(p, d)
        powers = np.stack((p, p+d, p+d+self.offset), axis=-1)
        transform = self.scale * np.exp(-self.log_max_x * powers)
        abc = coefficients[..., 1:] * transform
        log_weights = log_weights + np.log(transform).sum(axis=-1)
        z = self.y[0] + self.scale * coefficients[..., 0]
        return p, d, abc, z, log_weights

    def four_samples(self, outer_order=500, inner_order=100):
        pmin, pmax = .25, 3.6
        delta, delta_weight = gauss_grid(outer_order, .2, 3.1)
        p_scan = np.linspace(pmin, pmax, 100)
        matrix = self.design(p_scan[:, None], delta[None, :])
        coefs = np.linalg.solve(matrix, np.broadcast_to(self.yn, matrix.shape[:-1]))
        aa = self.convert(p_scan[:, None], delta[None, :], coefs, np.zeros(coefs.shape[:-1]))[2]
        energy_grid = np.sum((aa / np.array((1.6, 1., .4)))**2, axis=-1)

        left_rows, right_rows, ds, ws = [], [], [], []
        for j, d in enumerate(delta):
            if j % 32 == 0:
                self.check_time()
            def energy(p):
                coeff = np.linalg.solve(self.design(p, d), self.yn)
                powers = np.array((p, p+d, p+d+self.offset))
                abc = coeff[1:] * self.scale * np.exp(-self.log_max_x * powers)
                return np.sum((abc / np.array((1.6, 1., .4)))**2)

            eg = energy_grid[:, j]
            minima = np.flatnonzero(
                (eg <= np.r_[np.inf, eg[:-1]]) & (eg <= np.r_[eg[1:], np.inf])
            )
            intervals = []
            for k in minima:
                lindex, rindex = max(0, k-1), min(len(p_scan)-1, k+1)
                opt = minimize_scalar(
                    energy, bounds=(p_scan[lindex], p_scan[rindex]),
                    method="bounded", options={"xatol": 2e-12},
                )
                mode, emin = opt.x, opt.fun
                if emin > 100 or not np.isfinite(emin):
                    continue
                cut = max(16., emin+9.)
                li = min(k, np.searchsorted(p_scan, mode)-1)
                ri = max(k, np.searchsorted(p_scan, mode))
                while li >= 0 and eg[li] < cut:
                    li -= 1
                while ri < len(p_scan) and eg[ri] < cut:
                    ri += 1
                left = pmin if li < 0 else brentq(
                    lambda p: energy(p)-cut, p_scan[li], mode, xtol=2e-12
                )
                right = pmax if ri == len(p_scan) else brentq(
                    lambda p: energy(p)-cut, mode, p_scan[ri], xtol=2e-12
                )
                intervals.append((left, right))
            intervals.sort()
            merged = []
            for left, right in intervals:
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                else:
                    merged.append((left, right))
            for left, right in merged:
                left_rows.append(left)
                right_rows.append(right)
                ds.append(d)
                ws.append(delta_weight[j])
        if not ds:
            raise ValueError("No plausible four-point power models")
        nodes, weights = gauss_grid(inner_order, 0., 1.)
        lo, hi = np.array(left_rows), np.array(right_rows)
        pp = lo[:, None] + (hi-lo)[:, None]*nodes
        dd = np.broadcast_to(np.array(ds)[:, None], pp.shape)
        matrix = self.design(pp, dd)
        coefs = np.linalg.solve(matrix, np.broadcast_to(self.yn, matrix.shape[:-1]))
        logw = np.log((hi-lo)[:, None] * np.array(ws)[:, None] * weights)
        logw -= np.linalg.slogdet(matrix)[1]
        return self.convert(pp.ravel(), dd.ravel(), coefs.reshape(-1, 4), logw.ravel())

    def five_samples(self, outer_order=500, scan_order=130):
        pmin, pmax, dmin, dmax = .25, 3.6, .2, 3.1
        samples = []
        for solve_p in (True, False):
            if solve_p:
                outer, outer_weight = gauss_grid(outer_order, dmin, dmax)
                scan = np.linspace(pmin, pmax, scan_order)
                matrix = self.design(scan[:, None], outer[None, :])
            else:
                outer, outer_weight = gauss_grid(outer_order, pmin, pmax)
                scan = np.linspace(dmin, dmax, scan_order)
                matrix = self.design(outer[None, :], scan[:, None])
            augmented = np.concatenate((
                matrix,
                np.broadcast_to(self.yn[:, None], matrix.shape[:-1] + (1,)),
            ), axis=-1)
            signs = np.linalg.slogdet(augmented)[0]
            scan_index, outer_index = np.nonzero(signs[:-1] * signs[1:] < 0)
            if len(scan_index) > 6 * outer_order:
                raise ValueError("Unresolved interpolation constraints")
            for j, (inner_i, outer_i) in enumerate(zip(scan_index, outer_index)):
                if j % 64 == 0:
                    self.check_time()
                fixed = outer[outer_i]

                def constraint(v):
                    mat = self.design(v, fixed) if solve_p else self.design(fixed, v)
                    return np.linalg.det(np.column_stack((mat, self.yn)))

                try:
                    root = brentq(constraint, scan[inner_i], scan[inner_i+1], xtol=2e-11)
                    p, d = (root, fixed) if solve_p else (fixed, root)
                    mat = self.design(p, d)
                    coeff = np.linalg.lstsq(mat, self.yn, rcond=None)[0]
                    dp = self.lx * (mat[:, 1:] @ coeff[1:])
                    dd = self.lx * (mat[:, 2:] @ coeff[2:])
                    logjp = np.linalg.slogdet(np.column_stack((mat, dp)))[1]
                    logjd = np.linalg.slogdet(np.column_stack((mat, dd)))[1]
                    # Two coordinate charts share the integral through a
                    # partition of unity, eliminating divergent fold weights.
                    logj = logjp if solve_p else logjd
                    logw = np.log(outer_weight[outer_i]) + logj - np.logaddexp(2*logjp, 2*logjd)
                    samples.append((p, d, coeff, logw))
                except (ValueError, np.linalg.LinAlgError):
                    continue
        if not samples:
            raise ValueError("No positive-order five-point models")
        p = np.array([s[0] for s in samples])
        d = np.array([s[1] for s in samples])
        coefficients = np.array([s[2] for s in samples])
        logw = np.array([s[3] for s in samples])
        return self.convert(p, d, coefficients, logw)

    def samples(self, outer_order=500, inner_order=100):
        if self.n == 4:
            return self.four_samples(outer_order, inner_order)
        return self.five_samples(outer_order)
