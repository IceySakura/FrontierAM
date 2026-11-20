#!/usr/bin/env python3
"""Adaptive black-box integration over a box, via the JSON Lines oracle."""

import json
import math
import sys
import warnings

import numpy as np

from core import OracleError, Samples, StopSampling, agrees, fit_exp_quadratic, fit_polynomial
from cusps import try_cusps
from numerical import adaptive_cubature, qmc_regression
from peaks import fit_peaks, integrate_peak_residual
from projection import face_periods, periodic_mean, try_projection
from ridges import (central_line, line_knots, spectral_line, try_hinges,
                    try_shifted_waves, try_waves, vertex_points)
from separable import try_separable


def validate(s, m, n=96, rel=8.e-11, corners=False):
    if m is None or not np.isfinite(m.mean):
        return False
    if s.left < n:
        return False
    z = s.design(n)
    if corners and s.left > n+16:
        z = np.vstack((z, vertex_points(s, min(256, s.left-n))))
    q = s.ask(z)/s.scale
    return agrees(m.predict(z), q, rel, 6.e-13)


def answer(s, estimate, name=None):
    if name:
        s.route = name
    return float(estimate*s.scale)


def integrate(s):
    if s.budget < 120 or s.d > 10:
        s.uniform(s.left)
        return s.quick()
    n = min(256, 1 << max(5, (s.budget//3).bit_length()-1))
    s.uniform(n)
    s.set_scale()
    x, y = s.data()
    if np.max(np.abs(y), initial=0.) == 0.:
        s.uniform(min(512, s.left))
        s.set_scale()
        x, y = s.data()
        n = len(y)
        if np.max(np.abs(y), initial=0.) == 0.:
            s.uniform(s.left)
            return s.quick()
    m = fit_polynomial(x, y)
    if m is not None and validate(s, m, corners=True):
        return answer(s, m.mean, m.name)
    m = fit_exp_quadratic(x, y)
    if m is not None and validate(s, m):
        return answer(s, m.mean, m.name)
    try:
        f = try_separable(s, n, small=(s.budget < 900))
        if f is not None:
            return answer(s, f)
    except (StopSampling, ArithmeticError, ValueError, np.linalg.LinAlgError):
        pass
    line = None
    if s.left >= 366 and s.d >= 2 and s.time_left() > 8.:
        # This line is also used for spectral fitting if it is smooth.
        state = s.rng.bit_generator.state
        line = central_line(s)
        p = line_knots(line, line.y)
        if p[0] and (p[1] or p[0] == 1):
            try:
                m = try_hinges(s, line, p)
            except (StopSampling, ArithmeticError, ValueError, np.linalg.LinAlgError):
                m = None
            if m is not None and np.isfinite(m.mean):
                return answer(s, m.mean, m.name)
            hint = getattr(s, "hint", None)
            if hint is not None and s.left >= 96:
                z = s.design(s.left)
                y = s.ask(z)/s.scale
                corr = float(np.mean(y-hint.predict(z)))
                if math.isfinite(corr):
                    return answer(s, hint.mean+corr, "hinge residual")
        else:
            s.rng.bit_generator.state = state
    if s.left > 500 and s.time_left() > 7.:
        try:
            periodic = face_periods(s)
            if np.all(periodic):
                f = periodic_mean(s)
                if f is not None and math.isfinite(f):
                    return answer(s, f, "periodic lattice")
        except (StopSampling, ArithmeticError, ValueError, np.linalg.LinAlgError):
            pass
        try:
            f = try_projection(s, n)
            if f is not None and math.isfinite(f):
                return answer(s, f, "single projection")
        except (StopSampling, ArithmeticError, ValueError, np.linalg.LinAlgError):
            pass
    if s.left >= 300 and s.d >= 2 and s.time_left() > 7.:
        if line is None:
            line = central_line(s)
        p = spectral_line(line)
        m = None
        if p is not None:
            try:
                m = try_waves(s, line, p)
            except (ValueError, np.linalg.LinAlgError, ArithmeticError):
                pass
        if m is not None and validate(s, m, 104, rel=1.5e-10):
            return answer(s, m.mean, m.name)
        if s.left > 575 and s.time_left() > 5.:
            try:
                m = try_shifted_waves(s, line)
            except (StopSampling, ArithmeticError, ValueError, np.linalg.LinAlgError):
                m = None
            if m is not None and validate(s, m, 91, rel=1.5e-10):
                return answer(s, m.mean, m.name)
    if s.left > 600 and line is not None and s.time_left() > 9.:
        try:
            m = try_cusps(s, line)
        except (StopSampling, ArithmeticError, ValueError, np.linalg.LinAlgError):
            m = None
        if m is not None and np.isfinite(m.mean):
            return answer(s, m.mean, m.name)
        hint = getattr(s, "hint", None)
        if hint is not None and s.left >= 132:
            z = s.design(s.left)
            y = s.ask(z)/s.scale
            corr = float(np.mean(y-hint.predict(z)))
            if math.isfinite(corr):
                return answer(s, hint.mean+corr, "cusp residual")
    best = None
    if s.left > 512 and s.time_left() > 6.:
        try:
            m, best = fit_peaks(s)
        except (StopSampling, ArithmeticError, ValueError, np.linalg.LinAlgError):
            m = None
        if m is not None and np.isfinite(m.mean):
            return answer(s, m.mean, m.name)
    if best is not None and s.left >= 300 and s.time_left() > 3.:
        if s.d == 2 and best.quality > .035:
            f, _ = adaptive_cubature(s, best)
        else:
            f = integrate_peak_residual(s, best)
        if f is not None and np.isfinite(f):
            return answer(s, f, "peak residual")
    if s.d == 2 and s.left > 500 and s.time_left() > 3.:
        y = s.uy[:s.nu]/s.scale
        if np.max(np.abs(y)) > 7.*np.mean(np.abs(y)):
            f, _ = adaptive_cubature(s)
            if np.isfinite(f):
                return answer(s, f, "adaptive cubature")
    f, s.route = qmc_regression(s)
    return f


def main():
    msg = json.loads(sys.stdin.readline())
    if msg.get("type") != "init":
        raise ValueError("expected init")
    s = Samples(msg)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            result = integrate(s)
        except OracleError:
            result = s.quick()
        except Exception:
            if s.left and s.time_left() > 1.:
                s.uniform(s.left)
            result = s.quick()
        if not math.isfinite(result):
            result = s.quick()
        result *= s.volume
    print(json.dumps({"type": "result", "estimate": result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
