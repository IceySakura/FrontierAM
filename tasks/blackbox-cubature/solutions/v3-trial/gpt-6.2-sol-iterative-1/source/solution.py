"""Black-box Gaussian cubature dispatcher."""
import math

import numpy as np

from common import Session, log, main_fallback, sobol
from options import continuous_option, digital_option


def solve(s):
    x0 = sobol(512, s.d, s.rng)
    x = np.vstack((x0, np.zeros((1, s.d))))
    y = s.query(x)
    s.best = float(y[:-1].mean())
    if np.all(y >= 0) and (np.any(y == 0) or np.all(y == y[0])):
        positive = y[y > 0]
        if len(positive) >= 2 and np.ptp(positive) < 1e-12 * positive[0]:
            z = digital_option(s, x, y)
        else:
            z = continuous_option(s, x, y)
    else:
        try:
            from hard import hard_integral
        except ModuleNotFoundError:
            return main_fallback(s, x, y)
        z = hard_integral(s, x, y)
    if z is not None and math.isfinite(z):
        return z
    return main_fallback(s, x, y)


def main():
    s = Session()
    try:
        value = solve(s)
    except Exception as exc:
        log("numerical fallback:", repr(exc))
        if s.best is not None and s.remaining >= 512 and s.elapsed < 23.0:
            try:
                value = main_fallback(s, np.zeros((1, s.d)), np.zeros(1), basis=np.eye(s.d))
            except Exception:
                value = s.best
        else:
            value = s.best
    s.finish(value)


if __name__ == "__main__":
    main()
