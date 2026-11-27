"""Adaptive Gaussian cubature using value-only pilots."""

import math
import sys

import numpy as np

from core import Oracle, gaussian, grad_basis, net_integral
from ellipsoid import detect_jump, estimate_jump, estimate_tail
from kinks import collection, finish_kinks
from options import payoff


def solve(f, X, Y, at_zero):
    if np.any(Y == 0) or at_zero == 0:
        return payoff(f, X, Y)
    jump,X,Y,anchor = detect_jump(f, X, Y, at_zero)
    if jump is not None and f.remaining>=11000:
        if jump>0:
            est = estimate_tail(f, X, Y, at_zero, jump)
        else:
            est = estimate_jump(f, X, Y, at_zero, anchor)
        if est is not None and math.isfinite(est):
            return est
    if jump is None and f.remaining >= 11500:
        model = collection(f, X, Y)
        if model is not None:
            est = finish_kinks(f, X, Y, model)
            if est is not None and math.isfinite(est):
                return est
    V, _, _ = grad_basis(f, X, Y, min(32, f.remaining//(4*f.d)))
    return net_integral(f, V, fallback=float(Y.mean()))


def main():
    f = Oracle()
    X = gaussian(min(512, max(1, 1 << ((f.remaining//8).bit_length()-1))), f.d, f.rng)
    vals = f.query(np.vstack((np.zeros(f.d), X)))
    Y = vals[1:]
    base = float(np.median(Y))
    try:
        ans = solve(f, X, Y, vals[0])
    except Exception as e:
        print(f"recovering from fit: {type(e).__name__}: {e}", file=sys.stderr)
        ans = net_integral(f, np.eye(f.d), fallback=base) if f.remaining >= 128 and f.left() > 5 else base
    f.result(ans if ans is not None and math.isfinite(ans) else base)


if __name__ == "__main__":
    main()
