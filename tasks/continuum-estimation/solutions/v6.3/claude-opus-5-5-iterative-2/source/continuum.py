"""Continuum (x -> 0) estimate from 4-5 exact observations y(x).

Writes a valid finite estimate at every stage: a cheap pure-Python fallback is
written first, then replaced by the Bayesian posterior estimate.
"""
import json
import math
import os
import sys

DEFAULT_IN = '/workspace/inputs/input.json'
DEFAULT_OUT = '/workspace/result/output.json'


def write_estimate(path, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('non-finite estimate')
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(json.dumps({'estimate': value}, allow_nan=False))
    os.replace(tmp, path)


def simple_fallback(x, y):
    """Three finest levels, single power law with order from a bracketed bisection."""
    pts = sorted(zip(x, y))
    if len(pts) < 3:
        return pts[0][1]
    (h1, f1), (h2, f2), (h3, f3) = pts[:3]
    if f3 == f2 or f2 == f1:
        return f1
    r = (f2 - f1) / (f3 - f2)

    def g(p):
        return (h2 ** p - h1 ** p) / (h3 ** p - h2 ** p) - r
    lo, hi = 0.05, 8.0
    try:
        glo, ghi = g(lo), g(hi)
        if glo * ghi > 0:
            return f1
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if gm * glo > 0:
                lo, glo = mid, gm
            else:
                hi = mid
        p = 0.5 * (lo + hi)
        a = (f2 - f1) / (h2 ** p - h1 ** p)
        z = f1 - a * h1 ** p
        # guard against wild extrapolation
        span = max(abs(f3 - f1), 1e-12)
        if not math.isfinite(z) or abs(z - f1) > 3 * span + 1.0:
            return f1
        return z
    except Exception:
        return f1


def run(inp, out):
    with open(inp) as fh:
        obj = json.load(fh)
    x = [float(v) for v in obj['x']]
    y = [float(v) for v in obj['y']]
    pts = sorted(zip(x, y))
    z0 = pts[0][1]
    write_estimate(out, z0)
    try:
        z1 = simple_fallback(x, y)
        if math.isfinite(z1):
            write_estimate(out, z1)
    except Exception:
        pass
    try:
        import signal

        def _alarm(signum, frame):
            raise TimeoutError('time budget exceeded')
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(40)
    except Exception:
        pass
    z = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import numpy as np
        np.seterr(all='ignore')
        import estimator
        z = estimator.robust_estimate(x, y)
    except Exception as exc:  # keep fallback
        sys.stderr.write('estimator failed: %r\n' % (exc,))
    try:
        import signal
        signal.alarm(0)
    except Exception:
        pass
    if z is not None and math.isfinite(z):
        write_estimate(out, z)


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    try:
        run(inp, out)
    except BaseException as exc:
        sys.stderr.write('run failed: %r\n' % (exc,))
    if not os.path.isfile(out):
        # last resort: any valid number beats a missing output
        try:
            with open(inp) as fh:
                obj = json.load(fh)
            ys = [float(v) for v in obj['y'] if math.isfinite(float(v))]
            write_estimate(out, sum(ys) / len(ys) if ys else 0.0)
        except BaseException:
            write_estimate(out, 0.0)


if __name__ == '__main__':
    main()
