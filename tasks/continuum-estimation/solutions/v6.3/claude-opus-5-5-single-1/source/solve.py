import json
import math
import os
import sys


def fallback(x, y):
    """Generalized Aitken/Richardson on the three finest levels, else finest value."""
    pts = sorted(zip(x, y))
    z = pts[0][1]
    try:
        (x3, y3), (x2, y2), (x1, y1) = pts[:3]
        r = (y1 - y2) / (y2 - y3)
        f = lambda p: (x1 ** p - x2 ** p) / (x2 ** p - x3 ** p) - r
        lo, hi = 0.05, 10.0
        if f(lo) * f(hi) < 0:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            p = 0.5 * (lo + hi)
            cand = y3 - (y2 - y3) * x3 ** p / (x2 ** p - x3 ** p)
            if math.isfinite(cand) and abs(cand - y3) < 10 * (abs(y1 - y3) + 1e-12):
                z = cand
    except Exception:
        pass
    return z


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else '/workspace/inputs/input.json'
    out = sys.argv[2] if len(sys.argv) > 2 else '/workspace/result/output.json'
    z = 0.0
    try:
        with open(inp) as fh:
            d = json.load(fh)
        x = [float(v) for v in d['x']]
        y = [float(v) for v in d['y']]
        z = y[x.index(min(x))]
        z = fallback(x, y)
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import numpy as np
            from est import estimate_any
            e = estimate_any(np.array(x), np.array(y))
            if e is not None and math.isfinite(e):
                z = float(e)
        except Exception as exc:
            sys.stderr.write('estimator failed: %r\n' % (exc,))
    except Exception as exc:
        sys.stderr.write('input failure: %r\n' % (exc,))
    if not math.isfinite(z):
        z = 0.0
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out, 'w') as fh:
        fh.write(json.dumps({"estimate": float(z)}))


if __name__ == '__main__':
    main()
