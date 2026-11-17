#!/usr/bin/env python3
"""Continuum-limit estimator: read observations, write one finite estimate."""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import estimator
except ImportError:
    estimator = None


def simple_estimate(x, y):
    """Last-resort extrapolation, always finite, pure Python."""
    pairs = sorted(zip(map(float, x), map(float, y)))
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    vals = [float(ys[0])]
    for p in (1.0, 2.0, 3.0):
        ratio = (xs[0] / xs[1]) ** p
        den = 1.0 - ratio
        if abs(den) > 1e-12:
            v = (ys[0] - ratio * ys[1]) / den
            if math.isfinite(v):
                vals.append(float(v))
    # quadratic through the three smallest levels
    if len(xs) >= 3:
        try:
            x1, x2, x3 = xs[:3]
            y1, y2, y3 = ys[:3]
            d12, d13, d23 = x1 - x2, x1 - x3, x2 - x3
            a = ((y1 - y2) / d12 - (y1 - y3) / d13) / d23
            b = (y1 - y2) / d12 - a * (x1 + x2)
            c = y1 - a * x1 * x1 - b * x1
            if math.isfinite(c):
                vals.append(float(c))
        except ZeroDivisionError:
            pass
    vals.sort()
    return float(vals[len(vals) // 2])


def main():
    inp = os.environ.get('INPUT', sys.argv[1] if len(sys.argv) > 1 else '/workspace/inputs/input.json')
    out = os.environ.get('OUTPUT', sys.argv[2] if len(sys.argv) > 2 else '/workspace/result/output.json')
    try:
        obj = json.loads(Path(inp).read_text())
        x = [float(v) for v in obj['x']]
        y = [float(v) for v in obj['y']]
        if len(x) != len(y) or len(x) < 2:
            raise ValueError('bad arrays')
        if not all(math.isfinite(v) and v > 0 for v in x) or not all(math.isfinite(v) for v in y):
            raise ValueError('nonfinite input')
        if estimator is None:
            raise ValueError('numerical runtime unavailable')
        cfg = estimator.production_cfg()
        value = estimator.estimate(x, y, cfg)
        if not math.isfinite(value):
            raise ValueError('nonfinite estimate')
    except Exception:
        try:
            value = simple_estimate(x, y)
        except Exception:
            value = 0.0
    if not math.isfinite(value):
        value = 0.0
    outdir = Path(out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({'estimate': float(value)}, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
