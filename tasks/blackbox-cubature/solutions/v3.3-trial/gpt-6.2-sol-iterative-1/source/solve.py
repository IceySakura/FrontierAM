"""Point-query cubature driver."""
import sys
import time

_START = time.monotonic()

import numpy as np

from cubature_protocol import Session
from numerics import balanced_rule, log, normal


class Context(Session):
    def __init__(self):
        super().__init__()
        self.start = _START
        self.rng = np.random.default_rng([self.seed % (2**64), 4589])
        self.limit = min(29., float(self.init.get("time_limit_s", 30.)))
        self.last = None

    def time_left(self):
        return self.limit-(time.monotonic()-self.start)

    def query(self, points):
        if self.time_left() < 1.35:
            raise TimeoutError("leaving time to write the result")
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if not np.all(np.isfinite(pts)):
            raise ValueError("invalid internal query")
        return super().query(np.clip(pts, -39.99, 39.99))


def solve(ctx):
    np.seterr(over="ignore", under="ignore", invalid="ignore")
    n = min(384, max(1, ctx.budget//12))
    x = normal(n, ctx.d, ctx.rng)
    y = ctx.query(x)
    ctx.last = float(np.mean(y))
    zeros = np.mean(y == 0)
    is_option = zeros >= .025 and np.min(y) >= 0
    if is_option:
        from options import options
        return options(ctx, x, y)
    from hard import hard
    return hard(ctx, x, y)


def main():
    ctx = Context()
    try:
        ans = solve(ctx)
        if not np.isfinite(ans):
            raise ValueError("nonfinite cubature")
    except Exception as exc:
        log("fallback:", repr(exc))
        if "CUBATURE_DEBUG" in __import__("os").environ:
            import traceback
            traceback.print_exc(file=sys.stderr)
        ans = ctx.last
        if ctx.remaining > 0 and ctx.time_left() > 3:
            try:
                n = min(ctx.remaining, 8192 if ctx.time_left() > 8 else 1024)
                xx = balanced_rule(n, ctx.d, ctx.rng)
                rr = ctx.query(xx)
                if np.isfinite(rr).all():
                    ans = rr.mean()
            except (ValueError, RuntimeError, TimeoutError, np.linalg.LinAlgError):
                pass
    if ans is None or not np.isfinite(ans):
        ans = 0.
    ctx.result(float(ans))


if __name__ == "__main__":
    main()
