"""Point-only integration of a fresh Gaussian oracle instance."""

import sys
import time

started = time.monotonic()

import numpy as np

from cubature_protocol import Session
from numerics import Work, local_gradient, rotation
import hard
import kinks
import options


def basic(w, x, y):
    d = w.d
    ids = np.round(np.linspace(0, len(x)-1, min(20, len(x)))).astype(int)
    g = local_gradient(w, x[ids], y[ids])
    # Cap unusually large numerical derivatives, e.g. a pilot straddling a jump.
    gn = np.linalg.norm(g, axis=1)
    cap = max(np.median(gn)*4., 1.e-9)
    g = g / np.maximum(1., gn/cap)[:,None]
    gram = g.T @ g
    geom = getattr(w,"kink_cov",None)
    if geom is not None:
        gram += geom*.56*np.trace(gram)/max(np.trace(geom),1.e-100)
    b = rotation(gram)
    return w.query(w.normal(w.remaining) @ b.T).mean()


def main():
    session = Session()
    deadline = started + float(session.init.get("time_limit_s", 30.)) - 1.3
    w = Work(session, deadline)
    x = w.normal(512)
    y = w.query(x)
    w.fallback = float(np.mean(y))
    zero = y == 0.
    try:
        if np.any(zero) and np.all(y >= 0.):
            p = y[~zero]
            if len(p) > 1 and np.ptp(p) <= 2.e-12*abs(p[0]):
                answer = options.digital(w, x, y)
            else:
                answer = options.continuous(w, x, y)
        else:
            jump = hard.find_jump(w,x,y)
            answer = hard.jump_estimate(w,x,y,jump) if jump is not None else None
            if answer is None and jump is None and w.remaining>10500:
                answer = kinks.estimate(w,x,y,w.line_survey)
            if answer is None:
                answer = basic(w, x, y)
        if not np.isfinite(answer):
            raise FloatingPointError("nonfinite quadrature")
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, RuntimeError) as exc:
        print(f"Estimator fallback: {exc}", file=sys.stderr)
        answer = w.fallback
        if w.remaining and time.monotonic() - started < 16.:
            try:
                yy = w.query(w.normal(w.remaining))
                answer = (len(y)*answer + np.sum(yy)) / (len(y)+len(yy))
            except RuntimeError:
                pass
    w.session.result(answer)


if __name__ == "__main__":
    main()
