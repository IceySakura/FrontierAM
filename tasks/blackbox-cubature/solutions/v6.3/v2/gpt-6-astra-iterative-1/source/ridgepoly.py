"""Polynomial ridge interpolation with a fitted active direction."""
import time
import numpy as np
from scipy import linalg, optimize
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.legendre import leggauss


def integrate(oracle, x, y, direction, density, deadline):
    if oracle.remaining < 180:
        return None
    d = x.shape[1]
    scale = max(np.max(np.abs(y)), 1e-300)
    orientation = np.where(direction >= 0, 1.0, -1.0)
    t = np.cos(np.arange(129)*np.pi/128)
    line_x = 0.5+t[:, None]*orientation/2
    line_y = oracle.ask(line_x)/scale
    nt = len(x)*3//4
    xx = np.vstack((x[:nt], line_x))
    yy = np.r_[y[:nt]/scale, line_y]
    z = xx-0.5
    zv = x[nt:]-0.5
    yv = y[nt:]/scale
    fixed = int(np.argmax(np.abs(direction)))
    ids = [j for j in range(d) if j != fixed]
    initial = direction[ids]/direction[fixed]
    background = np.column_stack((np.ones(len(xx)), z))
    validation_background = np.column_stack((np.ones(len(zv)), zv))
    best = None
    for degree in (12, 20, 32, 48, 72, 104):
        if time.monotonic() > deadline-2:
            break
        def get_direction(params):
            w = np.zeros(d)
            w[fixed] = 1
            w[ids] = params
            return w/np.sum(np.abs(w))
        def calc(params):
            w = get_direction(params)
            a = np.column_stack((background, chebvander(2*z@w, degree)[:, 2:]))
            c = linalg.lstsq(a, yy, cond=2e-14, lapack_driver="gelsy")[0]
            return a@c-yy, c, w
        def fun(params):
            if time.monotonic() > deadline:
                raise TimeoutError("ridge interpolation deadline")
            return calc(params)[0]
        try:
            if d == 1:
                params = np.empty(0)
            else:
                fit = optimize.least_squares(fun, initial, method="lm",
                                            max_nfev=35, ftol=1e-14,
                                            xtol=1e-14, gtol=1e-14,
                                            diff_step=1e-5)
                params = fit.x
            residual, c, w = calc(params)
            pred = validation_background@c[:d+1]+chebvander(2*zv@w, degree)[:, 2:]@c[d+1:]
            err = max(np.max(np.abs(residual)), np.max(np.abs(pred-yv)))
            if err < 3e-13:
                best = (degree, c, w)
                break
        except (ValueError, linalg.LinAlgError, TimeoutError):
            return None
    if best is None:
        return None
    degree, c, w = best
    za = np.vstack((x, line_x))-0.5
    ya = np.r_[y/scale, line_y]
    aa = np.column_stack((np.ones(len(za)), za, chebvander(2*za@w, degree)[:, 2:]))
    c = linalg.lstsq(aa, ya, cond=2e-14, lapack_driver="gelsy")[0]
    breaks = np.zeros(1)
    for v in np.abs(w):
        breaks = np.r_[breaks, breaks+v]
    breaks = np.unique(np.r_[0, np.clip(breaks, 0, 1), 1])
    glx, glw = leggauss((degree+d+3)//2)
    left, right = breaks[:-1], breaks[1:]
    qt = ((left+right)[:, None]/2+(right-left)[:, None]/2*glx).ravel()
    qw = ((right-left)[:, None]/2*glw).ravel()*density(qt, w)
    moments = chebvander(2*qt-1, degree)[:, 2:].T@qw
    return float(scale*(c[0]+moments@c[d+1:]))
