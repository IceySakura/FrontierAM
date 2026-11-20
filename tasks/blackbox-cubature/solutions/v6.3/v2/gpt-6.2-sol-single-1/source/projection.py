"""The distribution of a linear projection of a point in the unit cube."""
from decimal import Decimal, localcontext
import math

import numpy as np
from scipy.interpolate import PPoly

from common import gl
from numpy.polynomial.legendre import legvander


class Projection:
    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)
        self.span = float(np.sum(np.abs(w)))
        self.v = self.w / self.span
        self.low = np.minimum(self.w, 0.)
        weights = self.w[np.abs(self.w) > 2e-17*self.span]
        if len(weights) > 12:
            raise ValueError("too many projection weights")
        # A fitted plane may have several very small components. The
        # inclusion-exclusion formula divides by their product.
        self.precision = max(72,44+int(-np.sum(np.log10(np.abs(weights)/self.span))))
        with localcontext() as ctx:
            ctx.prec = self.precision
            raw = [Decimal.from_float(abs(float(i))) for i in weights]
            total = sum(raw)
            self.u = [a / total for a in raw]
            self.offsets = [(Decimal(0), 1)]
            for a in self.u:
                self.offsets += [(s + a, -sgn) for s, sgn in self.offsets]
            self.prod = math.prod(self.u)
        self.n = len(weights)
        self.poly = None

    def param(self, x):
        return (np.asarray(x) - .5) @ self.v + .5

    def points(self, t):
        t = np.asarray(t).reshape(-1, 1)
        return np.where(self.w >= 0, t, 1.-t) + np.zeros_like(self.w)[None, :]

    def moment(self, t, power=1):
        """E[(t-S)_+^power]; power=0 is the CDF."""
        if t <= 0.:
            return 0.
        if power == 0 and t >= .5:
            return 1. - self.moment(1.-t, 0) if t > .5 else .5
        if power == 1 and t > .5:
            return t - .5 + self.moment(1.-t, 1)
        with localcontext() as ctx:
            ctx.prec = self.precision
            u = Decimal.from_float(float(t))
            if power == int(power):
                p = int(power)
                n = self.n + p
                sm = sum(sgn * (u-s)**n for s, sgn in self.offsets if s < u)
                m = sm / (self.prod * math.prod(range(p+1, n+1)))
            else:
                p = Decimal.from_float(float(power))
                n = self.n + p
                sm = sum(sgn * (u-s)**n for s, sgn in self.offsets if s < u)
                m = sm / (self.prod * math.prod(p+i+1 for i in range(self.n)))
        return max(0., float(m))

    def _build(self):
        n = self.n - 1
        with localcontext() as ctx:
            ctx.prec = self.precision
            merged = {Decimal(".5"): 0}
            for t, sign in self.offsets:
                if t < Decimal(".5"):
                    merged[t] = merged.get(t, 0) + sign
            knots = sorted(t for t, sgn in merged.items() if sgn or t == Decimal(".5"))
            fact = math.factorial(n) * self.prod
            cur = [Decimal(0)] * (n + 1)
            coeff = []
            for a, b in zip(knots[:-1], knots[1:]):
                cur[n] += merged[a]
                coeff.append([float(c / fact) for c in cur[::-1]])
                h = b - a
                cur = [sum(math.comb(k, j) * cur[k] * h**(k-j) for k in range(j,n+1))
                       for j in range(n+1)]
        # PPoly needs strictly ordered intervals for evaluation; unique
        # Decimal knots can round to the same double.
        keep = [i for i,(a,b) in enumerate(zip(knots,knots[1:])) if float(b) > float(a)]
        br = [float(knots[i]) for i in keep] + [.5]
        self.poly = PPoly(np.array([coeff[i] for i in keep]).T, br, extrapolate=False)
        self.knots = np.unique(np.r_[br, 1.-np.array(br)])

    def panel_moments(self, a, b):
        if self.poly is None:
            self._build()
        cuts = self.knots[(self.knots > a) & (self.knots < b)]
        ab = np.r_[a, cuts, b]
        h = np.diff(ab)[:, None] / 2
        m = ab[:-1, None] + h
        node, weight = gl(12)
        t = m + h*node
        pdf = np.maximum(0., self.poly(np.minimum(t, 1.-t)))
        v = legvander(((t-a)*(2/(b-a))-1).ravel(), 14)
        return (h * weight * pdf).ravel() @ v
