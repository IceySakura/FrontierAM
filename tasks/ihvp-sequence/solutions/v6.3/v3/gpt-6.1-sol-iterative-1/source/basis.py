"""An orthogonal preconditioning basis learned from current operator actions."""
import ctypes
from pathlib import Path
import time
import numpy as np


try:
    _lib = ctypes.CDLL(str(Path(__file__).with_name("_rotations.so")))
    _lib.rotate_layers.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3 + [
        ctypes.c_void_p] * 3 + [ctypes.c_int]
    _lib.rotate_layers.restype = None
except OSError:
    _lib = None


class BasisBudget(Exception):
    pass


class LearnedBasis:
    def __init__(self, n, seed, iterations=5, block=16):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.iterations = iterations
        self.block = block
        self.bits = np.empty(n.bit_length() - 1, dtype=np.int32)
        self.cosines = np.empty((len(self.bits), n // 2))
        self.sines = np.empty_like(self.cosines)
        self.depth = 0
        self.diagonal = None
        self.fraction = 0.
        self.oracle_seconds = .003

    def transform(self, v, inverse):
        a = np.array(v, dtype=np.float64, order="C", copy=True)
        k = 1 if a.ndim == 1 else a.shape[1]
        if _lib is not None:
            _lib.rotate_layers(a.ctypes.data, self.n, k, self.depth,
                               self.cosines.ctypes.data, self.sines.ctypes.data,
                               self.bits.ctypes.data, int(inverse))
            return a
        order = range(self.depth) if inverse else reversed(range(self.depth))
        work = a.reshape(self.n, k)
        for j in order:
            bit = int(self.bits[j])
            pair = work.reshape(-1, 2, bit, k)
            c = self.cosines[j].reshape(-1, bit, 1)
            s = self.sines[j].reshape(-1, bit, 1)
            x, y = pair[:, 0].copy(), pair[:, 1].copy()
            if inverse:
                pair[:, 0] = c * x + s * y
                pair[:, 1] = -s * x + c * y
            else:
                pair[:, 0] = c * x - s * y
                pair[:, 1] = s * x + c * y
        return a

    def _action(self, hvp, v, end):
        columns = 1 if v.ndim == 1 else v.shape[1]
        if time.monotonic() + max(.008, 1.6 * columns * self.oracle_seconds) > end:
            raise BasisBudget()
        start = time.monotonic()
        result = self.transform(hvp(self.transform(v, False)), True)
        self.oracle_seconds = max(.0001, (time.monotonic() - start) / columns)
        return result

    def refresh(self, hvp, mu, end):
        columns = 64
        v = self.rng.choice([-1., 1.], size=(self.n, columns))
        y = self._action(hvp, v, end)
        estimates = v * y
        diagonal = np.mean(estimates, axis=1)
        deviation = np.std(estimates, axis=1)
        fraction = float(np.sum(diagonal ** 2) / (np.sum(y ** 2) / columns))
        candidate = np.maximum(np.maximum(diagonal + .25 * deviation,
                                          .25 * deviation), mu)
        if not np.isfinite(candidate).all():
            return False
        self.diagonal = candidate
        self.fraction = fraction
        return fraction > .35

    def fit(self, hvp, mu, end):
        start = time.monotonic()
        initial = None
        for level, shift in enumerate(reversed(range(len(self.bits)))):
            # Reserve enough time to validate the completed basis.
            reserve = max(.2, 100 * self.oracle_seconds)
            if time.monotonic() + reserve > end:
                return False
            bit = 1 << shift
            lo = np.arange(self.n)
            lo = lo[(lo & bit) == 0]
            hi = lo + bit
            refs = np.arange(0, self.n, 2 * bit)
            probe = np.zeros((self.n, 2))
            probe[refs, 0] = 1
            probe[refs + bit, 1] = 1
            a = self._action(hvp, probe, end)
            angle = .5 * np.arctan2(a[refs, 1] + a[refs + bit, 0],
                                   a[refs, 0] - a[refs + bit, 1])
            group = lo // (2 * bit)
            cr, sr = np.cos(angle)[group], np.sin(angle)[group]
            a00 = a[lo, 0] * cr + a[lo, 1] * sr
            a01 = -a[lo, 0] * sr + a[lo, 1] * cr
            a10 = a[hi, 0] * cr + a[hi, 1] * sr
            a11 = -a[hi, 0] * sr + a[hi, 1] * cr
            nc = a00 ** 2 - a01 ** 2 - a10 ** 2 + a11 ** 2
            ns = 2 * (a00 * a10 - a01 * a11)
            den = np.maximum(np.hypot(nc, ns), 1e-100)
            c, s = nc / den, ns / den
            v = self.rng.normal(size=(self.n, self.block))
            y = self._action(hvp, v, end)
            if level == 0:
                diagonal = np.mean(v * y, axis=1)
                initial = float(np.sum(diagonal ** 2) /
                                (np.sum(y ** 2) / self.block))
            # A commuting trace-zero block projector reveals each rotation layer.
            for iteration in range(self.iterations):
                pv = np.empty_like(v)
                pv[lo] = c[:, None] * v[lo] + s[:, None] * v[hi]
                pv[hi] = s[:, None] * v[lo] - c[:, None] * v[hi]
                z = self._action(hvp, pv, end)
                nc = np.sum(y[lo] * z[lo] - y[hi] * z[hi], axis=1)
                ns = np.sum(y[hi] * z[lo] + y[lo] * z[hi], axis=1)
                den = np.maximum(np.hypot(nc, ns), 1e-100)
                c, s = nc / den, ns / den
            py = np.empty_like(y)
            py[lo] = c[:, None] * y[lo] + s[:, None] * y[hi]
            py[hi] = s[:, None] * y[lo] - c[:, None] * y[hi]
            residual = float(np.linalg.norm(z - py) / np.linalg.norm(y))
            if level == 0:
                if residual > .7:
                    return False
                estimate = (time.monotonic() - start) * len(self.bits) * 1.1 + reserve
                if start + estimate > end:
                    return False
            angle = .5 * np.arctan2(s, c)
            self.bits[level] = bit
            self.cosines[level] = np.cos(angle)
            self.sines[level] = np.sin(angle)
            self.depth += 1
        return self.refresh(hvp, mu, end) and self.fraction > max(initial * 1.5, .35)

    def apply(self, r):
        return self.transform(self.transform(r, True) / self.diagonal, False)
