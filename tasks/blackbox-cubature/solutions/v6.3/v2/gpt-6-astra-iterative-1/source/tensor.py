"""Cross-interpolatory tensor quadrature and residual correction."""
import time
import numpy as np
from scipy import linalg
from numpy.polynomial.legendre import leggauss, legvander
import ridges
from sampling import average as qmc_average
from sampling import query_uniform


def integrate(oracle, x, y, extra_points, deadline, max_rank=None):
    d = oracle.d
    if d < 2 or oracle.remaining < 500:
        return None
    scale = max(np.max(np.abs(y)), 1e-300)
    cache = {}
    for xx, yy in zip(oracle.xs, oracle.ys):
        cache.update((tuple(p), float(v)/scale) for p, v in zip(xx, yy))

    def ask(points):
        points = np.asarray(points).reshape(-1, d)
        keys = [tuple(p) for p in points]
        unseen = list(dict.fromkeys(k for k in keys if k not in cache))
        if len(unseen) > oracle.remaining:
            raise ValueError("tensor budget exhausted")
        if unseen:
            values = oracle.ask(np.asarray(unseen))/scale
            cache.update(zip(unseen, values))
        return np.asarray([cache[k] for k in keys])

    ids = np.arange(8)
    anchors = x[ids]
    crosses = {}
    order = sorted(range(1, d), key=lambda k: abs(k-d/2))
    for cut in order:
        points = np.empty((64, d))
        for a in range(8):
            for b in range(8):
                points[a*8+b, :cut] = anchors[a, :cut]
                points[a*8+b, cut:] = anchors[b, cut:]
        matrix = ask(points).reshape(8, 8)
        singular = linalg.svdvals(matrix)
        if singular[0] < 1e-200:
            return None
        rank = int(np.count_nonzero(singular > singular[0]*2e-13))
        if rank >= 8 or rank == 0:
            return None
        if max_rank is not None and rank > max_rank:
            return None
        if cut == order[0]:
            cost = 2*rank+(d-2)*rank*rank
            future = 56*(d-2)+4*cost+256
            required = max(12, 0.75*getattr(oracle, "characteristic_frequency", 0))
            if (oracle.remaining-future)//max(cost, 1) < required:
                return None
        rows = linalg.qr(matrix.T, pivoting=True, mode="economic")[2][:rank]
        cols = linalg.qr(matrix, pivoting=True, mode="economic")[2][:rank]
        cross = matrix[np.ix_(rows, cols)]
        if np.linalg.cond(cross) > 2e9:
            return None
        inverse = linalg.inv(cross)
        crosses[cut] = (anchors[rows, :cut], anchors[cols, cut:], inverse)

    def core_points(j, nodes):
        left = np.empty((1, 0)) if j == 0 else crosses[j][0]
        right = np.empty((1, 0)) if j == d-1 else crosses[j+1][1]
        a, b = len(left), len(right)
        points = np.empty((len(nodes)*a*b, d))
        points[:, j] = np.repeat(nodes, a*b)
        if j:
            points[:, :j] = np.tile(np.repeat(left, b, axis=0), (len(nodes), 1))
        if j < d-1:
            points[:, j+1:] = np.tile(right, (len(nodes)*a, 1))
        return points, a, b

    validation_ids = [int(np.argmin(y)), int(np.argmax(y)), len(x)//3, 2*len(x)//3]
    validation_x = x[validation_ids]
    vector = np.ones((len(validation_ids), 1))
    cost = 0
    for j in range(d):
        points, a, b = core_points(j, validation_x[:, j])
        cost += a*b
        core = ask(points).reshape(len(validation_ids), a, b)
        vector = np.einsum("na,nab->nb", vector, core)
        if j < d-1:
            vector = vector@crosses[j+1][2]
    if np.max(np.abs(vector[:, 0]-y[validation_ids]/scale)) > 2e-10:
        return None
    nq = min(128, (oracle.remaining-256)//cost)
    if nq < max(8, 0.75*getattr(oracle, "characteristic_frequency", 0)):
        return None
    nodes, weights = leggauss(nq)
    nodes = (nodes+1)/2
    weights /= 2
    basis = legvander(2*nodes-1, nq-1)
    coefficients = []
    vector = np.ones(1)
    for j in range(d):
        points, a, b = core_points(j, nodes)
        core = ask(points).reshape(nq, a*b)
        if not np.all(np.isfinite(core)):
            return None
        coef = (basis.T@(core*weights[:, None]))*(2*np.arange(nq)+1)[:, None]
        if not np.all(np.isfinite(coef)):
            return None
        coefficients.append((coef, a, b))
        vector = vector@coef[0].reshape(a, b)
        if j < d-1:
            vector = vector@crosses[j+1][2]
    estimate = float(vector[0])

    def predict(points):
        vector = np.ones((len(points), 1))
        for j, (coef, a, b) in enumerate(coefficients):
            core = (legvander(2*points[:, j]-1, nq-1)@coef).reshape(len(points), a, b)
            vector = np.einsum("na,nab->nb", vector, core)
            if j < d-1:
                vector = vector@crosses[j+1][2]
        return vector[:, 0]

    if time.monotonic() < deadline-8:
        class InterpolatedFunction:
            d = oracle.d
            seed = oracle.seed
            remaining = 8192

            def ask(self, points):
                return predict(points)

        periodic = ridges.fit_fourier(InterpolatedFunction(), x, y/scale, deadline)
        if periodic is not None:
            return periodic*scale
    n = min(oracle.remaining, len(extra_points))
    points = extra_points[:n]
    start = getattr(oracle, "uniform_count", len(x))
    residual = query_uniform(oracle, points)/scale-predict(points)
    if n:
        estimate += qmc_average(residual, start)
    if np.isfinite(estimate):
        return estimate*scale
    return None
