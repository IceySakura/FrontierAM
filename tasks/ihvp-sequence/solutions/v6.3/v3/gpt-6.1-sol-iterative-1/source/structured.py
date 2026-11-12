"""A probed diagonal-plus-low-rank inverse; no matrix entries are assumed."""
import numpy as np


class DiagonalLowRank:
    def __init__(self, dimension, seed, rank=32):
        self.d = dimension
        self.rank = min(rank, dimension // 2)
        rng = np.random.default_rng(seed)
        ids = rng.permutation(dimension)[:2 * self.rank]
        self.j = ids[:self.rank]
        self.k = ids[self.rank:]
        self.v = rng.choice([-1., 1.], size=dimension)
        self.dinv = None
        self.u = None
        self.left = None
        self.residual = np.inf
        self.singular = None
        self.columns = None

    def fit(self, hvp, mu, L, first=None):
        n, m = self.d, self.rank
        if first is None:
            probes = np.zeros((n, m))
            probes[self.j, np.arange(m)] = 1
            cj = hvp(probes)
        else:
            cj = first
        self.columns = cj
        core = cj.copy()
        core[self.j] = 0
        u, singular, _ = np.linalg.svd(core, full_matrices=False)
        self.singular = singular
        cutoff = max(singular[0] * 1e-7, L * 1e-13)
        r = int(np.count_nonzero(singular > cutoff))
        if r >= m - 3:
            return False
        if r == 0:
            diag = hvp(self.v) / self.v
            if np.min(diag) < mu * .5:
                return False
            self.dinv = 1 / diag
            self.u = np.empty((n, 0))
            self.left = self.u.copy()
            self.residual = 0.
            return True
        probes = np.zeros((n, m))
        probes[self.k, np.arange(m)] = 1
        ck = hvp(probes)
        u = u[:, :r]
        mask = np.ones(n, dtype=bool)
        mask[self.j] = False
        mask[self.k] = False
        ui = u[mask]
        t = np.linalg.solve(ui.T @ ui, ui.T @ ck[mask])
        c = t @ np.linalg.pinv(u[self.k].T, rcond=1e-10)
        c = (c + c.T) * .5
        u[self.j] = ck[self.j] @ np.linalg.pinv(t, rcond=1e-10)
        hv = hvp(self.v)
        diag = (hv - u @ (c @ (u.T @ self.v))) / self.v
        if np.min(diag) <= max(mu * .1, L * 1e-12):
            return False
        self.dinv = 1 / diag
        weighted = self.dinv[:, None] * u
        g = u.T @ weighted
        vals, rot = np.linalg.eigh((g + g.T) * .5)
        if vals[0] <= 0:
            return False
        whitening = rot / np.sqrt(vals)
        reduced = (rot.T @ c @ rot) * np.sqrt(vals[:, None] * vals[None, :])
        e = np.eye(r) + (reduced + reduced.T) * .5
        if np.linalg.eigvalsh(e)[0] <= 1e-9:
            return False
        self.u = np.asfortranarray(weighted @ whitening)
        self.left = np.asfortranarray(self.u @ (np.linalg.inv(e) - np.eye(r)))
        test = np.ones(n)
        prediction = diag + u @ (c @ (u.T @ test))
        actual = hvp(test)
        self.residual = np.linalg.norm(prediction - actual) / np.linalg.norm(actual)
        return bool(np.isfinite(self.residual) and self.residual < 1e-7)

    def apply(self, r):
        return self.dinv * r + self.left @ (self.u.T @ r)
