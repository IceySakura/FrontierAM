import json
import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy import sparse


def maxabs(x):
    return float(np.max(np.abs(x), initial=0.0))


def normalized(M, scale):
    if scale == 1.:
        return M
    result = M.copy()
    result.data /= scale
    return result


@lru_cache(32)
def upper(d):
    j, i = np.tril_indices(d)
    return i, j, np.where(i == j, 1., math.sqrt(2.))


def smat(x, d):
    i, j, w = upper(d)
    r = np.empty((d, d), dtype=np.float64, order='F')
    r[i, j] = r[j, i] = x / w
    return r


def svec(x):
    i, j, w = upper(x.shape[0])
    return x[i, j] * w


def exp_error(a, b, c):
    with np.errstate(divide='ignore', invalid='ignore'):
        interior = a - b * (np.log(c) - np.log(b))
    boundary = np.maximum.reduce([np.abs(b), a, -c])
    return np.maximum(0., np.where((b > 0) & (c > 0),
                                   np.minimum(boundary, interior), boundary))


class Layout:
    def __init__(self, spec):
        self.spec = list(spec)
        self.blocks = []
        self.nonlinear = []
        self.degree = 0
        pos = 0
        ng, eg = [], []
        grouped = defaultdict(list)
        for k, d in spec:
            width = d * (d + 1) // 2 if k == 'psd' else d
            stop = pos + width
            self.blocks.append((k, d, pos, stop))
            if k == 'zero':
                eg.append((pos, stop))
            elif k == 'nonnegative' or (k == 'psd' and d == 1):
                ng.append((pos, stop))
                self.degree += width
            else:
                self.nonlinear.append((k, d, pos, stop))
                grouped[k, d].append(pos)
                self.degree += d if k != 'soc' else 1
            pos = stop
        self.m = pos
        self.eq = self._inds(eg)
        self.lpos = self._inds(ng)
        self.linear_blocks = ng
        self.exp = any(k == 'exponential' for k, d, a, b in self.nonlinear)
        self.psd = any(k == 'psd' for k, d, a, b in self.nonlinear)
        self.groups = {}
        for (k, d), starts in grouped.items():
            if k != 'psd':
                self.groups[k, d] = np.asarray(starts, dtype=np.int32)

    @staticmethod
    def _inds(ranges):
        if not ranges:
            return np.empty(0, dtype=np.int32)
        if len(ranges) == 1:
            return np.arange(*ranges[0], dtype=np.int32)
        return np.concatenate([np.arange(a, b, dtype=np.int32) for a, b in ranges])

    def cone_error(self, u, dual):
        error = maxabs(u[self.eq]) if not dual and self.eq.size else 0.
        if len(self.linear_blocks) <= 8:
            for a, b in self.linear_blocks:
                v = u[a:b]
                lo = float(v.min())
                if lo < 0:
                    error = max(error, -lo / max(1., abs(lo), float(v.max())))
        elif self.lpos.size:
            # Use unscaled violations on the common exactly feasible path.
            vals = u[self.lpos]
            if np.any(vals < 0):
                for a, b in self.linear_blocks:
                    v = u[a:b]
                    lo = float(v.min())
                    if lo < 0:
                        error = max(error, -lo / max(1., abs(lo), float(v.max())))
        for (k, d), starts in self.groups.items():
            if starts.size == 1:
                a = starts[0]
                v = u[a:a+d].reshape(1, -1)
            else:
                v = u[starts[:, None] + np.arange(d, dtype=np.int32)]
            v = v / np.maximum(1., np.max(np.abs(v), axis=1, keepdims=True))
            if k == 'soc':
                err = np.sqrt(np.einsum('ij,ij->i', v[:, 1:], v[:, 1:])) - v[:, 0]
            elif dual:
                err = exp_error(-v[:, 1], -v[:, 0], math.e * v[:, 2])
            else:
                err = exp_error(*v.T)
            error = max(error, float(err.max()))
        if self.psd:
            from scipy.linalg import eigvalsh
            for k, d, a, b in self.nonlinear:
                if k == 'psd':
                    v = u[a:b]
                    mat = smat(v / max(1., maxabs(v)), d)
                    eig = eigvalsh(mat, subset_by_index=(0, 0), check_finite=False)[0]
                    error = max(error, -float(eig))
        return error

    def project_roundoff(self, v, dual):
        """Small exact-cone corrections when a computed residual is near tolerance."""
        u = v.copy()
        if not dual:
            u[self.eq] = 0.
        u[self.lpos] = np.maximum(u[self.lpos], 0.)
        for (k, d), starts in self.groups.items():
            idx = starts[:, None] + np.arange(d, dtype=np.int32)
            w = u[idx]
            if k == 'soc':
                w[:, 0] = np.maximum(w[:, 0], np.linalg.norm(w[:, 1:], axis=1))
            else:
                a, b, c = (-w[:, 1], -w[:, 0], math.e*w[:, 2]) if dual else w.T
                bnd = np.maximum.reduce([np.abs(b), np.maximum(a, 0.), np.maximum(-c, 0.)])
                good = (b > 0) & (c > 0)
                deficit = np.full(b.shape, np.inf)
                deficit[good] = a[good] - b[good]*(np.log(c[good])-np.log(b[good]))
                use = good & (deficit < bnd)
                # Adjust the first primal coordinate, or use the closed face.
                a = np.where(use, a - np.maximum(0., np.where(use, deficit, 0.)),
                             np.minimum(a, 0.))
                b = np.where(use, b, 0.)
                c = np.maximum(c, 0.)
                w = np.column_stack([-b, -a, c/math.e] if dual else [a, b, c])
            u[idx] = w
        if self.psd:
            from scipy.linalg import eigh
            for k, d, a, b in self.nonlinear:
                if k == 'psd':
                    lam, U = eigh(smat(u[a:b], d), check_finite=False)
                    if lam[0] < 0:
                        u[a:b] = svec((U * np.maximum(lam, 0.)) @ U.T)
        return u


class Problem:
    def __init__(self, P, q, A, b, layout, offset=0.):
        self.P, self.q, self.A, self.b = P, q, A, b
        self.K = layout
        self.offset = offset
        self.m, self.n = A.shape
        self.an, self.pn = maxabs(A.data) or 1., maxabs(P.data) or 1.
        self.bn, self.qn = maxabs(b), maxabs(q)
        self._At = A.T
        self.tol = 9.7e-9
        self.best = np.inf

    def validate_optimal(self, x, s, z, Ax=None, Px=None, Atz=None, monitor=False):
        if (not np.isfinite(x).all() or not np.isfinite(s).all()
                or not np.isfinite(z).all()):
            return False
        Ax = self.A @ x if Ax is None else Ax
        Px = self.P @ x if Px is None else Px
        Atz = self._At @ z if Atz is None else Atz
        rp = maxabs(Ax + s - self.b) / (1. + max(maxabs(Ax), maxabs(s), self.bn))
        rd = maxabs(Px + self.q + Atz) / (1. + max(maxabs(Px), self.qn, maxabs(Atz)))
        xx = float(x @ Px)
        p0, d0 = 0.5 * xx + float(self.q @ x), -0.5 * xx - float(self.b @ z)
        g = abs(p0-d0)/max(1., abs(p0), abs(d0))
        metric = max(rp, rd, g)
        if not np.isfinite([rp, rd, p0, d0, g, p0+self.offset, d0+self.offset]).all():
            return False
        if metric > self.tol and not monitor:
            return False
        cp = self.K.cone_error(s, False)
        cd = self.K.cone_error(z, True)
        metric = max(metric, cp, cd)
        if monitor:
            self.best = min(self.best, metric)
            if monitor > 1:
                print(" check %.2e %.2e %.2e %.2e %.2e" % (rp, rd, g, cp, cd), flush=True)
        return metric <= self.tol and np.isfinite(s @ z)

    def optimal(self, x, z, s=None, *, polish_slack=True, monitor=False):
        Ax, Px, Atz = self.A @ x, self.P @ x, self._At @ z
        exact = self.b - Ax
        if s is not None and self.validate_optimal(x, s, z, Ax, Px, Atz, monitor):
            return {'status': 'solved', 'x': x, 's': s, 'z': z}
        if self.validate_optimal(x, exact, z, Ax, Px, Atz, monitor if s is None else False):
            return {'status': 'solved', 'x': x, 's': exact, 'z': z}
        if (polish_slack and np.isfinite(exact).all() and np.isfinite(Atz).all()
            and maxabs(Px+self.q+Atz) <= self.tol*(1.+max(maxabs(Px), self.qn, maxabs(Atz)))):
            projected = self.K.project_roundoff(exact, False)
            if self.validate_optimal(x, projected, z, Ax, Px, Atz):
                return {'status': 'solved', 'x': x, 's': projected, 'z': z}
        return None

    def primal_cert(self, z):
        zn = maxabs(z)
        if zn == 0. or not np.isfinite(zn) or self.bn == 0.:
            return None
        z = z / zn
        beta = -float((self.b / self.bn) @ z)
        if not np.isfinite(beta) or beta <= 1.01e-8:
            return None
        A = normalized(self.A,self.an)
        if (maxabs(A.T @ z) <= self.tol * beta
                and self.K.cone_error(z, True) <= self.tol * beta):
            return {'status': 'primal_infeasible', 'z': z}
        return None

    def dual_cert(self, x, s=None):
        xn = maxabs(x)
        if xn == 0. or not np.isfinite(xn) or self.qn == 0.:
            return None
        x = x / xn
        alpha = -float((self.q / self.qn) @ x)
        if not np.isfinite(alpha) or alpha <= 1.01e-8:
            return None
        if self.P.nnz:
            P = normalized(self.P,self.pn)
            if not maxabs(P @ x) <= self.tol * alpha:
                return None
        A = normalized(self.A,self.an)
        Ax = A @ x
        s0 = -Ax
        limit = self.tol * alpha
        if s is not None:
            s = s / xn / self.an
            if maxabs(Ax + s) <= limit and self.K.cone_error(s, False) <= limit:
                return {'status': 'dual_infeasible', 'x': x, 's': s * self.an}
        if np.isfinite(s0).all() and self.K.cone_error(s0, False) <= limit:
            return {'status': 'dual_infeasible', 'x': x, 's': s0 * self.an}
        return None


def read_problem(path):
    with np.load(path, allow_pickle=False) as f:
        q, b = f['q'], f['b']
        mat = []
        for k, shape in [('P', (len(q), len(q))), ('A', (len(b), len(q)))]:
            M = sparse.csc_matrix((f[k+'_data'], f[k+'_indices'], f[k+'_indptr']),
                                  shape=shape)
            M.sum_duplicates()
            M.eliminate_zeros()
            mat.append(M)
        meta = json.loads(str(f['metadata_json']))
        spec = meta['cones']
    return Problem(mat[0], q, mat[1], b,
                   Layout([(d['type'], d['dim']) for d in spec]),
                   float(meta.get('objective_offset',0.)))


def write_result(path, result):
    result = (result or {'status': 'unresolved'}).copy()
    metadata = np.array(json.dumps({'schema_version': 1, 'status': result.pop('status')}))
    with open(path, 'wb') as f:
        np.savez(f, metadata_json=metadata, **result)


class Scaled:
    """Cone-preserving diagonal Ruiz equilibration."""
    def __init__(self, p, passes=8, scale_rhs=False):
        self.base = p
        self.K, self.n, self.m = p.K, p.n, p.m
        self.D, self.E, self.c = np.ones(p.n), np.ones(p.m), 1.
        self.A, self.P = p.A.copy(), p.P.copy()
        self.q, self.b = p.q.copy(), p.b.copy()
        if not p.n:
            return
        ir = self.A.indices
        ac = np.repeat(np.arange(p.n, dtype=np.int32), np.diff(self.A.indptr))
        pr = self.P.indices
        pc = np.repeat(np.arange(p.n, dtype=np.int32), np.diff(self.P.indptr))
        for _ in range(passes):
            cmax = np.zeros(p.n)
            rmax = np.zeros(p.m)
            aabs = np.abs(self.A.data)
            np.maximum.at(rmax, ir, aabs)
            np.maximum.at(cmax, ac, aabs)
            if pr.size:
                np.maximum.at(cmax, pc, np.abs(self.P.data))
            for k, _, s, e in self.K.nonlinear:
                rmax[s:e] = rmax[s:e].max()
            d = np.clip(1./np.sqrt(np.where(cmax > 1.e-15, cmax, 1.)),
                        1.e-10/self.D, 1.e10/self.D)
            e = np.clip(1./np.sqrt(np.where(rmax > 1.e-15, rmax, 1.)),
                        1.e-10/self.E, 1.e10/self.E)
            self.A.data *= d[ac]*e[ir]
            self.P.data *= d[pc]*d[pr]
            self.q *= d
            self.b *= e
            self.D *= d
            self.E *= e
            # Avoid making the quadratic curvature tiny relative to the gradient.
            freq = max(maxabs(self.q), maxabs(self.P.data)*0.1, 1.e-14)
            if freq > 1.e-13:
                c = float(np.clip(1./freq, 1.e-12/self.c, 1.e12/self.c))
                self.P.data *= c
                self.q *= c
                self.c *= c
            if np.max(np.abs(d-1.), initial=0.) < .06 and np.max(np.abs(e-1.), initial=0.) < .06:
                break
        if scale_rhs:
            big = maxabs(self.b)
            if big > 30. or 0. < big < .03:
                gam = float(np.clip(big, 1.e-7, 1.e7))
                self.D *= gam
                self.E /= gam
                self.b /= gam
                self.P.data *= gam
                self.c /= gam
                freq = max(maxabs(self.q), maxabs(self.P.data)*0.1, 1.e-10)
                self.q /= freq
                self.P.data /= freq
                self.c /= freq
