"""Analytic elimination of singleton inequality rows in a fixed QP KKT system."""

import os

import numpy as np
from scipy import sparse

from linear import SparseFactors


class BoxKKT:
    def __init__(self, P, A, ne, sigma, bound):
        self.P, self.A, self.n, self.m, self.sigma = P, A, P.shape[0], A.shape[0], sigma
        self.ne = ne
        self.bound = bound
        keep = np.ones(self.m, dtype=bool)
        keep[bound] = False
        self.keep = np.flatnonzero(keep)
        csr = A.tocsr()
        positions = csr.indptr[bound]
        self.cols, self.coeff = csr.indices[positions], csr.data[positions]
        self.C = A[self.keep, :].tocsc()
        self.CT = self.C.T.tocsc()
        self.diagonal = (P.nnz <= self.n and
                         not np.any(P.indices != np.repeat(np.arange(self.n), np.diff(P.indptr))))
        self.Pdiag = P.diagonal() if self.diagonal else None
        unbounded_null = (self.diagonal and np.any((self.Pdiag == 0) &
                         (np.bincount(self.cols, minlength=self.n) == 0)))
        column_counts = np.diff(self.C.indptr).astype(np.int64)
        self.schur = (self.diagonal and not unbounded_null and len(self.keep) <= max(2000, self.n) and
                      np.sum(column_counts * column_counts) <= 10_000_000)
        self.factory = SparseFactors(native=False)
        self.core = None

    def equality_guess(self, q, b):
        if not self.diagonal or len(self.keep) != self.ne or self.ne > 2000:
            return None
        if np.min(self.Pdiag, initial=np.inf) <= 1e-12 * max(1., float(np.max(self.Pdiag, initial=0.))):
            return None
        inverse = 1 / self.Pdiag
        if self.ne:
            S = (self.C @ sparse.diags(inverse) @ self.CT).tocsc()
            regularization = 1e-12 * max(1., float(np.max(np.abs(S.diagonal()), initial=0.)))
            core = self.factory(S + regularization * sparse.eye(self.ne, format="csc"))
            dual_eq = core.solve(-b[:self.ne] - self.C @ (inverse * q))
        else:
            dual_eq = np.empty(0)
        x = -inverse * (q + self.CT @ dual_eq)
        dual = np.zeros(self.m)
        dual[:self.ne] = dual_eq
        return x, dual

    def refactor(self, rho):
        self.rho = rho.copy()
        added = np.bincount(self.cols, weights=rho[self.bound] * self.coeff**2,
                           minlength=self.n)
        if self.schur:
            self.inverse = 1 / (self.Pdiag + self.sigma + added)
            if len(self.keep):
                S = (self.C @ sparse.diags(self.inverse) @ self.CT +
                     sparse.diags(1 / rho[self.keep])).tocsc()
                self.core = self.factory(S)
                p = self.factory.perm
                if p is None:
                    p = np.argsort(self.core.perm_c)
            else:
                p = np.empty(0, dtype=int)
            self.ordering = np.r_[self.n + self.bound, np.arange(self.n), self.n + self.keep[p]]
        else:
            H = self.P + sparse.diags(self.sigma + added)
            D = (sparse.diags(1 / rho[self.keep]) if len(self.keep)
                 else sparse.csc_matrix((0, 0)))
            K = sparse.bmat([[H, self.CT], [self.C, -D]], format="csc")
            self.core = self.factory(K)
            p = self.factory.perm
            if p is None:
                p = np.argsort(self.core.perm_c)
            mapping = np.r_[np.arange(self.n), self.n + self.keep]
            self.ordering = np.r_[self.n + self.bound, mapping[p]]
        raw = getattr(self.core, "factor", self.core)
        self.nnz = (raw.nnz if raw is not None else 0) + 2 * len(self.bound) + self.n
        return self

    def solve(self, rhs):
        n, rows = self.n, rhs[self.n:]
        rb = self.rho[self.bound]
        effective = rhs[:n] + np.bincount(self.cols,
            weights=rb * self.coeff * rows[self.bound], minlength=n)
        if self.schur:
            free = self.inverse * effective
            nu = (self.core.solve(self.C @ free - rows[self.keep]) if len(self.keep)
                  else np.empty(0))
            x = free - self.inverse * (self.CT @ nu)
        else:
            reduced = self.core.solve(np.r_[effective, rows[self.keep]])
            x, nu = reduced[:n], reduced[n:]
        result = np.empty(n + self.m)
        result[:n] = x
        result[n + self.keep] = nu
        result[n + self.bound] = rb * (self.coeff * x[self.cols] - rows[self.bound])
        return result


def make(P, A, ne, sigma):
    if os.getenv("ADMM_BOUND_SPLIT", "1") == "0":
        return None
    m, n = A.shape
    counts = np.bincount(A.indices, minlength=m)
    bound = np.flatnonzero((np.arange(m) >= ne) & (counts == 1))
    if len(bound) < max(10, min(n, m) // 4):
        return None
    return BoxKKT(P, A, ne, sigma, bound)
