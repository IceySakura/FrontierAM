import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
from native_ldl import LDL
from problem import norm


class PolyKKT:
    """Reuse the entire ordered QP matrix; only cone diagonals change."""
    def __init__(self, P, A, cone):
        self.P, self.A, self.At, self.cone = P, A, A.T, cone
        self.m, self.n = A.shape
        n, m = self.n, self.m
        self.reg = 1e-12
        diagonal = np.full(m, -self.reg)
        diagonal[cone.nn] -= 1.
        K = sparse.bmat([[P+sparse.eye(n,format="csc")*self.reg, A.T],
                         [A, sparse.diags(diagonal,format="csc")]],format="csc")
        self.order = reverse_cuthill_mckee(K, symmetric_mode=True)
        inverse = np.empty(n+m, dtype=np.int32)
        inverse[self.order] = np.arange(n+m, dtype=np.int32)
        self.upper = sparse.triu(K[self.order,:][:,self.order],format="csc")
        self.upper.sort_indices()
        positions = self.upper.indptr[1:]-1
        self.dual_positions = positions[inverse[n:]]
        sign = np.r_[np.ones(n), -np.ones(m)]
        self.factorization = LDL(self.upper, sign[self.order])
        self.w = np.zeros(m)

    def factor(self, scaling):
        self.w.fill(0.)
        self.w[self.cone.nn] = scaling.diag
        self.upper.data[self.dual_positions] = -self.w-self.reg
        self.factorization.factor(self.upper)

    def linear(self, rhs):
        ordered = self.factorization.solve(rhs[self.order], overwrite=True)
        out = np.empty_like(rhs)
        out[self.order] = ordered
        return out

    def solve(self, rx, rz):
        rhs = np.r_[rx,rz]
        out = self.linear(rhs)
        previous = np.inf
        changed = False
        for _ in range(6):
            changed = False
            x, z = out[:self.n], out[self.n:]
            r1 = rx-self.P@x-self.At@z
            r2 = rz-self.A@x+self.w*z
            error = max(norm(r1)/(1+norm(rx)), norm(r2)/(1+norm(rz)))
            if error < 2e-12 or error >= previous*.95:
                break
            previous = error
            out += self.linear(np.r_[r1,r2])
            changed = True
        if changed:
            x, z = out[:self.n], out[self.n:]
            error = max(norm(rx-self.P@x-self.At@z)/(1+norm(rx)),
                        norm(rz-self.A@x+self.w*z)/(1+norm(rz)))
        if not np.isfinite(out).all() or error > 2e-10:
            raise ArithmeticError("QP refinement requires pivoting")
        return out[:self.n], out[self.n:]
