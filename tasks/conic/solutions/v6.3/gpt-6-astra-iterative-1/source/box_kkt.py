import numpy as np
from scipy import linalg
from problem import norm


class BoxKKT:
    def __init__(self, P, A, cone):
        self.P, self.A, self.At, self.cone = P, A, A.T, cone
        self.m, self.n = A.shape
        counts = np.bincount(A.indices,minlength=self.m)[cone.nn]
        if np.max(counts, initial=0) > 1:
            raise ValueError("Not a bound-constrained Hessian")
        B = A[cone.nn, :].tocsc()
        self.columns = np.zeros(len(cone.nn), dtype=np.int32)
        self.coefficients = np.zeros(len(cone.nn))
        self.columns[B.indices] = np.repeat(np.arange(self.n), np.diff(B.indptr))
        self.coefficients[B.indices] = B.data
        self.dense_P = P.toarray(order="F")
        self.eq = A[cone.zero, :].toarray()
        self.diagonal = np.arange(self.n)
        self.w = np.zeros(self.m)

    def factor(self, scaling):
        self.w.fill(0.)
        self.w[self.cone.nn] = scaling.diag
        self.inverse_w = 1/scaling.diag
        weights = self.coefficients**2*self.inverse_w
        H = self.dense_P.copy(order="F")
        H[self.diagonal,self.diagonal] += np.bincount(
            self.columns, weights=weights, minlength=self.n)+1e-12
        self.chol = linalg.cho_factor(H, lower=True, overwrite_a=True, check_finite=False)
        if len(self.cone.zero):
            self.V = linalg.cho_solve(self.chol, self.eq.T, check_finite=False)
            S = self.eq@self.V
            S = .5*(S+S.T)
            S.flat[::len(S)+1] += 1e-12
            self.schur = linalg.cho_factor(S, lower=True, check_finite=False)

    def linear(self, rx, rz):
        rhs = rx+np.bincount(self.columns,
            weights=self.coefficients*self.inverse_w*rz[self.cone.nn], minlength=self.n)
        x = linalg.cho_solve(self.chol, rhs, check_finite=False)
        z = np.zeros(self.m)
        if len(self.cone.zero):
            ze = linalg.cho_solve(self.schur, self.eq@x-rz[self.cone.zero],
                                 check_finite=False)
            x -= self.V@ze
            z[self.cone.zero] = ze
        z[self.cone.nn] = self.inverse_w*(
            self.coefficients*x[self.columns]-rz[self.cone.nn])
        return x, z

    def solve(self, rx, rz):
        x, z = self.linear(rx,rz)
        previous = np.inf
        for _ in range(5):
            r1 = rx-self.P@x-self.At@z
            r2 = rz-self.A@x+self.w*z
            error = max(norm(r1)/(1+norm(rx)),norm(r2)/(1+norm(rz)))
            if error < 2e-12 or error >= previous*.95:
                break
            previous = error
            dx, dz = self.linear(r1,r2)
            x += dx
            z += dz
        error = max(norm(rx-self.P@x-self.At@z)/(1+norm(rx)),
                    norm(rz-self.A@x+self.w*z)/(1+norm(rz)))
        if not np.isfinite(x).all() or not np.isfinite(z).all() or error > 2e-10:
            raise ArithmeticError("Bound Hessian requires pivoting")
        return x,z
