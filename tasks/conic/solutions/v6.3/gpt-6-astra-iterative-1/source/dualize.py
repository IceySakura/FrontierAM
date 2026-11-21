import numpy as np
from scipy import sparse
from problem import Problem


class Dualized:
    def __init__(self, original, cone_rows, equality_rows, cones, inverse):
        self.original = original
        self.cone_rows = cone_rows
        self.equality_rows = equality_rows
        self.inverse = inverse
        bK = original.b[cone_rows]
        self.constant = inverse@bK
        E = original.A[equality_rows, :].tocsc()
        B = E@inverse
        d = E@self.constant-original.b[equality_rows]
        h = -inverse.T@original.q
        self.order = np.arange(original.n)
        self.scale = np.ones(original.n)
        start = 0
        for cone in cones:
            dim = cone["dim"]
            width = dim*(dim+1)//2 if cone["type"] == "psd" else dim
            if cone["type"] == "exponential":
                self.order[start:start+2] = [start+1, start]
                self.scale[start:start+3] = [-1., -1., np.e]
            start += width
        A = (-B.T)[self.order, :].multiply(self.scale[:, None]).tocsc()
        b = self.scale*h[self.order]
        r = len(equality_rows)
        self.problem = Problem(sparse.csc_matrix((r,r)), d, A, b, cones)

    @classmethod
    def create(cls, problem):
        if problem.P.nnz or problem.n < 100:
            return None
        K, eq, cones, start = [], [], [], 0
        for cone in problem.cones:
            dim = cone["dim"]
            width = dim*(dim+1)//2 if cone["type"] == "psd" else dim
            rows = np.arange(start, start+width)
            if cone["type"] == "zero":
                eq.append(rows)
            else:
                K.append(rows)
                cones.append(cone)
            start += width
        cone_rows = np.concatenate(K) if K else np.empty(0, dtype=int)
        equality_rows = np.concatenate(eq) if eq else np.empty(0, dtype=int)
        n = problem.n
        if len(cone_rows) != n or n < 2*len(equality_rows):
            return None
        AK = problem.A[cone_rows, :].tocsc()
        if (AK.nnz != n or np.any(np.diff(AK.indptr) != 1)
                or len(np.unique(AK.indices)) != n):
            return None
        inverse = sparse.csc_matrix(
            (1/AK.data, (np.arange(n), AK.indices)), shape=(n,n))
        return cls(problem, cone_rows, equality_rows, cones, inverse)

    def recover(self, result):
        status = result["status"]
        if status == "unresolved":
            return result
        p = self.original
        if status in ("solved", "primal_infeasible"):
            u = np.empty(p.n)
            u[self.order] = self.scale*result["z"]
            x = -self.inverse@u
            if status == "solved":
                x += self.constant
            s = np.zeros(p.m)
            s[self.cone_rows] = u
            out = {"status": "solved" if status == "solved" else "dual_infeasible",
                   "x": x, "s": s}
        else:
            out = {"status": "primal_infeasible"}
        if status in ("solved", "dual_infeasible"):
            cone_dual = np.empty(p.n)
            cone_dual[self.order] = result["s"]/self.scale
            z = np.zeros(p.m)
            z[self.cone_rows] = cone_dual
            z[self.equality_rows] = -result["x"]
            out["z"] = z
        return out
