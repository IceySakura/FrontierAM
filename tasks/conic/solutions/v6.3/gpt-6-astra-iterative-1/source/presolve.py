from collections import deque
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from problem import Problem, norm, cone_defect
from cones import PSD, SQRT2


class Identity:
    def __init__(self, problem, result=None):
        self.reduced = problem
        self.result = result

    def recover(self, result):
        return result


def prepare(problem, validate=None):
    from problem import assess
    if validate is None:
        validate = lambda result, cones=True: assess(problem,result,cones=cones)
    if problem.q.any():
        free = ((np.diff(problem.A.indptr) == 0)
                & (np.diff(problem.P.indptr) == 0) & (problem.q != 0))
        if np.any(free):
            x = np.zeros(problem.n)
            x[free] = -np.sign(problem.q[free])
            result = {"status":"dual_infeasible","x":x,"s":np.zeros(problem.m)}
            if validate(result)[0]:
                return Identity(problem,result)
    rows = np.bincount(problem.A.indices, minlength=problem.m)
    bad_constant = (rows == 0) & (problem.b < 0)

    def constant_proof(ids, zero=False):
        ids = np.asarray(ids,dtype=np.int32)
        if not len(ids):
            return None
        z = np.zeros(problem.m)
        z[ids] = -np.sign(problem.b[ids]) if zero else 1.
        result = {"status":"primal_infeasible","z":z}
        if validate(result)[0]:
            return Identity(problem,result)
        return None

    start = 0
    removable = 0
    soc, exp = [], []
    for cone in problem.cones:
        kind, dim = cone["type"], cone["dim"]
        if kind == "psd":
            j = np.arange(dim)
            ids = start+j*(j+3)//2
            proof = constant_proof(ids[bad_constant[ids]])
            if proof is not None:
                return proof
            return Presolve(problem)
        end = start+dim
        if kind == "zero":
            bad = start+np.flatnonzero((rows[start:end] == 0)
                                      & (problem.b[start:end] != 0))
            proof = constant_proof(bad,True)
            if proof is not None:
                return proof
            if np.any(rows[start:end] == 1):
                return Presolve(problem)
            removable += np.count_nonzero((rows[start:end] == 0)
                                          & (problem.b[start:end] == 0))
        elif kind == "nonnegative":
            bad = start+np.flatnonzero(bad_constant[start:end])
            proof = constant_proof(bad)
            if proof is not None:
                return proof
            removable += np.count_nonzero((rows[start:end] == 0)
                                          & (problem.b[start:end] >= 0))
        elif kind == "soc":
            if bad_constant[start]:
                proof = constant_proof([start])
                if proof is not None:
                    return proof
            if dim == 2:
                return Presolve(problem)
            soc.append(start)
        elif kind == "exponential":
            if bad_constant[start+1] or bad_constant[start+2]:
                ids = [i for i in (start+1,start+2) if bad_constant[i]]
                proof = constant_proof(ids)
                if proof is not None:
                    return proof
            exp.append(start+1)
        start = end
    for entries in (soc, exp):
        if entries:
            ids = np.asarray(entries)
            if np.any((rows[ids] == 0) & (problem.b[ids] == 0)):
                return Presolve(problem)
    if removable > max(100, problem.m//20):
        return Presolve(problem)
    return Identity(problem)


class Presolve:
    """Exact substitutions and orthogonal row transformations, with dual recovery."""
    def __init__(self, problem):
        self.result = None
        self.original = problem
        A, b, n, m = problem.A, problem.b, problem.n, problem.m
        self.Ar = A.tocsr()
        zero = np.zeros(m, dtype=bool)
        start = 0
        for block in problem.cones:
            dim = block["dim"]
            width = dim*(dim+1)//2 if block["type"] == "psd" else dim
            if block["type"] == "zero":
                zero[start:start+width] = True
            start += width
        self.zero = zero
        fixed = np.zeros(n, dtype=bool)
        values = np.zeros(n)
        records = []
        bn = b.copy()
        counts = np.diff(self.Ar.indptr).copy()
        if n <= 100000 and A.nnz <= 3000000:
            queue = deque(np.flatnonzero(zero & (counts == 1)))
            while queue:
                row = queue.popleft()
                if counts[row] != 1:
                    continue
                a, e = self.Ar.indptr[row:row+2]
                cols = self.Ar.indices[a:e]
                vals = self.Ar.data[a:e]
                free = ~fixed[cols]
                if np.count_nonzero(free) != 1:
                    continue
                col, coef = int(cols[free][0]), float(vals[free][0])
                value = bn[row]/coef
                if not np.isfinite(value) or abs(value) > 1e100:
                    continue
                fixed[col], values[col] = True, value
                records.append((row, col, coef))
                ca, ce = A.indptr[col:col+2]
                rows = A.indices[ca:ce]
                bn[rows] -= A.data[ca:ce]*value
                counts[rows] -= 1
                newly = rows[zero[rows] & (counts[rows] == 1)]
                queue.extend(newly)
                bn[row] = 0.
        self.values, self.records = values, records
        self.free = np.flatnonzero(~fixed)
        Af = A[:, self.free].tocsc() if records else A
        P = problem.P[self.free, :][:, self.free].tocsc() if records else problem.P
        q = (problem.q + problem.P@values)[self.free] if records else problem.q
        rowcount = np.diff(Af.tocsr().indptr)
        rr, cc, vv, blocks = [], [], [], []
        target = 0
        self.soc_faces = []
        self.psd_faces = []
        self.exp_faces = []

        def append(kind, dim, ids, transform=None):
            nonlocal target
            width = dim*(dim+1)//2 if kind == "psd" else dim
            blocks.append({"type": kind, "dim": dim})
            if transform is None:
                rr.extend(range(target, target+width))
                cc.extend(ids)
                vv.extend(np.ones(width))
            else:
                r, c = np.nonzero(transform)
                rr.extend(target+r)
                cc.extend(np.asarray(ids)[c])
                vv.extend(transform[r, c])
            target += width

        def psd_component(vertices, original_start):
            dim = len(vertices)
            c, r = np.tril_indices(dim)
            r, c = vertices[r], vertices[c]
            ids = original_start+c*(c+1)//2+r
            if dim == 1:
                if rowcount[ids[0]] == 0 and bn[ids[0]] >= 0:
                    return
                append("nonnegative", 1, ids)
            elif dim == 2:
                T = np.array([[1/SQRT2, 0, 1/SQRT2],
                              [1/SQRT2, 0, -1/SQRT2], [0, 1, 0]])
                append("soc", 3, ids, T)
            else:
                append("psd", dim, ids)

        start = 0
        for block in problem.cones:
            kind, dim = block["type"], block["dim"]
            width = dim*(dim+1)//2 if kind == "psd" else dim
            ids = np.arange(start, start+width)
            used = rowcount[ids] != 0
            if kind == "zero":
                keep = used | (bn[ids] != 0)
                ids = ids[keep]
                if len(ids):
                    append(kind, len(ids), ids)
            elif kind == "nonnegative":
                keep = used | (bn[ids] < 0)
                ids = ids[keep]
                if len(ids):
                    append(kind, len(ids), ids)
            elif not np.any(used) and cone_defect(bn[ids], [block]) <= 1e-12:
                pass
            elif kind == "soc" and not used[0] and bn[start] == 0:
                append("zero", dim-1, ids[1:])
                self.soc_faces.append(ids)
            elif kind == "soc" and dim == 2:
                T = np.array([[1., 1.], [1., -1.]])/SQRT2
                append("nonnegative", 2, ids, T)
            elif kind == "exponential" and not used[1] and bn[start+1] == 0:
                append("nonnegative", 2, ids,
                       np.array([[-1., 0., 0.], [0., 0., 1.]]))
                self.exp_faces.append(ids)
            elif kind == "psd" and dim <= 2000:
                p = PSD(start, dim)
                activity = used | (bn[ids] != 0)
                removed = np.flatnonzero(~activity[p.diag])
                vertices = np.flatnonzero(activity[p.diag])
                is_removed = np.zeros(dim, dtype=bool)
                is_removed[removed] = True
                if len(removed):
                    face = (is_removed[p.r] | is_removed[p.c]) & p.off
                    zero_ids = ids[face & activity]
                    if len(zero_ids):
                        append("zero", len(zero_ids), zero_ids)
                    self.psd_faces.append((p, removed, vertices))
                off = activity & p.off & ~is_removed[p.r] & ~is_removed[p.c]
                r, c = p.r[off], p.c[off]
                if not len(vertices):
                    start += width
                    continue
                inv = np.empty(dim, dtype=int)
                inv[vertices] = np.arange(len(vertices))
                r, c = inv[r], inv[c]
                graph = sparse.coo_matrix((np.ones(2*len(r)),
                    (np.r_[r,c], np.r_[c,r])),
                    shape=(len(vertices),len(vertices))).tocsr()
                count, labels = connected_components(graph, directed=False)
                if count == 1 and dim > 2 and not len(removed):
                    append(kind, dim, ids)
                else:
                    for label in range(count):
                        psd_component(vertices[labels == label], start)
            else:
                append(kind, dim, ids)
            start += width
        self.R = sparse.csc_matrix((vv, (rr, cc)), shape=(target, m))
        self.reduced = Problem(P, q, (self.R@Af).tocsc(), self.R@bn, blocks)

    def recover(self, result):
        status = result["status"]
        if status == "unresolved":
            return result
        p = self.original
        out = {"status": status}
        if status in ("solved", "dual_infeasible"):
            x = self.values.copy() if status == "solved" else np.zeros(p.n)
            x[self.free] = result["x"]
            out["x"] = x
            base = p.b - p.A@x if status == "solved" else -p.A@x
            s = base + self.R.T@(result["s"] - self.R@base)
            s[self.zero] = 0.
            out["s"] = s
        if status in ("solved", "primal_infeasible"):
            z = np.asarray(self.R.T@result["z"])
            for ids in self.soc_faces:
                z[ids[0]] = np.linalg.norm(z[ids[1:]])
            eps = max(1e-300, 1e-12/(1+norm(p.b)+norm(p.A.data)))
            for ids in self.exp_faces:
                a = max(0., -z[ids[0]])
                z[ids[0]] = -a
                z[ids[2]] = max(0., z[ids[2]])
                if a > 0:
                    z[ids[2]] = max(eps, z[ids[2]])
                    z[ids[1]] = max(0., a*(np.log(a)-np.log(z[ids[2]])-1)) + eps
                else:
                    z[ids[1]] = max(0., z[ids[1]])
            for psd, removed, vertices in self.psd_faces:
                Z = psd.mat(z[psd.sl])
                D = Z[np.ix_(removed, removed)]
                if len(vertices):
                    C = Z[np.ix_(vertices, vertices)]
                    eig, U = np.linalg.eigh(C)
                    shift = max(0., -eig[0]) + eps
                    C += np.eye(len(vertices))*shift
                    B = Z[np.ix_(removed, vertices)]
                    V = (B@U)/np.sqrt(np.maximum(eig+shift, eps))[None, :]
                    schur = V@V.T
                    Z[np.ix_(vertices, vertices)] = C
                else:
                    schur = np.zeros_like(D)
                off = D-schur
                np.fill_diagonal(off, 0)
                Z[removed, removed] = np.diag(schur)+np.sum(np.abs(off), axis=1)+eps
                z[psd.sl] = psd.vec(Z)
            residual = p.A.T@z
            if status == "solved":
                residual += p.P@out["x"]+p.q
            for row, col, coef in reversed(self.records):
                delta = -residual[col]/coef
                z[row] += delta
                a, e = self.Ar.indptr[row:row+2]
                residual[self.Ar.indices[a:e]] += self.Ar.data[a:e]*delta
            out["z"] = z
        return out
