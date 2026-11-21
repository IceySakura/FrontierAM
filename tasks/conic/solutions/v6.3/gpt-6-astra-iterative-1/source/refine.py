import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import lsmr
from problem import norm, assess


def project_solution(problem, result):
    if result["status"] != "solved":
        return result
    x, z = result["x"].copy(), result["z"].copy()
    P, q, A, b = problem.P, problem.q, problem.A, problem.b
    Px = P @ x
    C = sparse.vstack((A.T, sparse.csc_matrix(b.reshape(1, -1))), format="csc")
    rhs = np.r_[-Px-q-A.T@z, -x.dot(Px)-q.dot(x)-b.dot(z)]
    cn = np.sqrt(np.asarray(C.power(2).sum(axis=1)).ravel())
    cn = np.maximum(cn, 1e-12)
    C = C.multiply((1/cn)[:, None]).tocsc()
    rhs /= cn
    if C.shape[0] <= 500:
        gram = (C@C.T).toarray()
        multipliers = linalg.lstsq(gram, rhs, cond=1e-14, check_finite=False,
                                  lapack_driver="gelsy")[0]
        delta = C.T@multipliers
        if norm(C@delta-rhs) > 1e-7*norm(rhs):
            delta = lsmr(C, rhs, atol=1e-13, btol=1e-13, maxiter=200)[0]
    else:
        delta = lsmr(C, rhs, atol=1e-12, btol=1e-12, maxiter=150)[0]
    for frac in (1., .75, .5, .25):
        zn = z + frac*delta
        refined = {"status": "solved", "x": x, "s": b-A@x, "z": zn}
        if assess(problem, refined)[0]:
            return refined
        refined["s"] = result["s"]
        if assess(problem, refined)[0]:
            return refined
    return result
