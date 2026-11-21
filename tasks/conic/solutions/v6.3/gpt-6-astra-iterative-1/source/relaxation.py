import numpy as np
from problem import assess, norm
from highs_solver import make_highs


def probe(problem,validate):
    if problem.n < 20000 and problem.m < 200000:
        return None
    rows, equality = [], []
    start = 0
    for cone in problem.cones:
        kind, dim = cone["type"],cone["dim"]
        width = dim*(dim+1)//2 if kind == "psd" else dim
        if kind in ("zero","nonnegative"):
            rows.extend(range(start,start+width))
            equality.extend([kind == "zero"]*width)
        elif kind == "soc":
            rows.append(start)
            equality.append(False)
        elif kind == "exponential":
            rows.extend((start+1,start+2))
            equality.extend((False,False))
        else:
            rows.extend(start+np.cumsum(np.arange(1,dim+1))-1)
            equality.extend([False]*dim)
        start += width
    rows = np.asarray(rows,dtype=np.int32)
    equality = np.asarray(equality,dtype=bool)
    A, b = problem.A[rows,:].tocsc(),problem.b[rows]
    scale = max(1.,norm(problem.q))
    q = np.zeros(problem.n) if problem.P.nnz else problem.q/scale
    solver = make_highs(None,q,A,np.where(equality,b,-np.inf),b,
                       options={"solver":"simplex","time_limit":.05})
    solver.run()
    status = str(solver.getModelStatus())
    if "Optimal" in status:
        sol = solver.getSolution()
        x = np.asarray(sol.col_value)
        z = np.zeros(problem.m)
        z[rows] = -scale*np.asarray(sol.row_dual)
        s = problem.b-problem.A@x
        s[rows[equality]] = 0.
        result = {"status":"solved","x":x,"s":s,"z":z}
        if validate(result)[0]:
            return result
    if "Infeasible" in status and norm(b):
        certificate = make_highs(None,b/norm(b),A.T.tocsc(),
            np.zeros(problem.n),np.zeros(problem.n),np.where(equality,-1.,0.),
            np.ones(len(rows)),options={"solver":"simplex","time_limit":.15})
        certificate.run()
        if "Optimal" in str(certificate.getModelStatus()):
            z = np.zeros(problem.m)
            z[rows] = np.asarray(certificate.getSolution().col_value)
            result = {"status":"primal_infeasible","z":z}
            if validate(result)[0]:
                return result
    return None
