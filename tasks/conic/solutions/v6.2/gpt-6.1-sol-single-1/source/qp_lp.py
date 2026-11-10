"""Simplicial decomposition for polyhedral QPs with a small quadratic support."""
import os
import sys
import time

import numpy as np
from scipy import linalg, sparse


def solve_qp_lp(P,q,A,b,cones):
    from solver import highs_model,highs_core,polyhedral,poly_candidate,norm
    hc=highs_core()
    active=np.flatnonzero(np.diff(P.indptr))
    if not len(active) or len(active)>512:
        return None
    start=time.monotonic()
    budget=float(os.environ.get("QPLP_SECONDS","5.0"))
    eq=np.zeros(len(b),dtype=bool)
    st=0
    for c in cones:
        if c["type"]=="zero":
            eq[st:st+c["dim"]]=True
        st+=c["dim"]
    lo=np.full(len(b),-hc.kHighsInf)
    lo[eq]=b[eq]
    h=highs_model(None,q,A,lo,b,time_limit=budget,feas_tol=1e-10,
                  method="ipm",crossover=True)
    if os.environ.get("SOLVER_DEBUG"):
        print("FW initial",h.modelStatusToString(h.getModelStatus()),
              time.monotonic()-start,file=sys.stderr,flush=True)
    if h.getModelStatus()==hc.HighsModelStatus.kInfeasible:
        result=polyhedral(sparse.csc_matrix(P.shape),np.zeros(len(q)),A,b,cones)
        return result if result["status"]=="primal_infeasible" else None
    if h.getModelStatus() in (hc.HighsModelStatus.kUnbounded,hc.HighsModelStatus.kUnboundedOrInfeasible):
        h=highs_model(None,np.zeros(len(q)),A,lo,b,time_limit=budget,feas_tol=1e-10)
    if h.getModelStatus()!=hc.HighsModelStatus.kOptimal:
        return None
    h.setOptionValue("solver","simplex")
    point=np.asarray(h.getSolution().col_value)
    vertices=point[:,None]
    weights=np.ones(1)
    pp=P[active][:,active].toarray()
    debug=os.environ.get("SOLVER_DEBUG")
    previous=np.inf
    stagnant=0

    def master(X,old):
        V=X[active]
        G=V.T@pp@V
        c=q@X
        scale=max(1.,norm(G),norm(c))
        a=old.copy()
        for minor in range(len(a)+8):
            k=len(a)
            K=np.block([[G/scale,np.ones((k,1))],
                        [np.ones((1,k)),np.zeros((1,1))]])
            rhs=np.r_[-c/scale,1.]
            sol=linalg.lstsq(K,rhs,cond=1e-15,lapack_driver="gelsy",
                             check_finite=False)[0]
            residual=norm(K@sol-rhs)
            trial=sol[:k]
            if residual>1e-9:
                from scipy.optimize import minimize
                rr=minimize(lambda v:.5*v@G@v/scale+c@v/scale,a,
                    jac=lambda v:(G@v+c)/scale,method="SLSQP",
                    bounds=[(0.,1.)]*k,
                    constraints={"type":"eq","fun":lambda v:np.sum(v)-1.,
                                 "jac":lambda v:np.ones(k)},
                    options={"ftol":1e-14,"maxiter":150})
                trial=np.maximum(rr.x,0.)
                trial/=np.sum(trial)
                keep=trial>1e-9
                if np.any(~keep):
                    X=X[:,keep]
                    a=trial[keep]
                    V=X[active]
                    G=V.T@pp@V
                    c=q@X
                    continue
                return X,trial
            if np.min(trial,initial=0.)>=-1e-11:
                trial=np.maximum(trial,0.)
                trial/=np.sum(trial)
                keep=trial>1e-12
                return X[:,keep],trial[keep]/np.sum(trial[keep])
            bad=trial<0
            step=np.min(a[bad]/(a[bad]-trial[bad]))
            a=(1.-step)*a+step*trial
            keep=a>1e-12
            if not np.any(keep):
                keep[np.argmax(a)]=True
            X=X[:,keep]
            a=a[keep]
            a/=np.sum(a)
            V=X[active]
            G=V.T@pp@V
            c=q@X
        return X,a

    for iteration in range(120):
        px=P@point
        gradient=q+px
        remaining=budget-(time.monotonic()-start)
        if remaining<=0:
            break
        h.changeColsCost(len(active),active.astype(np.int32),gradient[active])
        h.setOptionValue("time_limit",h.getRunTime()+remaining)
        h.run()
        if h.getModelStatus()!=hc.HighsModelStatus.kOptimal:
            break
        sol=h.getSolution()
        vertex=np.asarray(sol.col_value)
        z=-np.asarray(sol.row_dual)
        s=b-A@point
        gap=float(gradient@(point-vertex))
        if debug:
            print("FW",iteration,len(weights),"gap %.3e time %.3f" %
                  (gap,time.monotonic()-start),file=sys.stderr,flush=True)
        if poly_candidate(P,q,A,b,cones,point,s,z,tol=8e-9):
            return {"status":"solved","x":point,"s":s,"z":z}
        stagnant=stagnant+1 if gap>=previous*(1.-1e-8) else 0
        previous=gap
        if stagnant>5:
            break
        vertices=np.column_stack([vertices,vertex])
        weights=np.r_[weights,0.]
        vertices,weights=master(vertices,weights)
        point=vertices@weights
    return None
