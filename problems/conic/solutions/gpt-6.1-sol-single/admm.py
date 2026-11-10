"""Operator splitting for linearly constrained QPs, with active-set polishing."""
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

from cones import Layout
from ipm import Factor,equilibrate,norm,certificate


def solve_admm(P0,q0,A0,b0,cones):
    from solver import poly_candidate
    begin=time.monotonic()
    n,m=len(q0),len(b0)
    budget=float(os.environ.get("ADMM_SECONDS",str(max(.4,min(2.,8e-5*n)))))
    layout=Layout(cones,m)
    P,q,A,b,d,e,cost=equilibrate(P0,q0,A0,b0,layout)
    eq=np.zeros(m,dtype=bool)
    eq[layout.eq]=True
    lower=np.full(m,-np.inf)
    lower[eq]=b[eq]
    x=np.zeros(n)
    z=np.minimum(np.maximum(np.zeros(m),lower),b)
    y=np.zeros(m)
    sigma=1e-6
    rho=float(os.environ.get("ADMM_RHO",".1"))
    factors=None
    last_rho=None
    debug=os.environ.get("SOLVER_DEBUG")

    def result_at(xx,yy):
        original_x=d*xx
        original_z=e*yy/cost
        s=b0-A0@original_x
        projected=s.copy()
        projected[eq]=0.
        projected[~eq]=np.maximum(projected[~eq],0.)
        original_z[~eq]=np.maximum(original_z[~eq],0.)
        if poly_candidate(P0,q0,A0,b0,cones,original_x,projected,original_z,tol=8e-9):
            return {"status":"solved","x":original_x,"s":projected,"z":original_z}
        return None

    def polish(xx,yy):
        slack=b-A@xx
        threshold=max(1e-7,1e-5*norm(yy))
        use=eq|((yy>threshold)&(slack<1e-2*(1+np.abs(b))))
        rows=np.flatnonzero(use)
        if len(rows)>n*1.5+100:
            return None
        try:
            E=A[rows].tocsc()
            fac=Factor(P,E,reg=1e-8)
            sol=fac.solve(np.r_[-q,b[rows]])
            zz=np.zeros(m)
            zz[rows]=sol[n:]
            return result_at(sol[:n],zz)
        except (ValueError,RuntimeError,np.linalg.LinAlgError,MemoryError):
            return None

    for iteration in range(20000):
        if last_rho!=rho:
            r=np.full(m,rho)
            r[eq]=rho*100.
            M=(P+sparse.diags(np.full(n,sigma))+A.T@(sparse.diags(r)@A)).tocsc()
            factors=sla.splu(M,permc_spec="MMD_AT_PLUS_A",diag_pivot_thresh=0.,
                              options={"SymmetricMode":True})
            last_rho=rho
        old=x
        xt=factors.solve(sigma*x-q+A.T@(r*z-y))
        ax=A@xt
        relaxed=1.6*ax-.6*z
        zn=np.minimum(np.maximum(relaxed+y/r,lower),b)
        y+=r*(relaxed-zn)
        z=zn
        x=1.6*xt-.6*old
        if iteration%25:
            continue
        Ax=A@x
        Px=P@x
        Ay=A.T@y
        rp=norm(Ax-z)/(1+max(norm(Ax),norm(z)))
        rd=norm(Px+q+Ay)/(1+max(norm(Px),norm(q),norm(Ay)))
        if debug:
            print("ADMM",iteration,"rp %.2e rd %.2e rho %.1e time %.3f" %
                  (rp,rd,rho,time.monotonic()-begin),file=sys.stderr,flush=True)
        candidate=result_at(x,y)
        if candidate:
            return candidate
        if iteration>=75 and iteration%100==0 and max(rp,rd)<5e-3:
            candidate=polish(x,y)
            if candidate:
                return candidate
        if iteration%200==0 and iteration:
            cert=certificate(P0,q0,A0,b0,layout,d*x,np.zeros(m),e*y/cost,tol=7e-9)
            if cert:
                return cert
        if time.monotonic()-begin>budget:
            break
        if iteration%50==0 and iteration:
            ratio=np.sqrt(max(rp,1e-12)/max(rd,1e-12))
            if ratio>5 or ratio<.2:
                rho=min(1e4,max(1e-5,rho*ratio))
    return None
