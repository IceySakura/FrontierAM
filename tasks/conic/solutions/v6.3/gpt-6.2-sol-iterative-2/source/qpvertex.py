"""Warm-started linear minimization for vertex-dominated QPs."""
from time import perf_counter

import numpy as np


def solve(p, sp, high, deadline, warm=None):
    from fastlp import Bounds, create, norm
    if p.n<70 or p.n+p.m>100000 or len(p.A.val)>400000 or not len(p.P.val):
        return None,warm
    q = p.q
    counts=np.bincount(p.A.idx,minlength=p.m)
    simple=counts==1
    a,lo,hi,e,b=p.A,None,None,p.eq,p.b
    B=None
    if int(simple.sum())>max(10,p.m//9):
        B1=Bounds(p,p.A,simple)
        if B1.ok and np.all(B1.lb<=B1.ub):
            B=B1
            a,lo,hi,e,b=B.M,B.lb,B.ub,p.eq[B.keep],p.b[B.keep]
    l=np.where(e,b,-np.inf)
    start=perf_counter()
    left=deadline-start
    if left<.024:
        return None,warm
    st=h=create(high,a,q,l,b,lo,hi,None,left,'choose')
    if h is None:
        return None,warm
    x=None
    if warm is not None and len(warm[0])==p.n:
        x=warm[0].copy()
    cols=np.arange(p.n,dtype=np.int32)
    spread=np.inf
    best=warm
    S=high.HighsModelStatus
    for k in range(10):
        left=deadline-perf_counter()
        if left<.009:
            break
        g=q if x is None else q+p.P.mv(x,sp)
        if k or x is not None:
            h.changeColsCost(p.n,cols,g)
        h.setOptionValue('time_limit',left+h.getRunTime())
        h.run()
        if h.getModelStatus()!=S.kOptimal:
            break
        sol=h.getSolution()
        if not sol.value_valid:
            break
        a0,z=np.asarray(sol.col_value),-np.asarray(sol.row_dual)
        if B is not None:
            z=B.restore(z,np.asarray(sol.col_dual),p.m)
        ans=p.optimal(a0,z,sp)
        if ans is None and x is not None:
            ans=p.optimal(x,z,sp)
        if ans is not None:
            return ans,(ans['x'],ans['z'])
        if x is None:
            x=a0
            continue
        d=a0-x
        diff=-float(g@d)
        mag=max(1.,abs(float(g@x)),abs(float(g@a0)))
        if diff/mag<spread:
            best=x.copy(),z
            spread=diff/mag
        if spread<2.e-5 and p.n<=300:
            from fastlp import dense_polish
            ans=dense_polish(p,x,z,sp)
            if ans is not None:
                return ans,best
        curv=float(d@p.P.mv(d,sp))
        if curv<=0. or diff<=5.e-14*mag:
            break
        alpha=min(1.,diff/curv)
        x=x+alpha*d
    return None,best
