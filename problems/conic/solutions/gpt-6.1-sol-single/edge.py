"""Zero-dimensional and unconstrained cases."""
import math

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sla

from cones import Layout
from ipm import certificate


def simple_case(P,q,A,b,cones):
    from solver import poly_candidate,norm
    n,m=len(q),len(b)
    layout=Layout(cones,m)
    if not np.any(q) and layout.violation(b)<=5e-9:
        return {"status":"solved","x":np.zeros(n),"s":b.copy(),"z":np.zeros(m)}
    if not m:
        diagonal=P.diagonal()
        off=P-sparse.diags(diagonal,format="csc")
        if not off.nnz:
            x=np.zeros(n)
            positive=diagonal>0
            x[positive]=-q[positive]/diagonal[positive]
            if poly_candidate(P,q,A,b,cones,x,b.copy(),b.copy()):
                return {"status":"solved","x":x,"s":b.copy(),"z":b.copy()}
            direction=np.zeros(n)
            direction[~positive]=-np.sign(q[~positive])
        else:
            x=sla.lsmr(P,-q,atol=1e-13,btol=1e-13,maxiter=min(4000,10*n+10))[0]
            if poly_candidate(P,q,A,b,cones,x,b.copy(),b.copy()):
                return {"status":"solved","x":x,"s":b.copy(),"z":b.copy()}
            direction=-(P@x+q)
        return certificate(P,q,A,b,layout,direction,np.empty(0),np.empty(0),tol=8e-9)
    if n:
        return None
    st=0
    for cone in cones:
        kind,d=cone["type"],cone["dim"]
        width=d*(d+1)//2 if kind=="psd" else d
        v=b[st:st+width]
        u=v/max(1.,norm(v))
        z=np.zeros(m)
        zz=None
        if kind=="zero":
            zz=-u
        elif kind=="nonnegative":
            zz=np.maximum(-u,0.)
        elif kind=="soc":
            a=-u
            vn=np.linalg.norm(a[1:])
            if vn<=a[0]:
                zz=a
            elif vn>abs(a[0]):
                zz=np.r_[(vn+a[0])*.5,a[1:]*(vn+a[0])/(2*vn)]
        elif kind=="psd":
            block=next(p for p in layout.psd if p.start==st)
            ev,U=linalg.eigh(block.mat(u),subset_by_index=[0,0],check_finite=False)
            if ev[0]<0:
                zz=block.vec(U@U.T)
        else:
            a,y,w=map(float,u)
            if y<0:
                zz=np.array([0.,1.,0.])
            elif w<0:
                zz=np.array([0.,0.,1.])
            else:
                if y>0 and w>0:
                    t=math.log(w)-math.log(y)
                elif y==0 and a>0:
                    t=math.log(max(w,1e-300))+math.log(4.)-math.log(a)
                elif y>0:
                    t=a/y
                else:
                    t=0.
                t=max(-600.,min(600.,t))
                zz=(np.array([-1.,t-1.,math.exp(-t)]) if t>=0 else
                    np.array([-math.exp(t),math.exp(t)*(t-1.),1.]))
        if zz is not None:
            z[st:st+width]=zz
            result=certificate(P,q,A,b,layout,np.empty(0),np.zeros(m),z,tol=8e-9)
            if result:
                return result
        st+=width
    return None
