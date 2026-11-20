"""Proximal augmented-Lagrangian QP fallback."""
from time import perf_counter

import numpy as np
from scipy.linalg import lapack
from scipy.sparse.linalg import splu

from model import Scaled, maxabs


def solve_alm(p,deadline,point=None,*,maxiter=36,verbose=0):
    n,m = p.n,p.m
    if not p.P.nnz or p.K.nonlinear or not n or n+m>110_000 or p.A.nnz>550000:
        return None
    w=Scaled(p,passes=6,scale_rhs=True)
    A=w.A.tocsr()
    P,q,b,K=w.P,w.q,w.b,w.K
    At=A.T
    costs = np.diff(A.indptr).astype(np.int64)
    direct = (n<=950 or int(costs@costs)<3_000_000)
    mask=np.zeros(m,dtype=bool)
    mask[K.eq]=True
    x=np.zeros(n)
    z=np.zeros(m)
    if point is not None:
        xx,zz = (point[0],point[-1])
        x = xx/w.D
        z = zz*w.c/w.E
        z[~mask] = np.maximum(0.,z[~mask])
    rho=np.full(m,.45)
    rho[mask]*=8.
    sig=8.e-6
    rp_prev=np.full(m,np.inf)
    prepared=None
    polishes=0
    last_polish=np.inf
    dz0=z
    for it in range(maxiter):
        if perf_counter()>deadline or not np.isfinite(x).all():
            break
        x0=x.copy()
        center=z.copy()
        res=A@x-b
        Ax=res+b
        Px=P@x
        base=Px+q
        inner_eps=max(2.e-11,min(2.e-2,.001*.13**min(it,9)))
        for jt in range(10):
            if perf_counter()>deadline:
                return None
            v=center+rho*res
            act=mask | (v>0.)
            y=np.where(act,v,0.)
            grad=base+At@y+sig*(x-x0)
            gnorm=maxabs(grad)
            if gnorm<inner_eps*(1.+max(maxabs(Px),maxabs(q),maxabs(At@y))):
                break
            if not np.isfinite(gnorm):
                return None
            S=A[act,:].tocsr()
            R=rho[act]
            f=None
            if direct:
                if prepared is not None and np.array_equal(act,prepared[0]):
                    f=prepared[1]
                else:
                    H=(P + S.T@S.multiply(R[:,None])).tocsc()
                    H.setdiag(H.diagonal()+sig)
                    try:
                        if n<100 or (n<1400 and H.nnz>.12*n*n):
                            L,flag=lapack.dpotrf(H.toarray(order='F'),lower=1,overwrite_a=1)
                            if flag==0:
                                f=lambda v,L=L:lapack.dpotrs(L,v,lower=1)[0]
                        else:
                            L=splu(H,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,
                                   options={'SymmetricMode':True})
                            f=L.solve
                    except (RuntimeError,ValueError,np.linalg.LinAlgError):
                        pass
                    if f is not None:
                        prepared=(act.copy(),f)
            if f is not None:
                dx=f(-grad)
            else:
                diag=P.diagonal()+sig+np.asarray(S.multiply(S).T@R).ravel()
                di=1./np.maximum(sig,diag)
                r=-grad
                dx=np.zeros(n)
                c=di*r
                rr=float(r@c)
                for j in range(32):
                    h=P@c+S.T@(R*(S@c))+sig*c
                    den=float(c@h)
                    if not den>0. or not rr>0.:
                        break
                    a=rr/den
                    dx+=a*c
                    r-=a*h
                    if maxabs(r)<.03*gnorm or perf_counter()>deadline:
                        break
                    u=di*r
                    nr=float(u@r)
                    c*=nr/rr
                    c+=u
                    rr=nr
            ad,pd=A@dx,P@dx
            v=ad*rho
            dd=float(dx@pd)+sig*float(dx@dx)
            w0=float(dx@(base+sig*(x-x0)))

            def slope(a):
                u=center+rho*res+a*v
                u[~mask]=np.maximum(0.,u[~mask])
                return w0+a*dd+float(u@ad)

            alpha=1.
            if slope(1.)>0.:
                lo,hi=0.,1.
                fl,fh=float(grad@dx),slope(1.)
                if not fl<0.:
                    break
                for k in range(15):
                    mid=np.clip((lo*fh-hi*fl)/(fh-fl),lo+.09*(hi-lo),hi-.09*(hi-lo))
                    fm=slope(mid)
                    if fm>0.:
                        hi,fh=mid,fm
                    else:
                        lo,fl=mid,fm
                    if abs(fm)<1.e-5*max(1.e-12,-fl) or hi-lo<5.e-5:
                        break
                alpha=mid
            x+=alpha*dx
            res+=alpha*ad
            base+=alpha*pd
            Px+=alpha*pd
        z=center+rho*res
        z[~mask]=np.maximum(0.,z[~mask])
        ax=A@x
        rp=ax-b
        rv=np.where(mask | (z>1.e-9),np.abs(rp),np.maximum(0.,rp))
        rn=maxabs(rv)/(1.+max(maxabs(ax),maxabs(b)))
        az=At@z
        dn=maxabs(P@x+q+az)/(1.+max(maxabs(az),maxabs(Px),maxabs(q)))
        if verbose and it%5==0:
            print(' alm',it,'%.3e %.3e'%(rn,dn),flush=True)
        if max(rn,dn)<.0003:
            x1,z1=w.D*x,w.E/w.c*z
            ans=p.optimal(x1,z1,polish_slack=False)
            if ans is not None:
                return ans
            crit=max(rn,dn)
            if polishes<2 and crit<.1*last_polish and perf_counter()<deadline-.03:
                from poly import polish_polyhedral
                polishes+=1
                last_polish=crit
                ans=polish_polyhedral(p,x1,z1,min(deadline,perf_counter()+.18))
                if ans is not None:
                    return ans
        if it%6==5:
            zz=w.E*(z-dz0)
            ans=p.primal_cert(zz)
            if ans is None:
                ans=p.dual_cert(w.D*(x-x0))
            if ans is not None:
                return ans
            dz0=z.copy()
        if it>=1:
            hard=(rv>np.maximum(.35*rp_prev,5.e-12)) & (rv>.06*maxabs(rv))
            if np.any(hard):
                inc=np.minimum(12.,np.maximum(2.,5.*rv[hard]/max(1.e-20,maxabs(rv))))
                rho[hard]=np.minimum(rho[hard]*inc,1.e9)
                prepared=None
        rp_prev=rv
        if rn>1.e8 or maxabs(z)>1.e14:
            break
    return None
