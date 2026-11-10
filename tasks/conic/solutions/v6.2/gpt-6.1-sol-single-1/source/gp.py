"""Eliminate canonical exponential epigraphs into smooth log-sum-exp rows."""
import math
import os
import sys

import numpy as np
from scipy import linalg, sparse

from cones import Layout
from ipm import accepted, certificate, norm


class LogModel:
    def __init__(self, P, q, A, b, cones):
        if P.nnz:
            raise ValueError("nonlinear quadratic objective")
        self.P,self.q,self.A,self.b,self.cones=P,q,A,b,cones
        ar=A.tocsr()
        st=0
        ex,nn=[],[]
        for c in cones:
            if c["type"]=="exponential":
                ex.append(np.arange(st,st+3))
            elif c["type"]=="nonnegative":
                nn.extend(range(st,st+c["dim"]))
            else:
                raise ValueError("not an exponential epigraph model")
            st+=c["dim"]
        self.ex=np.asarray(ex,dtype=np.int32)
        self.nn=np.asarray(nn,dtype=np.int32)
        k=len(ex)
        if not k:
            raise ValueError("empty exponential model")
        ue,ve,we=self.ex.T
        if np.any(np.diff(ar.indptr)[ve]!=0) or np.any(b[ve]!=1):
            raise ValueError("nonconstant perspective coordinates")
        if np.any(np.diff(ar.indptr)[we]!=1) or np.any(b[we]!=0):
            raise ValueError("noncanonical epigraph coordinates")
        wpos=ar.indptr[we]
        self.wcol=ar.indices[wpos]
        self.h=-ar.data[wpos]
        if len(np.unique(self.wcol))!=k or np.any(self.h<=0) or np.any(q[self.wcol]!=0):
            raise ValueError("coupled epigraph variables")
        if A[ue][:,self.wcol].nnz:
            raise ValueError("recursive epigraph")
        mask=np.ones(len(q),dtype=bool)
        mask[self.wcol]=False
        self.keep=np.flatnonzero(mask)
        if len(self.keep)>1600:
            raise ValueError("dense reduced model too large")
        W=A[self.nn][:,self.wcol].tocsc()
        if np.any(np.diff(W.indptr)!=1) or np.any(W.data<=0):
            raise ValueError("epigraph must enter exactly one positive sum")
        wrow=W.indices
        unique,inverse=np.unique(wrow,return_inverse=True)
        self.group_rows=unique
        self.group=inverse
        self.weight=W.data/self.h
        L=A[self.nn][:,self.keep].tocsr()
        if L[unique].nnz or np.any(b[self.nn[unique]]<=0):
            raise ValueError("nonconstant sum bounds")
        self.bound=b[self.nn[unique]]
        self.logbound=np.log(self.bound)
        self.B=(-A[ue][:,self.keep]).tocsr()
        self.c=b[ue]
        self.C=self.c+np.log(self.weight)
        self.order=np.argsort(inverse,kind="stable")
        self.starts=np.r_[0,np.flatnonzero(np.diff(inverse[self.order]))+1]
        self.linear_rows=np.setdiff1d(np.arange(len(nn)),unique)
        self.L=L[self.linear_rows].tocsr()
        self.lb=b[self.nn[self.linear_rows]]
        self.lscale=np.maximum(1.0,np.maximum(np.abs(self.lb),
            np.asarray(abs(self.L).max(axis=1).toarray()).ravel())) if len(self.lb) else np.empty(0)
        self.L=sparse.diags(1/self.lscale)@self.L
        self.lb=self.lb/self.lscale
        self.Ld=self.L.toarray()
        self.qr=q[self.keep]
        self.ng=len(unique)
        self.n=len(self.keep)
        self.m=self.ng+len(self.lb)
        self.layout=Layout(cones,len(b))

    def evaluate(self,x,phase=False):
        main=x[:-1] if phase else x
        values=self.B@main+self.C
        vmax=np.maximum.reduceat(values[self.order],self.starts)
        expv=np.exp(values-vmax[self.group])
        sums=np.bincount(self.group,weights=expv,minlength=self.ng)
        probs=expv/sums[self.group]
        logs=vmax+np.log(sums)
        G=sparse.csr_matrix((probs,(self.group,np.arange(len(probs)))),
                             shape=(self.ng,len(probs)))
        Jg=(G@self.B).toarray()
        f=np.r_[logs-self.logbound,self.L@main-self.lb]
        J=np.vstack([Jg,self.Ld])
        if phase:
            f-=x[-1]
            J=np.column_stack([J,-np.ones(self.m)])
        return f,J,probs,logs,Jg

    def hessian(self,lam,probs,Jg,phase=False):
        ww=lam[:self.ng][self.group]*probs
        H=(self.B.T@(sparse.diags(ww)@self.B)).toarray()
        H-=(Jg.T*lam[:self.ng])@Jg
        H=(H+H.T)*0.5
        if phase:
            H=np.pad(H,((0,1),(0,1)))
        return H

    def restore(self,x,lam):
        f,J,prob,logs,Jg=self.evaluate(x)
        u=self.B@x+self.c
        exps=np.exp(np.clip(u,-690.0,690.0))
        fullx=np.zeros(len(self.q))
        fullx[self.keep]=x
        fullx[self.wcol]=exps/self.h
        z=np.zeros(len(self.b))
        zn=lam[:self.ng]*np.exp(np.clip(-logs,-690.,690.))
        z[self.nn[self.group_rows]]=zn
        z[self.nn[self.linear_rows]]=lam[self.ng:]/self.lscale
        zp=lam[:self.ng][self.group]*prob
        z[self.ex[:,0]]=-zp
        z[self.ex[:,1]]=zp*(u-1.0)
        z[self.ex[:,2]]=zn[self.group]*self.weight
        s=self.b-self.A@fullx
        return fullx,s,z

    def run(self,phase=False,start=None):
        n=self.n+int(phase)
        x=np.zeros(n)
        if start is not None:
            x[:self.n]=start[:self.n]
        if phase:
            f0=self.evaluate(x,phase=True)[0]
            x[-1]=max(1.,float(np.max(f0))+1.)
        q=np.r_[np.zeros(self.n),1.] if phase else self.qr
        f,J,prob,logs,Jg=self.evaluate(x,phase)
        s=np.maximum(1.,-f)
        lam=np.ones(self.m)
        best=None
        previous=np.inf
        stalled=0
        debug=os.environ.get("SOLVER_DEBUG")
        for it in range(100):
            f,J,prob,logs,Jg=self.evaluate(x,phase)
            rd=q+J.T@lam
            rp=f+s
            mu=(s@lam)/self.m
            measure=max(norm(rd),norm(rp),mu)
            stalled=stalled+1 if measure>=previous*(1.-1e-8) else 0
            previous=measure
            if debug:
                print("GP",int(phase),it,"rd %.2e rp %.2e mu %.2e obj %.9g" %
                    (norm(rd),norm(rp),mu,q@x),file=sys.stderr,flush=True)
            if (norm(rd)<1e-6 and norm(rp)<1e-6) or mu<1e-10:
                xx,ss,zz=self.restore(x[:self.n],lam)
                if phase:
                    cert=certificate(self.P,self.q,self.A,self.b,self.layout,
                                     xx,ss,zz,tol=6e-9)
                    if cert:
                        return cert,x[:self.n]
                    if x[-1]<-1e-5 and max(norm(rd),norm(rp))<1e-7:
                        return None,x[:self.n]
                elif accepted(self.P,self.q,self.A,self.b,self.layout,xx,ss,zz,tol=6e-9):
                    return {"status":"solved","x":xx,"s":ss,"z":zz},x
            if not np.isfinite(mu) or np.max(np.abs(x))>1e12:
                break
            if stalled>=6:
                break
            H=self.hessian(lam,prob,Jg,phase)
            D=lam/s
            augmented=mu<1e-4 or np.max(D)>1e8
            M=H+(J.T*D)@J if not augmented else np.block(
                [[H,J.T],[J,-np.diag(s/lam)]])
            reg=1e-12 if augmented else max(1e-12,1e-13*norm(M.diagonal()))
            fac=None
            for k in range(7):
                try:
                    Mr=M.copy()
                    if augmented:
                        Mr[np.arange(n),np.arange(n)]+=reg
                        Mr[n+np.arange(self.m),n+np.arange(self.m)]-=reg
                        fac=linalg.lu_factor(Mr,check_finite=False)
                    else:
                        Mr.flat[::n+1]+=reg
                        fac=linalg.cho_factor(Mr,check_finite=False)
                    break
                except np.linalg.LinAlgError:
                    reg*=10
            if fac is None:
                break

            def direction(rc):
                if augmented:
                    rr=np.r_[-rd,-rp-rc/lam]
                    out=linalg.lu_solve(fac,rr,check_finite=False)
                    for k in range(3):
                        res=rr-M@out
                        if norm(res)<1e-13*(1+norm(rr)):
                            break
                        out+=linalg.lu_solve(fac,res,check_finite=False)
                    dx,dl=out[:n],out[n:]
                    ds=-rp-J@dx
                    return dx,ds,dl
                rr=-rd-J.T@((rc+lam*rp)/s)
                dx=linalg.cho_solve(fac,rr,check_finite=False)
                for k in range(2):
                    residual=rr-M@dx
                    if norm(residual)<1e-12*(1+norm(rr)):
                        break
                    dx+=linalg.cho_solve(fac,residual,check_finite=False)
                ds=-rp-J@dx
                dl=(rc-lam*ds)/s
                return dx,ds,dl

            def step(dr,fraction):
                alpha=1.
                for v,dv in ((s,dr[1]),(lam,dr[2])):
                    mask=dv<0
                    if np.any(mask):
                        alpha=min(alpha,float(np.min(-v[mask]/dv[mask])))
                return min(1.,fraction*alpha)

            da=direction(-s*lam)
            aa=step(da,1.)
            mua=(s+aa*da[1])@(lam+aa*da[2])/self.m
            sigma=min(1.,max(0.,(mua/mu)**3))
            dr=direction(sigma*mu-s*lam-da[1]*da[2])
            alpha=step(dr,.995)
            rpscale=1.+norm(f)
            rdscale=1.+norm(q)
            objscale=1.+abs(q@x)
            merit=norm(rp)/rpscale+norm(rd)/rdscale+mu/objscale
            for ls in range(35):
                xn=x+alpha*dr[0]
                sn=s+alpha*dr[1]
                ln=lam+alpha*dr[2]
                fn,Jn,*_=self.evaluate(xn,phase)
                mun=sn@ln/self.m
                newmerit=(norm(fn+sn)/rpscale+norm(q+Jn.T@ln)/rdscale
                          +mun/objscale)
                if np.isfinite(newmerit) and newmerit<merit*(1.-.001*alpha):
                    break
                alpha*=.65
            if alpha<1e-12:
                break
            x,s,lam=xn,sn,ln
        return None,x[:self.n]

    def recession(self):
        from solver import highs_model,highs_core
        hc=highs_core()
        F=sparse.vstack([self.B,self.L],format="csc")
        h=highs_model(None,self.qr,F,np.full(F.shape[0],-hc.kHighsInf),
            np.zeros(F.shape[0]),-np.ones(self.n),np.ones(self.n),time_limit=5.0)
        if h.getModelStatus()!=hc.HighsModelStatus.kOptimal:
            return None
        d=np.asarray(h.getSolution().col_value)
        if self.qr@d>=-1e-8:
            return None
        x=np.zeros(len(self.q))
        x[self.keep]=d
        return certificate(self.P,self.q,self.A,self.b,self.layout,
                           x,-self.A@x,np.zeros(len(self.b)),tol=6e-9)

    def cut_certificate(self,start):
        from solver import highs_model,highs_core
        hc=highs_core()
        point=np.clip(start,-40.,40.) if np.isfinite(start).all() else np.zeros(self.n)
        points=[]
        rows=[]
        bounds=[]
        origins=[]
        for iteration in range(35):
            f,J,prob,logs,Jg=self.evaluate(point)
            pi=len(points)
            points.append(point.copy())
            selected=list(range(self.ng))
            if iteration==0:
                selected+=list(range(self.ng,self.m))
            for row in selected:
                if row<self.ng:
                    if logs[row]<-600:
                        continue
                    value=-np.expm1(np.clip(-f[row],-690.,690.))
                else:
                    value=f[row]
                rhs=J[row]@point-value
                scale=max(1.,abs(rhs),norm(J[row]))
                rows.append(J[row]/scale)
                bounds.append(rhs/scale)
                origins.append((pi,row,scale))
            mat=sparse.csc_matrix(np.column_stack([np.asarray(rows),-np.ones(len(rows))]))
            radius=40.*(2.**min(5,iteration//6))
            low=np.r_[np.full(self.n,-radius),-hc.kHighsInf]
            high=np.r_[np.full(self.n,radius),hc.kHighsInf]
            h=highs_model(None,np.r_[np.zeros(self.n),1.],mat,
                np.full(len(rows),-hc.kHighsInf),np.asarray(bounds),low,high,time_limit=3.)
            if h.getModelStatus()!=hc.HighsModelStatus.kOptimal:
                break
            sol=h.getSolution()
            val=np.asarray(sol.col_value)
            if os.environ.get("SOLVER_DEBUG"):
                print("GP cuts",iteration,len(rows),"xi",val[-1],file=sys.stderr,flush=True)
            if val[-1]>1e-9:
                weights=[np.zeros(self.m) for p in points]
                for lam,(p,r,scale) in zip(-np.asarray(sol.row_dual),origins):
                    weights[p][r]+=max(0.,lam)/scale
                z=np.zeros(len(self.b))
                for p,ww in zip(points,weights):
                    if np.any(ww):
                        z+=self.restore(p,ww)[2]
                cert=certificate(self.P,self.q,self.A,self.b,self.layout,
                    np.zeros(len(self.q)),np.zeros(len(self.b)),z,tol=6e-9)
                if cert:
                    return cert,point
            point=val[:self.n]
            ff=self.evaluate(point)[0]
            if np.max(ff)<-1e-5:
                return None,point
        return None,point


def solve_gp(P,q,A,b,cones):
    try:
        model=LogModel(P,q,A,b,cones)
    except ValueError:
        return None
    if os.environ.get("SOLVER_DEBUG"):
        print("log reduction",A.shape,"->",model.n,model.m,
              file=sys.stderr,flush=True)
    result,x=model.run()
    if result is not None:
        return result
    result=model.recession()
    if result is not None:
        return result
    result,warm=model.cut_certificate(x)
    if result is not None:
        return result
    if np.max(model.evaluate(warm)[0])<-1e-5:
        result,x=model.run(start=warm)
        if result is not None:
            return result
    result,start=model.run(phase=True,start=x)
    if result is not None:
        return result
    result,x=model.run(start=start)
    return result
