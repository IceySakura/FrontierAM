"""Proximal splitting fallback, with local acceleration and certificates."""
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import eigh, lapack
from scipy.sparse.linalg import splu

from model import Scaled, maxabs, smat, svec
import native


class Projection:
    def __init__(self,K):
        self.K = K
        self.indices = {k:starts[:,None]+np.arange(k[1],dtype=np.int32)
                        for k,starts in K.groups.items()}
        self.psd = [(d,a,b) for k,d,a,b in K.nonlinear if k=='psd']

    def cone(self,x):
        K = self.K
        s = x.copy()
        s[K.eq] = 0.
        s[K.lpos] = np.maximum(s[K.lpos],0.)
        for (k,d), ix in self.indices.items():
            q = s[ix]
            if k=='soc':
                r = np.sqrt(np.einsum('ij,ij->i',q[:,1:],q[:,1:]))
                mask = q[:,0] < r
                t = np.maximum(0.,.5*(q[mask,0]+r[mask]))
                denom = np.maximum(r[mask],1.e-100)
                q[mask,0] = t
                q[mask,1:] *= (t/denom)[:,None]
                s[ix] = q
            else:
                s[ix] = native.project_exp(np.ascontiguousarray(q))
        for d,a,b in self.psd:
            X = smat(x[a:b],d)
            lam,U = eigh(X,check_finite=False)
            if lam[0] < 0.:
                ind = lam>0.
                s[a:b] = svec((U[:,ind]*lam[ind])@U[:,ind].T)
        return s


class Linear:
    def __init__(self,A,P,eq,rho,deadline):
        self.A,self.At,self.P = A,A.T,P
        self.eq = eq
        self.d = A.shape[1]
        self.rho = rho
        lens = np.diff(A.tocsr().indptr).astype(np.int64)
        self.direct = (
            self.d <= 1050 or
            (self.d < 48000 and int(lens@lens) < 2_500_000 and P.nnz+A.nnz < 1_100_000))
        self.sigma = 5.e-7 if self.direct else 3.e-5
        self.Acols = np.repeat(np.arange(self.d,dtype=np.int32),np.diff(A.indptr))
        self.setup(rho)

    def setup(self,rho):
        self.rho = rho
        A,P = self.A,self.P
        R = np.full(A.shape[0],rho)
        R[self.eq] *= 120.
        self.R = R
        dd = self.sigma+P.diagonal()+np.bincount(self.Acols,weights=A.data*A.data*R[A.indices],
                                                minlength=self.d)
        self.diagonal = 1./np.maximum(dd,self.sigma)
        self.inv = None
        if not self.direct:
            return
        As = A.copy()
        As.data *= R[A.indices]
        M = (A.T@As+P).tocsc()
        M.setdiag(M.diagonal()+self.sigma)
        try:
            if self.d < 115 or (self.d < 1500 and M.nnz > .17*self.d*self.d):
                mat, info = lapack.dpotrf(M.toarray(order='F'),lower=1,overwrite_a=1)
                if info==0:
                    self.inv = lambda v: lapack.dpotrs(mat,v,lower=1)[0]
            else:
                fac = splu(M,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,
                           options={'SymmetricMode':True})
                self.inv = fac.solve
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            pass

    def mv(self,x):
        return self.P@x+self.sigma*x+self.At@(self.R*(self.A@x))

    def step(self,x,q,v,g,deadline):
        r = self.sigma*x-q+self.At@(self.R*v)
        if self.inv is not None:
            return self.inv(r)
        y = x.copy()
        z = r-self.mv(y)
        tol = max(5.e-11,min(4.e-4,g*3.e-3))*(1.+maxabs(r))
        h = self.diagonal*z
        k = float(z@h)
        ap = h.copy()
        for i in range(44 if self.d<70000 else 20):
            if k<=0. or not np.isfinite(k) or maxabs(z)<tol:
                break
            mul = self.mv(ap)
            den = float(ap@mul)
            if not den>0.:
                break
            a = k/den
            y += a*ap
            z -= a*mul
            h = self.diagonal*z
            nk = float(z@h)
            ap *= nk/k
            ap += h
            k = nk
            if i%8==7 and perf_counter()>deadline:
                break
        return y


def solve_split(p,deadline,point=None,maxiter=1700,verbose=0):
    if p.n==0 or p.m==0 or (p.K.exp and native.lib is None):
        return None,None
    if any(k=='psd' and d>145 for k,d,_,_ in p.K.nonlinear):
        return None,None
    w = Scaled(p,passes=6 if p.n+p.m<300000 else 4,scale_rhs=True)
    A,P,q,b,K = w.A,w.P,w.q,w.b,w.K
    n,m = p.n,p.m
    rho = .3
    l = Linear(A,P,K.eq,rho,deadline)
    proj = Projection(K)
    x = np.zeros(n)
    v = b-proj.cone(b)
    if point is not None:
        x0,s0,z0 = point
        x = x0/w.D
        v = b-s0*w.E+(z0/w.E*w.c)/l.R
    weight = .08
    dot0 = 10.
    mixx, mixv, oldsz = [],[],None
    pre_g = 0.
    restore = None
    best = np.inf
    best_loc = None
    cert_x = x.copy()
    cert_z = np.zeros(m)
    polishes = 0
    last_adapt = -200
    depth = 5 if n+m < 40_000 else (3 if n+m<200_000 else 0)
    freq = 8 if n+m < 10000 else 5
    At = A.T
    for k in range(maxiter):
        if k%3==0 and perf_counter() > deadline:
            break
        s = proj.cone(b-v)
        u = b-s
        d = v-u
        y = l.R*d
        nx = l.step(x,q,u-d,dot0,deadline)
        au = A@nx
        dr = au-u
        gx,gv = (nx-x)*1.55,dr*1.55
        dot0 = max(maxabs(gx),maxabs(gv))
        if not np.isfinite(dot0) or dot0>1.e15:
            break
        if restore is not None:
            if dot0>1.6*pre_g+1.e-11:
                x,v = restore
                restore = None
                oldsz = None
                mixx.clear(); mixv.clear()
                continue
            restore = None
        if k%freq==0:
            px, aty = P@x,At@y
            rn = maxabs(A@x+s-b)/(1.+max(maxabs(u),maxabs(s),maxabs(b)))
            dn = maxabs(px+q+aty)/(1.+max(maxabs(px),maxabs(q),maxabs(aty)))
            xx, qx, bz = float(x@px),float(q@x),float(b@y)
            gp = abs(xx+qx+bz)/max(w.c,abs(.5*xx+qx),abs(.5*xx+bz))
            measure = max(rn,dn,gp)
            if max(rn,dn)<=2.e-4 and measure < max(.1,best):
                x0,s0,z0 = w.D*x,s/w.E,w.E/w.c*y
                r = p.optimal(x0,z0,s0,polish_slack=False)
                if r is not None:
                    return r,(x0,s0,z0)
                if measure < best:
                    best,best_loc = measure,(x0,s0,z0)
                if (polishes<3 and perf_counter()<deadline-.02 and k>=15
                        and (k>=100*polishes or max(rn,dn)<3.e-6)
                        and n+m<28000):
                    polishes += 1
                    if not K.nonlinear:
                        from poly import polish_polyhedral
                        r=polish_polyhedral(p,x0,z0,min(deadline,perf_counter()+.18))
                    elif not K.psd:
                        from polish import polish_cone
                        r=polish_cone(p,(x0,s0,z0),min(deadline,perf_counter()+.18))
                    else:
                        r=None
                    if r is not None:
                        return r,(x0,s0,z0)
            if verbose and k%max(32,freq)==0:
                print(" split %4d %.2e %.2e %.2e rho %.3g"%(k,rn,dn,gp,rho),flush=True)
            if k>=30 and (k//freq)%4==0:
                cy = w.E*(y-cert_z)
                cx = w.D*(x-cert_x)
                if maxabs(cy)>1.e-8:
                    r=p.primal_cert(cy)
                    if r is not None:
                        return r,best_loc
                if maxabs(cx)>1.e-8:
                    r=p.dual_cert(cx)
                    if r is not None:
                        return r,best_loc
                cert_x,cert_z = x.copy(),y.copy()
            if 24<=k<600 and k-last_adapt>45 and dn>0. and rn>0.:
                change = np.clip((rn/dn)**.55,.15,7.)
                new = np.clip(rho*change,2.e-4,2.e3)
                if (new/rho>4. or new/rho<.25) and perf_counter()<deadline-.2:
                    v = u+(rho/new)*d
                    rho = new
                    l.setup(rho)
                    oldsz = None
                    mixx.clear(); mixv.clear()
                    last_adapt = k
                    continue
        if depth and k>5:
            if oldsz is not None:
                a,b0,g0,g1 = oldsz
                mixx.append(np.r_[weight*(x-a),v-b0])
                mixv.append(np.r_[weight*(gx-g0),gv-g1])
                if len(mixx)>depth:
                    mixx.pop(0); mixv.pop(0)
            oldsz = x,v,gx,gv
            if len(mixv)>=2 and k%3==0:
                F=np.asarray(mixv)
                G=F@F.T
                tr=float(np.trace(G))
                G[np.diag_indices(len(G))] += 1.e-8*tr+1.e-29
                try:
                    gamma=np.linalg.solve(G,F@np.r_[weight*gx,gv])
                except np.linalg.LinAlgError:
                    gamma=None
                if gamma is not None and maxabs(gamma)<14.:
                    u0,u1 = x+gx,v+gv
                    dx = gamma@(np.asarray(mixx)+F)
                    if maxabs(dx)<10.*(1.+dot0):
                        x,v = u0-dx[:n]/weight,u1-dx[n:]
                        restore = (u0,u1)
                        pre_g = dot0
                        continue
        x,v = x+gx,v+gv
    return None,best_loc
