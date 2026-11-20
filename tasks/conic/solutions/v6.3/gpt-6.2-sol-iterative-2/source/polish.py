"""Active-manifold Newton refinement for the orthant, SOC and exp cones."""
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import lapack
from scipy.sparse.linalg import splu

from model import Scaled, maxabs


def polish_sdp(p,point,deadline):
    if not p.K.psd or p.n>2200 or p.m>145000 or p.n==0:
        return None
    x,s,z = point
    t = p.P@x
    gap_value = float(x@(t+p.q))
    # Intersect the dual cone with stationarity and the objective hyperplane.
    F = sparse.vstack((p.A.T,sparse.csr_matrix(p.b.reshape(1,-1))),format='csr')
    norm = np.sqrt(np.asarray(F.multiply(F).sum(axis=1)).ravel())
    scale = 1./np.maximum(norm,1.e-14)
    F.data *= np.repeat(scale,np.diff(F.indptr))
    target = -np.r_[t+p.q,gap_value]*scale
    M = (F@F.T).tocsc()
    if p.n < 510 or M.nnz > .20*p.n*p.n:
        from scipy.linalg import eigh
        e,U = eigh(M.toarray(),check_finite=False)
        use = e > 2.e-13*e[-1]
        U = U[:,use]
        R = U/e[use]
        inverse = lambda v: R@(U.T@v)
    else:
        try:
            fact = splu(M+sparse.eye(M.shape[0],format='csc')*1.e-12,
                        permc_spec='MMD_AT_PLUS_A')
        except RuntimeError:
            return None
        inverse = fact.solve
    v = z
    correction = np.zeros_like(z)
    for k in range(18):
        if perf_counter() > deadline:
            return None
        u = v+F.T@inverse(target-F@v)
        r = p.optimal(x,u,s,polish_slack=False)
        if r is not None:
            return r
        v = p.K.project_roundoff(u+correction,True)
        correction += u-v
        if k%3==0:
            r = p.optimal(x,v,s,polish_slack=False)
            if r is not None:
                return r
    return None


class Rows:
    def __init__(self,m):
        self.i,self.j,self.v = [],[],[]
        self.m = m
        self.count = 0

    def add(self, j, v, dense=False):
        j = np.asarray(j,dtype=np.int32).reshape(-1)
        if dense:
            v = np.asarray(v)
            k = len(v)
            self.i.append(np.repeat(np.arange(self.count,self.count+k,dtype=np.int32),len(j)))
            self.j.append(np.tile(j,k))
            self.v.append(v.reshape(-1))
        else:
            k = len(j)
            self.i.append(np.arange(self.count,self.count+k,dtype=np.int32))
            self.j.append(j)
            self.v.append(np.broadcast_to(v,(k,)))
        self.count += k

    def csc(self):
        if not self.count:
            return sparse.csr_matrix((0,self.m))
        return sparse.csr_matrix((np.concatenate(self.v),(np.concatenate(self.i),
                   np.concatenate(self.j))),shape=(self.count,self.m))


def polish_cone(p,point, deadline, *, steps=4):
    if p.K.psd or not p.K.nonlinear or not p.n or p.n+p.m>25000 or p.A.nnz>130000:
        return None
    x0,s0,z0 = point
    w = Scaled(p,passes=5,scale_rhs=True)
    K, A, P, b, q = w.K, w.A, w.P, w.b, w.q
    n,m = p.n,p.m
    x = x0/w.D
    z = z0*w.c/w.E
    sref = s0*w.E
    s = b-A@x
    apre = np.diff(A.tocsr().indptr)
    peq = K.eq
    los = K.lpos
    smag = max(1.,maxabs(s))
    zmag = max(1.,maxabs(z))
    bound = 5.e-6*smag
    lsh = (s[los] < bound) & ((sref[los] < bound) | (z[los] > 1.e-6*zmag))
    mask = np.r_[peq,los[lsh]]
    modes = []
    guess = [z[mask]]
    for kind,d,a,e in K.nonlinear:
        ss, zz = s[a:e], z[a:e]
        sn, zn = maxabs(ss), maxabs(zz)
        sv = maxabs(sref[a:e])
        scale = max(1.,sn,sv)
        r = np.arange(a,e,dtype=np.int32)
        if max(sn,sv) < 2.e-6*smag and zn > 2.e-7*zmag:
            modes.append(('flat',r))
            guess.append(zz)
        elif kind == 'soc':
            norm = np.linalg.norm(ss[1:])
            if (ss[0]-norm) > 2.e-5*scale and zn < 1.e-4*zmag:
                continue
            if norm <= 2.e-8*scale:
                if zn > 1.e-7*zmag or ss[0] < 0.:
                    modes.append(('flat',r))
                    guess.append(zz)
                continue
            if d*d > 400_000 or int(apre[r].sum()) > 2500:
                return None
            modes.append(('soc',r))
            guess.append(np.array([max(0.,zz[0])]))
        else:
            y,z3 = ss[1:]
            if min(y,z3) <= 1.e-9*scale or (y <= 1.e-6*scale and ss[0] < -1.e-5*scale):
                r = r[[1]] if z3 > 2.e-6*scale else r[[1,2]]
                modes.append(('flat',r))
                guess.append(z[r])
                continue
            interior = y*(np.log(z3)-np.log(y))-ss[0]
            if interior > 2.e-5*scale and zn < 1.e-4*zmag:
                continue
            modes.append(('exp',r))
            guess.append(np.array([max(0.,-zz[0])]))

    lam = np.concatenate(guess)

    def make(xx, ll, curvature):
        s = b-A@xx
        T = Rows(m)
        C = Rows(m)
        h = [s[mask]]
        T.add(mask,1.)
        i = len(mask)
        for mode,j in modes:
            ss = s[j]
            if mode == 'flat':
                T.add(j,1.)
                h.append(ss)
                i += len(j)
                continue
            if mode == 'soc':
                r = np.linalg.norm(ss[1:])
                if not r > 1.e-16:
                    return None
                v = ss[1:]/r
                T.add(j,np.r_[1.,-v].reshape(1,-1),True)
                h.append(np.array([ss[0]-r]))
                if curvature and ll[i] > 0.:
                    C.add(j[1:],np.sqrt(ll[i]/r)*(np.eye(len(v))-np.outer(v,v)),True)
            else:
                y,zv = ss[1:]
                if min(y,zv) <= 1.e-16:
                    return None
                L = np.log(zv)-np.log(y)
                T.add(j,np.array([[-1.,L-1.,y/zv]]),True)
                h.append(np.array([y*L-ss[0]]))
                if curvature and ll[i] > 0.:
                    C.add(j[1:],np.sqrt(ll[i]/y)*np.array([[1.,-y/zv]]),True)
            i += 1
        return T.csc(),C.csc(),np.concatenate(h),s

    for it in range(steps+1):
        if perf_counter() > deadline:
            return None
        ret = make(x,lam,it<steps)
        if ret is None:
            return None
        T,C,g,s = ret
        z = T.T@lam
        ans = p.optimal(x*w.D,z*w.E/w.c,polish_slack=(it==steps))
        if ans is not None:
            return ans
        if it == steps:
            break
        J = (T@A).tocsc()
        k = J.shape[0]
        D = (C@A).tocsc()
        if D.nnz > 350_000:
            return None
        lens = np.diff(D.tocsr().indptr).astype(np.int64)
        if int(lens@lens) > 35_000_000:
            return None
        H = P+(D.T@D)
        r = -(P@x+q+A.T@z)
        mat = sparse.bmat([[H,J.T],[J,None]],format='csc')
        res = np.r_[r,g]
        base = np.r_[x,lam]
        mx = np.maximum.reduceat(
            np.r_[np.abs(mat.data),0.],
            np.minimum(mat.indptr[:-1],mat.nnz))
        sc = 1./np.sqrt(np.maximum(mx,1.e-7))
        sc = np.minimum(sc,1.e3)
        eq = mat.multiply(sc[:,None]).multiply(sc[None,:]).tocsc()
        v = eq+sparse.diags(np.r_[np.full(n,1.e-11),np.full(k,-1.e-11)],format='csc')
        try:
            if len(res) <= 115:
                F,S,info = lapack.dsytrf(v.toarray(order='F'),lower=1,overwrite_a=1)
                if info:
                    return None
                solve = lambda r: lapack.dsytrs(F,S,r,lower=1)[0]
            else:
                lu = splu(v,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=.05)
                solve = lu.solve
            res *= sc
            d = solve(res)
            err = res-eq@d
            if maxabs(err) > 1.e-12:
                du = solve(err)
                if maxabs(res-eq@(d+du)) < maxabs(err):
                    d += du
            d *= sc
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            return None
        if not np.isfinite(d).all() or maxabs(d) > 200.*(1.+maxabs(base)):
            return None
        xn = x+d[:n]
        sn = b-A@xn
        alpha = 1.
        for mode,j in modes:
            if mode=='exp':
                for idx in j[1:]:
                    if sn[idx] < .05*s[idx]:
                        alpha = min(alpha,.95*s[idx]/(s[idx]-sn[idx]))
        x += alpha*d[:n]
        lam += alpha*d[n:]
    return None
