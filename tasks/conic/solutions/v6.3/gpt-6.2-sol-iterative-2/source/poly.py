"""Bound-constrained fast paths and active-set polishing."""
from time import perf_counter

import numpy as np
from scipy import sparse

from model import Scaled, maxabs


class Box:
    def __init__(self, p, rowcount):
        A, b, n = p.A, p.b, p.n
        eq = np.zeros(p.m,dtype=bool)
        eq[p.K.eq] = True
        zero = rowcount == 0
        bad = np.flatnonzero(zero & ((b < 0.) | (eq & (b != 0.))))
        self.certificate = None
        if bad.size:
            z = np.zeros(p.m)
            k = bad[np.argmax(np.abs(b[bad]))]
            z[k] = -np.sign(b[k])
            self.certificate = p.primal_cert(z)
            return
        self.col = col = np.repeat(np.arange(n, dtype=np.int32), np.diff(A.indptr))
        self.row = row = A.indices
        self.a = a = A.data
        val = b[row]/a
        self.lo, self.hi = lo, hi = np.full(n,-np.inf), np.full(n,np.inf)
        low = (a < 0.) | eq[row]
        high = (a > 0.) | eq[row]
        np.maximum.at(lo, col[low], val[low])
        np.minimum.at(hi, col[high], val[high])
        # Among tied bounds prefer a large pivot.
        li = np.where(low & (val == lo[col]))[0]
        ui = np.where(high & (val == hi[col]))[0]
        self.li = self._choose(li,n)
        self.ui = self._choose(ui,n)
        if np.any(lo > hi):
            j = np.argmax(np.where(lo > hi, lo-hi, -np.inf))
            l, u = self.li[j], self.ui[j]
            zl, zu = a[l], a[u]
            mn = min(abs(zl), abs(zu))
            z = np.zeros(p.m)
            z[row[l]], z[row[u]] = -mn/zl, mn/zu
            self.certificate = p.primal_cert(z)

    def _choose(self,i,n):
        res = np.full(n,-1,dtype=np.int32)
        score = np.zeros(n)
        np.maximum.at(score,self.col[i],np.abs(self.a[i]))
        i = i[np.abs(self.a[i]) == score[self.col[i]]]
        res[self.col[i]] = i
        return res

    def verify(self,p,x,g=None):
        if g is None:
            g = p.P@x+p.q
        z = np.zeros(p.m)
        for act, ids in [((x == self.lo) & (g > 0.), self.li),
                         ((x == self.hi) & (g < 0.), self.ui)]:
            sel = ids[act]
            if np.any(sel < 0):
                return None
            z[self.row[sel]] = -g[act]/self.a[sel]
        return p.optimal(x,z,polish_slack=False)


def solve_boxes(p, deadline):
    if p.K.nonlinear or not p.n:
        return None
    A, P = p.A, p.P
    if A.nnz > p.m:
        return None
    rows = np.bincount(A.indices,minlength=p.m)
    if np.any(rows > 1):
        return None
    B = Box(p,rows)
    if B.certificate is not None:
        return B.certificate
    if not hasattr(B,'lo') or np.any(B.lo > B.hi):
        return None
    lo, hi = B.lo, B.hi
    diag = P.diagonal()
    if np.count_nonzero(diag) == P.nnz and np.all(diag >= 0.):
        q = p.q
        direction = ((diag == 0.) & (q > 0.) & ~np.isfinite(lo))
        unbounded = ((diag == 0.) & (q < 0.) & ~np.isfinite(hi)).astype(float) - direction
        if np.any(unbounded):
            return p.dual_cert(unbounded)
        x = np.zeros(p.n)
        np.divide(-q, diag, out=x, where=diag>0.)
        x = np.where((diag==0.) & (q>0.),lo,x)
        x = np.where((diag==0.) & (q<0.),hi,x)
        return B.verify(p,np.clip(x,lo,hi))
    # A projected Newton method for genuinely coupled, lightly filled box QPs.
    dense = p.n <= 1400 and P.nnz > 14*p.n
    if not np.all(diag > 0.) or (not dense and P.nnz > 14*p.n) or p.n < 20:
        return None
    from scipy.sparse.linalg import spsolve
    if dense:
        from scipy.linalg import lapack
        full = P.toarray(order='F')
    den = np.maximum(diag,1.e-15*maxabs(diag))
    x = np.clip(-p.q/den,lo,hi)
    prev = None
    for it in range(11):
        if perf_counter() > deadline:
            break
        g = P@x+p.q
        trial = x-g/den
        low = trial <= lo
        high = (trial >= hi) & ~low
        fixed = low | high
        ans = B.verify(p,x,g)
        if ans is not None:
            return ans
        if prev is not None and np.array_equal(prev, fixed):
            return None
        prev = fixed
        x = np.zeros(p.n)
        x[low], x[high] = lo[low], hi[high]
        free = ~fixed
        if free.any():
            rhs = -p.q[free]-(P@x)[free]
            if dense:
                ij = np.flatnonzero(free)
                lhs = np.asfortranarray(full[np.ix_(ij,ij)])
                fac,flag = lapack.dpotrf(lhs,lower=1,clean=0,overwrite_a=1)
                if flag != 0:
                    lhs[:,:] = full[np.ix_(ij,ij)]
                    lhs.flat[::len(ij)+1] += 5.e-12*max(1.,maxabs(diag))
                    fac,flag = lapack.dpotrf(lhs,lower=1,clean=0,overwrite_a=1)
                    if flag!=0:
                        return None
                x[free] = lapack.dpotrs(fac,rhs,lower=1)[0]
            else:
                lhs = P[free,:][:,free].tocsc()
                x[free] = spsolve(lhs,rhs,permc_spec='MMD_AT_PLUS_A')
            if not np.isfinite(x).all():
                return None
        x = np.clip(x,lo,hi)
    return None


def polish_polyhedral(p,x,z,deadline):
    if not p.n or p.K.nonlinear or p.n+p.m > 100000:
        return None
    from scipy.sparse.linalg import splu
    from scipy.linalg import lapack
    w = Scaled(p,passes=4,scale_rhs=False)
    E, D, c = w.E, w.D, w.c
    X, Z = x/D, z/E*c
    s = w.b-w.A@X
    act = np.zeros(p.m,dtype=bool)
    act[p.K.eq] = True
    l = p.K.lpos
    margin = 1.e-6*np.maximum(1.,np.abs(w.b[l]))
    act[l] = (s[l] < margin) | (Z[l] > max(1.e-8,1.e-6*maxabs(Z[l])))
    for it in range(4):
        if perf_counter() > deadline:
            break
        A = w.A[act,:].tocsc()
        m, n = A.shape
        N = n+m
        z0 = Z[act].copy()
        sys = sparse.bmat([[w.P,A.T],[A,None]],format='csc')
        reg = sys+sparse.diags(np.r_[np.full(n,1.e-11),np.full(m,-1.e-11)],format='csc')
        if N <= 110:
            R, pivot, info = lapack.dsytrf(reg.toarray(order='F'),lower=1,overwrite_a=1)
            if info:
                return None
            f = lambda r: lapack.dsytrs(R,pivot,r,lower=1)[0]
        else:
            try:
                fact = splu(reg,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=.05)
            except RuntimeError:
                return None
            f = fact.solve
        rhs = np.r_[-w.q,w.b[act]]
        u = np.r_[X,z0]
        for _ in range(3):
            r = rhs-sys@u
            if maxabs(r) < 3.e-13:
                break
            u += f(r)
        X = u[:n]
        Z = np.zeros(p.m)
        Z[act] = u[n:]
        ans = p.optimal(X*D,Z*E/c)
        if ans is not None:
            return ans
        s = w.b-w.A@X
        new = act.copy()
        new[l] = (act[l] & (Z[l] >= -1.e-9)) | (s[l] < -1.e-8)
        if np.array_equal(new,act):
            if it > 0:
                break
            new[l] = act[l] & ((Z[l] > 1.e-7) | (s[l] < -1.e-8))
        act = new
    return None


def few_active(p,deadline):
    """A short active-set attempt when the unconstrained point is nearly feasible."""
    n,m=p.n,p.m
    P,A=p.P,p.A
    if (p.K.nonlinear or n<90 or n>13500 or p.K.eq.size>.28*n or
            P.nnz==0 or (P.nnz>16*n and n>850) or A.nnz>400000):
        return None
    from scipy.sparse.linalg import splu
    from scipy.linalg import lapack
    eq=np.zeros(m,dtype=bool)
    eq[p.K.eq]=True
    act=eq.copy()
    itr=0
    side=p.K.lpos
    tol=5.e-10*(1.+p.bn)
    for it in range(5):
        if perf_counter()>deadline:
            break
        rows=A[act,:].tocsc()
        k=rows.shape[0]
        if k==0 and P.nnz==np.count_nonzero(P.diagonal()):
            diag=P.diagonal()
            if np.any(diag<=0.):
                return None
            x=-p.q/diag
            za=np.empty(0)
        else:
            H=sparse.bmat([[P,rows.T],[rows,None]],format='csc') if k else P
            rhs=np.r_[-p.q,p.b[act]]
            try:
                if n+k<=110 or (n+k<1100 and H.nnz>.13*(n+k)**2):
                    Z=H.toarray(order='F')
                    if k:
                        R,S,flag=lapack.dsytrf(Z,lower=1,overwrite_a=1)
                        z=lapack.dsytrs(R,S,rhs,lower=1)[0] if not flag else None
                    else:
                        R,flag=lapack.dpotrf(Z,lower=1,overwrite_a=1)
                        z=lapack.dpotrs(R,rhs,lower=1)[0] if not flag else None
                    if z is None:
                        return None
                else:
                    R=splu(H,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=.02)
                    z=R.solve(rhs)
                x,za=z[:n],z[n:]
                if not np.isfinite(z).all():
                    return None
            except (ValueError,RuntimeError,np.linalg.LinAlgError):
                return None
        s=p.b-A@x
        z=np.zeros(m)
        z[act]=za
        bad=(s[side]<-tol)
        if not np.any(bad):
            result=p.optimal(x,z,s,polish_slack=False)
            if result is not None:
                return result
        neg=(z[side]<-4.e-11*(1.+p.qn))
        changes=np.count_nonzero(bad|neg)
        if changes>min(120,.26*n) or not changes:
            break
        updated=act.copy()
        updated[side]|=bad
        updated[side]&=~neg
        if not np.any(updated!=act):
            break
        act=updated
    return None
