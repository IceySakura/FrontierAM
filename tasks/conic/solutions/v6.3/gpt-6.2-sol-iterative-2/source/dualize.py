"""Dualize LP-conic models with explicit cone variables."""
import numpy as np
from scipy import sparse

from model import Layout, Problem


class Undo:
    def __init__(self,p,red,cols,rows,a,ep):
        self.p,self.red = p,red
        self.cols,self.rows,self.a,self.ep=cols,rows,a,ep

    def primal_cone(self,z):
        u = z.copy()
        i = self.ep
        u[i],u[i+1],u[i+2] = -z[i+1],-z[i],np.e*z[i+2]
        return u

    def dual_cone(self,z):
        u = z.copy()
        i = self.ep
        u[i],u[i+1],u[i+2] = -z[i+1],-z[i],z[i+2]/np.e
        return u

    def restore(self,r):
        p,g=self.p,self.red
        j,i,a=self.cols,self.rows,self.a
        status=r['status']
        if status=='solved':
            x=np.empty(p.n)
            s=self.primal_cone(r['z'])
            x[j]=(p.b[i]-s)/a
            z=np.empty(p.m)
            z[p.K.eq]=r['x']
            for c in (g.b-g.A@r['x'],r['s']):
                z[i]=self.dual_cone(c)
                ans=p.optimal(x,z)
                if ans is not None:
                    return ans
            return None
        if status=='primal_infeasible':
            x=np.empty(p.n)
            x[j]=-self.primal_cone(r['z'])/a
            return p.dual_cert(x)
        if status=='dual_infeasible':
            z=np.empty(p.m)
            z[p.K.eq]=r['x']
            z[i]=self.dual_cone(-g.A@r['x'])
            ans=p.primal_cert(z)
            if ans is None:
                z[i]=self.dual_cone(r['s'])
                ans=p.primal_cert(z)
            return ans
        return None


def apply(p):
    m,n=p.m,p.n
    ne=len(p.K.eq)
    if (p.P.nnz or n<60 or n!=m-ne or ne==0 or ne>.65*n or not p.K.nonlinear
            or (not p.K.psd and n<1200) or p.A.nnz>6_000_000):
        return p,None
    mask=np.ones(m,dtype=bool)
    mask[p.K.eq]=False
    count=np.bincount(p.A.indices,minlength=m)
    if np.any(count[mask]!=1):
        return p,None
    which=np.flatnonzero(mask)
    csr=p.A.tocsr()
    at=csr.indptr[which]
    cols=csr.indices[at]
    if np.any(np.bincount(cols,minlength=n)!=1):
        return p,None
    a=csr.data[at]
    if np.max(np.abs(a))>1.e12 or np.min(np.abs(a))<1.e-12:
        return p,None
    Aeq=csr[p.K.eq,:]
    A=(Aeq.T[cols,:].multiply((1./a)[:,None])).tocsc()
    b=-p.q[cols]/a
    v=np.empty(n)
    v[cols]=p.b[which]/a
    q=p.b[p.K.eq]-Aeq@v
    specs=[(k,d) for k,d in p.K.spec if k!='zero']
    K=Layout(specs)
    ep=K.groups.get(('exponential',3),np.empty(0,dtype=np.int32))
    if ep.size:
        row=np.arange(n,dtype=np.int32)
        mul=np.ones(n)
        row[ep],row[ep+1]=ep+1,ep
        mul[ep]=mul[ep+1]=-1.
        mul[ep+2]=np.e
        A=(A[row,:].multiply(mul[:,None])).tocsc()
        b=b[row]*mul
    g=Problem(sparse.csc_matrix((ne,ne)),q,A,b,K)
    return g,Undo(p,g,cols,which,a,ep)
