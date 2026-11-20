"""Feasible equality coordinates for moderate, equality-heavy models."""
import numpy as np
from scipy import sparse
from scipy.linalg import qr, solve_triangular

from model import Layout, Problem, maxabs


class Undo:
    def __init__(self,p,keep,scale,piv,Q,R,x0):
        self.p,self.keep,self.scale,self.piv=p,keep,scale,piv
        self.Q,self.R,self.x0=Q,R,x0
        self.rank=R.shape[0]
        self.N=Q[:,self.rank:]

    def dual(self,y,x=None):
        p=self.p
        z=np.zeros(p.m)
        z[self.keep]=y
        r=-(p.A.T@z)
        if x is not None:
            r-=p.P@x+p.q
        h=solve_triangular(self.R,self.Q[:,:self.rank].T@r,check_finite=False)
        eq=p.K.eq[self.piv]
        z[eq]=self.scale[self.piv]*h
        return z

    def restore(self,answer):
        p=self.p
        st=answer['status']
        if st=='solved':
            x=self.N@answer['x']+self.x0
            return p.optimal(x,self.dual(answer['z'],x))
        if st=='primal_infeasible':
            return p.primal_cert(self.dual(answer['z']))
        if st=='dual_infeasible':
            return p.dual_cert(self.N@answer['x'])
        return None


def reduce(p):
    n,m=p.n,p.m
    eq=p.K.eq
    ne=len(eq)
    if (n<36 or n>795 or ne<.48*n or ne>4700 or m>12000
            or (not p.P.nnz and not p.K.nonlinear)):
        return p,None,None
    if p.A.nnz<.055*n*m and p.P.nnz<.065*n*n and ne<.86*n:
        # Sparse KKT structure is usually more valuable than a dense
        # basis of comparable dimension.
        return p,None,None
    A=p.A[eq,:].toarray()
    v=np.max(np.abs(A),axis=1)
    sc=1./np.maximum(v,1.e-14)
    A*=sc[:,None]
    b=p.b[eq]*sc
    Q,R,per=qr(A.T,mode='full',pivoting=True,check_finite=False)
    diag=np.abs(np.diag(R))
    r=int(np.count_nonzero(diag>max(1.e-13,2.e-11*diag[0])))
    if r<.42*n:
        return p,None,None
    F=R[:r,:r]
    y=solve_triangular(F,b[per[:r]],trans='T',check_finite=False)
    err=b[per[r:]]-R[:r,r:].T@y
    if maxabs(err)>6.e-11*(1.+maxabs(b)):
        z=np.zeros(m)
        v=-err
        d=solve_triangular(F,-R[:r,r:]@v,check_finite=False)
        z[eq[per[:r]]]=sc[per[:r]]*d
        z[eq[per[r:]]]=sc[per[r:]]*v
        c=p.primal_cert(z)
        if c is not None:
            return p,None,c
        if maxabs(err)>8.e-9*(1.+maxabs(b)):
            return p,None,None
    x0=Q[:,:r]@y
    keep=np.ones(m,dtype=bool)
    keep[eq]=False
    N=Q[:,r:]
    if N.shape[1]*int(keep.sum())>5_400_000:
        return p,None,None
    Ap=p.A[keep,:]
    b0=p.b[keep]-Ap@x0
    q0=N.T@(p.q+p.P@x0)
    Ps=N.T@(p.P@N)
    g=Problem(sparse.csc_matrix((Ps+Ps.T)*.5),q0,sparse.csc_matrix(Ap@N),
              b0,Layout([(k,d) for k,d in p.K.spec if k!='zero']))
    return g,Undo(p,keep,sc,per[:r],Q,F,x0),None
