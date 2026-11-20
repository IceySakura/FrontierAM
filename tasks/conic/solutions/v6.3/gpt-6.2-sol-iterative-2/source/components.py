"""Independent blocks and identical subproblem reuse within one input."""
from time import perf_counter

import numpy as np
from scipy import sparse

import native
from model import Layout, Problem


CODE={'zero':0,'nonnegative':1,'soc':2,'exponential':3,'psd':4}
KIND=list(CODE)
BASE=1<<25


def decompose(p,solve,deadline):
    n,m=p.n,p.m
    if (native.lib is None or n<1800 or n>1_200_000 or m>2_800_000 or
        p.A.nnz+p.P.nnz>11_000_000 or not p.K.blocks or p.K.psd and m>600_000):
        return None
    starts,ends=[],[]
    code=np.ones(m,dtype=np.int32)
    code[p.K.eq]=0
    for k,d,a,b in p.K.nonlinear:
        starts.append(a)
        ends.append(b)
        code[a:b]=-1
        code[a]=CODE[k]*BASE+d
    codes=np.asarray(starts,dtype=np.int32),np.asarray(ends,dtype=np.int32),code
    ans=native.components(p,codes)
    if ans is None:
        return None
    nx,rs,sig = ans
    count=len(sig)
    nv,nr=np.bincount(nx,minlength=count),np.bincount(rs[rs>=0],minlength=count)
    s=np.column_stack((sig,nv.astype(np.uint64),nr.astype(np.uint64)))
    _,first,ind=np.unique(s,axis=0,return_index=True,return_inverse=True)
    if len(first)>32 or (len(first)>18 and len(first)>.6*count):
        return None
    void=np.flatnonzero(rs<0)
    if len(void):
        const=np.zeros(m)
        const[void]=p.b[void]
        if p.K.cone_error(const,False)>p.tol*.4:
            return None
    vi=np.argsort(nx,kind='stable')
    ri=np.argsort(rs,kind='stable')[len(void):]
    vp,rp=np.r_[0,np.cumsum(nv)],np.r_[0,np.cumsum(nr)]
    order=np.argsort(ind,kind='stable')
    groups=np.r_[0,np.cumsum(np.bincount(ind,minlength=len(first)))]
    x,z=np.zeros(n),np.zeros(m)
    for k,r in enumerate(first):
        if perf_counter()>deadline:
            return None
        cols=vi[vp[r]:vp[r+1]]
        rows=ri[rp[r]:rp[r+1]]
        A,P=p.A[:,cols],p.P[:,cols]
        ca=np.searchsorted(rows,A.indices).astype(np.int32)
        cp=np.searchsorted(cols,P.indices).astype(np.int32)
        arr=code[rows]
        spec=[]
        # Runs of scalar rows can be treated as a single linear block.
        pos=0
        while pos<len(arr):
            rcode=arr[pos]
            if rcode>=BASE:
                typ,d=divmod(int(rcode),BASE)
                spec.append((KIND[typ],d))
                pos += (d*(d+1)//2 if typ==4 else d)
            else:
                hi=pos+1
                while hi<len(arr) and arr[hi]==rcode:
                    hi+=1
                if rcode<0:
                    return None
                spec.append((KIND[rcode],hi-pos))
                pos=hi
        g=Problem(sparse.csc_matrix((P.data,cp,P.indptr),shape=(len(cols),len(cols))),
                   p.q[cols],sparse.csc_matrix((A.data,ca,A.indptr),shape=(len(rows),len(cols))),
                   p.b[rows],Layout(spec))
        res=solve(g)
        if res is None:
            return None
        sel=order[groups[k]:groups[k+1]]
        xrows=vi[(vp[sel,None]+np.arange(len(cols))).reshape(-1)]
        zrows=ri[(rp[sel,None]+np.arange(len(rows))).reshape(-1)]
        repeat=len(sel)
        st=res['status']
        if st=='primal_infeasible':
            a=np.zeros(m)
            a[zrows]=np.tile(res['z'],repeat)
            return p.primal_cert(a)
        if st=='dual_infeasible':
            a=np.zeros(n)
            a[xrows]=np.tile(res['x'],repeat)
            return p.dual_cert(a)
        x[xrows]=np.tile(res['x'],repeat)
        z[zrows]=np.tile(res['z'],repeat)
    return p.optimal(x,z,polish_slack=True)
