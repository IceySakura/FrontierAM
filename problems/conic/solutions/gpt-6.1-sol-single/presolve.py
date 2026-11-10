"""Exact sparse substitutions and original-coordinate reconstruction."""
import os
import sys

import numpy as np
from scipy import sparse


class Substitution:
    def __init__(self, P, q, A, b, cones):
        self.P, self.q, self.A, self.b = P,q,A,b
        self.cones = cones
        self.steps = []

    def remove_constants(self):
        A,b,cones=self.A,self.b,self.cones
        counts=np.diff(A.tocsr().indptr)
        keep=np.ones(len(b),dtype=bool)
        st=0
        newcones=[]
        for c in cones:
            kind,dim=c["type"],c["dim"]
            width=dim*(dim+1)//2 if kind=="psd" else dim
            sl=slice(st,st+width)
            if kind in ("zero","nonnegative"):
                good=(b[sl]==0) if kind=="zero" else (b[sl]>=0)
                drop=(counts[sl]==0)&good
                keep[sl]=~drop
                nd=int(np.count_nonzero(~drop))
                if nd:
                    newcones.append({"type":kind,"dim":nd})
            elif np.all(counts[sl]==0):
                bb=b[sl]
                if kind=="soc":
                    good=bb[0]>=np.linalg.norm(bb[1:])
                elif kind=="exponential":
                    a,v,w=bb
                    good=(v>0 and w>0 and a<=v*(np.log(w)-np.log(v))) or (
                        v==0 and a<=0 and w>=0)
                else:
                    good=False
                if good:
                    keep[sl]=False
                else:
                    newcones.append(c)
            else:
                newcones.append(c)
            st+=width
        if not np.all(keep):
            rows=np.flatnonzero(keep)
            self.steps.append(("rows",b,rows))
            self.A,self.b,self.cones=A[rows].tocsc(),b[rows],newcones
        return self

    def eliminate_equalities(self, rounds=4, max_degree=1):
        for repeat in range(rounds):
            P,q,A,b,cones = self.P,self.q,self.A,self.b,self.cones
            n,m = A.shape[1], A.shape[0]
            eq = []
            st = 0
            for c in cones:
                width = c["dim"]*(c["dim"]+1)//2 if c["type"]=="psd" else c["dim"]
                if c["type"]=="zero":
                    eq.extend(range(st,st+width))
                st += width
            eq = np.asarray(eq,dtype=np.int32)
            if not len(eq) or n < 2:
                break
            ae = A[eq].tocsc()
            counts = np.diff(ae.indptr)
            if max_degree>1:
                import ctypes
                lib=ctypes.CDLL(os.path.join(os.path.dirname(__file__),"native.so"))
                ip=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
                dp=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
                bp=np.ctypeslib.ndpointer(dtype=np.uint8,flags="C_CONTIGUOUS")
                lib.select_pivots.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,dp,ip,ip,bp,
                                            dp,ctypes.c_int,ip,ip,dp]
                piv=np.empty(n,dtype=np.int32)
                erows=np.empty(n,dtype=np.int32)
                coeff=np.empty(n)
                length=lib.select_pivots(n,len(eq),ae.indptr,ae.indices,ae.data,
                    np.diff(ae.tocsr().indptr).astype(np.int32),
                    np.diff(A.indptr).astype(np.int32),
                    (np.diff(P.indptr)==0).astype(np.uint8),q,max_degree,piv,erows,coeff)
                piv,erows,coeff=piv[:length],erows[:length],coeff[:length]
            else:
                candidates = np.flatnonzero((counts==1) & (np.diff(P.indptr)==0))
                if not len(candidates):
                    break
                positions = ae.indptr[candidates]
                erows = ae.indices[positions]
                coeffs = ae.data[positions]
                scores = np.diff(A.indptr)[candidates] + 100*(q[candidates]!=0)
                order = np.lexsort((scores, erows))
                candidates,erows,coeffs = candidates[order],erows[order],coeffs[order]
                first = np.r_[True,np.diff(erows)!=0]
                piv,erows,coeff = candidates[first],erows[first],coeffs[first]
            good = (np.abs(coeff)>1e-8) & (np.abs(coeff)<1e8)
            piv,erows,coeff = piv[good],erows[good],coeff[good]
            if len(piv) >= n:
                piv,erows,coeff = piv[:n-1],erows[:n-1],coeff[:n-1]
            if not len(piv):
                break
            prows = eq[erows]
            keepvar = np.ones(n,dtype=bool)
            keepvar[piv]=False
            keepvar = np.flatnonzero(keepvar)
            keeprow = np.ones(m,dtype=bool)
            keeprow[prows]=False
            keeprow = np.flatnonzero(keeprow)
            E = A[prows][:,keepvar].tocoo()
            tv = -E.data/coeff[E.row]
            T = sparse.csc_matrix((np.r_[np.ones(len(keepvar)),tv],
                (np.r_[keepvar,piv[E.row]],np.r_[np.arange(len(keepvar)),E.col])),
                shape=(n,len(keepvar)))
            offset = np.zeros(n)
            offset[piv] = b[prows]/coeff
            ar = A[keeprow]
            newA = (ar@T).tocsc()
            if (newA.nnz>max(A.nnz*5,10000) or
                np.max(np.abs(newA.data),initial=0.)>
                    100*max(1.,np.max(np.abs(A.data),initial=0.))):
                break
            newb = b[keeprow]-ar@offset
            newq = np.asarray(T.T@q).ravel()
            newP = (T.T@P@T).tocsc()
            newcones = []
            mask = np.ones(m,dtype=bool)
            mask[prows]=False
            st=0
            for c in cones:
                width = c["dim"]*(c["dim"]+1)//2 if c["type"]=="psd" else c["dim"]
                if c["type"]=="zero":
                    dim = int(np.count_nonzero(mask[st:st+width]))
                    if dim:
                        newcones.append({"type":"zero","dim":dim})
                else:
                    newcones.append(c)
                st+=width
            self.steps.append((P,q,A,b,T,offset,piv,prows,coeff,keeprow))
            self.P,self.q,self.A,self.b,self.cones = newP,newq,newA,newb,newcones
            self.remove_constants()
            if os.environ.get("SOLVER_DEBUG"):
                print("presolve", (m,n),"->",newA.shape,"nnz",newA.nnz,
                      file=sys.stderr,flush=True)
        return self

    def data(self):
        return self.P,self.q,self.A,self.b,self.cones

    def restore(self,result):
        status=result["status"]
        if status=="unresolved":
            return result
        for step in reversed(self.steps):
            if len(step)==3:
                _,b,rows=step
                if result.get("z") is not None:
                    zz=np.zeros(len(b))
                    zz[rows]=result["z"]
                    result["z"]=zz
                if result.get("s") is not None:
                    ss=b.copy() if status=="solved" else np.zeros(len(b))
                    ss[rows]=result["s"]
                    result["s"]=ss
                continue
            P,q,A,b,T,offset,piv,prows,coeff,keeprow=step
            x = result.get("x")
            z = result.get("z")
            if x is not None:
                x = T@x
                if status=="solved":
                    x += offset
                result["x"]=np.asarray(x).ravel()
            if z is not None:
                zz = np.zeros(len(b))
                zz[keeprow]=z
                residual=A.T@zz
                if status=="solved":
                    residual += P@x+q
                zz[prows]=-residual[piv]/coeff
                result["z"]=zz
            if status in ("solved","dual_infeasible"):
                ss = np.zeros(len(b))
                ss[keeprow] = result["s"]
                result["s"]=ss
        return result


def empty_columns(P,q,A):
    return np.flatnonzero((np.diff(A.indptr)==0)&(np.diff(P.indptr)==0)&(q==0))


def zero_cost_feasible(P,q,A,b,cones):
    """Peel unconstrained null-cost columns, then solve any remaining core."""
    from scipy.sparse import linalg as sla
    n,m=A.shape[1],A.shape[0]
    eq=np.zeros(m,dtype=bool)
    st=0
    for c in cones:
        if c["type"]=="zero":
            eq[st:st+c["dim"]]=True
        st+=c["dim"]
    ni=np.flatnonzero(~eq)
    ei=np.flatnonzero(eq)
    if not len(ei) or np.min(b[ni],initial=0.) < -1e-12:
        return None
    G=A[ni].tocsc()
    free=(q==0)&(np.diff(P.indptr)==0)&(np.diff(G.indptr)==0)&(np.diff(A.indptr)>0)
    free=np.flatnonzero(free)
    if not len(free):
        return None
    gr=G.tocsr()
    rr=np.flatnonzero(np.diff(gr.indptr)==1)
    pos=gr.indptr[rr]
    cc=gr.indices[pos]
    vv=gr.data[pos]
    good=q[cc]*vv<0
    rr,cc,vv=rr[good],cc[good],vv[good]
    selected=np.full(n,-1,dtype=np.int32)
    selected[cc]=rr
    use=np.flatnonzero(selected>=0)
    z=np.zeros(m)
    grows=selected[use]
    coeff=gr.data[gr.indptr[grows]]
    z[ni[grows]]=-q[use]/coeff
    if np.max(np.abs(q+A.T@z),initial=0.)>1e-11*(1+np.max(np.abs(q),initial=0.)):
        return None
    F=A[ei][:,free].tocsc()
    rhs=b[ei].copy()
    steps=[]
    for iteration in range(30):
        counts=np.diff(F.indptr)
        cand=np.flatnonzero(counts==1)
        if not len(cand):
            break
        pos=F.indptr[cand]
        rows=F.indices[pos]
        coeff=F.data[pos]
        order=np.lexsort((-np.abs(coeff),rows))
        cand,rows,coeff=cand[order],rows[order],coeff[order]
        first=np.r_[True,np.diff(rows)!=0]
        piv,rows,coeff=cand[first],rows[first],coeff[first]
        ar=F[rows].tocsr()
        steps.append((piv,ar,rhs[rows],coeff))
        keep=np.ones(F.shape[0],dtype=bool)
        keep[rows]=False
        F=F[keep].tocsc()
        rhs=rhs[keep]
        if not len(rhs):
            break
    xf=np.zeros(len(free))
    if len(rhs):
        scale=np.sqrt(np.asarray(F.multiply(F).sum(axis=1)).ravel())
        scale=np.maximum(scale,1e-15)
        Fs=sparse.diags(1/scale)@F
        xf=sla.lsmr(Fs,rhs/scale,atol=1e-12,btol=1e-12,maxiter=180)[0]
    for piv,ar,r,coeff in reversed(steps):
        xf[piv]=(r-ar@xf)/coeff
    x=np.zeros(n)
    x[free]=xf
    ax=A@x
    s=b-ax
    scale=1+max(np.max(np.abs(ax),initial=0.),np.max(np.abs(s),initial=0.),
                np.max(np.abs(b),initial=0.))
    gap=abs(float(b@z))
    if (np.max(np.abs(s[ei]),initial=0.)<5e-9*scale and
        np.min(s[ni],initial=0.)>-5e-9 and gap<5e-9):
        s[ei]=0.
        return {"status":"solved","x":x,"s":s,"z":z}
    return None
