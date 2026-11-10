"""Polyhedral outer approximation with reversible cone-dual aggregation."""
import os
import sys
import time

import numpy as np
from scipy import sparse

from cones import Layout
from ipm import accepted, certificate, norm


def outer_approx(P,q,A,b,cones):
    from solver import highs_model,highs_core,polyhedral
    hc=highs_core()
    start=time.monotonic()
    budget=float(os.environ.get("OA_SECONDS",str(max(.25,min(15.,8e-5*len(q))))))
    layout=Layout(cones,len(b))
    base=np.r_[layout.eq,layout.nn]
    C=sparse.csr_matrix((np.ones(len(base)),(np.arange(len(base)),base)),
                        shape=(len(base),len(b)))
    chunks=[C]
    for d,ids in layout.soc.items():
        count=len(ids)*2*(d-1)
        rr=np.arange(count)
        cc=np.repeat(ids[:,0],2*(d-1))
        vv=np.repeat(ids[:,1:],2,axis=1).ravel()
        chunks.append(sparse.csr_matrix((np.r_[np.ones(count),
            np.tile([-1.,1.],len(ids)*(d-1))],
            (np.r_[rr,rr],np.r_[cc,vv])),shape=(count,len(b))))
    C=sparse.vstack(chunks,format="csr")
    AA=(C@A).tocsc()
    bb=C@b
    lo=np.full(len(bb),-hc.kHighsInf)
    lo[:len(layout.eq)]=bb[:len(layout.eq)]
    h=highs_model(None,q,AA,lo,bb,method=os.environ.get("OA_METHOD","ipm"),
                  time_limit=max(.01,budget-(time.monotonic()-start)))
    warm=None

    def infeasible_certificate():
        a=norm(AA.data) or 1.
        bs=norm(bb)
        if not bs:
            return None
        alternative=sparse.vstack([AA.T/a,sparse.csr_matrix(bb/bs)],format="csc")
        rhs=np.r_[np.zeros(len(q)),-1.]
        cl=np.zeros(len(bb))
        cl[:len(layout.eq)]=-hc.kHighsInf
        hh=highs_model(None,np.zeros(len(bb)),alternative,rhs,rhs,cl,time_limit=20.)
        if hh.getModelStatus()!=hc.HighsModelStatus.kOptimal:
            return None
        lam=np.asarray(hh.getSolution().col_value)
        z=C.T@lam
        return certificate(P,q,A,b,layout,np.zeros(len(q)),np.zeros(len(b)),z,tol=7e-9)

    for iteration in range(80):
        status=h.getModelStatus()
        if os.environ.get("SOLVER_DEBUG"):
            print("OA",iteration,AA.shape[0],h.modelStatusToString(status),
                  "%.3f"%(time.monotonic()-start),file=sys.stderr,flush=True)
        if status==hc.HighsModelStatus.kInfeasible:
            result=infeasible_certificate()
            if result:
                return result,None
            break
        sol=h.getSolution()
        ray=False
        if status in (hc.HighsModelStatus.kUnbounded,hc.HighsModelStatus.kUnboundedOrInfeasible):
            cs=[]
            if len(layout.eq):
                cs.append({"type":"zero","dim":len(layout.eq)})
            if len(bb)>len(layout.eq):
                cs.append({"type":"nonnegative","dim":len(bb)-len(layout.eq)})
            rr=polyhedral(sparse.csc_matrix(P.shape),q,AA,bb,cs)
            if rr["status"]=="primal_infeasible":
                result=certificate(P,q,A,b,layout,np.zeros(len(q)),np.zeros(len(b)),
                                   C.T@rr["z"],tol=7e-9)
                if result:
                    return result,None
            if rr["status"]!="dual_infeasible":
                break
            x=rr["x"]
            s=-A@x
            result=certificate(P,q,A,b,layout,x,s,np.zeros(len(b)),tol=7e-9)
            if result:
                return result,None
            ray=True
        elif len(sol.col_value)==len(q) and len(sol.row_dual)==len(bb):
            x=np.asarray(sol.col_value)
            lam=-np.asarray(sol.row_dual)
            z=C.T@lam
            s=b-A@x
            if accepted(P,q,A,b,layout,x,s,z,tol=7e-9):
                return {"status":"solved","x":x,"s":s,"z":z},None
            linear_ok=(norm(s[layout.eq])<1e-6*(1+norm(b)) and
                (not len(layout.nn) or np.min(s[layout.nn])>
                    -1e-6*(1+norm(b))) and
                norm(q+A.T@z)<1e-5*(1+max(norm(q),norm(A.T@z))))
            if (linear_ok and np.isfinite(x).all() and np.isfinite(z).all()
                and norm(x)<1e12):
                warm={"x":x,"z":z}
        else:
            break
        if time.monotonic()-start>=budget:
            break
        rows,cols,vals=[],[],[]
        added=0
        for d,ids in layout.soc.items():
            ss=s[ids]
            vn=np.linalg.norm(ss[:,1:],axis=1)
            defect=(vn-ss[:,0])/np.maximum(1.,np.max(np.abs(ss),axis=1))
            pick=np.flatnonzero(defect>2e-9)
            if not len(pick):
                continue
            chosen=ids[pick]
            direction=ss[pick,1:]/np.maximum(vn[pick,None],1e-300)
            rows.append(np.repeat(np.arange(added,added+len(pick)),d))
            cols.append(chosen.ravel())
            vals.append(np.column_stack([np.ones(len(pick)),-direction]).ravel())
            added+=len(pick)
        if not added:
            break
        cut=sparse.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                              shape=(added,len(b)))
        ca=(cut@A).tocsr()
        cb=cut@b
        h.addRows(added,np.full(added,-hc.kHighsInf),cb,ca.nnz,ca.indptr,ca.indices,ca.data)
        C=sparse.vstack([C,cut],format="csr")
        AA=sparse.vstack([AA,ca],format="csc")
        bb=np.r_[bb,cb]
        remaining=budget-(time.monotonic()-start)
        if remaining<=0:
            break
        h.setOptionValue("time_limit",h.getRunTime()+remaining)
        h.run()
    return None,warm
