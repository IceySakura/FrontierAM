"""Exact recognition and recovery for weighted least-squares/L1 epigraphs."""
import ctypes
import os
import sys

import numpy as np
from scipy import sparse


def solve_lasso(P,q,A,b,cones):
    if not all(c["type"] in ("zero","nonnegative") for c in cones):
        return None
    pd=P.diagonal()
    rcols=np.flatnonzero(pd)
    if P.nnz!=len(rcols) or np.any(pd[rcols]<=0):
        return None
    eq=np.zeros(len(b),dtype=bool)
    st=0
    for c in cones:
        if c["type"]=="zero":
            eq[st:st+c["dim"]]=True
        st+=c["dim"]
    er=np.flatnonzero(eq)
    nr=np.flatnonzero(~eq)
    if len(rcols)!=len(er) or not len(er) or np.any(b[nr]!=0):
        return None
    E=A[er].tocsc()
    G=A[nr].tocsc()
    gr=G.tocsr()
    if np.any(np.diff(gr.indptr)!=2) or G[:,rcols].nnz:
        return None
    R=E[:,rcols].tocoo()
    if R.nnz!=len(er) or len(np.unique(R.row))!=len(er):
        return None
    if len(np.unique(R.col))!=len(rcols):
        return None
    coef=np.empty(len(er))
    rc=np.empty(len(er),dtype=np.int32)
    coef[R.row]=R.data
    rc[R.row]=rcols[R.col]
    if np.any(np.abs(coef)<1e-12):
        return None
    cols=gr.indices.reshape(-1,2)
    vals=gr.data.reshape(-1,2)
    role=q[cols]>0
    if np.any(np.sum(role,axis=1)!=1):
        return None
    tcol=np.where(role[:,0],cols[:,0],cols[:,1])
    wcol=np.where(role[:,0],cols[:,1],cols[:,0])
    tc=np.where(role[:,0],vals[:,0],vals[:,1])
    wc=np.where(role[:,0],vals[:,1],vals[:,0])
    if np.any(tc>=0) or np.any(q[wcol]!=0) or E[:,np.unique(tcol)].nnz:
        return None
    candidates=np.flatnonzero((np.diff(G.indptr)==2)&(q==0)&(pd==0))
    if len(candidates)*2!=len(nr):
        return None
    unique_tabs=np.unique(tcol)
    if len(unique_tabs)!=len(candidates) or np.any(np.diff(G.indptr)[unique_tabs]!=2):
        return None
    remaining=q!=0
    remaining[rc]=False
    remaining[unique_tabs]=False
    if np.any(remaining):
        return None
    pos=G.indptr[candidates]
    rows0=G.indices[pos]
    rows1=G.indices[pos+1]
    if np.any(tcol[rows0]!=tcol[rows1]):
        return None
    k0=wc[rows0]/tc[rows0]
    k1=wc[rows1]/tc[rows1]
    if np.max(np.abs(k0+k1)/np.maximum(1.,np.abs(k0)),initial=0.)>1e-12:
        return None
    live=np.diff(E.indptr)[candidates]>0
    wvars=candidates[live]
    rows0,rows1=rows0[live],rows1[live]
    tabs=tcol[rows0]
    kabs=np.abs(k0[live])
    penalty=q[tabs]*kabs
    if np.any(penalty<=0):
        return None
    D=E[:,wvars].tocsc()
    weight=np.sqrt(pd[rc]/2.)/coef
    B=(sparse.diags(weight)@D).tocsc()
    B.sort_indices()
    y=weight*b[er]+q[rc]/np.sqrt(2.*pd[rc])
    xw=np.zeros(len(wvars))
    residual=np.empty(len(er))
    diagnostics=np.zeros(4)
    lib=ctypes.CDLL(os.path.join(os.path.dirname(__file__),"native.so"))
    ip=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    dp=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
    lib.lasso_cd.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,dp,dp,dp,dp,dp,
                           ctypes.c_double,ctypes.c_double,ctypes.c_double,dp]
    offset=float(np.sum(q[rc]**2/(2.*pd[rc])))
    seconds=float(os.environ.get("LASSO_SECONDS","6.0"))
    success=lib.lasso_cd(len(er),len(wvars),B.indptr,B.indices,B.data,
        np.ascontiguousarray(y),np.ascontiguousarray(penalty),xw,residual,
        seconds,2e-9,offset,diagnostics)
    if os.environ.get("SOLVER_DEBUG"):
        print("Lasso",len(wvars),B.nnz,"success",success,diagnostics,file=sys.stderr,flush=True)
    x=np.zeros(len(q))
    x[wvars]=xw
    x[tabs]=kabs*np.abs(xw)
    x[rc]=(b[er]-D@xw)/coef
    z=np.zeros(len(b))
    z[er]=-(pd[rc]*x[rc]+q[rc])/coef
    zg=-q[tcol]/(2.*tc)
    grad=D.T@z[er]
    zg[rows0]-=grad/(2.*wc[rows0])
    zg[rows1]-=grad/(2.*wc[rows1])
    z[nr]=zg
    s=b-A@x
    from solver import poly_candidate
    result={"status":"solved","x":x,"s":s,"z":z}
    return result if poly_candidate(P,q,A,b,cones,x,s,z) else {"status":"unresolved"}
