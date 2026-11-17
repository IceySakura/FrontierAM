import numpy as np, sys
from numpy.lib.stride_tricks import sliding_window_view as swv
from common import *
PH=[(0,0),(0,1),(1,0),(1,1)]
def box(a, r):
    p=np.pad(a,((r+1,r),(r+1,r)),mode='reflect')
    c=p.cumsum(0); c=c[2*r+1:]-c[:-2*r-1]
    c=c.cumsum(1); c=c[:,2*r+1:]-c[:,:-2*r-1]
    return c/((2*r+1)**2)
def bp(a):
    b1=box(a,1); b2=box(b1,3)
    return b1-b2
def gmap(raw, r):
    x=raw.astype(np.float64)
    Hs=[bp(x[pr::2,pc::2]) for c,(pr,pc) in NATIVE.items()]
    V=[box(h*h,r) for h in Hs]
    acc=0
    for a in range(4):
        for b in range(a+1,4):
            acc=acc+box(Hs[a]*Hs[b],r)/np.sqrt(V[a]*V[b]+1e-2)
    g=acc/6
    return np.repeat(np.repeat(g,2,0),2,1)
def feats(raw, g, K, ph, step=1):
    R=K//2; x=raw.astype(np.float64)-128.0
    v=swv(x,(K,K)); pr,pc=ph
    sub=v[(pr-R)%2::2*step, (pc-R)%2::2*step]
    r0=(pr-R)%2+R; c0=(pc-R)%2+R
    n0,n1=sub.shape[:2]
    X=sub.reshape(n0*n1,K*K)
    gg=g[r0:r0+2*n0*step:2*step, c0:c0+2*n1*step:2*step].reshape(-1,1)
    return np.hstack([X, X*gg, np.ones_like(gg), gg]), r0, c0, n0, n1
def accum(K, r, step=1):
    st={}
    for s in SCENES:
        raw,ref=load(s); g=gmap(raw,r)
        for ph in PH:
            X,r0,c0,n0,n1=feats(raw,g,K,ph,step)
            Y=ref[:, r0:r0+2*n0*step:2*step, c0:c0+2*n1*step:2*step].reshape(4,-1).T.astype(np.float64)-128.0
            st[(s,ph)]=(X.T@X, X.T@Y)
    return st
def solve(st, scenes, lam=1e-4):
    W={}
    for ph in PH:
        A=sum(st[(s,ph)][0] for s in scenes); b=sum(st[(s,ph)][1] for s in scenes)
        n=A.shape[0]; A=A+lam*np.trace(A)/n*np.eye(n)
        W[ph]=np.linalg.solve(A,b)
    return W
def apply(raw, W, K, r):
    R=K//2; H,Wd=raw.shape; P=R+(R%2)
    g=gmap(raw,r)
    p=np.pad(raw.astype(np.float64),P,mode='reflect')-128.0
    gp=np.pad(g,P,mode='reflect')
    out=np.empty((4,H,Wd))
    v=swv(p,(K,K))
    for ph in PH:
        pr,pc=ph
        sub=v[P-R+pr:P-R+pr+H:2, P-R+pc:P-R+pc+Wd:2]
        n0,n1=sub.shape[:2]
        X=sub.reshape(n0*n1,K*K)
        gg=g[pr::2,pc::2].reshape(-1,1)
        Y=X@W[ph][:K*K]+(X@W[ph][K*K:2*K*K])*gg+W[ph][2*K*K]+gg*W[ph][2*K*K+1]+128.0
        out[:,pr::2,pc::2]=Y.T.reshape(4,n0,n1)
    for c,(pr,pc) in NATIVE.items(): out[c,pr::2,pc::2]=raw[pr::2,pc::2]
    return out
if __name__=='__main__':
    for arg in sys.argv[1:]:
        K,r=map(int,arg.split(','))
        st=accum(K,r)
        errs=[]
        for s in SCENES:
            W=solve(st,[t for t in SCENES if t!=s])
            raw,ref=load(s); errs.append(score(apply(raw,W,K,r),ref)[1])
        Wall=solve(st,SCENES)
        tr=[score(apply(load(s)[0],Wall,K,r),load(s)[1])[1] for s in SCENES]
        print(f'K={K} r={r} LOO={np.round(errs,4)} mean={np.mean(errs):.5f} train={np.round(tr,4)} {np.mean(tr):.5f}', flush=True)
