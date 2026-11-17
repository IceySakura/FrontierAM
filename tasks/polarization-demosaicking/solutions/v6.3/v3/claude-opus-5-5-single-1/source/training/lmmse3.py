import numpy as np, sys
from numpy.lib.stride_tricks import sliding_window_view as swv
from common import *
from lmmse2 import box, bp, PH
def gates(raw, r, r2, c0):
    x=raw.astype(np.float64)
    S=[x[pr::2,pc::2] for c,(pr,pc) in NATIVE.items()]
    Hs=[bp(s) for s in S]
    V=[box(h*h,r) for h in Hs]
    acc=0
    for a in range(4):
        for b in range(a+1,4):
            acc=acc+box(Hs[a]*Hs[b],r)/np.sqrt(V[a]*V[b]+1e-2)
    g=acc/6
    dev=S[0]+S[2]-S[1]-S[3]
    d=np.sqrt(box(dev*dev,r2)); h=d/(d+c0)
    up=lambda a: np.repeat(np.repeat(a,2,0),2,1)
    return [up(g), up(h)]
def build(raw, G, K, ph, sl=None):
    R=K//2; H,Wd=raw.shape; P=R+(R%2); pr,pc=ph
    p=np.pad(raw.astype(np.float64),P,mode='reflect')-128.0
    v=swv(p,(K,K))
    sub=v[P-R+pr:P-R+pr+H:2, P-R+pc:P-R+pc+Wd:2]
    n0,n1=sub.shape[:2]
    X=sub.reshape(n0*n1,K*K)
    gs=[g[pr::2,pc::2].reshape(-1,1) for g in G]
    F=[X]+[X*g for g in gs]+[np.ones_like(gs[0])]+gs
    return np.hstack(F), n0, n1
def run(K, r, r2, c0, lam=1e-4):
    st={}; Fs={}
    for s in SCENES:
        raw,ref=load(s); G=gates(raw,r,r2,c0)
        for ph in PH:
            F,n0,n1=build(raw,G,K,ph)
            Y=ref[:,ph[0]::2,ph[1]::2].reshape(4,-1).T.astype(np.float64)-128.0
            st[(s,ph)]=(F.T@F,F.T@Y)
    def solve(scenes):
        W={}
        for ph in PH:
            A=sum(st[(s,ph)][0] for s in scenes); b=sum(st[(s,ph)][1] for s in scenes)
            n=A.shape[0]; W[ph]=np.linalg.solve(A+lam*np.trace(A)/n*np.eye(n),b)
        return W
    def apply(raw,W):
        G=gates(raw,r,r2,c0); out=np.empty((4,)+raw.shape)
        for ph in PH:
            F,n0,n1=build(raw,G,K,ph); out[:,ph[0]::2,ph[1]::2]=(F@W[ph]+128.0).T.reshape(4,n0,n1)
        for c,(pr,pc) in NATIVE.items(): out[c,pr::2,pc::2]=raw[pr::2,pc::2]
        return out
    loo=[score(apply(load(s)[0],solve([t for t in SCENES if t!=s])),load(s)[1])[1] for s in SCENES]
    Wa=solve(SCENES); tr=[score(apply(load(s)[0],Wa),load(s)[1])[1] for s in SCENES]
    print(f'K={K} r={r} r2={r2} c0={c0} LOO={np.round(loo,4)} {np.mean(loo):.5f} train={np.round(tr,4)} {np.mean(tr):.5f}',flush=True)
    return Wa
if __name__=='__main__':
    for arg in sys.argv[1:]:
        K,r,r2,c0=map(float,arg.split(',')); run(int(K),int(r),int(r2),c0)
