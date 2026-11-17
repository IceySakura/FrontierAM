import numpy as np, sys
from numpy.lib.stride_tricks import sliding_window_view as swv
from common import *
from lmmse3 import gates
from lmmse2 import PH
def gfeat(g,h,mode):
    if mode=='A': return [g,h]
    return [g,h,g*g,h*h,g*h]
def build(raw, G, K, ph, mode):
    R=K//2; H,Wd=raw.shape; P=R+(R%2); pr,pc=ph
    p=np.pad(raw.astype(np.float64),P,mode='reflect')-128.0
    v=swv(p,(K,K)); sub=v[P-R+pr:P-R+pr+H:2, P-R+pc:P-R+pc+Wd:2]
    n0,n1=sub.shape[:2]; X=sub.reshape(n0*n1,K*K)
    gs=[a[pr::2,pc::2].reshape(-1,1) for a in gfeat(G[0],G[1],mode)]
    return np.hstack([X]+[X*a for a in gs]+[np.ones_like(gs[0])]+gs), n0, n1
def run(K, mode, lam=1e-4):
    st={}; cacheF={}
    for s in SCENES:
        raw,ref=load(s); G=gates(raw,8,4,5.0)
        for ph in PH:
            F,n0,n1=build(raw,G,K,ph,mode)
            Y=ref[:,ph[0]::2,ph[1]::2].reshape(4,-1).T.astype(np.float64)-128.0
            half=n0*n1//2
            st[(s,0,ph)]=(F[:half].T@F[:half],F[:half].T@Y[:half])
            st[(s,1,ph)]=(F[half:].T@F[half:],F[half:].T@Y[half:])
    def solve(keys):
        W={}
        for ph in PH:
            A=sum(st[k+(ph,)][0] for k in keys); b=sum(st[k+(ph,)][1] for k in keys)
            n=A.shape[0]; W[ph]=np.linalg.solve(A+lam*np.trace(A)/n*np.eye(n),b)
        return W
    def apply(raw,W):
        G=gates(raw,8,4,5.0); out=np.empty((4,)+raw.shape)
        for ph in PH:
            F,n0,n1=build(raw,G,K,ph,mode); out[:,ph[0]::2,ph[1]::2]=(F@W[ph]+128.0).T.reshape(4,n0,n1)
        for c,(pr,pc) in NATIVE.items(): out[c,pr::2,pc::2]=raw[pr::2,pc::2]
        return np.clip(out,0,255)
    allk=[(s,hh) for s in SCENES for hh in (0,1)]
    loo=[score(apply(load(s)[0],solve([k for k in allk if k[0]!=s])),load(s)[1])[1] for s in SCENES[1:]]
    W=solve([k for k in allk if k!=('43',1)]); raw,ref=load('43')
    o=apply(raw,W); h43=score(o[:,512:],ref[:,512:])[1]
    Wa=solve(allk); tr=[score(apply(load(s)[0],Wa),load(s)[1])[1] for s in SCENES]
    print(f'K={K} mode={mode} LOOcons={np.round(loo,4)} {np.mean(loo):.5f} 43half={h43:.4f} train={np.round(tr,4)} {np.mean(tr):.5f}',flush=True)
    return Wa
if __name__=='__main__':
    for arg in sys.argv[1:]:
        K,mode=arg.split(','); run(int(K),mode)
