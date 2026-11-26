"""Cross-fitted conditional-moment controls for normal probability strata."""
import numpy as np
from scipy import linalg, special

from numerics import balanced_blocks, log


def folded_strata(ctx, R, model, plan, mass, cuts, root):
    d=ctx.d
    u0=special.ndtr(cuts)
    latent=[]
    val=[]
    ident=[]
    grp=[]
    ww=[]
    count=[0,0]
    for i,num in enumerate(plan):
        if num<=0 or mass[i]<=0:
            continue
        blocks=balanced_blocks(num,d,ctx.rng)
        n=sum(map(len,blocks))
        for block in blocks:
            j=int(count[1]<count[0])
            count[j]+=len(block)
            p=u0[i]+mass[i]*special.ndtr(block[:,0])
            block[:,0]=special.ndtri(np.clip(p,1e-15,1-1e-15))
            latent.append(block)
            ident.append(np.full(len(block),i,dtype=int))
            grp.append(np.full(len(block),j,dtype=int))
            ww.append(np.full(len(block),mass[i]/n))
    Z=np.concatenate(latent)
    ident=np.concatenate(ident)
    grp=np.concatenate(grp)
    ww=np.concatenate(ww)
    X=Z@R.T
    r=ctx.query(X)-model(X)
    basic=float(ww@r)
    if min(count)<1200 or ctx.time_left()<4.8:
        return basic
    k=min(d-1,9)
    nl=min(d-1,31)
    i,j=np.triu_indices(k,1)
    nodes=root+np.array([.05,.46,.87,1.35,2.02,2.79,3.6])
    band=np.array([.29,.30,.31,.38,.49,.57,.78])
    sig=band/np.sqrt(1+band*band)
    mu=nodes/(1+band*band)
    p0=special.ndtr((cuts[:,None]-mu[None])/sig[None])
    base=p0[1:]-p0[:-1]
    S0=base/np.maximum(1e-14,mass[:,None])*(sig*np.exp(-.5*nodes*nodes/(1+band*band)))[None]
    u=Z[:,1:]
    basis=np.column_stack([np.ones(len(u)),u[:,:nl],
                           (u[:,:nl]**2-1)/np.sqrt(2),u[:,i]*u[:,j],
                           (u[:,:min(6,k)]**3-3*u[:,:min(6,k)])/np.sqrt(6)])
    H=np.exp(-.5*((Z[:,0,None]-nodes[None])/band[None])**2)
    p=basis.shape[1]
    F=np.concatenate([basis*H[:,j,None] for j in range(H.shape[1])],axis=1)
    adj=np.empty_like(r)
    penalty=np.full(p,.065)
    penalty[0]=.009
    penalty[1:nl+1]=.034
    penalty=np.tile(penalty,H.shape[1])
    test_var=[]
    for test in range(2):
        sel=grp!=test
        w=ww[sel]/ww[sel].mean()
        # Protect against extremely sparse groups and regression of long tails.
        w=np.clip(w,.10,6.)
        M=F[sel]
        A=M.T@(M*w[:,None])
        A.flat[::len(A)+1]+=penalty*len(M)/H.shape[1]
        try:
            c=linalg.solve(A,M.T@(w*r[sel]),assume_a="pos")
        except (ValueError,np.linalg.LinAlgError):
            return basic
        pred=F[~sel]@c
        cond=S0[ident[~sel]]@c[::p]
        adj[~sel]=pred-cond
        test_var.append(np.sum(ww[~sel]*(r[~sel]-pred)**2))
    raw_var=np.sum(ww*(r-basic)**2)
    if sum(test_var)>1.08*raw_var or not np.isfinite(adj).all():
        return basic
    ret=basic-.92*np.dot(ww,adj)
    log("cross-fitted strata",basic,ret,"CV ratio",np.sqrt(sum(test_var)/max(1e-30,raw_var)))
    return float(ret)
