"""Defensive mixture fitting to weighted successful option paths."""
import numpy as np
from scipy import linalg, special

from numerics import log


def choose_mixture(x,y,train_density,p,base,log_density_rel):
    m,R,var,ess=p
    d=len(m)
    valid=np.arange(len(y))%4==0
    good=y>0
    if ess<79 or np.count_nonzero(good&valid)<65 or np.count_nonzero(good&~valid)<100:
        return None
    sz=min(d,5 if ess>150 else 3)
    wh=np.sqrt(var[:sz])
    X=((x[good&~valid]-m)@R[:,:sz])/wh
    place=(.82*X[:,0]+.37*X[:,1])
    W=y[good&~valid]**.92/np.maximum(1e-30,train_density[good&~valid])
    W=np.minimum(W,np.quantile(W,.94)*2.)
    W/=W.sum()
    eff=1/np.sum(W*W)
    if eff<56:
        return None
    parts=3 if eff<550 else 4
    ord=np.argsort(place)
    weight0=np.cumsum(W[ord])
    cc=np.argsort(ord)
    labels=np.minimum(parts-1,(weight0[cc]*parts).astype(int))
    weights=np.ones(parts)/parts
    means=np.zeros((parts,sz))
    C=np.zeros((parts,sz,sz))
    df=np.eye(sz)
    A=np.eye(parts)[labels]*.96+.04/parts
    for _ in range(16):
        WW=A*W[:,None]
        mass=WW.sum(axis=0)
        if np.min(mass)<.025:
            return None
        weights=.94*mass+.06/parts
        means=(WW.T@X)/mass[:,None]
        G=[]
        for k in range(parts):
            u=X-means[k]
            K=(u.T*(WW[:,k]/mass[k]))@u
            e,V=np.linalg.eigh(K)
            C[k]=(V*np.clip(.90*e+.075,.19,2.8))@V.T
            t=linalg.solve_triangular(linalg.cholesky(C[k],lower=True),u.T,lower=True)
            gd=np.log(weights[k])-.5*(np.linalg.slogdet(C[k])[1]+np.sum(t*t,axis=0))
            G.append(gd)
        L=np.array(G).T
        A=special.softmax(L,axis=1)
    new=[]
    for k in range(parts):
        aa=wh[:,None]*C[k]*wh[None,:]
        e,v=np.linalg.eigh(aa)
        rank=np.argsort(np.abs(e-1.))[::-1]
        v=v[:,rank]
        e=e[rank]
        rot=np.column_stack([R[:,:sz]@v,R[:,sz:]])
        new.append((m+R[:,:sz]@(wh*means[k]),rot,np.r_[e,var[sz:]]))
    wide=(m,R,np.maximum(.75,var)*1.04)
    old=[(m,R,var),wide,base]
    a=valid
    xx=x[a]
    value=y[a]**2/np.maximum(1e-30,train_density[a])
    r0=sum(f*np.exp(np.clip(log_density_rel(xx,q),-150,100)) for f,q in zip([.74,.20,.06],old))
    rm=sum(w*np.exp(np.clip(log_density_rel(xx,q),-150,100)) for w,q in zip(weights,new))
    rw=np.exp(np.clip(log_density_rel(xx,wide),-150,100))
    rp=np.exp(np.clip(log_density_rel(xx,old[0]),-150,100))
    best=None
    for mix in (.49,.68,.78):
        b0=.8*(.86-mix)
        b1=.96-mix-b0
        den=mix*rm+b0*rp+b1*rw+.04
        ra=(value/den).sum()/(value/r0).sum()
        if best is None or ra<best[0]:
            best=(ra,mix,b0,b1)
    ra,mix,b0,b1=best
    if ra>.875:
        return None
    log("rare mixture",parts,sz,eff,"validation second moment ratio",ra)
    return new+old,[*(mix*weights),b0,b1,.04]
