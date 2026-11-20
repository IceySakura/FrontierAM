"""Defensive importance cubature for concentrated residuals."""
import math

import numpy as np
from scipy.special import ndtr, ndtri

from common import stable_mean


class ProductProposal:
    def __init__(self,center,width,cauchy=False):
        self.center=np.clip(center,0.,1.)
        self.width=np.clip(width,.002,.55)
        self.cauchy=cauchy
        a,b=-self.center/self.width,(1.-self.center)/self.width
        if cauchy:
            self.low,self.high=np.arctan(a),np.arctan(b)
        else:
            self.low,self.high=ndtr(a),ndtr(b)
        self.delta=self.high-self.low
        self.normalization=math.prod(self.width*self.delta)

    def transform(self,p):
        q=self.low+self.delta*p
        v=np.tan(q) if self.cauchy else ndtri(np.clip(q,2e-300,1.-5e-16))
        return np.clip(self.center+self.width*v,0.,1.)

    def density(self,x):
        z=(x-self.center)/self.width
        if self.cauchy:
            return 1./(np.prod(1.+z*z,axis=1)*self.normalization)
        return np.exp(-.5*np.sum(z*z,axis=1))/self.normalization/(2*np.pi)**(x.shape[1]*.5)


def _empirical(x,v,raw):
    w=np.abs(v)**.72
    if w.sum()==0. or np.max(w)>.68*np.sum(w):
        if np.max(np.abs(v))==0.:
            return None
        i=np.argmax(np.abs(v))
        return [ProductProposal(x[i],np.full(x.shape[1],.18),True)],np.ones(1)
    w=w/np.sum(w)
    p=w@x
    s=np.sqrt(w@((x-p)**2)+.020**2)
    g=ProductProposal(p,np.maximum(s*.95,.043),False)
    # A broad-tailed companion at a high point limits the effect of a
    # weighted center located between two distinct small supports.
    t=x[np.argmax(np.abs(v))]
    g2=ProductProposal(t,np.maximum(s*.63,.043),True)
    return [g,g2],np.array([.66,.34])


def _model_proposal(model, d):
    a=model.peak_params
    k=len(a)
    if not 1 <= k<=5:
        return None
    center=a[:,1:d+1]
    size=np.exp(-.5*a[:,d+1:])
    rr="rational" in model.name
    wt=(a[:,0]+np.sum(np.log(np.minimum(size,.6)),axis=1))
    weight=np.exp((wt-np.max(wt))*.63)+.02
    return [ProductProposal(mu,ss*(1.10 if rr else .94),rr)
            for mu,ss in zip(center,size)],weight/weight.sum()


def focused_residual(orc,scale,x,y,models=()):
    """Use a proposal only if it improves a separate variance proxy."""
    if orc.left<245 or orc.d>10 or len(x)<80:
        return None
    m0=np.median(y)
    raw=y-m0
    if np.ptp(raw)<5e-15:
        return None
    idx=np.argsort(abs(raw))[len(raw)*94//100:]
    if np.sum(raw[idx]**2) < .48*np.sum(raw**2):
        return None
    n=len(x)*5//8
    choices=[]
    if np.max(np.abs(raw))>12.*max(1e-200,np.median(np.abs(raw))):
        vals=_empirical(x[:n],raw[:n],y[:n])
        if vals is not None:
            choices.append((None,raw,*vals))
    for mod in models:
        if not hasattr(mod,"peak_params"):
            continue
        r=y-mod.predict(x)
        if not np.isfinite(r).all():
            continue
        vals=_model_proposal(mod,orc.d)
        if vals is not None:
            choices.append((mod,r,*vals))
    pick=None
    reference=np.mean(raw[n:]**2)
    if reference<=1e-20*np.max(raw**2):
        return None
    for mod,rr,props,w in choices:
        if np.std(rr[:n])>.65*np.std(raw[:n]) and mod is not None:
            continue
        p=sum(ww*g.density(x[n:]) for g,ww in zip(props,w))
        score=float(np.mean(rr[n:]**2/(.4+.6*p))/reference)
        reduction=np.mean(rr[n:]**2/(.4+.6*p))/max(1e-290,np.mean(rr[n:]**2))
        if score<.49 and reduction<.75 and (pick is None or score<pick[0]):
            pick=score,mod,rr,props,w
    if pick is None:
        return None
    _,model,rr,props,weight=pick
    if model is not None:
        centers=np.array([g.center for g in props])
        seen=orc.sample(centers)/scale
        v=model.predict(centers)
        if np.max(np.abs(v-seen))>.35*max(np.max(np.abs(seen)),np.max(np.abs(y))):
            model=None
    cap=orc.left
    floor=(1<<int(.50*cap).bit_length()-1)
    # Reserve a uniform component even when the fitted tail is very small.
    n0=max(floor,cap//3)
    if n0>cap-2*len(props):
        return None
    remaining=cap-n0
    counts=np.floor(weight*remaining).astype(int)
    order=np.argsort(weight*remaining-counts)[::-1]
    counts[order[:remaining-int(counts.sum())]] += 1
    ref=orc.new_sobol(n0)
    pts=[ref]
    for g,c in zip(props,counts):
        if c:
            pts.append(g.transform(orc.new_sobol(int(c))))
    u=np.vstack(pts)
    fv=orc.sample(u)/scale
    density=np.full(len(u),n0/cap)
    for g,c in zip(props,counts):
        if c:
            density += (c/cap)*g.density(u)
    base=(np.ones(len(u))*m0 if model is None else model.predict(u))
    mean=m0 if model is None else model.integral
    vv=(fv-base)/density
    if not np.isfinite(vv).all():
        return float(np.mean(fv[:n0]))
    return float(mean+stable_mean(vv))
