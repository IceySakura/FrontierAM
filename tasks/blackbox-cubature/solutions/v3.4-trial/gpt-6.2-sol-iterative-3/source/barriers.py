"""Defensive Gaussian mixtures for nonnegative payoffs with rare survival."""

import math
import numpy as np
from scipy import linalg, special

from numerics import powers, quadratic_features, quantile, ridge, rotation


class NormalProposal:
    def __init__(self, center, basis, scale):
        self.center, self.basis, self.scale = center,basis,scale
        self.logdet = np.log(scale).sum()

    def sample(self,z):
        return self.center+(z*self.scale)@self.basis.T

    def log_ratio(self,z):
        xs = ((z-self.center)@self.basis)/self.scale
        return .5*(np.sum(z*z,axis=1)-np.sum(xs*xs,axis=1))-self.logdet


def _logmix(z, props, weights):
    return special.logsumexp(np.column_stack([p.log_ratio(z)+math.log(weight)
                                           for p,weight in zip(props,weights) if weight>0.]),axis=1)


def _weighted_logit(x,y,weights,ridge_pen=.025):
    x = np.column_stack([np.ones(len(y)),x])
    reg = np.r_[.0003,np.full(x.shape[1]-1,ridge_pen)]
    weights = weights/weights.sum()
    frac = np.clip(weights @ y,.003,.997)
    c = np.zeros(x.shape[1])
    c[0] = np.log(frac/(1.-frac))
    for _ in range(11):
        p = special.expit(x@c)
        h = x.T@((weights*p*(1.-p))[:,None]*x)
        h.flat[::len(h)+1] += reg
        dw = linalg.solve(h,x.T@(weights*(y-p))-reg*c,assume_a="pos",check_finite=False)
        c += np.clip(dw,-2.3,2.3)
        if np.max(np.abs(dw))<1.e-5:
            break
    return c


def _feature(z,keep):
    sf = (np.sum(z*z,axis=1)-z.shape[1])/math.sqrt(2.*z.shape[1])
    t = z[:,0]
    return np.column_stack([quadratic_features(z,width=keep),sf, np.maximum(0.,t),
                            np.maximum(0.,t-.9),np.maximum(0.,-t-.6)])


def _balanced(n,rate):
    if rate<.03:
        return max(512, min(1024,n//10)), min(2048,n//5)
    if rate<.15:
        return min(2048,n//6), min(2048,n//5)
    return min(4096,n//4), min(2048,n//5)


def estimate(w,x,y,scales,basis,axis,mean,dev,rate):
    d = w.d
    std = NormalProposal(np.zeros(d),np.eye(d),np.ones(d))
    if np.linalg.norm(mean)<.04 and np.linalg.norm(dev-1.)<.04:
        good = y>0.
        weight = np.power(y[good],.66)
        weight /= max(1.e-90,weight.sum())
        z = x[good] @ basis
        mean = np.clip(.80*(weight @ z),-.95,1.25)
        mean[5:] *= .37
        rms = weight @ ((z-mean)**2)
        dev = np.sqrt(np.clip(.48+.52*rms,.51,1.32))
        dev[10:] = 1.
    p1 = NormalProposal(basis@mean,basis,dev)

    n = min(1536,w.remaining//6)
    xp = p1.sample(w.normal(n))
    yp = w.query(xp)
    zz, yy = np.vstack([x,xp]),np.r_[y,yp]
    old,ct = np.unique(scales,return_counts=True)
    base_props = [NormalProposal(np.zeros(d),np.eye(d),np.full(d,s)) for s in old]+[p1]
    base_weights = np.r_[ct,n] / len(zz)
    log_r = _logmix(zz,base_props,base_weights)
    hit = yy>0.
    if np.sum(hit)>=12:
        zh = zz[hit]
        ls = .89*np.log(yy[hit]) - log_r[hit]
        ls = np.minimum(ls, np.quantile(ls,.87)+1.37)
        weights = np.exp(ls-np.max(ls))
        weights /= weights.sum()
        neff = 1./(weights@weights)
        c = weights@zh
        cv = ((zh-c).T*weights)@(zh-c)
        oldcov = (basis*dev**2)@basis.T
        alpha = float(np.clip(neff/(neff+1.55*d+12),.12,.78))
        cov = alpha*cv + (1.-alpha)*oldcov
        c = alpha*c + (1.-alpha)*p1.center
        if rate>.18:
            c = c+.16*axis
        c *= min(1.,2.45/max(np.linalg.norm(c),1.e-100))
        vals, a = linalg.eigh(cov,check_finite=False)
        # No violent contraction in a weakly observed direction. Preserving
        # some Gaussian tails matters for the weighted residual.
        floor = .34 if rate<.095 else .49
        s2 = np.sqrt(np.clip(vals,floor,1.31))
        order = np.argsort(np.abs(np.log(s2))+.30*np.abs(a.T@axis))[::-1]
        p2 = NormalProposal(c,a[:,order],s2[order])
        rot = rotation((np.eye(d)-cov)@(np.eye(d)-cov)+.12*np.outer(c,c),axis)
    else:
        p2,rot = p1,basis

    n = w.remaining
    m0,m1 = _balanced(n,rate)
    counts = np.array([m0,m1,n-m0-m1])
    weights = counts/n
    props = [std, p1, p2]

    # A smooth exercised-payoff control. Its expectation is computed without
    # consuming point values, and checked using held-out pilot residuals.
    cv_ctrl = np.sum(hit)>=66
    if cv_ctrl:
        keep = min(9,d)
        a = zz @ rot
        feat = _feature(a,keep)
        med = float(np.median(yy[hit]))
        top = float(np.quantile(yy[hit],.98)*2.4)
        tax = np.exp(np.clip(-log_r,-32.,20.))
        fair = np.minimum(tax,np.quantile(tax,.88)*3.5)
        val = np.arange(len(zz)) % 7 == 0
        train = ~val
        jh = hit&train
        sc2 = (np.sum(a*a,axis=1)-d)/math.sqrt(2.*d)
        aw = np.column_stack([a[:,:min(18,d)],a[:,0]**2, sc2,
                             feat[:,d:(d+keep*(keep-1)//2)],
                             (a[:,:keep]**2-1.)*2.**-.5])
        cw = _weighted_logit(aw[train],hit[train],.45+.55*fair[train],.024)
        cb,bi = ridge(feat[jh],yy[jh],.042,weights=np.minimum(fair[jh]+.45,10.))

        def pred(z):
            a = z@rot
            b = _feature(a,keep)
            feats = np.column_stack([a[:,:min(18,d)],a[:,0]**2,
                                    (np.sum(a*a,axis=1)-d)/math.sqrt(2.*d),
                                    b[:,d:(d+keep*(keep-1)//2)],
                                    (a[:,:keep]**2-1.)*2.**-.5])
            return special.expit(cw[0]+feats@cw[1:])*np.clip(b@cb+bi,0.,top)

        zval,yval = zz[val],yy[val]
        rho = _logmix(zval,props,weights)
        vtar = np.exp(np.clip(-rho-log_r[val],-25.,22.))
        w0 = np.minimum(vtar,np.quantile(vtar,.91)*3.5)
        a0 = pred(zval)
        m2 = np.sum(w0*a0*a0)
        beta = np.clip(np.sum(w0*a0*yval)/max(m2,1.e-30),0.,1.6)
        cv_ctrl = m2>1.e-12 and np.sum(w0*(yval-beta*a0)**2)<.91*np.sum(w0*yval*yval)
        if cv_ctrl:
            outer = w.normal(1<<16) @ rot.T
            ex = beta*sum(pred(outer[j:j+4096]).sum() for j in range(0,len(outer),4096))/len(outer)
        else:
            ex = 0.
    else:
        ex = 0.

    total = 0.
    for p, k in zip(props,counts):
        for batch in powers(int(k),192):
            z = p.sample(w.normal(batch))
            obs = w.query(z)
            den = _logmix(z,props,weights)
            val = obs-beta*pred(z) if cv_ctrl else obs
            total += np.sum(val*np.exp(np.clip(-den,-65.,18.)))
    return float(ex+total/n)
