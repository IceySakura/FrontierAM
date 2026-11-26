"""A within-line interpolant with an independent importance correction."""

import math
import numpy as np
from scipy import special

from numerics import powers, quantile


def _phi(x):
    return np.exp(-.5*x*x)/math.sqrt(2.*math.pi)


def _moments(l,r):
    m0 = np.where(l>0.,special.ndtr(-l)-special.ndtr(-r),special.ndtr(r)-special.ndtr(l))
    fl,fr = _phi(l),_phi(r)
    return m0,fl-fr,m0+l*fl-r*fr


def _line_rows(w,q,n,anchor,beta):
    d = w.d
    U = w.unit(n,d+1)
    xp = quantile(U[:,:d-1])
    v = xp@q[:,1:].T
    th = np.clip(anchor+v@beta,-1.6,1.75)
    t = np.column_stack([th-.49,th+.24,th+.87,th+1.51,np.maximum(th+2.52,3.02)])
    pts = (v[:,None,:]+t[:,:,None]*q[:,0]).reshape(-1,d)
    obs = w.query(pts).reshape(n,5)
    m0 = special.ndtr(t)
    v0 = .05*np.maximum(obs[:,:-1],obs[:,1:])
    v0 += .55*((obs[:,:-1]>0.)!=(obs[:,1:]>0.))*(obs[:,:-1]+obs[:,1:])
    pr = np.argmax(np.diff(m0,axis=1)*(v0+.006*np.max(obs,axis=1)[:,None]),axis=1)
    ir = np.arange(n)
    extra = quantile(.5*(m0[ir,pr]+m0[ir,pr+1]))
    new = w.query(v+extra[:,None]*q[:,0])
    tk = np.column_stack([t,extra])
    xr = np.argsort(tk,axis=1)
    t = np.take_along_axis(tk,xr,axis=1)
    obs = np.take_along_axis(np.column_stack([obs,new]),xr,axis=1)
    nk = t.shape[1]
    lo = np.column_stack([np.full(n,-8.),t])
    hi = np.column_stack([t,np.full(n,8.)])
    yl = np.column_stack([obs[:,0],obs])
    yh = np.column_stack([obs,obs[:,-1]])
    dx = np.maximum(.01,hi-lo)
    cuts_l,cuts_r = lo.copy(),hi.copy()
    c0 = np.zeros_like(lo)
    c1 = np.zeros_like(lo)
    c2 = np.zeros_like(lo)
    slopes=(obs[:,1:]-obs[:,:-1])/np.diff(t,axis=1)
    pos=(obs[:,1:]>0.)&(obs[:,:-1]>0.)
    sr = np.max(obs,axis=1)
    for j in range(1,nk):
        a,b,y0,y1 = lo[:,j],hi[:,j],yl[:,j],yh[:,j]
        both = (y0>0.)&(y1>0.)
        l1 = (y0==0.)&(y1>0.)
        r1 = (y0>0.)&(y1==0.)
        u = (y1-y0)/dx[:,j]
        cleft = np.zeros(n)
        cright = np.zeros(n)
        have_l,have_r = np.zeros(n,bool),np.zeros(n,bool)
        if j>1:
            have_l = pos[:,j-2]
            cleft = np.where(have_l,(u-slopes[:,j-2])/(b-t[:,j-2]),0.)
        if j<nk-1:
            have_r = pos[:,j]
            cright = np.where(have_r,(slopes[:,j]-u)/(t[:,j+1]-a),0.)
        cc = (cleft+cright)/np.maximum(1,have_l.astype(int)+have_r)
        # When an interval has only one positive endpoint, extrapolate the
        # adjacent positive slope if it predicts a plausible exercise point.
        mid = quantile(.5*(special.ndtr(a)+special.ndtr(b)))
        sp = slopes[:,j] if j<nk-1 else np.zeros(n)
        sl = slopes[:,j-2] if j>1 else np.zeros(n)
        pl = b-y1/np.maximum(sp,1.e-10)
        pr = a-y0/np.minimum(sl,-1.e-10)
        contl = have_r&(pl>=a-.17*dx[:,j])&(pl<=b-.04*dx[:,j])
        contr = have_l&(pr<=b+.17*dx[:,j])&(pr>=a+.04*dx[:,j])
        ll = np.where(contl,np.maximum(a,pl),mid)
        rr = np.where(contr,np.minimum(b,pr),mid)
        cuts_l[:,j] = np.where(l1,ll,a)
        cuts_r[:,j] = np.where(r1,rr,b)
        swp = y1/np.maximum(b-ll,.03)
        swl = -y0/np.maximum(rr-a,.03)
        speed_l = np.where(contl,swp,np.clip(sp,-.8*swp,.88*swp))
        speed_r = np.where(contr,swl,np.clip(sl,.88*swl,-.8*swl))
        cc = np.clip(cc,-np.inf,3.6*np.minimum(y0,y1)/dx[:,j]**2)
        c2[:,j] = both*cc
        c1[:,j] = both*(u-cc*(a+b)) + l1*speed_l+r1*speed_r
        c0[:,j] = both*(y0-u*a+cc*a*b) + l1*(y1-speed_l*b) + r1*(y0-speed_r*a)
    c0[:,0] = obs[:,0]
    up = np.where(pos[:,-1],np.maximum(slopes[:,-1],0.),0.)
    up = np.minimum(up,obs[:,-1]+.12*sr)
    c0[:,-1] = obs[:,-1]-up*t[:,-1]
    c1[:,-1] = up
    m0,m1,m2 = _moments(cuts_l,cuts_r)
    mean = np.sum(c0*m0+c1*m1+c2*m2,axis=1)
    prob = _moments(lo,hi)[0]
    jump = (yl>0.) != (yh>0.)
    risk = .015*sr[:,None] + (.055+.34*jump)*(yl+yh)
    risk[:,0] += .12*yl[:,0]
    risk[:,-1] += .36*yh[:,-1]
    mass = prob*(risk+1.e-15)
    mass /= mass.sum(axis=1)[:,None]
    mass = .42*prob+.58*mass
    choice = np.sum(U[:,d-1,None]>np.cumsum(mass,axis=1)[:,:-1],axis=1)
    idx = np.arange(n)
    u1 = special.ndtr(lo[idx,choice])+U[:,d]*prob[idx,choice]
    g = quantile(u1)
    pred = c0[idx,choice]+g*(c1[idx,choice]+g*c2[idx,choice])
    pred *= (g>=cuts_l[idx,choice])&(g<=cuts_r[idx,choice])
    y = w.query(v+g[:,None]*q[:,0])
    corr = (y-pred)*prob[idx,choice]/np.maximum(mass[idx,choice],1.e-19)
    return mean+corr,corr


def experiment(w,q,x,y,logit):
    if logit is None or w.remaining<7200 or np.mean(y>0.)<.085:
        return None
    b = float(logit[1:]@q[:,0])
    if b<.10:
        return None
    a = np.clip(-logit[0]/b,-1.26,1.7)
    beta = -logit[1:]/b
    beta *= min(1.,1.65/max(np.linalg.norm(beta - (beta@q[:,0])*q[:,0]),1.e-10))
    pilot,cr = _line_rows(w,q,96,a,beta)
    base = max(float(np.std(y)),1.e-9)
    if np.std(pilot) > .435*base or np.std(cr) > .175*base:
        return None
    total,k0 = 0.,0
    for n in powers(w.remaining//7):
        sol,_ = _line_rows(w,q,n,a,beta)
        total += np.sum(sol)
        k0 += n
    return float(total/k0)
