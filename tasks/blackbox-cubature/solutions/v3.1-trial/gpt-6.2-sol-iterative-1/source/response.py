"""Projected first variation and a second-order radial residual."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, ndtri, roots_legendre

from cubature import norm_rows, uniform
from kinks import debug


def spectral_gradient(model,anchor,mean):
    """Derivative of P(q<0) w.r.t. A about the anchor, with b,c fixed."""
    lam,b,c,U=model.shifted(mean).spectrum()
    mu=U.T@(mean-anchor)
    scale=max(1e-12,np.sqrt(2*(lam@lam)+b@b))
    lam,b,c=lam/scale,b/scale,c/scale

    def slope(t):
        a=1-2*lam*t
        return c+np.sum(lam/a+t*b*b*(1-t*lam)/(a*a))

    s=-.10
    if slope(s)>0 and slope(-24.)<0:
        s=brentq(slope,-24.,s)
    x,w=roots_legendre(52)
    ends=[0.,6.,17.,43.,100.,260.]
    t=np.concatenate([.5*(l+h)+.5*(h-l)*x for l,h in zip(ends[:-1],ends[1:])])
    wt=np.concatenate([.5*(h-l)*w for l,h in zip(ends[:-1],ends[1:])])
    z=s+1j*t
    a=1-2*z[:,None]*lam
    b2=b*b
    mg=np.exp(z*c+(.5*(z*z)*(1/a@b2))-.5*np.log(a).sum(axis=1))
    cm=mu+z[:,None]*b/a
    w2=-mg*wt/(np.pi*scale)
    H=(cm.T@(w2[:,None]*cm)).real
    H.flat[::len(H)+1]+=(w2@(1/a)).real
    return U@H@U.T


def _directions(s,jump,model,anchor,width,c):
    H=spectral_gradient(model,anchor,jump.tilt)
    ev,U=np.linalg.eigh(H)
    k=s.d
    if not np.isfinite(H).all() or np.max(ev)>max(1e-5,.006*np.max(abs(ev))):
        return None
    V=U.T.copy()
    linear=model.b+2*model.A@anchor
    t=V@jump.tilt
    sg=np.where(abs(t)>1e-9,np.where(t<0,-1.,1.),np.where(V@linear<0,-1.,1.))
    V*=sg[:,None]
    aa=np.sum((V@model.A)*V,axis=1)
    bb=V@linear
    margin=np.maximum(.20*aa,10*width*np.trace(model.A)/k)
    l=np.maximum(1e-11,aa-margin)
    r=aa+margin

    def label(coeff):
        d=np.sqrt(bb*bb-4*c*coeff)
        t=np.where(bb<0,(d-bb)/(2*coeff),-2*c/np.maximum(d+bb,1e-27))
        z=anchor+t[:,None]*V
        return jump.indicator(z,s.query(z))

    lower,upper=~label(l),label(r)
    if not np.all(lower&upper):
        debug("response projections not bracketed",np.sum(~(lower&upper)))
        return None
    for _ in range(10):
        m=.5*(l+r)
        hit=label(m)
        l=np.where(hit,l,m)
        r=np.where(hit,m,r)
    return float(ev@(.5*(l+r)-aa))


def radial_response(s,rng,jump,model,anchor,width,nmax):
    """Small nonlinear radial residual after matching the slope functional."""
    d=s.d
    if nmax<12*d+1300 or width>.020:
        return None
    A=model.shifted(anchor)
    lam,b,c,U=A.spectrum()
    if abs(c+1)>.0005 or lam.min() < .18*np.mean(lam):
        return None
    w=jump.tilt-anchor
    if np.linalg.norm(w)>1.1:
        return None
    # Start by verifying that model-centred brackets find the expected jump.
    m=min(9,((nmax-12*d)//5).bit_length()-1)
    N=1<<m
    u=uniform(rng,m,d+1)
    x=norm_rows(ndtri(u[:,:d]))
    V=x@U.T
    a=(x*x)@lam
    b=x@b
    disc=np.sqrt(b*b-4*a*c)
    r0=-2*c/(b+disc)
    pr=V@w
    sh=.5*(w@w)
    norm=(1-.5*d)*np.log(2)-gammaln(.5*d)-sh

    def dens(r,ind=slice(None)):
        pr0=pr[ind]
        if r.ndim>pr0.ndim:
            pr0=pr0[:,None]
        return np.exp(np.clip(norm+(d-1)*np.log(r)-.5*r*r+pr0*r,-745,650))

    k0=dens(r0)/disc
    margin=np.clip(4.4*width,.008,.10)
    l,h=r0*(1-margin),r0*(1+margin)

    def label(r,i=slice(None)):
        z=anchor+r[:,None]*V[i]
        return jump.indicator(z,s.query(z))

    yes,no=label(l),~label(h)
    if np.mean(~(yes&no))>.017:
        debug("response initial radial rejection",np.mean(~(yes&no)))
        return None
    for factor in (2.5,6.):
        ii=np.flatnonzero(~yes)
        jj=np.flatnonzero(~no)
        if not len(ii)+len(jj):
            break
        l[ii]=r0[ii]*np.maximum(.2,1-factor*margin)
        h[jj]=r0[jj]*(1+factor*margin)
        yes[ii],no[jj]=label(l[ii],ii),~label(h[jj],jj)
    if not np.all(yes&no):
        return None
    linear=_directions(s,jump,model,anchor,width,c)
    if linear is None:
        return None
    for _ in range(2):
        r=.5*(l+h)
        hit=label(r)
        l=np.where(hit,r,l)
        h=np.where(hit,h,r)
    r=l+(h-l)*u[:,-1]
    y=label(r)
    W=dens(r)+k0*r0*r0*(b/(r*r)+2*c/(r*r*r))
    # Integrate the weighted Gaussian from r0 to l directly; doing so avoids
    # subtracting two nearly equal CDFs.
    xx,ww=roots_legendre(10)
    mid=.5*(r0+l);radius=.5*(l-r0)
    base=radius*(dens(mid[:,None]+radius[:,None]*xx)@ww)
    base-=k0*((a*l+b)*l+c)*(r0/l)**2
    corr=np.mean(base+(h-l)*y*W)
    debug("first variation",linear,"radial second order",corr,"N",N)
    return float(linear+corr)
