"""Gaussian product-spline controls for option payoffs."""

import itertools

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import ndtr,ndtri
from scipy.linalg import solve

from cubature import net_sizes,normals,ordered_basis,uniform
from kinks import debug
from payoff_axes import positive_directions
from rare import estimate_rare


def _powers(p, degree):
    return [a for k in range(1,degree+1) for a in
            itertools.combinations_with_replacement(range(p),k)]


class TensorControl:
    def __init__(self,basis,offset):
        self.basis = basis
        self.offset = offset
        self.dimension = len(basis)
        self.p = min(4,self.dimension-1)
        self.low,self.high = -4.6,5.3
        # Denser near exercise, coarser in small-probability tails.
        atlas = np.r_[-2.7,-1.75,-1.12,-.65,-.34,-.10,.16,.43,.78,1.26,1.93,2.75]+offset
        internal = np.unique(np.clip(atlas,self.low+.05,self.high-.05))
        self.knots=np.r_[np.repeat(self.low,4),internal,np.repeat(self.high,4)]
        self.knots_count=len(self.knots)-4
        self.ps=_powers(self.p,3)
        self.ps.extend(a for a in _powers(self.p,4) if len(a)==4 and max(a)<3)
        self.group = np.array([len(a) for a in self.ps])
        self.size1=1+len(self.ps)
        edges = np.unique(np.r_[self.low, internal,self.high])
        x,w = np.polynomial.legendre.leggauss(14)
        mids,ran = .5*(edges[1:]+edges[:-1]),.5*np.diff(edges)
        z = (mids[:,None]+ran[:,None]*x).ravel()
        mass = (ran[:,None]*w).ravel()*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        self.bmeans = mass @ self.t_basis(z)
        self.bmeans[0] += ndtr(self.low)*(1+.08*max(0.,self.low)**2)
        self.bmeans[-1] += ndtr(-self.high)*(1+.08*max(0.,self.high)**2)

    def t_basis(self,t):
        tc=np.clip(t,self.low,self.high)
        B = BSpline.design_matrix(tc,self.knots,3,extrapolate=True).toarray()
        return B*(1+.08*np.maximum(tc,0)**2)[:,None]

    def features(self,z):
        t=z[:,0]
        x=z[:,1:]
        B=self.t_basis(t)
        r=x[:,:self.p]
        u=2*ndtr(r)-1
        P=[np.ones_like(u),u]
        for j in range(2,5):
            P.append(((2*j-1)*u*P[-1]-(j-1)*P[-2])/j)
        H=[np.sqrt(2*j+1)*p for j,p in enumerate(P)]
        u=[np.ones(len(z))]
        for i in self.ps:
            it = np.bincount(i,minlength=self.p)
            v=np.ones(len(z))
            for rj in np.flatnonzero(it):
                v*=H[it[rj]][:,rj]
            u.append(v)
        phi=np.column_stack(u)
        top = (B[:,:,None]*phi[:,None,:]).reshape(len(z),-1)
        xrest=x[:,self.p:]
        dt=np.clip(t-self.offset,-2.5,4.)
        sig=1/(1+np.exp(np.clip(-6*dt,-40,40)))
        R=np.c_[np.ones(len(t)),.6*dt,.28*(dt*dt-1)]*sig[:,None]
        L=np.c_[x, xrest*H[1][:,0,None]*.65, .55*(x*x-1)/np.sqrt(2),
                H[1][:,0,None]*(r[:,1:] - .977*H[1][:,1:])*.4]
        tail=(R[:,:,None]*L[:,None,:]).reshape(len(z),-1)
        return np.c_[top,tail]

    def train(self,z,y,weights=None):
        n = len(z)
        phi = self.features(z)
        # Bound the leverage of rare polynomial rows without altering the
        # Gaussian expectation of the fitted predictor.
        if weights is None:
            gram = phi.T@phi
            rhs = phi.T@y
        else:
            w=np.clip(weights,.2,3.3)
            gram=phi.T@(w[:,None]*phi)
            rhs=phi.T@(w*y)
        q=self.knots_count
        diag = np.r_[np.tile(np.r_[.00004,np.where(self.group>3,.0014,.00034)],q),
                      np.full(phi.shape[1]-q*self.size1,.0022)]
        gram.flat[::len(gram)+1] += n*diag
        self.coef = solve(gram,rhs,assume_a="pos",check_finite=False)
        self.expect = self.bmeans@self.coef[:q*self.size1:self.size1]

    def __call__(self,z):
        return self.features(z)@self.coef


def _refined_direction(s,rng,v,g,z,y):
    # Extra gradient snapshots from usable paths limit the rotation error
    # when the minor eigenspectrum is active in the price.
    if len(z[y>0]) < 18 or s.remaining < 8500:
        return v,g
    from cubature import grad_batch
    idx=np.flatnonzero(y>0)
    p = rng.choice(idx, size=min(len(idx),20),replace=False)
    g2=grad_batch(s,z[p])
    norm=np.linalg.norm(g2,axis=1)
    ok=(norm>0)&(norm<8*max(1e-10,np.median(norm)))
    if np.any(ok):
        g=np.r_[g,g2[ok]/norm[ok,None]]
        v=g.mean(axis=0)
        v/=np.linalg.norm(v)
    return v,g


def residual_transport(t,err,defense=.24):
    cut=np.r_[-np.inf,-2.55,-1.85,np.arange(-1.50,2.61,.23),3.05,3.65,np.inf]
    grid=ndtr(cut)
    mass=np.diff(grid)
    bins=len(mass)
    b=np.clip(np.searchsorted(cut,t,side="right")-1,0,bins-1)
    n=np.bincount(b,minlength=bins)
    power=np.bincount(b,weights=err**2,minlength=bins)
    sd=np.sqrt((power+2*np.mean(err**2))/(n+2))
    # Smooth adjacent allocation estimates and retain full Gaussian support.
    sd=np.convolve(np.r_[sd[0],sd,sd[-1]],[.18,.64,.18],mode="valid")
    pr=(1-defense)*(mass*sd)/max(1e-15,mass@sd)+defense*mass
    return grid,pr/pr.sum()


def _crossfit(s,rng,basis,offset,pr):
    d=s.d
    grid,pr=pr
    mass=np.diff(grid)
    den=pr/mass
    den=np.r_[den[0],.5*(den[:-1]+den[1:]),den[-1]]
    pr=.5*(den[1:]+den[:-1])*mass
    den/=pr.sum()
    pr/=pr.sum()
    edges=np.r_[0.,np.cumsum(pr)]
    edges[-1]=1.
    slope=np.diff(den)/mass
    total=s.remaining
    n=1<<(total.bit_length()-1)
    if n>.77*total:
        n//=2

    def block(N):
        xx,yy,ww=[],[],[]
        for k in net_sizes(N):
            if s.elapsed>s.limit-4.5 and xx:
                break
            u=uniform(rng,k,d)
            b=np.searchsorted(edges,u[:,0],side="right")-1
            q=u[:,0]-edges[b]
            discr=np.maximum(1e-15,den[b]**2+2*slope[b]*q)
            step=2*q/(den[b]+np.sqrt(discr))
            u[:,0]=grid[b]+step
            w=1/(den[b]+step*slope[b])
            x=ndtri(np.clip(u,2.**-47,1.-2.**-47))
            xx.append(x)
            yy.append(s.query(x@basis.T))
            ww.append(w)
        return np.vstack(xx),np.concatenate(yy),np.concatenate(ww)

    a,y,w=block(n)
    if s.elapsed>s.limit-4.5:
        return float(np.mean(y*w))
    b,v,q=block(total-n)
    if min(len(y),len(v))<1024 or s.elapsed>s.limit-4.0:
        return (y@w+v@q)/(len(a)+len(b))
    f,g=TensorControl(basis,offset),TensorControl(basis,offset)
    f.train(a,y,w*w)
    g.train(b,v,q*q)
    r1=y-g(a)
    r2=v-f(b)
    debug("cross option residual",np.std(w*r1),np.std(q*r2))
    return (len(a)*g.expect+len(b)*f.expect+w@r1+q@r2)/(len(a)+len(b))


def regression_option(s,rng,zp,yp):
    fit=positive_directions(s,zp,yp,rng)
    if fit is None:
        return None,None
    v,g,z,y=fit
    basis=ordered_basis(g.T@g,v)
    pos=y>0
    if np.mean(pos)<.13:
        return estimate_rare(s,rng,v,g,z,y),basis
    if s.remaining<10000:
        return None,basis
    if np.mean(pos)>.24:
        v,g=_refined_direction(s,rng,v,g,z,y)
        basis=ordered_basis(g.T@g,v)
    off = ndtri(np.clip(np.mean(yp<=0),.03,.95))
    vX=normals(rng,9,s.d)
    vX[256:,0]=1.05+1.13*vX[256:,0]
    vY=s.query(vX@basis.T)
    t=np.r_[z@basis[:,0],vX[:,0]]
    f=np.r_[y,vY]
    # Use an exercise-aligned pilot proxy for the residual scale. A broad
    # component prevents over-allocating to the uncorrected far-right tail.
    pulse=np.exp(-.5*((t-off)/.60)**2)*np.sqrt(np.mean(f*f))*.34
    pr=residual_transport(t,np.hypot(.42*f,pulse),defense=.39)
    return _crossfit(s,rng,basis,off,pr),basis
