"""Analytically centered kernels for smooth Gaussian remainders."""

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from common import eigenbasis, gradient, moment_rotation, perp, unit


def _sqdist(x, y, a):
    return np.maximum(0., (x*x)@a[:,None]+((y*y)@a)[None,:]-2*(x*a)@y.T)


class SmoothModel:
    """A mixture of two Gaussian kernels, quadratic and linear kernels."""
    def __init__(self, X, y, scales, noise=8.e-5, grad_points=None, grad_values=None):
        self.X = np.asarray(X)
        self.sc = np.asarray(scales)
        self.linear = .30 / max(1.,np.sum(scales))
        self.quad = .35 / max(1.,np.sum(scales))**2
        self.c = np.mean(y)
        D = self.kernel(X, center=False)
        D.flat[::len(D)+1] += noise
        self.Xd = grad_points
        if grad_points is not None:
            cross = self.gradient_kernel(X,grad_points).reshape(len(X),-1)
            grad = self.mixed_kernel(grad_points).reshape(grad_points.size,-1)
            grad.flat[::len(grad)+1] += np.tile(np.maximum(1.e-6, noise*18.*self.sc),len(grad_points))
            D = np.block([[D,cross],[cross.T,grad]])
            y = np.r_[y,np.ravel(grad_values)+self.c]
        self.coef = cho_solve(cho_factor(D,overwrite_a=True,check_finite=False),
                             y-self.c, check_finite=False)
        self.coef_value = self.coef[:len(X)]
        self.mean = float(self.c+self.kernel_mean()@self.coef)

    def kernel_mean(self):
        Y = self.X
        ans = np.ones(len(Y))*2. + self.quad*((Y*Y)@(self.sc*self.sc))
        for g,k in ((1.,.82),(.25,.18)):
            a = g*self.sc
            ans += k*np.exp(-.5*(np.log1p(a).sum()+(Y*Y)@(a/(1+a))))
        if self.Xd is not None:
            A=self.Xd
            gd=2.*self.quad*A*(self.sc*self.sc)
            for g,k in ((1.,.82),(.25,.18)):
                a = g*self.sc
                cov = a/(1+a)
                mu = k*np.exp(-.5*(np.log1p(a).sum()+(A*A)@cov))
                gd -= mu[:,None]*A*cov
            ans=np.r_[ans,gd.ravel()]
        return ans

    def kernel(self, X, center=False):
        Y = self.X
        G = (X*self.sc)@Y.T
        norm = (X*X)@self.sc
        dist = np.maximum(0.,norm[:,None]+((Y*Y)@self.sc)[None,:]-2*G)
        ret = .82*np.exp(-.5*dist)+.18*np.exp(-.125*dist)
        ret += 2. + self.linear*G + self.quad*G*G
        if center:
            ret -= self.kernel_mean()[None,:len(Y)]
        return ret

    def gradient_kernel(self, X, Y):
        a = self.sc
        Z = X*a
        dot = Z@Y.T
        diff = (X[:,None,:]-Y[None,:,:])*a
        dist = _sqdist(X,Y,a)
        rf = .82*np.exp(-.5*dist)+.045*np.exp(-.125*dist)
        K = rf[:,:,None]*diff
        K += (self.linear+2*self.quad*dot)[:,:,None]*Z[:,None,:]
        return K

    def mixed_kernel(self, X):
        a = self.sc
        Z=X*a
        dot=Z@X.T
        diff=(X[:,None,:]-X[None,:,:])*a
        dist=_sqdist(X,X,a)
        rf1=.82*np.exp(-.5*dist)
        rf2=.18*np.exp(-.125*dist)
        K=-(rf1+.0625*rf2)[:,None,:,None]*(diff[:,:,:,None]*diff[:,:,None,:]).transpose(0,2,1,3)
        K+=2.*self.quad*(Z.T[None,:,:,None]*Z[:,None,None,:])
        diag=(rf1+.25*rf2+(self.linear+2*self.quad*dot))
        for j,s in enumerate(a):
            K[:,j,:,j]+=diag*s
        return K

    def __call__(self,x):
        x = np.atleast_2d(x)
        out = np.empty(len(x))
        b=1024 if self.Xd is None else 384
        for i in range(0,len(x),b):
            X=x[i:i+b]
            out[i:i+b] = self.c+self.kernel(X)@self.coef_value
            if self.Xd is not None:
                K=self.gradient_kernel(X,self.Xd).reshape(len(X),-1)
                out[i:i+b] += K@self.coef[len(self.X):]
        return out


def infer_frame(w, z, f, control, nmax=32, returning_gradients=False):
    y_sc = max(1.e-9,np.std(f))
    d = w.d
    n = min(nmax, max(0,w.left//(7*(d+1))), len(z))
    if n<3:
        fallback=moment_rotation(z,f),np.ones(d)
        return (*fallback,None,None) if returning_gradients else fallback
    z_g = z[w.rng.permutation(len(z))[:n]]
    vv, gg = gradient(w,z_g,h=.0014)
    E = np.eye(d)*.0014
    sub = control((z_g[:,None,:]+E).reshape(-1,d))
    sub0 = control(z_g)
    gg -= (np.broadcast_to(sub,(n*d,)).reshape(n,d)-np.broadcast_to(sub0,(n,))[:,None])/.0014
    # Crossed jumps are filtered by the upper tail of the gradient norms.
    norm = np.linalg.norm(gg,axis=1)
    keep = (norm < max(np.median(norm)*4.5,.00001)) & np.all(np.isfinite(gg),axis=1)
    if np.sum(keep)<3:
        fallback=moment_rotation(z,f),np.ones(d)
        return (*fallback,None,None) if returning_gradients else fallback
    gg = gg[keep]
    cov = gg.T@gg/len(gg)
    g = (f-f.mean())@z/len(f)
    cov += .22*np.outer(g,g)
    R = eigenbasis(cov)
    vals = np.maximum(0.,np.sum(R*(cov@R),axis=0))
    normed=vals/max(y_sc*y_sc, 1.e-18)
    return (R,normed,z_g[keep]@R,gg@R) if returning_gradients else (R,normed)


def smooth_integrate(w,x,f,ctrl):
    d = w.d
    R,importance,gpos,gval = infer_frame(w,x,f,ctrl,returning_gradients=True)
    if w.left<750:
        z=w.normal(w.left)@R.T
        return float(np.mean(w.ask(z)-ctrl(z)))
    nq = min(1400, max(300,w.left//6))
    more = max(0,nq-len(x))
    if more:
        xt = w.normal(more)@R.T
        x = np.vstack([x,xt])
        f = np.r_[f,w.ask(xt)-ctrl(xt)]
    z = x@R
    # Broad isotropic floor for the small directions of genuinely full-dimensional backgrounds.
    scales = np.clip(.033+1.20*importance, .035, .9)
    if d>=25:
        scales *= max(.45, min(1.,3.2/scales.sum()))
    use = w.rng.permutation(len(z))
    chk = use[:max(50,len(z)//8)]
    train = use[len(chk):]
    yvar = max(1.e-18,np.mean((f[chk]-np.mean(f[train]))**2))
    opts=[]
    for noise, sm in ((2.e-6, .47),(2.e-4,.85),(.012,.85)):
        s = np.maximum(.008,scales*sm)
        gp = SmoothModel(z[train],f[train],s,noise)
        pred = gp(z[chk])
        opts.append((np.mean((pred-f[chk])**2),noise,s,False))
    if gpos is not None:
        # Exclude gradient observations at held-out anchor coordinates.
        dist=_sqdist(gpos,z[chk],np.ones(d))
        use_g=np.min(dist,axis=1)>1.e-9
        for noise,sm in ((2.e-5,.45),(5.e-4,.82)):
            s = np.maximum(.008,scales*sm)
            try:
                gp=SmoothModel(z[train],f[train],s,noise,gpos[use_g],gval[use_g])
                err=np.mean((gp(z[chk])-f[chk])**2)
                opts.append((err,noise,s,True))
            except np.linalg.LinAlgError:
                pass
    err,noise,s,with_gradients=min(opts,key=lambda a:a[0])
    n = w.left
    U = w.normal(n) if n>=2 else w.rng.standard_normal((n,d))
    X = U@R.T
    res = w.ask(X)-ctrl(X)
    if err >= yvar*.96:
        return float(np.mean(res))
    gp = SmoothModel(z,f,s,noise,gpos,gval) if with_gradients else SmoothModel(z,f,s,noise)
    return float(gp.mean+np.mean(res-gp(U)))
