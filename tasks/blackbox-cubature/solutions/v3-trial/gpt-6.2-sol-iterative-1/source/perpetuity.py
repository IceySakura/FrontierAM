"""Gaussian convolutions for one or two correlated lognormal path sums."""
import numpy as np
from scipy.fft import irfft, irfft2, next_fast_len, rfft, rfft2, rfftfreq, fftfreq
from scipy.special import ndtr, roots_legendre


class Axis:
    def __init__(self, dx):
        self.dx = float(dx)
        self.lo = -8.5
        self.n = int(np.ceil(18.5/dx)) + 1
        self.q = self.lo + self.dx*np.arange(self.n)
        self.offs = np.arange(6)
        self.denom = np.array([-120.,24.,-12.,12.,-24.,120.])

    def interp(self, q):
        pos = np.clip((q-self.lo)/self.dx, 0., self.n-1.)
        start = np.clip(np.floor(pos).astype(int)-2, 0, self.n-6)
        ids = start[..., None]+self.offs
        f = pos[..., None]-ids
        # Excluded products avoid a division at grid nodes.
        w = np.empty(f.shape)
        for j in range(6):
            w[...,j] = np.prod(f[...,self.offs!=j],axis=-1)/self.denom[j]
        return ids, w

    def grid(self, j):
        pad = int(np.ceil((8.0/np.sqrt(j))/self.dx))+4
        n = next_fast_len(self.n+2*pad, real=True)
        ext = self.lo+self.dx*(np.arange(n)-pad)
        return ext, pad, n


def centers(m, drift):
    return np.logaddexp.accumulate(np.arange(1,m+1).reshape((-1,)+(1,)*np.ndim(drift))*drift, axis=0)


def pull(ext, j, cs, vol, mu):
    u = cs[j-1]-mu + (np.sqrt(j)*vol)*ext
    v = u + np.log(-np.expm1(-np.maximum(u,1e-300)))
    return (v-cs[j-2])/(np.sqrt(j-1)*vol)


def cdf1(m, vol, mu, dx):
    axis = Axis(dx)
    cs = centers(m, mu)
    F = ndtr(axis.q)
    for j in range(2,m+1):
        ext, pad, n = axis.grid(j)
        ids,w = axis.interp(pull(ext,j,cs,vol,mu))
        u = np.einsum('ni,ni->n',F[ids],w)
        filt = np.exp(-0.5/j*(2*np.pi*rfftfreq(n,axis.dx))**2)
        F = irfft(rfft(u)*filt,n=n)[pad:pad+axis.n]
    return axis, cs[-1], np.sqrt(m)*vol, F


def bivariate_initial(q, rho):
    v = ndtr(q)
    F = np.multiply.outer(v,v)
    if rho == 0:
        return F
    xs, ws = roots_legendre(32)
    a = 0.5*np.arcsin(rho)
    q2s = q[:,None]**2+q[None,:]**2
    qq = np.multiply.outer(q,q)
    for t,w in zip(a*(xs+1), (a/(2*np.pi))*ws):
        F += w*np.exp(-(q2s-2*np.sin(t)*qq)/(2*np.cos(t)**2))
    return F


def cdf2(m, vol, mu, rho, dx):
    axis = Axis(dx)
    cs = centers(m, mu)
    F = bivariate_initial(axis.q, rho)
    for j in range(2,m+1):
        ext, pad, n = axis.grid(j)
        ii, wi = axis.interp(pull(ext,j,cs[:,0],vol[0],mu[0]))
        jj, wj = axis.interp(pull(ext,j,cs[:,1],vol[1],mu[1]))
        u = F[ii[:,0],:] * wi[:,0,None]
        for k in range(1,6):
            u += F[ii[:,k],:]*wi[:,k,None]
        v = np.ascontiguousarray(u[:,jj[:,0]]*wj[:,0])
        for k in range(1,6):
            v += u[:,jj[:,k]]*wj[:,k]
        a = 2*np.pi*fftfreq(n,axis.dx)
        b = 2*np.pi*rfftfreq(n,axis.dx)
        filt = np.exp(-0.5/j*(a[:,None]**2+2*rho*a[:,None]*b[None,:]+b[None,:]**2))
        F = irfft2(rfft2(v)*filt,s=(n,n))[pad:pad+axis.n,pad:pad+axis.n].copy()
    return axis, cs[-1], np.sqrt(m)*vol, F


def single_price(m, vol, mu, scale, strike, digital=False, dx=.011):
    if vol <= 0. or strike <= 0. or scale <= 0.:
        return float('nan')
    axis, c, sigma, F = cdf1(m,vol,mu,dx)
    a = (np.log(strike/scale)-c)/sigma
    if digital:
        ii,w = axis.interp(a)
        return float(np.clip(1.0-(w@F[ii]),0.,1.))
    b = axis.q[-1]
    if a >= b:
        return 0.
    if a < axis.q[0]:
        q = np.arange(1,m+1)
        return max(0.,float(scale*np.exp(q*(mu+0.5*vol*vol)).sum()-strike))
    xs,ws = roots_legendre(160)
    q = a+(xs+1)*(.5*(b-a))
    ii,w = axis.interp(q)
    surv = np.maximum(1.-np.einsum('ni,ni->n',F[ii],w),0.)
    return float(.5*(b-a)*sigma*scale*np.dot(ws, np.exp(c+sigma*q)*surv))


def pair_price(m, vol, mu, scale, rho, strike, rainbow, dx=None):
    if dx is None:
        dx=max(.024,min(.034,.063*np.sqrt(1.0-abs(rho))))
    axis, c, sig, F = cdf2(m,vol,mu,rho,dx)
    base = c+np.log(scale)

    def at(x,y,surv=False):
        i, wi = axis.interp((x-base[0])/sig[0])
        j, wj = axis.interp((y-base[1])/sig[1])
        z = np.einsum('ni,nij,nj->n',wi,F[i[:,:,None],j[:,None,:]],wj)
        if surv:
            z += 1.-np.einsum('ni,ni->n',wi,F[i,-1])-np.einsum('ni,ni->n',wj,F[-1,j])
        return np.clip(z,0.,1.)

    if rainbow:
        lo = np.log(strike)
        hi = float(np.min(base+sig*axis.q[-1]))
        if hi <= lo:
            return 0.
        x, w = roots_legendre(180)
        a = lo+(x+1)*(.5*(hi-lo))
        return float(.5*(hi-lo)*(w@(np.exp(a)*at(a,a,True))))
    # The put is the integral of the bivariate CDF along a segment.
    nodes, w = roots_legendre(224)
    p = (nodes+1)*.5
    f = at(np.log(strike*p), np.log(strike*(1-p)))
    day = np.arange(1,m+1)[:,None]
    forward = np.sum(scale*np.exp(day*(mu+0.5*vol**2)))
    return float(forward-strike+0.5*strike*(w@f))
