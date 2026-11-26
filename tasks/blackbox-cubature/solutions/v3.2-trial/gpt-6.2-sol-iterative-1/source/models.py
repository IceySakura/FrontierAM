"""Low-degree control models, integrated without additional oracle calls."""

import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.linalg import eigh
from scipy.special import ndtr

from common import eigenbasis, normal_moments


class Quadratic:
    def __init__(self, c, b, A, drop=0.):
        self.c = float(c)
        self.b = np.asarray(b, dtype=float)
        self.A = .5*(np.asarray(A, dtype=float)+np.asarray(A, dtype=float).T)
        if drop:
            val, vec = eigh(self.A, check_finite=False)
            keep = np.abs(val)>max(1.e-12,drop*np.max(np.abs(val)))
            self.A = (vec[:, keep]*val[keep])@vec[:,keep].T

    def __call__(self, z):
        return self.c+z@self.b+np.sum((z@self.A)*z,axis=-1)

    def translated(self, a):
        return Quadratic(self(a),self.b+2*self.A@a,self.A)

    def scale(self, a):
        return Quadratic(a*self.c,a*self.b,a*self.A)

    def minus(self, other):
        return Quadratic(self.c-other.c,self.b-other.b,self.A-other.A)

    def plus(self, other):
        return self.minus(other.scale(-1.))

    def std(self):
        return np.sqrt(np.dot(self.b,self.b)+2.*np.sum(self.A**2))

    def mean(self):
        return self.c+np.trace(self.A)


def quadratic_law(q, mode='hinge'):
    """Positive-part mean or positive probability by inversion of a characteristic function."""
    lam,Q = eigh(q.A, check_finite=False)
    b = Q.T@q.b
    scl = q.std()
    if scl < 1.e-13:
        return max(q.c,0.) if mode == 'hinge' else float(q.c>0.)
    c = q.c/scl
    lam,b = lam/scl,b/scl
    zero = np.abs(lam)<2.e-10
    var = np.dot(b[zero],b[zero])
    lam = lam[~zero]
    b2 = b[~zero]**2
    nc = b2/(4*lam)
    offset = c-np.sum(nc)
    if var < 1.e-16:
        if np.all(lam>=0) and offset >= 0:
            return q.mean() if mode=='hinge' else 1.
        if np.all(lam<=0) and offset <= 0:
            return 0.

    def log_cf(t):
        s = 1.-2.j*t*lam
        return 1.j*t*c-.5*t*t*(var+np.sum(b2/s))-.5*np.log(s).sum()

    def log_tail(t):
        s = 1.-2.j*t*lam
        return 1.j*t*np.sum(nc/s)-.5*t*t*var-.5*np.log(s).sum()

    if mode=='hinge':
        fun = lambda t: -np.expm1(log_cf(t)).real/(t*t)
        power = 2
    else:
        fun = lambda t: np.exp(log_cf(t)).imag/t
        power = 1
    atol = 5.e-10
    cut = 12.
    if abs(offset)>10:
        cut = min(cut, 200/abs(offset))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        low = quad(fun,0.,cut,epsabs=atol,epsrel=2.e-9,limit=250)[0]
        tail_at_cut = np.exp(log_tail(cut))
        if abs(tail_at_cut)/cut**(power-1) < atol*.02:
            tail = 0.
        elif abs(offset) > .025:
            a = lambda t: np.exp(log_tail(t)).real/t**power
            b = lambda t: np.exp(log_tail(t)).imag/t**power
            if mode == 'hinge':
                tail = quad(a,cut,np.inf,weight='cos',wvar=offset,epsabs=atol,limit=120,limlst=130)[0]
                tail -= quad(b,cut,np.inf,weight='sin',wvar=offset,epsabs=atol,limit=120,limlst=130)[0]
            else:
                tail = quad(a,cut,np.inf,weight='sin',wvar=offset,epsabs=atol,limit=120,limlst=130)[0]
                tail += quad(b,cut,np.inf,weight='cos',wvar=offset,epsabs=atol,limit=120,limlst=130)[0]
        else:
            k = 1.e-7
            if mode=='hinge':
                tail = quad(lambda u: np.exp(log_cf(1/u)).real, k, 1./cut,
                            points=[.001, .005, .02] if cut<50 else None,epsabs=atol,limit=240)[0]
            else:
                tail = quad(lambda u: np.exp(log_cf(1/u)).imag/u, k, 1./cut,epsabs=atol,limit=280)[0]
    if mode == 'hinge':
        return max(0., .5*q.mean() + scl/math.pi*(low+1./cut-tail))
    return float(np.clip(.5+(low+tail)/math.pi,0.,1.))


def real_roots(a, b, c):
    """The two roots clipped to the practically relevant Gaussian range."""
    a = np.broadcast_to(a, np.shape(b))
    mask = np.abs(a) > 1.e-13
    cond = b*b-4*a*c
    q = -.5*(b+np.where(b<0,-1.,1.)*np.sqrt(np.maximum(0.,cond)))
    x1 = np.divide(q, a, out=np.full_like(b,9.), where=mask & (cond>0))
    x2 = np.divide(c, q, out=np.full_like(b,9.), where=mask & (cond>0)&(np.abs(q)>1.e-14))
    lin = (~mask)&(np.abs(b)>1.e-14)
    x1[lin]=-c[lin]/b[lin]
    return np.clip(x1,-9.,9.),np.clip(x2,-9.,9.)


class Control:
    """A selector or a signed sum of hinges, in ordinary spatial coordinates."""
    def __init__(self, forms, rule='sum', signs=None, sets=None):
        self.forms = list(forms)
        self.rule = rule
        self.signs = signs if signs is not None else [1.]*len(forms)
        self.sets = sets

    def __call__(self, X):
        val = [q(X) for q in self.forms]
        if self.rule=='sum':
            return sum(s*np.maximum(0.,a) for s,a in zip(self.signs,val))
        return self.select(val)

    def select(self, values):
        return np.maximum.reduce([np.minimum.reduce([values[k] for k in s]) for s in self.sets])

    def basis(self):
        D = len(self.forms[0].b)
        cov = np.zeros((D,D))
        for q in self.forms:
            cov += q.A@q.A+.4*np.outer(q.b,q.b)
        return eigenbasis(cov)

    def mean(self, w):
        if self.rule=='sum':
            return sum(s*quadratic_law(q,'hinge') for s,q in zip(self.signs,self.forms))
        # Gaussian preintegration of a min/max of quadratics: all switching points are roots.
        B = self.basis()
        sub = B[:,1:]
        v = B[:,0]
        params = [(float(v@q.A@v),sub.T@q.b,sub.T@q.A@sub,float(v@q.b),
                    2*(sub.T@q.A@v),q.c) for q in self.forms]
        size = 2**19 if w.d <= 32 else 2**17
        if len(self.forms)>3:
            size //= 2
        ans = 0.
        # Generate one net, in blocks, to retain its balance at the final length.
        from scipy.stats import qmc
        from common import gauss
        qr = qmc.Sobol(w.d-1, scramble=True, seed=w.rng, bits=30)
        for start in range(0,size,8192):
            if w.elapsed() > min(19.,w.limit*.64):
                raise ArithmeticError('time reserve in control integration')
            Y = gauss(qr.random(8192))
            A = [p[0] for p in params]
            b = np.array([p[3]+Y@p[4] for p in params])
            c = np.array([p[5]+Y@p[1]+np.sum((Y@p[2])*Y,axis=1) for p in params])
            roots = [np.full(len(Y),-9.),np.full(len(Y),9.)]
            for i in range(len(A)):
                for j in range(i):
                    roots.extend(real_roots(A[i]-A[j],b[i]-b[j],c[i]-c[j]))
            roots = np.sort(np.asarray(roots),axis=0)
            for lo,hi in zip(roots[:-1],roots[1:]):
                mid = (lo+hi)*.5
                vals = np.asarray(A)[:,None]*mid**2+b*mid+c
                best = self.select(vals)
                k = np.argmin(np.abs(vals-best),axis=0)
                m0,m1,m2 = normal_moments(lo,hi,2)
                ans += np.sum(np.asarray(A)[k]*m2+b[k,np.arange(len(Y))]*m1+c[k,np.arange(len(Y))]*m0)
        return ans/size
