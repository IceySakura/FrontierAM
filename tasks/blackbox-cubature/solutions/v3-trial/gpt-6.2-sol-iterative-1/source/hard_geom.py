"""Localizes piecewise-smooth interfaces and estimates their local models."""
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import integrate, linalg
from scipy.ndimage import maximum_filter1d
from scipy.special import chndtr, roots_legendre

from common import log, normalize


@dataclass
class Event:
    p: np.ndarray
    v: np.ndarray
    t: float
    width: float
    score: float
    d: np.ndarray = None
    base: float = None

    @property
    def z(self):
        return self.p+self.t*self.v


_OFF = np.linspace(.34, 1.54, 6)
_PL = np.linalg.inv(np.polynomial.polynomial.polyvander(-_OFF,5))
_PR = np.linalg.inv(np.polynomial.polynomial.polyvander(_OFF,5))


def jet_batch(s, es, widths):
    q0 = np.stack([e.z for e in es])
    v = np.stack([e.v for e in es])
    w = np.asarray(widths)
    q = q0[:,None,:]+(w[:,None]*np.r_[-_OFF,_OFF])[:,:,None]*v[:,None,:]
    y = s.query(q.reshape(-1,s.d)).reshape(len(es),12)
    l = y[:,:6]@_PL.T
    r = y[:,6:]@_PR.T
    return l,r


def scan(s, xs, y, extra=False):
    d = s.d
    n = (10 if d<=16 else 13) if not extra else 14
    v = normalize(s.rng.normal(size=(n,d)))
    ix = np.argsort(y)
    center = np.zeros((n,d))
    center[n//2:] = xs[s.rng.integers(len(xs)-1,size=n-n//2)]
    if not extra:
        center[-3:] = np.r_[xs[ix[-1:]],xs[ix[-7:-6]],xs[ix[:1]]]
    else:
        center[:4] = xs[rng_idx(s,ix,n=4)]
        v[:3] = normalize(xs[ix[-3:]]-xs[ix[:3]])
        center[-4:]=xs[ix[-4:]]
    H = .205 if d<=16 else .242
    npts = 83 if d<=16 else 85
    t = H*(np.arange(npts)-npts//2)
    f = s.query((center[:,None]+t[None,:,None]*v[:,None]).reshape(-1,d)).reshape(n,npts)
    js=[]
    ks=[]
    for p,w,yy in zip(center,v,f):
        delta = np.diff(yy)
        q = np.abs(delta)
        for j in np.argsort(-q)[:6]:
            ll = max(0,j-2)
            rr = min(j+3,len(delta))
            other = np.abs(np.r_[delta[ll:j],delta[j+1:rr]])
            if q[j]>.20 and q[j]>4.5*max(.008,np.max(other,initial=0.)):
                js.append((Event(p,w,.5*(t[j]+t[j+1]),H,q[j]),yy[j],yy[j+1]))
        dd = np.abs(np.diff(yy,4))
        dd0 = max(9e-6, 9.*np.median(dd))
        ii = np.flatnonzero((dd==maximum_filter1d(dd,size=7))&(dd>dd0))
        for j in ii:
            ks.append(Event(p,w,t[j+2],H,dd[j]))
    return js,ks,(center,v,t,f)


def rng_idx(s,ix,n):
    return ix[s.rng.choice(np.arange(max(0,len(ix)-70),len(ix)),size=n,replace=False)]


def localize_jumps(s, rec):
    if not rec:
        return []
    rec = sorted(rec,key=lambda r:r[0].score,reverse=True)[:28]
    es, yl, yr = zip(*rec)
    a,b = np.array([-e.width*.5 for e in es]),np.array([e.width*.5 for e in es])
    left = (np.asarray(yl)>np.asarray(yr))
    thresh = np.asarray(yl)*.5+np.asarray(yr)*.5
    X,V = np.stack([e.z for e in es]),np.stack([e.v for e in es])
    for _ in range(27):
        p = (a+b)*.5
        ok = (s.query(X+p[:,None]*V)>thresh) == left
        a,b = np.where(ok,p,a),np.where(ok,b,p)
    es = list(es)
    for e,c in zip(es,(a+b)*.5):
        e.t += c
    h=.018*np.ones(len(es))
    l,r=jet_batch(s,es,h)
    out=[]
    for e,ll,rr,hh in zip(es,l,r,h):
        D=(rr-ll)/(hh**np.arange(6))
        if abs(D[0])>.16 and abs(D[0])>.78*e.score:
            e.d=D
            e.base=min(ll[0],rr[0])
            out.append(e)
    return out


def localize_kinks(s, rec, take=55):
    es = sorted(rec,key=lambda e:e.score,reverse=True)[:take]
    if not es:
        return []
    hs = np.array([e.width*1.68 for e in es])
    seen = None
    for h in (hs, .053, .016, .008):
        if not es:
            break
        h = np.broadcast_to(h,(len(es),))
        l,r = jet_batch(s,es,h)
        new=[]
        for e,d,hh in zip(es,r-l,h):
            if abs(d[1]) < 1e-5*hh:
                continue
            u = -d[0]/d[1]
            if abs(u)>.62:
                continue
            for _ in range(4):
                p = np.polynomial.polynomial.polyval(u,d)
                q = np.polynomial.polynomial.polyval(u,d[1:]*np.arange(1,6))
                if abs(q)<5e-8 or abs(p/q)>.45:
                    break
                u -= p/q
            if abs(u)>.56:
                continue
            deriv = np.array([
                np.polynomial.polynomial.polyval(u,d[1:]*np.arange(1,6))/hh,
                np.polynomial.polynomial.polyval(u,d[2:]*np.array([1.,3.,6.,10.]))/hh**2,
                np.polynomial.polynomial.polyval(u,d[3:]*np.array([1.,4.,10.]))/hh**3])
            if seen is not None and abs(e.d[1]-deriv[0]) > .09*abs(deriv[0])+.001:
                continue
            e.t += hh*u
            e.d = np.r_[0.,deriv]
            new.append(e)
        seen = True
        es = new
    return [e for e in es if abs(e.d[1])>.006 and abs(e.d[3])<.018+.01*abs(e.d[2])]


class Quad:
    """x^T A x + b^T x + c, with an optional isotropic cap factorization."""
    def __init__(self,A,b,c):
        self.A,self.b,self.c = (A+A.T)*.5,b,float(c)
        self.cap=None

    def __call__(self,X):
        return np.einsum('ni,ni->n',X@self.A,X) + X@self.b + self.c

    def change(self,k):
        return Quad(self.A*k,self.b*k,self.c*k)

    def plus(self,p,k=1.):
        return Quad(self.A+k*p.A,self.b+k*p.b,self.c+k*p.c)

    def matches(self,e):
        g = (2*self.A@e.z+self.b)@e.v
        a = e.v@self.A@e.v
        r = abs(g-e.d[1])+abs(a-e.d[2])
        t = abs(g+e.d[1])+abs(a+e.d[2])
        return abs(self(e.z[None,:])[0])<.006+.002*np.linalg.norm(e.z) and min(r,t)<.0035+.012*abs(g)

    def as_cap(self, tol=.018):
        eig,U=linalg.eigh(self.A,check_finite=False)
        scale=max(1e-15,-eig[0])
        keep = eig<-.018*scale
        if np.max(eig)>tol*scale or not np.any(keep):
            return None
        k=int(np.sum(keep))
        a=-float(np.mean(eig[keep]))
        V=U[:,keep]
        b=V@(V.T@self.b)
        if np.ptp(eig[keep])>tol*a or np.linalg.norm(self.b-b)>tol*max(.07,np.linalg.norm(self.b)):
            return None
        cap=Quad(-a*(V@V.T),b,self.c)
        cap.cap=(a,cap.c+(b@b)/(4*a),float((b@b)/(4*a*a)),k,V)
        return cap


def diffs(s, z, v, X, h, order):
    # At the center these reproduce the difference of the extensions
    # (through order-1).  Paired extrapolants cancel the common terms.
    T=np.arange(1,order+1)
    cn=np.array([comb(order,j)*(-1.)**(j+1) for j in T])
    step = (h*T)[:,None]*v
    z = z+X
    y=s.query((z[:,None]+np.r_[step,-step]).reshape(-1,s.d)).reshape(-1,2*order)
    return (y[:,:order]-y[:,order:])@cn


def event_normal(s,e):
    d=s.d
    eps=.00033
    X = np.r_[np.zeros((1,d)),eps*np.eye(d)]
    f=diffs(s,e.z,e.v,X,.008,2)
    ns=(f[1:]-f[0])/eps
    if abs(ns@e.v-e.d[1])>.008+.03*abs(e.d[1]):
        f=diffs(s,e.z,e.v,.43*X,.012,2)
        ns=(f[1:]-f[0])/(eps*.43)
    if abs(ns@e.v-e.d[1])>.008+.03*abs(e.d[1]):
        return None
    return normalize(ns)*np.sign(e.d[1])


def approximate_quad(s,e, thin=False):
    d=s.d
    stlen=(d+1)*(d+2)//2
    min_cost=2*stlen if not thin or d<=16 else 24*d
    if s.remaining<min_cost+1850:
        return None
    n=event_normal(s,e)
    if n is None:
        return None
    h=.00023
    I=np.eye(d)*h
    if not thin or d<=16:
        ii,jj=np.triu_indices(d)
        st=np.r_[np.zeros((1,d)),I,I[ii]+I[jj]]
        if s.remaining<2*len(st)+1000:
            return None
        y=s.query(e.z+np.r_[st+2.1*h*n,st-2.1*h*n]).reshape(2,-1)

        def unpack(f,back):
            g=f[1:1+d]-f[0]
            H=np.empty((d,d))
            H[ii,jj]=H[jj,ii]=((f[1+d:]-f[0])-g[ii]-g[jj])/h**2
            return H,g/h-.5*h*H.diagonal()+H@back

        P,p=unpack(y[0],-2.1*h*n)
        M,m=unpack(y[1],2.1*h*n)
        A=(P-M)*.5
        g=p-m
        plus_first=np.linalg.norm(P)>=np.linalg.norm(M)
    else:
        # Grow a column sketch of the symmetric difference. Diagonals
        # and independent directions detect projections it did not see.
        off=2.1*h*n
        st=np.r_[np.zeros((1,d)),I,2.*I]
        yp,ym=s.query(e.z+np.r_[st+off,st-off]).reshape(2,-1)
        ap,am=yp[1:d+1]-yp[0],ym[1:d+1]-ym[0]
        dp=(yp[d+1:]-yp[0]-2*ap)/h**2
        dm=(ym[d+1:]-ym[0]-2*am)/h**2
        D=.5*(dp-dm)
        g=(ap-am)/h-h*D
        plus_first=np.linalg.norm(dp)>=np.linalg.norm(dm)

        dirs=normalize(s.rng.normal(size=(9,d)))
        ee=.0032
        T=np.r_[np.zeros((1,d)),ee*dirs,-ee*dirs]
        f=diffs(s,e.z,n,T,.0045,2)
        test=(f[1:10]+f[10:]-2*f[0])/(2*ee**2)
        perm=s.rng.permutation(d)
        S=np.zeros((d,d))
        np.fill_diagonal(S,D)
        ready=np.eye(d,dtype=bool)
        A=None
        last=0
        for cols in (min(d,8), min(d,17), min(d,29), min(d,42), d):
            if cols<=last:
                continue
            sel=perm[:cols]
            need=~ready[sel,:]
            J,K=np.where(need)
            J=sel[J]
            valid=(~np.isin(K,sel)) | (J<K)
            J,K=J[valid],K[valid]
            if s.remaining<2*len(J)+max(1850,9*d):
                break
            y=s.query(e.z+np.r_[I[J]+I[K]+off,I[J]+I[K]-off]).reshape(2,-1)
            C=.5*((y[0]-yp[0])-ap[J]-ap[K]-(y[1]-ym[0])+am[J]+am[K])/h**2
            S[J,K]=S[K,J]=C
            ready[J,K]=ready[K,J]=True
            last=cols
            if cols==d:
                A=S
                break
            u,l,_=linalg.svd(S[:,sel],full_matrices=False,check_finite=False)
            valid=l>max(1e-5,3e-4*l[0])
            rank=np.count_nonzero(valid)
            if not (1<=rank<cols):
                continue
            V=u[:,valid]
            R=linalg.lstsq(V[sel], (V.T@S[:,sel]).T,cond=1e-4,check_finite=False)[0]
            A0=V@((R+R.T)*.5)@V.T
            thresh=7e-6+8e-4*np.max(np.abs(D))
            if (np.max(np.abs(A0.diagonal()-D))<thresh
                and np.max(np.abs(np.einsum('ni,ni->n',dirs@A0,dirs)-test))<thresh):
                A=A0
                log("quadratic column sketch",cols,"of",d,"rank",rank)
                break
        if A is None:
            return None
        # The common-center Rayleigh stencil below estimates g directly.
    if (np.linalg.norm(A)>15. or abs(e.v@A@e.v-e.d[2])>.008+.032*abs(e.d[2])
        or abs(g@e.v-e.d[1])>.008+.03*abs(e.d[1])):
        return None
    eig,U=linalg.eigh(A,check_finite=False)
    # Re-estimate spectral diagonals and the jump gradient with a wider
    # common-center stencil.  Errors in the first eigenspaces then enter
    # the eigenvalues only to second order.
    eps=.0031
    X=np.r_[np.zeros((1,d)),eps*U.T,-eps*U.T]
    dz=diffs(s,e.z,n,X,.0045,2)
    diagonal=(dz[1:d+1]+dz[d+1:]-2*dz[0])/(2*eps**2)
    if np.max(np.abs(eig-diagonal))<.015*max(.1,np.max(np.abs(eig))):
        g=U@((dz[1:d+1]-dz[d+1:])/(2*eps))
        eig=diagonal
        eig[np.abs(eig)<2.5e-6]=0.
        # Equal eigenvalues occur for projected spheres and cones.
        if np.any(eig!=0):
            oo=np.argsort(eig)
            j=0
            while j<d:
                i=j+1
                while i<d and abs(eig[oo[i]]-eig[oo[j]])<max(2e-6,.00035*abs(eig[oo[j]])):
                    i+=1
                eig[oo[j:i]]=np.mean(eig[oo[j:i]])
                j=i
        A=(U*eig)@U.T
    q=Quad(A,g-2*A@e.z, e.z@A@e.z-g@e.z)
    if not plus_first:
        q=q.change(-1.)
    log("kink eigenvalues",np.around(eig,6),"grad",np.linalg.norm(g))
    return q if q.matches(e) else None


def hinge_mean(q):
    lam,U=linalg.eigh(q.A,check_finite=False)
    b2=(U.T@q.b)**2
    mean=lam.sum()+q.c
    var=2*(lam@lam)+b2.sum()
    sig=max(np.sqrt(var),.03)
    l=lam/sig
    b=b2/sig**2
    c=q.c/sig
    def F(t):
        w=1.-2j*l*t
        lf=(1j*t*c-.5*np.log(w).sum()-.5*t*t*np.sum(b/w))
        return float(-np.real(np.expm1(lf))/(t*t))
    T=2000.
    vv=1./T
    for left,right in zip([0.,1.,5.,20.,85.,360.],[1.,5.,20.,85.,360.,T]):
        vv+=integrate.quad(F,left,right,epsabs=5e-10,limit=420)[0]
    return float(.5*mean + sig*vv/np.pi)


def cap_pair_mean(q,r,as_max):
    a,c,n,k,U=q.cap
    b,e,m,j,V=r.cap
    def survival(t):
        p=chndtr(np.maximum((c-t)/a,0.),k,n)
        q=chndtr(np.maximum((e-t)/b,0.),j,m)
        return p+q-p*q if as_max else p+q
    v = 0.
    l = 0.
    s,w=roots_legendre(96)
    for h in sorted((max(0.,c),max(0.,e))):
        if h>l:
            v += .5*(h-l)*(w@survival(l+.5*(h-l)*(s+1.)))
        l=h
    return float(v)
