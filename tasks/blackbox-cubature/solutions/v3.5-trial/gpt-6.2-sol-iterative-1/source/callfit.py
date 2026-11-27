"""A global control for a continuous exercise onset and smooth payoff."""

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from core import fit_ridge, gaussian, net_integral


def transverse_features(w, nquad=8, ncubic=4):
    n,d=w.shape
    z=w[:,:min(d,nquad)]
    i,j=np.triu_indices(z.shape[1],1)
    quad=np.column_stack(((z*z-1)/np.sqrt(2),z[:,i]*z[:,j]))
    h=w[:,:min(d,ncubic)]
    h3=(h*h-3)*h/np.sqrt(6.)
    return quad,h3


def exercise_fit(f,V,K,c,b,n=128):
    v, U=V[:,0],V[:,1:]
    bv=b@v
    if bv < .025 or f.remaining<11000:
        return None
    w=gaussian(n,f.d-1,f.rng)
    z=w@U.T
    pred=np.clip((np.log(K)-c-z@b)/bv,-4.,4.)
    lo,hi=pred-.92,pred+1.27
    yl=f.query(z+lo[:,None]*v)
    yh=f.query(z+hi[:,None]*v)
    ok=(yl==0)&(yh>0)
    if ok.mean()<.86:
        return None
    for _ in range(5):
        t=(lo+hi)*.5
        yp=f.query(z+t[:,None]*v)
        condition=yp>0
        lo=np.where(condition,lo,t)
        hi=np.where(condition,t,hi)
        yh=np.where(condition,yp,yh)
    h=.051
    ys=f.query(z+(hi+h)[:,None]*v)
    slope=(ys-yh)/h
    rest=yh/np.maximum(1e-9,slope)
    jump=(rest>hi-lo+.020) | (slope<=0)
    keep=ok&(~jump)
    if keep.mean() < .86:
        return None
    root=(hi-np.clip(rest,0,hi-lo+.020))[keep]
    w=w[keep]
    return U,v,w,root


def features_r(w):
    q,h=transverse_features(w,9,5)
    return np.column_stack((np.ones(len(w)), w, q, h))


def calc_basis(w):
    qq,h3=transverse_features(w, 9, 5)
    n,d=w.shape
    p=np.column_stack((np.ones(n), w))
    s1=np.column_stack((qq, h3))
    return p,s1


def power_cols(t, offset, scale):
    v=(t-offset)/scale
    return np.column_stack((np.ones(len(t)),v,v*v,v**3,v**4))


def integral_powers(r, offset, scale):
    t=ndtr(-r)
    pdf=np.exp(-.5*r*r)/np.sqrt(2*np.pi)
    m=[t,pdf]
    for j in range(2,6):
        m.append(pdf*r**(j-1)+(j-1)*m[-2])
    rel=[m[k+1]-r*m[k] for k in range(5)]
    return np.column_stack((
        rel[0],(rel[1]-offset*rel[0])/scale,
        (rel[2]-2*offset*rel[1]+offset**2*rel[0])/scale**2,
        (rel[3]-3*offset*rel[2]+3*offset**2*rel[1]-offset**3*rel[0])/scale**3,
        (rel[4]-4*offset*rel[3]+6*offset**2*rel[2]-4*offset**3*rel[1]+offset**4*rel[0])/scale**4))


class GlobalExercise:
    def __init__(self,U,v,co,limit):
        self.U=U
        self.v=v
        self.co=co
        self.limit=limit
        self.offset=1.25
        self.scale=1.65
        self.par=None

    def root(self, w):
        return np.clip(features_r(w)@self.co,*self.limit)

    def design(self, t, w, integration=False):
        r=self.root(w)
        g,h=calc_basis(w)
        if integration:
            q=integral_powers(r,self.offset,self.scale)
        else:
            q=np.maximum(t-r,0)[:,None]*power_cols(t,self.offset,self.scale)
        # Output columns have exact normal moments in the leading coordinate.
        return np.column_stack([q] +
          [g[:,1:]*q[:,j,None] for j in range(3)] +
          [h*q[:,j,None] for j in range(2)])

    def __call__(self,X):
        return self.design(X@self.v,X@self.U)@self.par

    def expected(self,f):
        v=[]
        ww=[]
        N=8192
        for _ in range(2):
            sob=qmc.Sobol(self.U.shape[1],scramble=True,bits=32,rng=f.rng)
            a=[]
            for _ in range(16):
                if f.left()<8.5:
                    break
                g=ndtri(np.clip(sob.random(N),1e-13,1-1e-13))
                a.append(float(np.mean(self.design(None,g,True)@self.par)))
            if a:
                v.append(np.mean(a))
                ww.append(len(a))
        return np.average(v,weights=ww) if v else None


def fitted_call(f, X, Y, V, K, c, b):
    if f.remaining<11750:
        return None
    tc=exercise_fit(f,V,K,c,b,n=128)
    if tc is None:
        return None
    U,v,w,root=tc
    n=1024 if f.remaining>=12300 else 640
    # Independent tilt augments the pilot over the positive tail.
    W=gaussian(n,f.d,f.rng)
    W[:,0]=1.58+.97*W[:,0]
    XP=W@V.T
    YP=f.query(XP)
    X,Y=np.vstack((X,XP)),np.r_[Y,YP]
    rr=features_r(w)
    # The exercise regression is regularized separately from the payoff one.
    co=fit_ridge(rr,root,1.65)
    radius=np.clip(4.4*np.std(root),.32,1.65)
    med=np.median(root)
    model=GlobalExercise(U,v,co,(med-radius,med+radius))
    RX=model.design(X@v,X@U)
    sy=max(1e-10,np.mean(np.maximum(Y,0.)))
    weights=1/np.sqrt(1+.90*np.maximum(Y,0.)/sy)
    weights[:len(Y)-n]*=1.4
    # Weighting limits the extrapolation contribution of very large values.
    model.par=fit_ridge(RX*weights[:,None],Y*weights,1.48)
    pred=RX@model.par
    ferr=np.abs(pred-Y)
    if np.mean(np.minimum(ferr,4.*sy)) > .34*sy:
        return None
    mean=model.expected(f)
    if mean is None or not np.isfinite(mean) or abs(mean) > 8*sy or f.remaining<4096:
        return None
    return net_integral(f,V,model,mean,float(np.mean(Y[:len(Y)-n])))
