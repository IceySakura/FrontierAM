"""A one-factor soft survival control learned from knockout observations."""

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import ndtr

from conditional import payoff_importance
from core import basis_from, gaussian, net_integral, unit


def survival_integral(K, c, b, w, h):
    norm=np.linalg.norm(b)
    e=b/norm
    cov=w@e
    sd=np.sqrt(max(1.,1+w@w-cov*cov))
    root=(np.log(K)-c)/norm
    nf=1/np.sqrt(2*np.pi)

    def tail(t, offset):
        if t>=10.:
            return 0.
        return nf*quad(lambda u: np.exp(-.5*u*u)*ndtr((offset+cov*u)/sd),
                        max(-10.,t),10.,epsabs=1.5e-12,limit=64)[0]

    return np.exp(c+.5*norm*norm)*tail(root-norm,h+w@b)-K*tail(root,h)


def barrier_control(f,X,Y,V,G,K,c,b,term,fallback):
    if f.remaining<10600:
        return None
    # The tilted pilot sees enough in-the-money paths to distinguish a
    # barrier failure from exercise, while the natural pilot diagnoses fit.
    Z=gaussian(640,f.d,f.rng)
    lead=Z[:,0].copy()
    Z[:,0]=1.02+.92*lead
    x=Z@V.T
    y=f.query(x)
    U=np.vstack((X,x))
    vals=np.r_[Y,y]
    pt=term(U)
    live=(pt>.38*max(1e-12,fallback))&(np.arange(len(U))>=len(X))
    status=vals[live]>0
    if np.sum(status)<24 or np.sum(~status)<12:
        return None
    A=np.column_stack((np.ones(np.sum(live)),U[live]))
    wgt=np.sqrt(np.clip(pt[live]/(3.*fallback+1e-12),.08,3.))
    v0=np.zeros(f.d+1)
    v0[0]=.8
    pen=1.7

    def ls(theta):
        x=A@theta
        p=ndtr(x)
        w=wgt*(p-status)
        return .5*(np.dot(p-status,w)+pen*np.dot(theta[1:],theta[1:])), (
            A.T@(w*np.exp(-.5*x*x)/math.sqrt(2*np.pi))+pen*np.r_[0.,theta[1:]])

    fit=minimize(ls,v0,jac=True,method="L-BFGS-B",
                 options={"maxiter":90,"ftol":1e-11}).x
    h,w=fit[0],fit[1:]
    if np.linalg.norm(w)<.12 or not np.all(np.isfinite(fit)):
        return None
    pro=pt*ndtr(U@w+h)
    keep=pt < np.quantile(pt,.993)
    q=pro-pt
    M=np.column_stack((pt,q))
    W=np.sqrt(1/(1+np.maximum(pt,0.)/(5.*fallback+1e-12)))
    K0=(M*(W*keep)[:,None]).T
    S=K0@(M*(W*keep)[:,None])
    S[1,1]+=.08*S[1,1]+1e-9
    u=np.linalg.solve(S+np.eye(2)*1e-9,K0@(vals*W*keep))
    coef=np.clip(u[0],.38,1.42)
    drop=np.clip(u[1],0.,coef)
    raw=pt[:len(X)]
    hat=coef*raw+drop*q[:len(X)]
    beta=np.clip(np.cov(Y,raw,ddof=0)[0,1]/max(1e-15,np.var(raw)),.0,1.2)
    old=np.clip(Y-beta*raw,-6*fallback,6*fallback)
    new=np.clip(Y-hat,-6*fallback,6*fallback)
    if drop<.12 or np.var(new)>.91*np.var(old):
        return None

    def proxy(z):
        return term(z)*(coef-drop+drop*ndtr(z@w+h))

    from core import linear_call_mean
    exact=(coef-drop)*linear_call_mean(c,b,K)+drop*survival_integral(K,c,b,w,h)
    if not np.isfinite(exact) or exact<0:
        return None
    v=V[:,0]
    vv=w-v*(w@v)
    S=G.T@G/max(1,len(G))
    S+=np.outer(unit(vv),unit(vv))*.068*np.trace(S)
    B=basis_from(S,v)
    est=payoff_importance(f,X,Y,B,proxy,exact,fallback)
    if est is not None:
        return est
    return net_integral(f,B,proxy,exact,fallback)
