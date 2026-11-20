"""Inference and analytic integration of a localized periodic profile."""
import math

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import ive

from common import Model, solve_ls


def peak_moment(kappa,w,phase):
    n=max(32,int(14*math.sqrt(kappa))+6)
    k=np.arange(1,n+1)
    damp=np.prod(np.sinc(k[:,None]*w[None,:]/(2*np.pi)),axis=1)
    return float(ive(0,kappa)+2*np.dot(ive(k,kappa)*np.cos(k*phase),damp))


def _frequency_starts(x,v,grid,top):
    if len(x)<8:
        return []
    vv=v-np.mean(v)
    X=grid[:,None]*x
    c,s=np.cos(X),np.sin(X)
    c-=c.mean(axis=1)[:,None]
    s-=s.mean(axis=1)[:,None]
    r1,r2,r3=np.sum(c*c,axis=1),np.sum(s*s,axis=1),np.sum(c*s,axis=1)
    q1,q2=c@vv,s@vv
    cor=(r2*q1*q1+r1*q2*q2-2*r3*q1*q2)/np.maximum(1e-16,r1*r2-r3*r3)
    peaks=np.flatnonzero((cor[1:-1]>cor[:-2])&(cor[1:-1]>cor[2:]))+1
    ix=peaks[np.argsort(cor[peaks])[-top:]][::-1]
    out=[]
    for i in ix:
        def cost(w):
            u=w*x
            A=np.column_stack((np.ones(len(u)),np.cos(u),np.sin(u)))
            c=solve_ls(A,v)
            return float(np.std(A@c-v))
        opt=minimize_scalar(cost,method="bounded",bounds=(grid[i-1],grid[i+1]),
                            options={"xatol":1e-8})
        out.append(opt.x)
    return out


def periodic_peak_profile(t, y, rough=False):
    """Return offset, amplitude, concentration, angular frequency, phase."""
    span=np.ptp(y)
    if span<1e-13 or len(t)<30:
        return None
    dt=np.min(np.diff(t))
    ridges=np.count_nonzero(np.diff(np.sign(np.diff(y))))
    if ridges<4:
        return None
    width=np.ptp(t)
    domain=np.linspace(3.8/width,min(.992*np.pi/dt,330./width),490)
    up=y-np.median(y)
    guess_sign=1. if np.max(up)>=-np.min(up) else -1.
    result=[]
    for sign in (guess_sign,-guess_sign):
        yn=y*sign
        low=float(np.min(yn))
        q=(yn-low)/span
        frequencies=[]
        for offset in (0.,2e-7,.00032):
            v=q+offset
            sel=v>5e-8
            if np.sum(sel)<9:
                sel=v>4e-12
            logv=np.log(v[sel])
            starts=_frequency_starts(t[sel],logv,domain,2)
            frequencies.extend((g,offset,True) for g in starts)
        if not result:
            frequencies.extend((w,0.,False) for w in _frequency_starts(t,q,domain,4))
        ranking=[]
        for om,off,logged in frequencies:
            ang=om*t
            M=np.column_stack([np.ones(len(t)),np.cos(ang),np.sin(ang)])
            if logged:
                v=q+off
                sel=v>5e-8
                if np.sum(sel)<9:
                    sel=v>4e-12
                c=solve_ls(M[sel],np.log(v[sel]))
                K=np.clip(np.linalg.norm(c[1:]),.36,280.)
                A0=math.exp(float(np.clip(c[0]+K,-4.,1.5)))
                phi=np.arctan2(-c[2],c[1])
            else:
                c=solve_ls(M,q)
                K=2.
                phi=np.arctan2(-c[2],c[1])
                A0=1.
            pp=np.r_[-off,np.log(A0),np.log(K),om,phi]
            def shapes(p):
                u=p[3]*t+p[4]
                s,c=np.sin(u),np.cos(u)-1
                k=np.exp(p[2])
                g=np.exp(np.clip(p[1]+k*c,-745,5.))
                return g,s,c,k
            def fun(p):
                g,_,_,_=shapes(p)
                return p[0]+g-q
            ee=np.std(fun(pp))
            ranking.append((ee,pp,fun,shapes))
        ranking.sort(key=lambda a:a[0])
        for ee,start,fun,shapes in ranking[:6]:
            if ee>.25 and (result or not rough):
                break
            def jac(p):
                g,s,c,k=shapes(p)
                m=-k*s*g
                return np.column_stack((np.ones(len(t)),g,k*c*g,m*t,m))
            lo=np.r_[-2.,-5.,-3.,.75*start[3],-6*np.pi]
            hi=np.r_[2.,3.,6.3,1.25*start[3],6*np.pi]
            sol=least_squares(fun,start,jac=jac,bounds=(lo,hi),max_nfev=42,
                              ftol=5e-11,xtol=3e-12,gtol=2e-12,x_scale="jac")
            err=np.max(np.abs(sol.fun))
            if err < (.027 if rough else .003):
                v=sol.x
                result.append((err,sign*(low+v[0]*span),sign*span*np.exp(v[1]),
                               np.exp(v[2]),v[3],np.remainder(v[4]+np.pi,2*np.pi)-np.pi))
                if err<3e-11:
                    return result[-1]
    return min(result,key=lambda a:a[0]) if result else None


def local_periodic_peak(orc,x,y,scale):
    """Estimate a narrow periodic profile from its local logarithmic jets."""
    d=orc.d
    if orc.left < 8*d+80:
        return None
    mask=np.flatnonzero(np.all((x>.014)&(x<.986),axis=1))
    if len(mask)<10:
        return None
    y0=y-np.median(y)
    sg=1. if np.max(y0)>=-np.min(y0) else -1.
    u=y*sg
    a,b=np.min(u),np.max(u)
    spread=b-a
    if spread <= 1e-12 or np.median(u)-a > .44*spread:
        return None
    n=int(len(x)*.82)
    im=mask[mask<n]
    idx=im[np.argmax(u[im])]
    center=x[idx]
    def probes(h):
        ax=np.eye(d)
        q=center+ax[:,None,:]*(h*np.array([-2.,-1.,1.,2.]))[None,:,None]
        v=sg*orc.sample(q.reshape(-1,d)).reshape(d,4)/scale
        return v
    step=.0047
    data=probes(step)
    if np.min(data)-a < .0016*spread:
        step=.0018
        data=probes(step)
    estimates=[]
    offsets=[a]
    if a>0. and a<.025*spread:
        offsets.insert(0,0.)
    offsets += [a-4e-5*spread,a-.0014*spread]
    for off in offsets:
        if np.min(data)-off<2e-12*spread or u[idx]-off<=0.:
            continue
        logf=np.log(data-off)
        log0=np.log(u[idx]-off)
        left, lm, rp, right=logf.T-log0
        curvature=lm+rp
        j=np.argmax(np.abs(curvature))
        if curvature[j] >= -2e-6:
            continue
        # The centered second differences are a sinusoid at the same frequency.
        c=(left+right-2*(lm+rp))/(2.*np.minimum(curvature,-1e-13))
        if not -1.+1e-8<c[j]<1.-2e-7:
            continue
        aj=np.arccos(c[j])
        C=curvature[j]/(2*(c[j]-1.))
        S=-(rp[j]-lm[j])/(2*np.sin(aj))
        kap=np.hypot(C,S)
        if not .16<kap<580 or not .004<aj<1.3:
            continue
        ac=np.clip(1.+curvature/(2*C),-1.,1.)
        ang=np.arccos(ac)
        if abs(S)>1e-7*max(C,1.):
            sign=np.sign((lm-rp)/S)
            w=ang*sign/step
        else:
            if orc.left < 2*(d-1)+76:
                continue
            kk=np.flatnonzero(np.arange(d)!=j)
            v=np.eye(d)[kk]+np.eye(d)[j]
            vals=sg*orc.sample(np.vstack((center+step*v,center-step*v)))/scale-off
            if np.min(vals)<=0.:
                continue
            vals=np.log(vals).reshape(2,-1)
            cos_sum=np.clip(1.+(vals.sum(axis=0)-2*log0)/(2*C),-1.,1.)
            w=np.zeros(d)
            w[j]=aj/step
            w[kk]=(np.arccos(cos_sum)-aj)/step
        ph=np.arctan2(S,C)-(center-.5)@w
        logA=log0-C+kap
        if logA>8. or not np.all(np.isfinite(w)):
            continue
        pred=off+np.exp(logA+kap*(np.cos((x[:n]-.5)@w+ph)-1.))
        err=np.std(pred-u[:n])/max(float(np.std(u[:n])),1e-200)
        estimates.append((err,off,logA,kap,w,ph))
    if not estimates:
        return None
    old,of,logA,kap,w,ph=min(estimates,key=lambda s:s[0])
    if old>.21:
        return None
    xa,ya=orc.data()
    xx=np.vstack((x[:n],xa[len(x):]))-.5
    yy=np.r_[u[:n],sg*ya[len(x):]/scale]
    def shapes(z,q):
        ang=z@q[4:]+q[3]
        cc,ss=np.cos(ang)-1.,np.sin(ang)
        k=np.exp(q[2])
        e=np.exp(np.clip(q[1]+k*cc,-745,26.))
        return e,cc,ss,k
    def fun(q):
        return q[0]+shapes(xx,q)[0]-yy
    def jac(q):
        e,c,s,k=shapes(xx,q)
        a=-k*s*e
        return np.column_stack((np.ones(len(e)),e,k*c*e,a,a[:,None]*xx))
    low=np.r_[-np.inf,-80,-3.,np.full(d+1,-np.inf)]
    high=np.r_[np.inf,25.,6.5,np.full(d+1,np.inf)]
    sol=least_squares(fun,np.r_[of,logA,np.log(kap),ph,w],jac=jac,
                      bounds=(low,high),max_nfev=41,xtol=2e-13,ftol=2e-12,gtol=2e-13,
                      x_scale="jac")
    q=sol.x
    def predict(p):
        return sg*(q[0]+shapes(p-.5,q)[0])
    err=np.max(np.abs(predict(x[n:])-y[n:]))
    if err>.065*spread:
        return None
    value=sg*(q[0]+np.exp(q[1])*peak_moment(np.exp(q[2]),q[4:],q[3]))
    return Model(predict,float(value),"periodic peak jets",float(err))
