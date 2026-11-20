"""Spectral fits with slowly varying polynomial amplitudes."""
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import spherical_jn

from common import Model, legmat, multiindices, solve_ls


def _linear_line(t,frequency,power,degree):
    p=[t**j for j in range(degree+1)]
    for a,m in zip(frequency,power):
        f=t*a
        co,si=np.cos(f),np.sin(f)
        for j in range(m+1):
            m0=t**j
            p.extend((m0*co,m0*si))
    return np.column_stack(p)


def _options(roots,t,y):
    found=[]
    for r in roots:
        use=r[(r.imag>1.15)&(abs(r.real)<.65)]
        if len(use)<2:
            continue
        for eps in (.045,.16,.36):
            groups=[]
            for z in sorted(use,key=lambda a:a.imag):
                if groups and z.imag-np.mean(groups[-1]).imag<eps:
                    groups[-1].append(z)
                else:
                    groups.append([z])
            if not any(len(g)>1 for g in groups) or not 1<=len(groups)<=5 or max(map(len,groups))>3:
                continue
            freq=np.array([np.mean(g).imag for g in groups])
            mod=np.array([len(g)-1 for g in groups])
            if any(np.std(g).real>.34 for g in groups) or len(y)*.44<3+2*np.sum(mod+1):
                continue
            A=_linear_line(t,freq,mod,2)
            c=solve_ls(A,y)
            e=np.std(A@c-y)/max(1e-200,np.std(y))
            if e>.012 or np.linalg.cond(A)>2e9:
                continue
            def fun(ff):
                XX=_linear_line(t,ff,mod,2)
                return XX@solve_ls(XX,y)-y
            opt=least_squares(fun,freq,max_nfev=21,x_scale="jac",ftol=6e-12,
                              xtol=5e-12,gtol=2e-13,
                              bounds=(np.maximum(.65,freq-.45),freq+.45))
            af=opt.x
            A=_linear_line(t,af,mod,2)
            e=np.std(opt.fun)/max(1e-200,np.std(y))
            weight=max(e,1e-13)*max(1.,np.linalg.cond(A)/1e5)**.14
            if e<8e-6:
                found.append((weight,mod,af))
    if not found:
        return None
    return min(found,key=lambda p:p[0])[1:]


def _wave_moments(w,idx):
    v=w/2.
    z=np.ones(idx.shape[1],dtype=complex)
    for j,row in enumerate(idx):
        for p in np.unique(row):
            z[row==p] *= (1j if v[j]>=0 else -1j)**int(p)*spherical_jn(int(p),abs(v[j]))
    return z


def modulated_spectrum(orc,line,x,y,scale,roots):
    center,v,t,raw=line
    u=raw/scale
    out=_options(roots,t,u)
    if out is None:
        return None
    mod,freq=out
    d=x.shape[1]
    n=min(43,max(21,2*int(np.sum(mod+1))+9))
    if orc.left < n*d+65:
        return None
    ids=np.rint(np.linspace(0,len(t)-1,n)).astype(int)
    t0=t[ids]
    gap=np.minimum(.026,.53*(1.-(center+.5*np.abs(v))))
    if np.min(gap)<.001:
        return None
    delta=np.eye(d)*gap
    pts=center+t0[None,:,None]*v+delta[:,None,:]
    sample=orc.sample(pts.reshape(-1,d)).reshape(d,n)/scale
    A=_linear_line(t0,freq,mod,2)
    obs=np.column_stack((u[ids],sample.T))
    C=solve_ls(A,obs)
    if np.max(np.abs(A@C-obs)) > .0014*np.ptp(u):
        return None
    p=3
    ww=[]
    for m,f in zip(mod,freq):
        k=p+2*m
        z=C[k]-1j*C[k+1]
        p+=2*(m+1)
        if np.min(np.abs(z))<1e-9*max(1e-210,np.std(u)):
            return None
        w=np.angle(z[1:]/z[0])/gap
        e=f-v@w
        alias=2*np.pi/(t[1]-t[0])
        e-=round(e/alias)*alias
        if abs(e)>1.5+.045*f:
            return None
        ww.append(w+(e/np.dot(v,v))*v)
    ww=np.asarray(ww)
    nt=int(.82*len(x))
    z=np.vstack((x[:nt],center+t[:,None]*v,pts.reshape(-1,d)))-.5
    vals=np.r_[y[:nt],u,sample.ravel()]
    best=None
    for degree in (0,1,2):
        i0=multiindices(d,degree,degree)
        idx=[multiindices(d,int(m),int(m)) for m in mod]
        p0=legmat(z+.5,i0)
        P=[legmat(z+.5,ix) for ix in idx]
        ncoef=p0.shape[1]+2*sum(k.shape[1] for k in P)
        if ncoef > .68*len(z):
            continue
        def design(z,w,p,H):
            U=z@w.T
            co,si=np.cos(U),np.sin(U)
            pf=[p]
            for j,q in enumerate(H):
                pf.extend((co[:,j,None]*q,si[:,j,None]*q))
            return np.column_stack(pf),co,si
        a=design(z,ww,p0,P)[0]
        c=solve_ls(a,vals)
        e=np.std(a@c-vals)
        if e > .13*np.std(vals):
            continue
        f=ww.copy()
        def fun(theta):
            c,f=theta[:ncoef],theta[ncoef:].reshape(-1,d)
            return design(z,f,p0,P)[0]@c-vals
        def jac(theta):
            c,f=theta[:ncoef],theta[ncoef:].reshape(-1,d)
            A,co,si=design(z,f,p0,P)
            p=p0.shape[1]
            ds=[]
            for j,mat in enumerate(P):
                n=mat.shape[1]
                grad=(-si[:,j]*(mat@c[p:p+n])+co[:,j]*(mat@c[p+n:p+2*n]))
                ds.append(grad[:,None]*z)
                p+=2*n
            return np.column_stack((A,*ds))
        if e>1.2e-12:
            opt=least_squares(fun,np.r_[c,f.ravel()],jac=jac,max_nfev=32,
                              x_scale="jac",ftol=4e-12,xtol=4e-12,gtol=2e-13)
            c,f=opt.x[:ncoef],opt.x[ncoef:].reshape(-1,d)
        def pred(x,c=c,f=f,i0=i0,idx=idx):
            return design(x-.5,f,legmat(x,i0),[legmat(x,ix) for ix in idx])[0]@c
        err=np.max(abs(pred(x[nt:])-y[nt:]))
        if (best is None or err<best.error) and err<.07*np.ptp(y):
            p=p0.shape[1]
            moment=[c[0]]
            for j,ix in enumerate(idx):
                U=_wave_moments(f[j],ix)
                n=len(U)
                moment.append(c[p:p+n]@U.real+c[p+n:p+2*n]@U.imag)
                p+=2*n
            best=Model(pred,float(math.fsum(moment)),"modulated spectrum",float(err))
        if err<1e-11:
            break
    return best
