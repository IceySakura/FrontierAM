"""Active-coordinate cubature for nonnegative, partly zero payoffs."""
import math

import numpy as np
from scipy import linalg, optimize, special

from numerics import (Quad, balanced_rule, fd_gradient, hessian, log, normal,
                      orthogonal_to, sobol)


def unit(x):
    return x / max(1e-30, np.linalg.norm(x))


def logistic(x, y, ridge=.01, weights=None, maxiter=110):
    n, p = x.shape
    lam = np.full(p,ridge)
    lam[0] = ridge*.03
    wt = np.ones(n) if weights is None else weights/np.mean(weights)

    def fun(b):
        a = x@b
        loss = (wt@(np.logaddexp(0.,a)-y*a))/n+.5*(lam*b)@b
        grad = x.T@(wt*(special.expit(a)-y))/n+lam*b
        return loss, grad
    return optimize.minimize(fun,np.zeros(p),jac=True,method="L-BFGS-B",
                             options={"maxiter":maxiter,"ftol":1e-10}).x


def digital_root(ctx, z, v, lo, hi, it, check=True):
    lo = np.broadcast_to(lo,(len(z),)).copy()
    hi = np.broadcast_to(hi,(len(z),)).copy()
    if check:
        P = np.vstack([z+lo[:,None]*v, z+hi[:,None]*v])
        L, H = (ctx.query(P) > 0).reshape(2,-1)
        ok = ~L & H
    else:
        ok = np.ones(len(z), dtype=bool)
    for _ in range(it):
        a = .5*(lo+hi)
        t = ctx.query(z + a[:,None]*v) > 0
        lo = np.where(t,lo,a)
        hi = np.where(t,a,hi)
    return .5*(hi+lo), ok


def bin_heading(ctx, x, f):
    d = x.shape[1]
    p = f>0
    cols = np.column_stack((np.ones(len(x)),x))
    b = logistic(cols,p,ridge=.007,maxiter=100)
    v = unit(b[1:])
    if np.linalg.norm(v)==0:
        return None
    zero = np.zeros((1,d))
    tt, ok = digital_root(ctx,zero,v,-4.,4.,20)
    if not ok[0]:
        return v
    a = tt[0]*v
    h = .075
    off = h*np.eye(d)
    z = np.vstack([off,-off])+a
    slope, ok = digital_root(ctx,z,v,-.15,.15,10)
    if ok.mean() < .92:
        return v
    w = unit(-(slope[:d]-slope[d:])/(2*h))
    if w@v > .65:
        v = w
    return v


def active_fit(t, z, ridge=.06, rank=None, orientation=None):
    """Recover a few large directions of a scalar boundary height."""
    n,d = z.shape
    if rank is None:
        rank = min(8, d)
    X = np.column_stack([np.ones(n), z])
    c = np.linalg.lstsq(X,t,rcond=None)[0]
    e = t-X@c
    H = (z.T * e) @ z / n
    eig, V = np.linalg.eigh(H)
    O = V[:, np.argsort(-np.abs(eig))] if orientation is None else orientation
    u = z@O
    i,j = np.triu_indices(rank,1)

    def features(x):
        w=x@O
        return np.column_stack([np.ones(len(w)),w,(w*w-1)/np.sqrt(2),
                                w[:,i]*w[:,j],
                                (w[:,:rank]**3-3*w[:,:rank])/np.sqrt(6)])

    X=features(z)
    W=X.T@X
    penalty=np.full(X.shape[1],n*ridge)
    penalty[0]=1e-7
    penalty[1:1+d]*=.1
    penalty[1+d:1+2*d]*=.45
    W.flat[::len(W)+1]+=penalty
    c2=linalg.solve(W,X.T@t,assume_a="pos")
    r=t-X@c2
    # Pull back cross terms to improve ordering of quadratically active coordinates.
    A=np.diag(c2[1+d:1+2*d]/np.sqrt(2))
    A[i,j]=A[j,i]=.5*c2[1+2*d:1+2*d+len(i)]
    e2,V=np.linalg.eigh(A@A+.25*np.outer(c2[1:d+1],c2[1:d+1]))
    return lambda t:features(t)@c2, O@V[:,::-1], r


def boundary_curvature(ctx,v,R,t0,prior):
    d=R.shape[1]
    ncol=min(3,d)
    T=R@prior
    stencil,i,j=__import__("numerics").quad_stencil(d,ncol)
    z=.55*stencil@T.T
    tt,ok=digital_root(ctx,np.zeros((1,d+1)),v,t0-1.5,t0+1.5,17)
    if not ok[0]:
        return None
    t=tt[0]
    vals,good=digital_root(ctx,z,v,t-.26,t+.26,9)
    if good.mean()<.985:
        return None
    _,_,M=__import__("numerics").unpack_stencil(vals,.55,d,ncol,i,j)
    if np.linalg.norm(M)>.48 or np.linalg.norm(M)<.010:
        return None
    _,U=np.linalg.eigh(M@M.T)
    return prior@U[:,::-1]


def digital(ctx,x,y):
    d=ctx.d
    a=float(np.max(y))
    v=bin_heading(ctx,x,y)
    if v is None or ctx.remaining < 4800:
        return ctx.query(balanced_rule(ctx.remaining,d,ctx.rng)).mean()
    R=orthogonal_to(v)[:,1:]
    n=224 if d>=25 else 176
    z=normal(n,d-1,ctx.rng)
    t0=-special.ndtri(np.clip(np.mean(y>0),.003,.997))
    t,ok=digital_root(ctx,z@R.T,v,t0-1.8,t0+1.8,11)
    if ok.mean()<.94:
        log("digital direction declined",ok.mean())
        return ctx.query(balanced_rule(ctx.remaining,d,ctx.rng)@orthogonal_to(v).T).mean()
    keep=ok & (np.arange(n)%4 != 0)
    pred, O, res=active_fit(t[keep],z[keep],.095)
    test=ok & ~keep
    simple=np.std(t[test])
    residual=np.std((t-pred(z))[test])
    orient=None
    if ctx.d>=28 and simple>.041 and ctx.remaining>10500:
        kk=boundary_curvature(ctx,v,R,t0,O)
        if kk is not None:
            model,rotation,rs=active_fit(t[keep],z[keep],.094,orientation=kk)
            sd=np.std((t-model(z))[test])
            if sd<min(.935*residual,.975*simple):
                pred,O,res,orient,residual=model,rotation,rs,kk,sd
                log("digital boundary curvature accepted",simple,sd)
    if residual>1.06*simple:
        Rhat=R@O
        tavg=np.median(t[ok])
        pred=lambda z: np.full(len(z),tavg)
        err=np.abs(t[ok]-tavg)
    else:
        pred,O,res=active_fit(t[ok],z[ok],.095,orientation=orient)
        Rhat=R@O
        err=np.abs(res)
    qq=np.quantile(err,.88)
    step=float(np.clip(max(.009,qq*1.3,.95*min(simple,residual)),.009,.65))
    count=min(ctx.remaining//5,3200)
    Z=balanced_rule(count,d,ctx.rng)
    X=Z[:,:d-1]@Rhat.T
    tp=np.clip(pred(X@R),-6.,6.)
    hit=ctx.query(X+tp[:,None]*v)>0
    direction=np.where(hit,-1.,1.)
    a0=np.where(hit,-8.,tp)
    b0=np.where(hit,tp,8.)
    widths=(1.,2.7,7.5)
    for k in range(3):
        unresolved=(b0-a0)>1+step*widths[-1]
        vtry=np.clip(tp+direction*widths[k]*step,-7.75,7.75)
        t=np.where(unresolved,vtry,.5*(a0+b0))
        t=np.clip(t,a0,b0)
        hit=ctx.query(X+t[:,None]*v)>0
        a0=np.where(hit,a0,t)
        b0=np.where(hit,t,b0)
    u=special.ndtr(a0)
    w=special.ndtr(b0)-u
    r=special.ndtri(np.clip(u+w*special.ndtr(Z[:,-1]),1e-15,1-1e-15))
    hit=ctx.query(X+r[:,None]*v)>0
    val=special.ndtr(-b0)+w*hit
    cv=special.ndtr(-tp)
    base=[]
    for _ in range(3):
        U=normal(2**14,d-1,ctx.rng)
        base.append(np.mean(special.ndtr(-np.clip(pred(U@O),-6.,6.))))
    if ctx.remaining >= count:
        # Additional stratification of the worst unresolved brackets only.
        log("unused digital queries",ctx.remaining)
    log("digital fit",simple,residual,step,count,"residual mean and sd",
        np.mean(val-cv),np.std(val-cv))
    return a*(np.mean(base)+np.mean(val-cv))


def active_heading(ctx,x,y):
    d=ctx.d
    pos=np.flatnonzero(y>0.)
    if not len(pos):
        return None,None
    pos=pos[np.argsort(y[pos])]
    sel=pos[(np.linspace(.22,.94,min(8,len(pos)))*(len(pos)-1)).astype(int)]
    G=fd_gradient(ctx.query,x[sel],h=3e-5)
    L=np.linalg.norm(G,axis=1)
    # A finite difference crossing a knock-out surface is not a path derivative.
    ok=(L >0.) & (L < max(1.,np.median(L))*5)
    if not np.any(ok):
        return unit(np.mean(x[pos],axis=0)),None
    ctx.live_gradients = (x[sel[ok]].copy(), y[sel[ok]].copy(), G[ok].copy())
    gn=G[ok]/L[ok,None]
    v=unit(np.mean(gn,axis=0))
    if len(gn) > 1:
        _, S, O=linalg.svd(gn,full_matrices=False)
        vn=O[0]*np.sign(O[0]@v)
        v=unit(v+vn)
    return v,gn


def gaussian_proposal(x,y,v,weights=None):
    d=x.shape[1]
    sel=y>0
    if np.count_nonzero(sel)<3:
        return None
    W=np.sqrt(y[sel]/max(np.mean(y[sel]),1e-15))
    W=np.minimum(W,4.)
    if weights is not None:
        wx=weights[sel]
        W*=np.minimum(wx,12*np.median(wx)+1e-30)
    W/=W.sum()
    m=W@x[sel]
    ess=1/np.sum(W**2)
    m*=ess/(ess+d*.22)
    a=x[sel]-m
    C=(a.T*W)@a
    if v is not None and np.linalg.norm(v)>.5:
        R=orthogonal_to(v)
    else:
        R=np.eye(d)
    M=R.T@C@R
    B,V=np.linalg.eigh(M[1:,1:])
    dirorder=np.argsort(np.abs(np.log(np.maximum(.05,B))))[::-1]
    V=V[:,dirorder]
    B=B[dirorder]
    s0=np.clip(M[0,0]*ess/(ess+13)+.62*13/(ess+13),.12,1.7)
    blend=ess/(ess+max(75,2.2*d))
    vals=1+blend*np.clip(B-1,-.86,.55)
    # The low-covariance end is reliable only after the proposal has acquired
    # enough effective observations.
    if ess>85:
        dkeep=min(d-1,max(7,int(np.sqrt(ess))))
        vals[:dkeep]=1+min(.85,ess/(ess+35))*np.clip(B[:dkeep]-1,-.86,.50)
    U=np.column_stack([R[:,0],R[:,1:]@V])
    return m, U, np.r_[s0,vals], ess


def draw_proposal(n, center, R, var, rng):
    return center+(normal(n,len(center),rng)*np.sqrt(var))@R.T


def log_density_rel(x,p):
    m,R,var=p[:3]
    z=(x-m)@R
    return -.5*(np.log(var).sum() + np.sum(z*z/var,axis=1) - np.sum(x*x,axis=1))


def importance_mix(ctx, plist, fractions, model=None):
    n=ctx.remaining
    dd=ctx.d
    ns=[int(n*f) for f in fractions]
    ns[-1]=n-sum(ns[:-1])
    ret=0.
    for j,p in enumerate(plist):
        if ns[j]<=0:
            continue
        Z=balanced_rule(ns[j],dd,ctx.rng)
        m,R,var=p[:3]
        x=m+(Z*np.sqrt(var))@R.T
        y=ctx.query(x)
        mix=sum(n0/n*np.exp(np.clip(log_density_rel(x,pp),-700,650)) for n0,pp in zip(ns,plist))
        if model is not None:
            y=y-model(x)
        v=y/np.maximum(1e-80,mix)
        ret+=ns[j]/n*np.mean(v)
        log("proposal component",j,np.std(v),np.mean(v),len(v))
    return ret


def rare_option(ctx,x,y,v):
    d=ctx.d
    base=(np.zeros(d),np.eye(d) if v is None else orthogonal_to(v),np.ones(d))
    if np.count_nonzero(y)>6:
        initial=gaussian_proposal(x,y,v)
    else:
        initial=None
    if initial is not None:
        m,R,s,_=initial
        p0=(m,R,np.maximum(s,.64))
    else:
        p0=(np.zeros(d),base[1],np.full(d,.80))
    n=1024
    x0=draw_proposal(n,*p0,ctx.rng)
    y0=ctx.query(x0)
    if np.count_nonzero(y0)==0 and np.count_nonzero(y)==0:
        # Complement small scaling by a wider one; also protects rare OTM cases.
        t=normal(1536,d,ctx.rng)*1.22
        yt=ctx.query(t)
        x0,y0=np.vstack([x0,t]),np.r_[y0,yt]
    if v is None and np.count_nonzero(y0)>0:
        v,_=active_heading(ctx,x0,y0)
    if not np.any(y0>0):
        return importance_mix(ctx,[base,p0],[.65,.35])
    w0=np.exp(-np.clip(log_density_rel(x0,p0),-100,100))
    pp=gaussian_proposal(np.vstack([x,x0]),np.r_[y,y0],v,np.r_[np.ones(len(y)),w0])
    if pp is None:
        return importance_mix(ctx,[base,p0],[.67,.33])
    m,R,s,ess=pp
    n1=2048 if ctx.remaining>10500 else 1024
    p1=(m,R,s)
    x1=draw_proposal(n1,*p1,ctx.rng)
    y1=ctx.query(x1)
    xxx=np.vstack([x,x0,x1]); yyy=np.r_[y,y0,y1]
    mix0=(len(x)+len(x0)*np.exp(log_density_rel(xxx,p0))+
          len(x1)*np.exp(log_density_rel(xxx,p1)))/len(xxx)
    pp2=gaussian_proposal(xxx,yyy,v,1/np.maximum(1e-20,mix0))
    if pp2 is None:
        return importance_mix(ctx,[p1,p0,base],[.68,.22,.10])
    m,R,s,ess=pp2
    log("rare proposal",ess,np.linalg.norm(m),s[:12],np.mean(y1>0))
    if ctx.remaining>=7700:
        from mixfit import choose_mixture
        try:
            proposal=choose_mixture(xxx,yyy,mix0,pp2,base,log_density_rel)
            if proposal is not None:
                return importance_mix(ctx,*proposal)
        except (ValueError,np.linalg.LinAlgError) as exc:
            log("rich rare mixture declined",repr(exc))
    return importance_mix(ctx,[(m,R,s),(m,R,np.maximum(.75,s)*1.04),base],[.74,.20,.06])


def payoff_features(z, rank=7):
    n,d=z.shape
    u=np.clip(z,-6.6,6.6)
    s=u[:,0]
    v=u[:,1:rank+1]
    i,j=np.triu_indices(v.shape[1],1)
    return np.column_stack((np.ones(n),u,(u*u-1)/np.sqrt(2),s[:,None]*u[:,1:],
                            (s**3-3*s)/np.sqrt(6),
                            (s**4-6*s*s+3)/np.sqrt(24),
                            (s*s-1)[:,None]*v/np.sqrt(2),
                            s[:,None]*(v*v-1)/np.sqrt(2),
                            v[:,i]*v[:,j], (v**3-3*v)/np.sqrt(6)))


def fit_payoff(ctx,xx,yy,center,jet,R,positive,scale):
    d=len(center)
    f,g,H=jet
    u=xx@R
    XF=payoff_features(u,min(9,d-1))
    idx=np.arange(len(yy))
    keep=positive & (idx%4 != 0)
    test=positive & ~keep
    wt=np.clip(np.sqrt(yy[keep]/max(scale,1e-10)),.3,3.5)
    # Stronger shrinkage along the poorly conditioned directions.
    reg=np.full(XF.shape[1],.003)
    reg[:1+2*d]=.00075
    reg[0]=.000035
    X=XF[keep]
    G=X.T@(X*wt[:,None])
    G.flat[::len(G)+1]+=reg*len(X)
    cf=linalg.cho_factor(G)
    best=None
    for K in (3.,7.,15.,35.):
        K=scale*K
        fac=K/(K+f)
        jet0=Quad.at(.5*(fac*H-fac/(K+f)*np.outer(g,g)),fac*g,center,K*np.log1p(f/K))
        if jet0.sd()>50*max(scale,1e-7):
            jet0=Quad.at(np.zeros((d,d)),fac*g,center,K*np.log1p(f/K))
        j=jet0(xx)
        c=linalg.cho_solve(cf,X.T@(wt*(K*np.log1p(yy[keep]/K)-j[keep])))
        zhat=j+XF@c
        hi=max(4.0,np.log1p(120*scale/K))
        yy0=K*np.expm1(np.clip(zhat/K,0.,hi))
        err=np.sqrt(np.mean((yy0[test]-yy[test])**2)) if np.any(test) else np.inf
        if best is None or err<best[0]:
            best=err,K,jet0,c,hi,yy0
    err,K,q,c,hi,yp=best
    log("payoff train",K,err,scale, "active",np.count_nonzero(positive))

    def predict(x):
        j=q(x)
        a=payoff_features(x@R,min(9,d-1))@c+j
        return K*np.expm1(np.clip(a/K,0.,hi))
    return predict,yp


def fit_alive(x,y,yp,R,scale):
    pos=y>0.
    relevant=(yp > .06*scale) | pos
    fail=np.mean(~pos[relevant])
    if fail < .025:
        return None,R
    X=x[relevant]
    Y=pos[relevant]
    base=R[:,0]
    X1=X-(X@base)[:,None]*base
    v=unit(np.mean(X1[Y],axis=0)-np.mean(X1[~Y],axis=0))
    # Select directions useful for both the payoff and its domain.
    if np.linalg.norm(v)>.5:
        Rnew=orthogonal_to(np.column_stack([base,v,R[:,1:]] )[:,:len(base)])
    else:
        Rnew=R

    def features(x):
        u=x@Rnew
        v=u[:,:min(7,u.shape[1])]
        i,j=np.triu_indices(v.shape[1],1)
        return np.column_stack([np.ones(len(u)),u,(u*u-1)*.707,v[:,i]*v[:,j]])

    z=features(X)
    w=np.clip((yp[relevant]/scale+.3)**.3,.5,4.)
    c=logistic(z,Y,ridge=.015,weights=w)
    log("barrier classifier",fail,np.mean((z@c>0)==Y), len(Y))
    def gate(x):
        return special.expit(features(x)@c)
    gate.score=lambda x:features(x)@c
    return gate,Rnew


def smooth_option(ctx,x,y,v):
    d=ctx.d
    pos=np.flatnonzero(y>0)
    scale=float(np.median(y[pos]))
    ts=np.array([.35,.60,.85,1.1,1.35,1.6,1.95,2.35])
    zz=ts[:,None]*v
    yc=ctx.query(zz)
    ok=(yc>.2*scale)&(yc<2.75*scale)
    if ok.any():
        good=np.flatnonzero(ok)[0]
        center=zz[good]
    else:
        select=pos[(y[pos]>.35*scale)&(y[pos]<2.4*scale)]
        if not len(select):
            select=pos
        center=x[select[np.argmin(np.linalg.norm(x[select],axis=1))]]
    fc,g,H=hessian(ctx.query,center,scale=.0025)
    if not 0.<np.linalg.norm(g)<1e6 or np.linalg.norm(H,ord="fro") > 100*max(1.,scale):
        fc=ctx.query(center)[0]
        g=v*max(1.,scale*.80)
        H=np.zeros((d,d))
    if unit(g)@v>.7:
        v=unit(.35*v+.65*unit(g))
    O=orthogonal_to(v)
    U=O[:,1:]
    M=U.T@(H@H+.4*np.outer(g,g))@U
    _,W=np.linalg.eigh(M)
    R=np.column_stack([v,U@W[:,::-1]])
    nt=1536 if d<36 else 2048
    zt=normal(nt,d,ctx.rng)
    xt=zt@R.T
    yt=ctx.query(xt)
    xx=np.vstack([x,xt]); yy=np.r_[y,yt]
    pos=yy>0
    if np.count_nonzero(pos)<100:
        return rare_option(ctx,xx,yy,v)
    model,yp=fit_payoff(ctx,xx,yy,center,(fc,g,H),R,pos,scale)
    alive,R0=fit_alive(xx,yy,yp,R,scale)
    if alive is not None:
        pp=model
        def model(z):
            return pp(z)*alive(z)
    # Surrogate-only quadrature is cheap compared with oracle evaluations.
    e=[]
    for _ in range(4):
        z=normal(2**15,d,ctx.rng)@R0.T
        e.append(model(z).mean())
    ctx.last=float(np.mean(e))
    z=balanced_rule(ctx.remaining,d,ctx.rng)@R0.T
    q=ctx.query(z)-model(z)
    log("option residual",q.mean(),q.std(),len(q), "surrogate means",e)
    return ctx.last+q.mean()


def options(ctx,x,y):
    d=ctx.d
    npos=np.count_nonzero(y)
    if npos >= 2 and np.unique(y[y>0]).size == 1 and npos < len(y):
        return digital(ctx,x,y)
    if npos < 9 and ctx.remaining > 1200:
        x0=normal(640,d,ctx.rng)
        y0=ctx.query(x0)
        x,y=np.vstack([x,x0]),np.r_[y,y0]
        if np.count_nonzero(y)>=2 and np.unique(y[y>0]).size==1:
            return digital(ctx,x,y)
    v,gn=active_heading(ctx,x,y)
    p=np.mean(y>0)
    log("option branch",p,"dims",d)
    if p < .14 or v is None:
        return rare_option(ctx,x,y,v)
    if ctx.remaining>11300:
        from warp import integrate_warp
        try:
            return integrate_warp(ctx,x,y,v,gn)
        except (ValueError,np.linalg.LinAlgError) as exc:
            log("warp declined",repr(exc))
    if ctx.remaining<4100:
        return ctx.query(balanced_rule(ctx.remaining,d,ctx.rng)@orthogonal_to(v).T).mean()
    return smooth_option(ctx,x,y,v)
