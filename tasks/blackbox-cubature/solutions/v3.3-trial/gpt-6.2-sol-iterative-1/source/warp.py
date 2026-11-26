"""Payoff warping and stratified residual cubature for smooth terminal branches."""
import math

import numpy as np
from scipy import integrate, interpolate, linalg, special

from numerics import (balanced_rule, fd_gradient, log, normal, orthogonal_to,
                      quad_stencil, unpack_stencil)


def axis_curve(ctx, v, scale):
    d=len(v)
    ts=np.r_[np.linspace(-5.,-2.,13)[:-1],np.linspace(-2.,3.8,59),np.linspace(3.9,7.,24)]
    ys=ctx.query(ts[:,None]*v)
    if ys[-1] <= max(.1*scale,np.max(ys)*.975) or np.count_nonzero(ys>0)<18:
        raise ValueError("axis is not a one-sided terminal curve")
    active=np.flatnonzero(ys>0)
    if active[0]<=0 or active[0] > len(ts)-20 or np.min(ys[active[0]:])<=0:
        raise ValueError("axis has no bracketed first transition")
    i=active[0]
    lo,hi=ts[i-1],ts[i]
    yhi=ys[i]
    for _ in range(11):
        p=.5*(lo+hi)
        y0=ctx.query(p*v)[0]
        if y0>0:
            hi,yhi=p,y0
        else:
            lo=p
    root=.5*(hi+lo)
    if yhi > .05*max(1e-12,scale):
        raise ValueError("axis has a sizeable knock-out jump")
    yr=ys[active]
    tr=ts[active]
    diff=np.diff(yr)
    if np.count_nonzero(diff <= 0)>3 or np.min(diff)<-.02*scale:
        raise ValueError("axis payoff does not support an inverse")
    keep=np.r_[True,np.diff(np.maximum.accumulate(yr))>1e-7*scale]
    yr,tr=yr[keep],tr[keep]
    a=interpolate.PchipInterpolator(np.r_[root,tr],np.r_[0.,yr],extrapolate=True)
    inv=interpolate.PchipInterpolator(np.r_[0.,yr],np.r_[root,tr],extrapolate=True)
    der=a.derivative()
    last_slope=max(.04*yr[-1],min(yr[-1],float(der(tr[-1]))))

    def f(t):
        t=np.asarray(t)
        u=np.clip(t,root,tr[-1])
        out=a(u)
        mask=t>tr[-1]
        if np.any(mask):
            out[mask]=yr[-1]*np.exp(np.minimum(3.,(t[mask]-tr[-1])*last_slope/yr[-1]))
        return np.maximum(0.,out)

    def g(y):
        y=np.asarray(y)
        u=np.clip(y,0.,yr[-1])
        out=inv(u)
        mask=y>yr[-1]
        if np.any(mask):
            out[mask]=tr[-1]+np.log(y[mask]/yr[-1])*yr[-1]/last_slope
        return out
    return root,f,g,der


def warp_rotation(ctx,v,gn,root,der):
    d=len(v)
    W=orthogonal_to(v)[:,1:]
    s0=root+np.array([.105,.28,.50,.76,1.00,1.30,1.67,2.10,2.62,3.15,3.85,4.60])
    s0=np.minimum(s0,6.7)
    if np.min(np.diff(s0))<.01:
        s0=np.linspace(root+.105,6.55,len(s0))
    x=s0[:,None]*v
    G=fd_gradient(ctx.query,x,h=3e-5)
    b=(G@W)/np.maximum(1e-7,der(s0))[:,None]
    b=np.clip(b,-1.,1.)
    # Add a curvature sketch in the transverse plane.
    rr=min(d,11)
    p=ctx.rng.standard_normal((d,d))
    p[:,0]=v
    Q=linalg.qr(p,mode="economic")[0]
    S,i,j=quad_stencil(d,rr)
    t0=np.clip(root+.7,root+.35,root+1.2)
    h=.0011
    Y=ctx.query(t0*v+h*(S@Q.T))
    _,_,H=unpack_stencil(Y,h,d,rr,i,j)
    H=W.T@Q@H/max(1e-8,der(t0))
    H[:,0] *= .32
    B=b.T@b/(len(s0)*1.0)
    if gn is not None:
        tab=(gn@W)
        B+=.7*(tab.T@tab)/max(1,len(tab))
    if np.linalg.norm(H)<10.:
        B+=.46*(H@H.T)*d/max(1,rr)
    _,V=np.linalg.eigh(B)
    R=W@V[:,::-1]
    b=b@V[:,::-1]
    bk=interpolate.PchipInterpolator(s0,b,axis=0,extrapolate=False)
    bdk=bk.derivative()

    def bprior(s,u):
        return np.sum(bk(np.clip(s,s0[0],s0[-1]))*u,axis=1)
    def gradient(s,u):
        t=np.clip(s,s0[0],s0[-1])
        ds=np.sum(bdk(t)*u,axis=1)*((s>s0[0])&(s<s0[-1]))
        return np.column_stack([1+ds,bk(t)])
    bprior.jet=gradient
    bprior.knots=s0
    return np.column_stack([v,R]),bprior


class WarpFeatures:
    def __init__(self,d,root):
        self.k=7
        self.knots=root+np.array([.08,.42,.77,1.14,1.58,2.20,3.04])
        self.band=np.array([.28,.30,.30,.33,.40,.53,.78])
        self.d=d
        self.r=min(d-1,9)
        self.i,self.j=np.triu_indices(self.r,1)

    def components(self,z):
        s=z[:,0]
        u=np.clip(z[:,1:],-5.6,5.6)
        a=np.clip(s,self.knots[0]-.22,self.knots[-1]+.48)
        T=np.exp(-.5*((a[:,None]-self.knots[None])/self.band[None])**2)
        T /= np.sum(T,axis=1,keepdims=True)
        U=np.column_stack([np.ones(len(u)),u,(u*u-1)/np.sqrt(2),u[:,self.i]*u[:,self.j],
                          (u[:,:self.r]**3-3*u[:,:self.r])/np.sqrt(6)])
        return s,u,T,U

    def design(self,z):
        _,_,T,U=self.components(z)
        return np.concatenate([T[:,i,None]*U for i in range(self.k)],axis=1)

    def jet(self,z):
        s,u,T,U=self.components(z)
        n,d=z.shape
        width=d-1
        h=np.arange(width)
        D=np.zeros((n,width,U.shape[1]))
        D[:,h,1+h]=1.
        D[:,h,1+width+h]=np.sqrt(2)*u
        k=1+2*width+np.arange(len(self.i))
        D[:,self.i,k]=u[:,self.j]
        D[:,self.j,k]=u[:,self.i]
        jj=np.arange(self.r)
        D[:,jj,1+2*width+len(self.i)+jj]=(3/np.sqrt(6))*(u[:,jj]*u[:,jj]-1)
        D*= (np.abs(z[:,1:])<5.6)[:,:,None]
        lo,hi=self.knots[0]-.22,self.knots[-1]+.48
        z0=np.clip(s,lo,hi)
        T0=(self.knots[None]-z0[:,None])/(self.band[None]**2)
        T0*=((s>lo)&(s<hi))[:,None]
        H=T*(T0-np.sum(T*T0,axis=1)[:,None])
        return np.concatenate([np.concatenate([(H[:,i,None]*U)[:,None],T[:,i,None,None]*D],
                                               axis=1) for i in range(self.k)],axis=2)

    def coefficients(self,p):
        d=self.d-1
        reg=np.full(p,.014)
        reg[0]=.00048
        reg[1:1+d]=.0055
        reg[1:1+min(8,d)]=.0025
        reg[1+d:1+2*d]=.014
        reg[1+d:1+d+min(8,d)]=.006
        return reg


def warped_model(ctx, x, y, v, gn):
    d=len(v)
    scale=np.median(y[y>0])
    root,f0,inv,der=axis_curve(ctx,v,scale)
    R,base=warp_rotation(ctx,v,gn,root,der)
    n0=1536 if d<40 else 2048
    n1=640 if d<40 else 1024
    Z=normal(n0,d,ctx.rng)
    Za=normal(n1,d,ctx.rng)
    Za[:,0] = root + .04 + (max(3.5,min(4.4,4.8-root))-.04)*special.ndtr(Za[:,0])
    Z=np.vstack([Z,Za])
    xx=Z@R.T
    yy=ctx.query(xx)
    s=np.r_[x@v,Z[:,0]]
    xc=np.vstack([x,xx])
    yc=np.r_[y,yy]
    Z=np.vstack([x@R,Z])
    active=yc>0.
    if np.count_nonzero(active)<210:
        raise ValueError("insufficient positive observations for a warp")
    W=WarpFeatures(d,root)
    X = W.design(Z)
    estinv = inv(yc)
    P=base(s,Z[:,1:])
    pri=s+P
    train=np.arange(len(yc))%4!=1
    vtr=active & train
    yhat=estinv-pri
    wp=np.clip((yc[vtr]/scale+.2)**.15,.7,2.0)
    F=X[vtr]
    nfeat=F.shape[1]//W.k
    reg=np.tile(W.coefficients(nfeat), W.k)*len(F)/W.k
    M=F.T@(F*wp[:,None])
    M.flat[::len(M)+1]+=reg
    rhs=F.T@(wp*yhat[vtr])
    beta=linalg.solve(M,rhs,assume_a="pos")
    grad_data=getattr(ctx,"live_gradients",None)
    if grad_data is not None and ctx.time_left()>17:
        xg,yg,dg=grad_data
        u0=inv(yg)
        gden=der(u0)
        raw=(dg@R)/np.maximum(1e-9,gden[:,None])
        use=(gden>.03*scale)&(raw[:,0]>.045)&(np.linalg.norm(raw,axis=1)<4.4)
        if use.sum()>=4:
            zz=xg[use]@R
            y0=raw[use]-base.jet(zz[:,0],zz[:,1:])
            wt=np.repeat((.50+.16*np.minimum(3.,yg[use]/scale))[:,None],d,axis=1)
            wt[:,0]*=.82
            ax=np.zeros((len(base.knots),d))
            ax[:,0]=base.knots
            xxj=np.vstack([zz,ax])
            yyj=np.vstack([y0,np.zeros_like(ax)])
            rw=np.vstack([wt,np.full_like(ax,.21)])
            J=W.jet(xxj).reshape(-1,X.shape[1])
            J*=np.sqrt(rw).ravel()[:,None]
            yyj=(yyj*np.sqrt(rw)).ravel()
            try:
                vbeta=linalg.solve(M+J.T@J,rhs+J.T@yyj,assume_a="pos")
                ok=(~train)&active
                old = f0(pri[ok]+X[ok]@beta)
                new = f0(pri[ok]+X[ok]@vbeta)
                wt0=np.where(np.flatnonzero(ok)<len(x)+n0,1.,.21)
                r0=np.mean(wt0*(old-yc[ok])**2)
                r1=np.mean(wt0*(new-yc[ok])**2)
                right=s>root+.6
                frac=np.count_nonzero(right&(yc==0))/max(1,np.count_nonzero(right))
                if r1<.992*r0 and frac<.020:
                    beta=vbeta
                    log("gradient fit selected",use.sum(),np.sqrt(r1/max(1e-20,r0)))
            except (ValueError,np.linalg.LinAlgError):
                pass
    h_est=pri+X@beta
    wide=(h_est>root+.095)&(~active)&train & (s>root+.55)
    near=(h_est>root+.003)&(~active)&train & (s<root+.52)
    gfrac=np.count_nonzero(wide)/max(1,np.count_nonzero(train&(s>root+.55)))
    log("warp root",root,"barrier fraction",gfrac,"near",np.count_nonzero(near),"holdout positive",
        np.std((h_est-estinv)[active&~train]))
    if gfrac < .028:
        # One-sided information from zero payoffs near the terminal exercise boundary.
        for _ in range(2):
            near=(h_est>root+.002)&(~active)&train & (s<root+.65)
            if not near.any():
                break
            H=X[near]
            beta=linalg.solve(M+H.T@H*.52,rhs+H.T@((root-pri[near])*.52),assume_a="pos")
            h_est=pri+X@beta
    yp=f0(s+np.clip(h_est-s,-2.0,2.1))
    from options import fit_alive
    if gfrac>.018:
        gate,Ract=fit_alive(xc[:len(x)+n0],yc[:len(x)+n0],yp[:len(x)+n0],R,scale)
    else:
        gate,Ract=None,R
    C=beta.reshape(W.k,nfeat).T

    def term(x):
        s,u,T,U=W.components(x@R)
        off=base(s,u)+np.sum(T*(U@C),axis=1)
        return f0(s+np.clip(off,-2.,2.1))

    def func(x):
        pred=term(x)
        if gate is not None:
            pred*=gate(x)
        return pred
    func.gate=gate
    func.term=term
    g_range=np.arange(len(x)+n0)
    wsd=np.std(yc[g_range[~train[g_range]]]-func(xc[g_range[~train[g_range]]]))
    log("warp validation",wsd,"vs raw",np.std(yc[g_range[~train[g_range]]]))
    return func,Ract,xc,yc,len(x)+n0,root,wsd,train,f0


def stratified_residual(ctx, R, fun, x, y, root):
    n=ctx.remaining
    d=ctx.d
    sp=x@R[:,0]
    err=y-fun(x)
    cuts=np.unique(np.clip(np.r_[[-np.inf,-1.0],root-.32,root+.12,root+.53,
                                  root+1.04,root+1.7,root+2.48,root+3.4,np.inf],-20,20))
    cuts[0],cuts[-1]=-np.inf,np.inf
    P=special.ndtr(cuts)
    mass=np.diff(P)
    sd=np.zeros(len(mass))
    overall=np.sqrt(np.mean(err*err))
    i0=np.searchsorted(cuts[1:],sp,side="left")
    for i in range(len(mass)):
        sel=i0==i
        if sel.sum()>=6:
            sd[i]=np.sqrt((np.sum(err[sel]**2)+4*(overall*.3)**2)/(sel.sum()+4))
        elif i>0:
            sd[i]=max(overall,sd[i-1]*1.25)
        else:
            sd[i]=overall
    score=mass*(.14*max(1e-15,overall)+sd)
    weights=score/max(score.sum(),1e-15)
    plan=np.maximum(64,(weights*(n-64*len(mass))).astype(int)+64)
    if plan.sum()>n:
        plan=(plan*(n/plan.sum())).astype(int)
    plan[np.argmax(plan)] += n-plan.sum()
    if ctx.remaining>=9100 and ctx.time_left()>9.0:
        from strata import folded_strata
        return folded_strata(ctx,R,fun,plan,mass,cuts,root)
    est=0.
    for i,num in enumerate(plan):
        if num <= 0 or mass[i] <=0:
            continue
        z=balanced_rule(num,d,ctx.rng)
        p=P[i]+mass[i]*special.ndtr(z[:,0])
        z[:,0]=special.ndtri(np.clip(p,1e-15,1-1e-15))
        xx=z@R.T
        val=ctx.query(xx)-fun(xx)
        est+=mass[i]*np.mean(val)
    log("warp strata",plan.tolist(),"correction",est)
    return est


def residual_importance(ctx,R,fun,x,y,root):
    """Use a defensive proposal for anisotropic or heavily skewed residuals."""
    from options import importance_mix, log_density_rel
    d=ctx.d
    r=y-fun(x)
    if len(y)<900 or ctx.remaining<4600 or np.std(r)<1e-8:
        return None
    idx=np.arange(len(y))
    valid=idx%4==1
    U=x@R
    Z=U[~valid]
    e=r[~valid]
    s0=np.sqrt(np.mean(r[valid]**2))
    if s0<=1e-14:
        return None
    ps=[]
    wt=[]
    for direction in (-1.,1.):
        e0=np.maximum(0.,direction*e)
        if np.count_nonzero(e0>.15*s0)<36:
            continue
        w=np.minimum(e0,6*s0)**.97
        w/=w.sum()
        ne=1/np.sum(w*w)
        if ne<36:
            continue
        m=w@Z
        a=Z-m
        var=w@(a*a)
        v2=np.clip(var,.47,2.5)
        rho0=ne/(ne+max(38,1.6*d))
        s=np.ones(d)
        s[:min(d,10)] = 1.+min(.78,rho0+.14)*(v2[:min(d,10)]-1.)
        s[0]=1+.82*(v2[0]-1)
        energy=np.linalg.norm(m[1:])
        m[1:]*=min(rho0,.82,1.05/max(energy,1e-12))
        m[0]=np.clip(m[0],-1.25,2.25)
        # A direction of failure close to the control plane is often more
        # important than a diffuse tilt across many axes.
        orient=R
        uu=a[:,1:]
        H=(uu.T*w)@uu
        B,V=np.linalg.eigh(H)
        tol=ne/(ne+max(110,3*d))
        sel=np.argsort(np.abs(np.log(np.maximum(.04,B))))[::-1]
        if ne>120 and np.linalg.norm(m[1:])>.12 and B[0]<.54:
            rot=V[:,sel]
            s[1:]=1+tol*np.clip(B[sel]-1,-.70,1.10)
            orient=np.column_stack([R[:,0],R[:,1:]@rot])
        ps.append((R@m,orient,s))
        wt.append(np.sum(e0))
    if not ps:
        return None
    wt=np.array(wt)/sum(wt)
    xv=x[valid]
    rv=r[valid]
    ratios=sum(w*np.exp(np.clip(log_density_rel(xv,p),-100,60)) for w,p in zip(wt,ps))
    cuts=np.unique(np.r_[-np.inf,-1.,root-.32,root+.12,root+.53,root+1.04,
                         root+1.7,root+2.48,root+3.4,np.inf])
    pp=np.diff(special.ndtr(cuts))
    grp=np.searchsorted(cuts[1:],U[:,0],side="left")
    q0=[]
    for i,a in enumerate(pp):
        keep=(grp==i)&~valid
        m2=np.mean(r[keep]**2) if keep.sum()>=8 else 1.5*s0*s0
        q0.append(a*(.15*s0+np.sqrt(m2)))
    q0=np.array(q0)
    prop=.92*q0/max(1e-15,q0.sum())+.08*pp
    denom=(prop/np.maximum(1e-20,pp))[grp[valid]]
    var0=(rv*rv)@(1/np.maximum(1e-20,denom))
    best=None
    for mix in [.36,.56,.72]:
        p=(1-mix)+mix*ratios
        ratio=float((rv*rv)@(1/p)/var0)
        if best is None or ratio<best[0]:
            best=(ratio,mix)
    ratio,mix=best
    if ratio>.81:
        return None
    log("option residual proposal",ratio,mix,[p[0]@R[:,0] for p in ps])
    return importance_mix(ctx,ps+[(np.zeros(d),R,np.ones(d))],
                          [*(mix*wt),1-mix],model=fun)


def integrate_warp(ctx,x,y,v,gn):
    fun,R,xx,yy,base_n,root,err,train,f0=warped_model(ctx,x,y,v,gn)
    if not np.isfinite(err) or err>.68*np.std(yy[:base_n]):
        raise ValueError("warp did not reduce holdout variation")
    axis=integrate.quad(lambda t: f0(np.array([t]))[0]*np.exp(-.5*t*t)/np.sqrt(2*np.pi),
                       root,max(10.,root+8),epsabs=3e-8,limit=250)[0]
    fun0=fun
    try:
        from kernels import kernel_update
        fun,kmean=kernel_update(ctx,fun0,R,xx,yy,base_n,train)
    except (ValueError,np.linalg.LinAlgError) as exc:
        log("declining kernel control",repr(exc))
        fun,kmean=fun0,0.
    means=[]
    for _ in range(5):
        z=normal(2**16,ctx.d,ctx.rng)
        means.append(np.mean([(fun0(zz@R.T)-f0(zz[:,0])).mean() for zz in np.split(z,8)]))
        if ctx.time_left()<10.:
            break
    ctx.last=float(axis+kmean+np.mean(means))
    q=residual_importance(ctx,R,fun,xx[:base_n],yy[:base_n],root)
    if q is not None:
        return ctx.last+q
    return ctx.last+stratified_residual(ctx,R,fun,xx[:base_n],yy[:base_n],root)
