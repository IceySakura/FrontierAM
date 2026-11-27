"""Ray-based geometric pilots for closed jump surfaces."""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import gammainc, gammaincinv, ndtr, ndtri

from core import basis_from, blocks, fd_gradients, fit_ridge, gaussian, net_integral, uniform
from quadratic import positive_probability, quad_values, tilted_normal


def ray_directions(n, d, rng):
    a = rng.standard_normal((n,d))
    return a / np.linalg.norm(a, axis=1)[:, None]


def xray(p,U,r):
    return p + U*r[...,None]


def limit_ray(p,U):
    return np.min((39.5-np.abs(p))/np.maximum(1.e-14,np.abs(U)),axis=1)


def roots(A, b, U):
    q = np.einsum("ij,ij->i", U @ A, U)
    l = U @ b
    return 2/np.maximum(np.sqrt(np.maximum(0., l*l+4*q))+l, 1e-9)


def classify_roots(f, p, U, threshold, radius, spread, steps):
    cap = limit_ray(p,U)
    lo = np.maximum(0., radius-spread) + np.zeros(len(U))
    hi = np.minimum(cap, radius+spread) + np.zeros(len(U))
    vl = f.query(xray(p,U,lo))
    vh = f.query(xray(p,U,hi))
    lo[vl <= threshold] = 0
    hi[vh > threshold] = cap[vh > threshold]
    lo,hi,vl,vh = refine_sections(f,p,U,(lo,hi,vl,vh),steps,threshold=threshold)
    ok = (vl>threshold) & (vh<=threshold) & (hi>0)
    return (lo,hi,vl,vh), ok


def bracket_jump(f,p,U,rad):
    vals = f.query((p+U[:,None,:]*rad[:,:,None])).reshape(rad.shape)
    dv = vals[:,:-1]-vals[:,1:]
    # Normalize slowly varying second-order trends without suppressing steps.
    j = np.argmax(dv,axis=1)
    i = np.arange(len(U))
    return rad[i,j],rad[i,j+1], vals[i,j], vals[i,j+1]


def refine_sections(f,p,U,bracket,steps,threshold=None,amplitude=None):
    lo,hi,vl,vh = bracket
    for _ in range(steps):
        t = .5*(lo+hi)
        x=xray(p,U,t)
        vt = f.query(x)
        if threshold is not None:
            lower = vt > threshold
        elif amplitude is not None:
            lower = np.abs((vt-amplitude(x))-(vl-amplitude(xray(p,U,lo)))) < np.abs(vt-vh)
        else:
            lower = np.abs(vt-vl) < np.abs(vt-vh)
        lo = np.where(lower,t,lo)
        hi = np.where(lower,hi,t)
        vl = np.where(lower,vt,vl)
        vh = np.where(lower,vh,vt)
    return lo,hi,vl,vh


def midpoint_data(br):
    lo,hi,vl,vh = br
    return .5*(lo+hi), .5*(hi-lo), vl-vh


def amplitude_fit(bx,J,d):
    med=np.median(J)
    ok=(J >= .13*med)&(J <= 7.5*med)&(np.linalg.norm(bx,axis=1)<np.sqrt(d)*3.0)
    Q=np.column_stack((np.ones(np.sum(ok)),bx[ok]))
    y=np.log(np.maximum(1e-15,J[ok]))
    if len(y)<max(10,d//2):
        return np.log(max(1e-15,med)), np.zeros(d), 1.
    co=fit_ridge(Q,y,.11)
    err=y-Q@co
    noise=max(.001,np.median(np.abs(err-np.median(err)))*3.4)
    wt=np.minimum(1.,noise/np.maximum(abs(err),1e-12))
    co=fit_ridge(wt[:,None]*Q,wt*y,.075)
    logj,l=co[0],co[1:]
    error=np.sqrt(np.mean(np.clip(y-Q@co,-1,1)**2))
    l *= min(1.,.82/max(1e-12,np.linalg.norm(l)))
    return logj,l,error


def features(x, ij=None):
    d=x.shape[1]
    if ij is None:
        ij=np.triu_indices(d,1)
    return np.column_stack((x*x, np.sqrt(2)*x[:,ij[0]]*x[:,ij[1]], x))


def from_coef(co, d, ij):
    A = np.diag(co[:d])
    A[ij] = A[ij[::-1]] = co[d:-d] / np.sqrt(2.)
    return A, co[-d:].copy()


def fit_rays(U, r, e, rscale, low_rank=True, use_nonlinear=True):
    """Weighted fit to section midpoints; select rank on held-out rays."""
    d=U.shape[1]
    n=len(U)
    ij=np.triu_indices(d,1)
    x=U*(r/rscale)[:,None]
    w=1/np.maximum(0.45, (r/rscale)**2)
    frac=e/r
    mid=np.median(frac)
    w /= np.sqrt(1+np.square(frac/max(1e-7,min(.016,mid*2))))
    F=features(x,ij)
    p=F.shape[1]
    ksplit=max(16,n//7)
    Xt, y=F*w[:,None], w
    prior=np.zeros(p)
    prior[:d] = np.sum(Xt[:,:d]@np.ones(d)*y)/max(1e-12,np.sum((Xt[:,:d]@np.ones(d))**2))
    reg=(2e-6 + .033 * d/max(d,n))*(min(.7,100*mid+.025) if n>=p*1.17 else 1.40)
    reg *= min(15.,(1+10*np.median(e/r)))
    co=prior+fit_ridge(Xt[ksplit:], y[ksplit:]-Xt[ksplit:]@prior,reg)
    A,b=from_coef(co,d,ij)
    lam,V=eigh(A)
    lam=lam[::-1]
    V=V[:,::-1].copy()

    def score(Ar, br):
        pred = roots(Ar,br,U[:ksplit])
        err = np.clip(np.log(np.maximum(1e-10,pred*rscale/r[:ksplit])), -2, 2)
        return np.sqrt(np.mean(err*err))

    def project(a):
        a2=np.maximum(lam,a)
        return (V*a2)@V.T

    A = project(max(1e-5,.009*max(0.,lam[0])))
    best = score(A,b)
    best_k=0
    if low_rank and use_nonlinear and lam[0] > 0:
        cand=set()
        ratios=lam[:-1]/np.maximum(.001*lam[0], lam[1:])
        order=np.argsort(-ratios[:min(d-1, max(2,int(n/d*1.4)))])
        for i in order[:3]:
            if ratios[i] > 1.36 and lam[i] > .055*lam[0]:
                cand.add(i+1)
        if n>=p:
            ix=np.sum(lam>.105*lam[0])
            if ix < d:
                cand.add(ix)
        if d>=31 and best>.009:
            dig=np.median(np.abs(np.log(r/np.median(r))))
            if n>=3*d and dig>.065:
                cand|={2,3,4,6}
            if n>=7*d and dig>.087:
                cand|={5,8,10}
        if not cand and best>.035:
            cand.add(2)
        train=slice(ksplit,None)
        for k in sorted(cand):
            if k < 1 or n < 1.2*k*d or k == d:
                continue
            At,bt=lowrank_fit(x[train],w[train],V[:,:k]*np.sqrt(np.maximum(lam[:k],1e-6)),b,100,
                             robust=(best>.034))
            se=score(At,bt)
            if se < best * .94:
                A,b=At,bt
                best,best_k=se,k
    if best_k:
        l,v=eigh(A,subset_by_index=[d-best_k,d-1])
        A,b=lowrank_fit(x,w,v*np.sqrt(np.maximum(l,1e-8)),b,145,robust=best>.022)
    else:
        co=prior+fit_ridge(Xt,y-Xt@prior,reg)
        A,b=from_coef(co,d,ij)
        lam,V=eigh(A)
        A=(V*np.maximum(lam, max(1e-6,.007*lam[-1])))@V.T
    return A,b,best


def lowrank_fit(x,w,L,b,steps,robust=False):
    """Optimize A=LL' and b=Lh so the fitted cylinders have bounded sections."""
    d,k=L.shape
    h=np.linalg.lstsq(L,b,rcond=1e-8)[0]
    w=w*w

    def obj(th):
        mat=th[:d*k].reshape(d,k)
        a=th[d*k:]
        u=x@mat
        res=np.einsum("ij,ij->i",u,u)+u@a-1
        wr=res*w
        if robust:
            r0=.065
            q=np.sqrt(1+np.square(res/r0))
            loss=r0*r0*((q-1)@w)
            wr=wr/q
        else:
            loss=.5*(res@wr)
        return loss, np.r_[(x.T @ (wr[:,None]*(2*u+a))).ravel(),wr@u]

    rr=minimize(obj,np.r_[L.ravel(),h],method="L-BFGS-B",jac=True,
                options={"maxiter":steps,"ftol":1e-13,"gtol":1e-9,"maxcor":12})
    L=rr.x[:d*k].reshape(d,k)
    return L@L.T,L@rr.x[d*k:]


def world_quadric(A,b,p,rscale):
    A=A/rscale**2
    b=b/rscale-2*(A@p)
    return A,b,float(-1-p@A@p-b@p)


def radius_from(q,b):
    return 2/np.maximum(np.sqrt(np.maximum(0.,b*b+4*q))+b,1e-9)


def gaussian_quantizer(k):
    edge=ndtri(np.linspace(0.,1.,k+1))
    for _ in range(64):
        ff=np.exp(-.5*edge*edge)/np.sqrt(2*np.pi)
        m=(ff[:-1]-ff[1:])/np.diff(ndtr(edge))
        edge[1:-1]=.5*(m[:-1]+m[1:])
    return edge,m


def fixed_diag_psd(B, diag, rounds=12):
    scale=np.sqrt(np.maximum(1e-14,diag))
    X=B/(scale[:,None]*scale)
    n=len(scale)
    index=np.arange(n)
    D=np.zeros_like(X)
    for _ in range(rounds):
        Y=X-D
        val,V=eigh(Y,check_finite=False)
        X=(V*np.maximum(val,2e-6))@V.T
        D=X-Y
        X[index,index]=1.
    val,V=eigh(X,check_finite=False)
    X=(V*np.maximum(val,max(1e-6,.00022*val[-1])))@V.T
    scale=scale/np.sqrt(np.diag(X))
    return X*(scale[:,None]*scale)


def frame_ellipsoid(f,p,X,Y,A0,b0,rscale,U0,r0,J0,threshold):
    d=f.d
    rare=threshold is not None
    E=np.eye(d)
    Ua=np.vstack((E,-E))
    pred=np.clip(roots(A0,b0,Ua)*rscale,.40*rscale,2.3*rscale)
    if not rare:
        logj,l,ae=amplitude_fit(xray(p,U0,r0),J0,d)
        def amp(z):
            return np.exp(logj+np.clip(z@l,-10,10))
        fac=np.array([.19,.52,.76,1.0,1.34,1.80,2.95])
        rad=np.minimum(pred[:,None]*fac,limit_ray(p,Ua)[:,None])
        bra=refine_sections(f,p,Ua,bracket_jump(f,p,Ua,rad),8,amplitude=amp if ae<.28 else None)
        ra,ea,ja=midpoint_data(bra)
        aj=amp(xray(p,Ua,ra))
        good=(ja>.46*aj)&(ja<2.1*aj)
    else:
        bra,good=classify_roots(f,p,Ua,threshold,pred,.65*pred,10)
        ra,ea,ja=midpoint_data(bra)
    if good.mean() < .92:
        return None
    da=np.where(good,rscale/np.maximum(.025*rscale,ra),np.r_[np.sqrt(np.diag(A0)),np.sqrt(np.diag(A0))])
    aa=da[:d]*da[d:]
    bl=da[:d]-da[d:]
    av=np.median(aa)
    if not .04<av<12 or np.std(aa)>av*.57 or np.min(ra)<.095*rscale:
        return None
    sig=max(.018*av,np.std(aa)*.68)
    i,j=np.triu_indices(d,1)
    di=.5*(aa[i]+aa[j])
    bi=(bl[i]+bl[j])/np.sqrt(2.)
    pairs=(E[i]+E[j])/np.sqrt(2.)
    dmin=.014*av
    bv=[xray(p,U0,r0),xray(p,Ua[good],ra[good])]
    jv=[J0,ja[good]]
    B=np.diag(aa)
    if rare:
        bits=min(8,(f.remaining-4100)//len(i))
        if bits<2:
            return None
        ed,ctr=gaussian_quantizer(1 << bits)
        lo=np.zeros(len(i),dtype=int)
        hi=np.full(len(i),len(ctr),dtype=int)
        for _ in range(bits):
            m=(lo+hi)//2
            q=np.maximum(dmin,di+sig*ed[m])
            z=xray(p,pairs,rscale*radius_from(q,bi))
            pos=f.query(z)>threshold
            hi=np.where(pos,m,hi)
            lo=np.where(pos,lo,m)
        aij=ctr[lo]*sig
    else:
        bits=min(8,(f.remaining-3830)//len(i)-2)
        if bits<1:
            return None
        spread=2.9*sig
        rlo=rscale*radius_from(di+spread,bi)
        rhi=rscale*radius_from(np.maximum(dmin,di-spread),bi)
        rhi=np.minimum(rhi,limit_ray(p,pairs))
        yl=f.query(xray(p,pairs,rlo))
        yh=f.query(xray(p,pairs,rhi))
        predj=amp(xray(p,pairs,.5*(rlo+rhi)))
        sel=(yl-yh>.48*predj)&(yl-yh<2.05*predj)
        if sel.mean()<.77:
            return None
        pairsel=pairs[sel]
        br=refine_sections(f,p,pairsel,(rlo[sel],rhi[sel],yl[sel],yh[sel]),bits,
                           amplitude=amp if ae<.28 else None)
        ri,ei,ji=midpoint_data(br)
        ul,uh=rscale/br[1],rscale/br[0]
        ll=(ul*ul-bi[sel]*ul-di[sel])/sig
        hh=(uh*uh-bi[sel]*uh-di[sel])/sig
        cond=(np.exp(-.5*ll*ll)-np.exp(-.5*hh*hh))/(np.sqrt(2*np.pi)*np.maximum(1e-10,ndtr(hh)-ndtr(ll)))
        goodi=np.isfinite(cond)&(ji>.45*amp(xray(p,pairsel,ri)))
        aij=np.clip(A0[i,j],-sig,sig)
        aij[np.flatnonzero(sel)[goodi]]=sig*np.clip(cond[goodi],-3.,3.)
        # Same-side outliers of the bracket can be distinguished when the
        # fitted smooth background is much narrower than the jump.
        qguess=quad_values((X-p)/rscale,A0,b0,-1.)
        Res=Y-amp(X)*(qguess<0)
        mid=np.median(Res)
        cl=max(1e-8,np.quantile(abs(Res-mid),.74)*1.55)
        Bg,g=standard_background(f,X,np.clip(Res,mid-cl,mid+cl))
        xl,xh=xray(p,pairs,rlo),xray(p,pairs,rhi)
        lb=(yl-mid-quad_values(xl,Bg,g,-np.trace(Bg)))/np.maximum(amp(xl),1e-9)
        hb=(yh-mid-quad_values(xh,Bg,g,-np.trace(Bg)))/np.maximum(amp(xh),1e-9)
        aij[(~sel)&(lb>.89)&(hb>.89)]=-3.12*sig
        aij[(~sel)&(lb<.11)&(hb<.11)]=3.12*sig
        bv.append(xray(p,pairsel[goodi],ri[goodi]))
        jv.append(ji[goodi])
    B[i,j]=B[j,i]=aij
    B=fixed_diag_psd(B,aa)

    def rms(A,b):
        ratio=roots(A,b,U0)*rscale/r0
        return float(np.sqrt(np.mean(np.clip(np.log(ratio),-1.5,1.5)**2)))

    cv=rms(B,bl)
    la,EV=eigh(B)
    la,EV=la[::-1],EV[:,::-1].copy()
    if cv>.009 and la[0]>0:
        floor=.023*la[0]
        diff=la[:-1]/np.maximum(la[1:],floor)
        ans=[]
        for k in np.argsort(-diff)[:3]+1:
            if k<4 or k>=d-3 or diff[k-1]<2.1:
                continue
            W=EV[:,:k]
            X1=W*np.sqrt(la[:k])
            X1*=np.sqrt(aa/np.sum(X1*X1,axis=1))[:,None]
            X2=X1@X1.T
            X3=W@(W.T@bl)
            se=rms(X2,X3)
            if se<cv*.88:
                ans.append((se,X2,X3))
        if ans:
            cv,B,bl=min(ans,key=lambda a:a[0])
    if not np.isfinite(cv) or cv>.50:
        return None
    return p,*world_quadric(B,bl,p,rscale),cv,np.vstack(bv),np.concatenate(jv)


def build_ellipsoid(f, X, Y, at_zero, threshold=None, anchor=None):
    rare=threshold is not None
    d=f.d
    if not rare:
        p=np.zeros(d) if anchor is None else anchor
    elif at_zero>threshold:
        p=np.zeros(d)
    else:
        inz=X[Y>threshold]
        if len(inz)==0:
            return None
        p=np.mean(inz,axis=0)
        if f.query(p)[0] <= threshold:
            p=inz[np.argmin(np.linalg.norm(inz,axis=1))]
    n0 = min(76,max(45,d+4)) if not rare else max(55,min(86,2*d+16))
    U0 = ray_directions(n0,d,f.rng)
    if not rare:
        # Wide pilot rays establish that the jump persists down to small spacing.
        rad = np.repeat((np.sqrt(d)*np.linspace(0.,2.20,13))[None,:],n0,axis=0)
        br0=refine_sections(f,p,U0,bracket_jump(f,p,U0,rad),8)
        _,_,j0=midpoint_data(br0)
        ok=j0>max(np.std(Y)*.36,1e-5)
    else:
        br0,ok=classify_roots(f,p,U0,threshold,np.sqrt(d)*.86,np.sqrt(d)*.75,10)
    r0,e0,j0=midpoint_data(br0)
    frac = np.mean(ok)
    if frac < .47:
        return None
    jump = float(np.median(j0[ok]))
    rscale = float(np.median(r0[ok]))
    if rscale < .001 or not np.isfinite(rscale):
        return None
    A,b,cv=fit_rays(U0[ok],r0[ok],e0[ok],rscale,use_nonlinear=False)
    if (d >= 48 and f.remaining>=12600 and frac>.83 and np.std(np.log(r0[ok]))<.215
            and np.max(abs(p))<3.5):
        frame=frame_ellipsoid(f,p,X,Y,A,b,rscale,U0[ok],r0[ok],j0[ok],threshold)
        if frame is not None:
            return frame
        if f.remaining<7500:
            return None
    width=.22 if cv<.18 else min(1.35,.33+2.7*cv)
    Uall=[U0[ok]]
    all_br=[tuple(arr[ok] for arr in br0)]
    npar=d*(d+3)//2
    if rare:
        steps=8 if npar<=180 else (7 if d<40 else 5)
        n1 = min(max(int(1.43*npar),150), 1180, max(1,(f.remaining-5950)//(steps+2)))
        U = ray_directions(n1,d,f.rng)
        pred = np.minimum(2.3*rscale, roots(A,b,U)*rscale)
        # Initial range is widened when the pilot has not captured anisotropy.
        ss = (np.minimum(1.4,width)+.025)*np.maximum(.6*rscale,pred)
        br,ok = classify_roots(f,p,U,threshold,pred,ss,steps)
        amp=None
    else:
        logj,l,ae=amplitude_fit(xray(p,U0[ok],r0[ok]),j0[ok],d)
        def amp(z):
            return np.exp(logj+np.clip(z@l,-10,10))
        if ae>.22:
            amp=None
        steps=7 if npar<=180 else 6 if d<40 else 4
        n1=min(max(int(1.45*npar),122),940,max(1,(f.remaining-6100)//(steps+5)))
        U=ray_directions(n1,d,f.rng)
        pred=np.clip(roots(A,b,U)*rscale,.20*rscale,2.0*rscale)
        # The extra exterior points handle poor radial predictions during training.
        w=min(.49,max(.12,2.25*cv))
        fac=np.array([.10, 1-w, 1., 1+w,2.68])
        rad=pred[:,None]*fac
        br=refine_sections(f,p,U,bracket_jump(f,p,U,rad),steps,amplitude=amp)
        rr,ee,jj=midpoint_data(br)
        if amp is None:
            ok=jj>max(jump*.145,1e-8)
        else:
            aa=amp(xray(p,U,rr))
            ok=(jj > .43*aa)&(jj < 2.3*aa)
    if np.sum(ok)>max(20,.36*n1):
        Uall.append(U[ok])
        all_br.append(tuple(arr[ok] for arr in br))
    U=np.vstack(Uall)
    bracket=tuple(np.concatenate(data) for data in zip(*all_br))
    r,e,J=midpoint_data(bracket)
    # Gross rounding failures are common when a ray nearly misses a low-rank body.
    good=(r>1e-6) & (e/r < .28)
    U=U[good]
    bracket=tuple(arr[good] for arr in bracket)
    r,e,J=midpoint_data(bracket)
    if len(r)<max(30,d//2):
        return None
    if f.left() > 12:
        nbits=min(10,max(0,(f.remaining-(4900 if rare else 5550))//len(U)))
        if nbits:
            bracket=refine_sections(f,p,U,bracket,nbits,threshold,amp)
            r,e,J=midpoint_data(bracket)
    good=(J>.12*jump)
    if not rare and amp is not None:
        aa=amp(xray(p,U,r))
        good &= (J>.46*aa)&(J<2.10*aa)
    U,r,e,J=U[good],r[good],e[good],J[good]
    if len(r)<max(30,d//2):
        return None
    A,b,cv=fit_rays(U,r,e,rscale)
    return p, *world_quadric(A,b,p,rscale), cv, xray(p,U,r), J


def standard_background(f, X, Y):
    """Basic quadratic control variate, trained on independent pilot points."""
    d=f.d
    i,j=np.triu_indices(d,1)
    DX=features(X,(i,j))
    DX[:,:d]-=1.
    yc=Y-np.mean(Y)
    beta=fit_ridge(DX,yc, max(12., .42*DX.shape[1]))
    A,b=from_coef(beta,d,(i,j))
    return A,b


def radial_tail_correction(f,A,b,c,threshold,cv,geom):
    """Estimate the discrepancy of two radial masses, using narrow sections."""
    n=f.remaining//6
    if n<256:
        return None
    mu,sd,V=geom
    d=f.d
    logsd=.73*np.log(sd)
    logsd-=np.mean(logsd)
    sd=np.exp(logsd)
    mat=(V.T@A)@V
    lv=V.T@b
    acc=[]
    counts=[]
    band=min(.34,max(.005,4.1*cv))
    for k in blocks(n):
        if k<32 or (acc and f.left()<4.):
            break
        W=uniform(k,d+2,f.rng)
        u=ndtri(np.clip(W[:,1:-1],1e-13,1-1e-13))
        tilted=W[:,0]<.75
        u[tilted]*=sd
        u/=np.linalg.norm(u,axis=1)[:,None]
        lq=-.5*d*np.log(np.sum((u/sd)**2,axis=1))
        rho=1/(.25+.75*np.exp(np.clip(lq,-600,600)))
        aq=np.einsum("ij,ij->i",u@mat,u)
        l=u@lv
        rc=(-2*c)/np.maximum(1e-12,np.sqrt(np.maximum(0.,l*l-4*aq*c))+l)
        uu=u@V.T
        maxr=limit_ray(np.zeros(d),uu)
        lo=np.minimum(maxr*.8,rc*np.exp(-band))
        hi=np.minimum(maxr,rc*np.exp(band))
        L=f.query(uu*lo[:,None])>threshold
        H=f.query(uu*hi[:,None])>threshold
        lo=np.where(L,lo,0.)
        hi=np.where(H,maxr,hi)
        for _ in range(3):
            mid=.5*(lo+hi)
            inside=f.query(uu*mid[:,None])>threshold
            lo=np.where(inside,mid,lo)
            hi=np.where(inside,hi,mid)
        F0=gammainc(d/2.,.5*lo*lo)
        F1=gammainc(d/2.,.5*hi*hi)
        rr=np.sqrt(2*gammaincinv(d/2.,np.clip(F0+(F1-F0)*W[:,-1],1e-300,1-1e-13)))
        hit=f.query(uu*rr[:,None])>threshold
        inside=F0+(F1-F0)*hit
        diff=inside-gammainc(d/2.,.5*rc*rc)
        acc.append(np.mean(diff*rho))
        counts.append(k**1.28)
    return np.average(acc,weights=counts) if acc else None


def estimate_tail(f, X, Y, at_zero, threshold):
    fit=build_ellipsoid(f,X,Y,at_zero,threshold)
    if fit is None:
        return None
    p,A,b,c,cv,bx,J=fit
    jump=float(np.median(J))
    jspread=float(np.quantile(J,.85)-np.quantile(J,.15))
    if jspread > .05*jump or cv > .80 or not np.isfinite(jump):
        return None
    P=1-positive_probability(A,b,c)
    if not 1e-10 <= P <= .65:
        return None
    nbase=2048 if f.remaining>=5400 else 1024
    # Estimate the smooth component separately, without multiplying its
    # variance by the rare-event likelihood ratio.
    B,g=standard_background(f,X,Y-jump*(Y>threshold))
    L,V=eigh(B@B+np.outer(g,g))
    Z=gaussian(nbase,f.d,f.rng) @ V[:,::-1].T
    Yz=f.query(Z)
    bg=np.mean(Yz-jump*(Yz>threshold) - quad_values(Z,B,g,-np.trace(B)))
    if f.remaining < 128 or f.left() < 4.:
        return bg+jump*P
    mu,sd,V=tilted_normal(A,b,c, max_shift=4.3)
    if f.d >= 48 and np.linalg.norm(p)<1e-12 and c<0 and cv<.073:
        corrected=radial_tail_correction(f,A,b,c,threshold,cv,(mu,sd,V))
        if corrected is not None and np.isfinite(corrected):
            return bg+jump*np.clip(P+corrected,0.,1.)
    n = int(f.remaining)
    n0=1 << (n.bit_length()-1)
    # Defensive mixture. Use a common mixture density for both components.
    n1 = 1 << (max(128,n-n0).bit_length()-1)
    if n0+n1>n:
        n0=n0//2
    if n0+n1>n:
        n1=n-n0
    pi=n0/(n0+n1)
    zz=[]
    for k,imp in ((n0,True),(n1,False)):
        if k<=0: continue
        w = gaussian(k,f.d,f.rng)
        if imp:
            w=w*sd+mu
        zz.append(w)
    z=np.vstack(zz)
    Xq=z@V.T
    q=quad_values(Xq,A,b,c)
    iq=(f.query(Xq)>threshold)
    lr=.5*np.sum(z*z-((z-mu)/sd)**2,axis=1)-np.sum(np.log(sd))
    balance=np.exp(-np.logaddexp(np.log(pi)+lr, np.log1p(-pi)))
    correction=np.mean((iq.astype(float)-(q<0))*balance)
    return bg+jump*np.clip(P+correction,0.,1.)


def estimate_jump(f,X,Y,at_zero,anchor=None):
    fit=build_ellipsoid(f,X,Y,at_zero,anchor=anchor)
    if fit is None:
        return None
    p,A,b,c,cv,bx,J=fit
    if not np.isfinite(cv) or cv>.48:
        return None
    logj,l,ae=amplitude_fit(bx,J,f.d)
    P=1-positive_probability(A,b,c)
    if not .006 < P < .995:
        return None

    def amp(z):
        return np.exp(logj+np.clip(z@l,-12,12))

    def alb(x):
        aq=np.einsum("ij,ij->i", x@A,x)
        bl=x@b
        z=(np.sqrt(np.maximum(0,bl*bl-4*aq*c))+bl)/(-2*c)
        return z, aq+bl+c<0

    cshift=float(l@A@l+b@l+c)
    Pshift=1-positive_probability(A, b+2*A@l, cshift)
    mu=np.exp(logj+.5*(l@l))*Pshift
    ax=amp(X)
    lab=quad_values(X,A,b,c)<0
    err=Y-ax*lab
    med=np.median(err)
    clip0=np.maximum(1e-8,np.quantile(np.abs(err-med),.79)*2.6)
    B,g=standard_background(f,X,np.clip(err,med-clip0,med+clip0))
    Vgeo=basis_from(A+np.outer(l,l)*.7)
    ngrad=min(30,max(0,(f.remaining-3900)//f.d))
    use=f.rng.permutation(len(X))[:ngrad]
    if len(use):
        G=fd_gradients(f,X[use],Y[use])-l[None,:]*(ax[use]*lab[use])[:,None]
        L=np.linalg.norm(G,axis=1)
        wt=np.minimum(1.,max(1e-9,2.4*np.median(L))/np.maximum(1e-12,L))
        GG=G*wt[:,None]
        Vbg=basis_from(GG.T@GG + 0.02*len(G)*(B@B+np.outer(g,g)))
        training=(X[use],G,wt)
    else:
        Vbg=basis_from(B@B+np.outer(g,g))
        training=None
    from kinks import fit_background
    signal=ax*lab
    quality=np.std(Y-signal)/max(1e-12,np.std(Y))
    if c >= -.001 or (cv>.092 and quality>.63):
        scl=np.clip(np.cov(Y,signal,ddof=0)[0,1]/max(1.e-12,np.var(signal)),.0,1.16)
        _,_,bg=fit_background(X,Y-scl*signal,Vbg,gradients=None)
        Vq=basis_from(B@B+.25*np.outer(g,g)+A@A*.10, Vbg[:,0])
        def surrogate(z):
            return scl*amp(z)*(quad_values(z,A,b,c)<0)+bg(z)
        return net_integral(f,Vq,surrogate,scl*mu,float(np.mean(Y)))
    # Determine the label only in a guard band. Outside it the surrogate is
    # used for the split and any error remains in the smooth-side summand.
    guard=min(.82,max(.013,4.5*cv))
    xr,lab=alb(X)
    inside=np.abs(np.log(np.maximum(xr,1e-15)))<guard
    for _ in range(2):
        m,_,bg=fit_background(X,Y-ax*lab,Vbg,degree=3)
        lab=np.where(inside,Y-(m+bg(X)) > .5*ax,lab)
    m,_,bg=fit_background(X,Y-ax*lab,Vbg,gradients=training)
    if len(use)>=12 and f.left()>9.:
        from smooth import smooth_kernel
        kern=smooth_kernel(X,Y-ax*lab-bg(X),Vbg,np.sum((GG@Vbg)**2,axis=0))
        if kern is not None:
            oldbg=bg
            bg=lambda z: oldbg(z)+kern(z)

    def classify(x,y,ax,xr,base):
        uncertain=np.abs(np.log(np.maximum(xr,1e-15))) < guard
        return np.where(uncertain,y-(m+bg(x))>.5*ax,base)

    n=f.remaining
    frac=.24 if cv<.00035 else (.46 if cv<.013 else .69)
    nrad=1 << (max(1,int(frac*n)).bit_length()-1)
    if nrad<128:
        return None
    band=min(.55,max(.0025,2.9*cv))
    d=f.d
    acc_bg=[]
    ng=[]
    correction=[]
    # Save the proposal density ratios until the actual block allocation is
    # known; a late time guard may drop an otherwise scheduled block.
    ordinary=n-nrad
    for k,is_rad in [(k,False) for k in blocks(ordinary)]+[(nrad,True)]:
        if k<4:
            continue
        if f.left()<3.0 and acc_bg:
            break
        if is_rad:
            uni=uniform(k,d+1,f.rng)
            U=ndtri(np.clip(uni[:,1:],1e-13,1-1e-13))
            U=U@Vgeo.T
            U/=np.linalg.norm(U,axis=1)[:,None]
            rho,_=alb(U)
            rcut=1/np.maximum(rho,1e-12)
        else:
            z=gaussian(k,d,f.rng)@Vbg.T
            r=np.linalg.norm(z,axis=1)
            U=z/np.maximum(r[:,None],1e-13)
            rho,_=alb(U)
            rcut=1/np.maximum(rho,1e-12)
        low=gammainc(d*.5,.5*(rcut*np.exp(-band))**2)
        high=gammainc(d*.5,.5*(rcut*np.exp(band))**2)
        empty=high-low<1e-6
        low[empty],high[empty]=0.,1.
        width=high-low
        if is_rad:
            uu=low+width*uni[:,0]
            r=np.sqrt(2*gammaincinv(.5*d,np.clip(uu,1e-13,1-1e-13)))
            z=U*r[:,None]
        else:
            uu=gammainc(d*.5,.5*r*r)
        xrad,base=alb(z)
        val=f.query(z)
        aa=amp(z)
        lab=classify(z,val,aa,xrad,base)
        diff=aa*(lab.astype(float)-base)
        correction.append((diff,((uu>=low)&(uu<=high))/width,is_rad))
        if not is_rad:
            acc_bg.append(np.mean(val-aa*lab-bg(z)))
            ng.append(k**1.28)
    if not acc_bg:
        return None
    total=sum(len(v) for v,_,_ in correction)
    pi=sum(len(v) for v,_,s in correction if s)/total
    resid=sum(np.sum(v/((1-pi)+pi*dens)) for v,dens,_ in correction)/total
    return mu+np.average(acc_bg,weights=ng)+resid


def detect_jump(f,X,Y,at_zero):
    med=np.median(Y)
    yh=np.max(Y)
    if max(at_zero,yh) > max(5*med,med+15):
        big=max(at_zero,yh)
        # A high payoff alone is not a jump. Verify a large empty value gap.
        ys=np.sort(np.r_[Y,at_zero])
        k=np.argmax(np.diff(ys))
        if ys[k+1]-ys[k] > .56*(big-med) and ys[k+1] > 4*max(1e-12,ys[k]):
            return .5*(ys[k+1]+ys[k]), X,Y,None
    # Small-scale rays distinguish a common jump from a continuous kink.
    d=f.d
    U=ray_directions(24,d,f.rng)
    rad=np.repeat((np.sqrt(d)*np.linspace(.25,1.95,12))[None,:],len(U),axis=0)
    lo,hi,vl,vh=bracket_jump(f,np.zeros(d),U,rad)
    drop0=vl-vh
    for _ in range(7):
        t=.5*(lo+hi)
        vt=f.query(U*t[:,None])
        lower=np.abs(vt-vl)<np.abs(vt-vh)
        lo=np.where(lower,t,lo)
        hi=np.where(lower,hi,t)
        vl=np.where(lower,vt,vl)
        vh=np.where(lower,vh,vt)
    jumpbig=(vl-vh)>max(4*med,14.)
    if jumpbig.any():
        aa=vl[jumpbig]
        zray=(U*lo[:,None])[jumpbig]
        threshold=0.5*(np.min(aa)+max(np.max(vh[jumpbig]),np.quantile(Y,.90)))
        if threshold>max(3.5*med,med+8):
            return threshold,np.vstack((X,zray)),np.r_[Y,aa],None
    eps=max(.22*np.std(Y), 2e-4*max(1.,abs(med)))
    if np.mean(vl-vh>eps) >= .5:
        return -1.,X,Y,None
    found=(vl-vh>eps)&(vl-vh>.29*drop0)
    if 3<=np.sum(found)<12 and f.remaining>13500:
        # Refine the interior anchor on lines that begin and end at observed
        # values rather than assuming a centred ellipsoid.
        n=28
        ind=np.argsort(Y)
        il=f.rng.choice(ind[len(ind)*2//3:],n,replace=False)
        ih=f.rng.choice(ind[:len(ind)//3],n,replace=False)
        B=X[il]
        D=X[ih]-B
        r=np.linspace(0,1,8)
        yy=f.query(B[:,None,:]+r[None,:,None]*D[:,None,:]).reshape(n,len(r))
        drops=yy[:,:-1]-yy[:,1:]
        im=np.argmax(drops,axis=1)
        L,H=r[im],r[im+1]
        h=np.arange(n)
        yL,yH=yy[h,im],yy[h,im+1]
        init=yL-yH
        for _ in range(8):
            M=.5*(L+H)
            ym=f.query(B+M[:,None]*D)
            ind=np.abs(ym-yL)<np.abs(ym-yH)
            L=np.where(ind,M,L)
            H=np.where(ind,H,M)
            yL=np.where(ind,ym,yL)
            yH=np.where(ind,yH,ym)
        live=(yL-yH > eps)&(yL-yH>.27*init)
        if np.sum(live)>=6:
            p=np.mean(np.vstack(((B+L[:,None]*D)[live],(lo[:,None]*U)[found])),axis=0)
            return -1.,X,Y,p
    return None,X,Y,None
