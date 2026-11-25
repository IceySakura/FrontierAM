"""Synthetic-track control variates built from changes of analytic pieces."""
import numpy as np
from scipy import linalg

from common import log, normalize
from hard_geom import (Quad, approximate_quad, cap_pair_mean, diffs, event_normal,
                       hinge_mean, localize_jumps, localize_kinks, scan)
from jump import jump_integral
from smooth import residual_mean


def projected_other(q,r):
    _,_,_,_,U=q.cap
    a,c,n,k,V=r.cap
    overlap=np.linalg.norm(U.T@V,ord=2)
    log("cap overlap",overlap)
    if overlap>.045 or U.shape[1]+V.shape[1]>len(r.b):
        return None
    W=linalg.qr(V-U@(U.T@V),mode='economic')[0]
    b=W@(W.T@r.b)
    z=Quad(-a*W@W.T,b,r.c)
    z.cap=(a,z.c+b@b/(4*a),float(b@b/(4*a*a)),k,W)
    return z


def split_internal(q, events, roughness):
    """The two signs of a switch Hessian can describe orthogonal caps."""
    lam,U=linalg.eigh(q.A,check_finite=False)
    tol=max(3e-6,.009*np.max(np.abs(lam)))
    pos=lam>tol
    neg=lam<-tol
    if not (np.any(pos) and np.any(neg)):
        return None
    L,R=U[:,neg],U[:,pos]
    bl=L@(L.T@q.b)
    br=-R@(R.T@q.b)
    if np.linalg.norm(q.b-bl+br)>.018*max(.1,np.linalg.norm(q.b)):
        return None
    left=Quad((L*lam[neg])@L.T,bl,0.).as_cap(tol=.019)
    right=Quad(-(R*lam[pos])@R.T,br,-q.c).as_cap(tol=.019)
    if left is None or right is None:
        return None
    fits=[]
    for e in events:
        if q.matches(e):
            continue
        z=e.z
        for a,b in ((left,right),(right,left)):
            g=(2.*a.A@z+a.b)@e.v
            A=e.v@a.A@e.v
            err=min(abs(g-e.d[1])+abs(A-e.d[2]),abs(g+e.d[1])+abs(A+e.d[2]))
            if err>.0015+.013*abs(g):
                continue
            offset=-a(z[None,:])[0]
            if b(z[None,:])[0]+offset>.008:
                continue
            l=Quad(left.A,left.b,left.c+offset).as_cap()
            r=Quad(right.A,right.b,right.c+offset).as_cap()
            if l.cap[1]>0 and r.cap[1]>0:
                score=roughness(lambda x:np.maximum(0.,np.maximum(l(x),r(x))))
                fits.append((score,l,r))
    if not fits:
        return None
    return min(fits,key=lambda v:v[0])


def complement_cap(s,q,e):
    """Try a cap whose space is the entire complement of the first one."""
    d=s.d
    V=q.cap[-1]
    P=np.eye(d)-V@V.T
    if V.shape[1]>=d or s.remaining<2700+9*d:
        return None
    here=q(e.z[None,:])[0]
    if abs(here)<.003:
        return None
    n=event_normal(s,e)
    if n is None:
        return None
    v=np.r_[e.v[None,:],normalize(s.rng.normal(size=(11,d)))]
    m=np.einsum('ni,ni->n',v@P,v)
    r=np.einsum('ni,ni->n',v@q.A,v) if here>0 else np.zeros(len(v))
    eps=.0028
    f=diffs(s,e.z,n,np.r_[np.zeros((1,d)),eps*v,-eps*v],.006,2)
    D=(f[1:13]+f[13:]-2*f[0])/(2*eps*eps)
    if abs(D[0]-e.d[2])>.002+.02*abs(e.d[2]):
        return None
    for sign in (1.,-1.):
        z=sign*D+r
        a=-float(z@m)/max(m@m,1e-20)
        if a<1e-5 or a>4. or np.max(abs(z+a*m))>1.5e-5+.0018*a:
            continue
        h=.0017
        E=np.eye(d)*h
        y=diffs(s,e.z,n,np.r_[E,-E],.005,2)
        g=sign*(y[:d]-y[d:])/(2*h)
        if here>0:
            g+=2*q.A@e.z+q.b
        gg=P@g
        if np.linalg.norm(g-gg)>.0025*max(.1,np.linalg.norm(g)):
            continue
        A=-a*P
        b=gg-2*A@e.z
        c=max(0.,here)-e.z@A@e.z-b@e.z
        cap=Quad(A,b,c).as_cap()
        if cap is not None and cap.cap[1]>0:
            return cap
    return None


def hard_integral(s,x,y):
    js,ks,lines=scan(s,x,y)
    if not js and (len(ks)<3 or np.max(y)>12*np.median(y)) and s.remaining>12000:
        n=880
        xp=s.rng.normal(size=(n,s.d))*np.resize([.28,.46,.64,.83,1.22,1.66,2.2,3.2],n)[:,None]
        yp=s.query(xp)
        jj,kk,ll=scan(s,np.r_[x,xp],np.r_[y,yp],extra=True)
        if jj:
            js,ks,lines=jj,kk,ll
        elif not ks and kk:
            # Keep the rays with an observed interface, not exploratory points.
            ks,lines=kk,ll
    if js:
        events=localize_jumps(s,js)
        log("jump events",len(events))
        if events:
            model=jump_integral(s,x,y,events,lines)
            if model is not None:
                integ,residual=model
                r=residual.__self__
                return residual_mean(s,residual,x[:-1],y[:-1],integ,
                                     try_sparse=r.aq is not None or np.linalg.norm(r.b)<1e-6)
            return None
    if not ks:
        return residual_mean(s,lambda x,y:y,x[:-1],y[:-1],0.)
    events=localize_kinks(s,ks,take=34 if s.d>=32 else 44)
    log("kink events",len(events))
    if not events:
        return None
    events.sort(key=lambda e:e.score)
    events=events[::-1]
    dim=s.d
    center,v,t,f=lines
    paths=(center[:,None]+t[None,:,None]*v[:,None]).reshape(-1,dim)

    def how_smooth(q):
        a=f-q(paths).reshape(f.shape)
        return float(np.mean(np.abs(np.diff(a,n=4,axis=1))**1.25))

    raw=how_smooth(lambda z:np.zeros(len(z)))
    detected=[]
    caps=[]
    failures=0
    complement_tried=False
    for e in events:
        if s.remaining<max(1900,24*dim+1800) or len(detected)>=2 or failures>=2:
            break
        if any(q.matches(e) for q in detected):
            continue
        if dim>=16 and len(caps)==1 and not complement_tried:
            complement_tried=True
            first=caps[0]
            cap=complement_cap(s,first,e)
            if cap is not None:
                best=how_smooth(lambda x:np.maximum(0.,np.maximum(first(x),cap(x))))
                if best<.74*how_smooth(lambda x:np.maximum(0.,first(x))):
                    caps.append(cap)
                    log("full complement cap",best/raw)
                    break
        q=approximate_quad(s,e,thin=True)
        if q is None:
            failures+=1
            continue
        detected.append(q)
        for g in (q,q.change(-1.)):
            c=g.as_cap()
            if c is not None:
                caps.append(c)
                break
        if len(detected)==1 and not caps:
            model=split_internal(q,events,how_smooth)
            one=how_smooth(lambda x:np.maximum(0.,q(x)))
            if model is not None and model[0]<.55*one:
                caps.extend(model[1:])
                log("caps from one internal switch",model[0]/raw)
                break
            if one<.008*raw:
                break
    if not detected:
        return None
    if len(caps)==1 and len(detected)==2:
        c=caps[0]
        rest=[g for g in detected if g.as_cap() is None and g.change(-1.).as_cap() is None]
        if rest:
            for k in (-1.,1.):
                a=c.plus(rest[0],k).as_cap(tol=.022)
                if a is not None:
                    caps.append(a)
                    break
    best=None
    if caps:
        primary=caps[0]
    else:
        primary=detected[0]
    fun=lambda z:np.maximum(primary(z),0.)
    best=(how_smooth(fun),fun,"single",primary,None)
    if len(caps)==2:
        q,r=caps
        rp=projected_other(q,r)
        if rp is not None:
            r=rp
            g=lambda z:np.maximum(np.maximum(q(z),r(z)),0.)
            candidate=(how_smooth(g),g,"max",q,r)
            if candidate[0]<best[0]:
                best=candidate
        g=lambda z:np.maximum(q(z),0.)+np.maximum(r(z),0.)
        cand=(how_smooth(g),g,"sum",q,r)
        if cand[0]<best[0]:
            best=cand
    elif len(detected)==2:
        q,r=detected
        g=lambda z:np.maximum(q(z),0.)+np.maximum(r(z),0.)
        cand=(how_smooth(g),g,"sum",q,r)
        if cand[0]<best[0]*.80:
            best=cand
    rough, fun, kind, q, r=best
    log("hinge model",kind,"roughness ratio",rough/raw)
    if rough>1.05*raw:
        return None
    if kind=="max":
        known=cap_pair_mean(q,r,True)
    elif kind=="sum":
        known=hinge_mean(q)+hinge_mean(r)
    else:
        known=hinge_mean(q)
    if not np.isfinite(known) or known<-.02:
        return None
    controls=(q,) if r is None else (q,r)
    return residual_mean(s,lambda z,f:f-fun(z),x[:-1],y[:-1],known,controls,rough<.012*raw)
