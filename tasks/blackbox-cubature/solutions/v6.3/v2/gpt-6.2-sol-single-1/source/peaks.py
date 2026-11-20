"""Positive mixtures of separable peaks inferred from function samples."""
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from common import Model, residual, solve_ls
from structure import log_gaussian_1d


def peak_logits(x,q,power=None):
    d = x.shape[1]
    z = x[:,None,:]-q[None,:,1:1+d]
    r = np.exp(q[:,1+d:])
    if power is not None:
        return q[None,:,0]-power*np.sum(np.log1p(z*z*r),axis=2)
    return q[None,:,0]-np.einsum("nkd,nkd,kd->nk",z,z,r)


def init_peak(orc, p, v, sign, scale, other=None, local=False, baseline=0.):
    d = len(p)
    D = np.eye(d)
    h = (np.array([-.050,-.018,-.006,.006,.018,.050]) if local else
         np.array([-.115,-.043,-.016,.016,.043,.115]))
    pp = np.clip(p + D[:,None,:]*h[None,:,None],0.,1.)
    offset = baseline(pp.reshape(-1,d)).reshape(d,len(h)) if callable(baseline) else baseline
    vals = sign*(orc.sample(pp.reshape(-1,d)).reshape(d,len(h))/scale-offset)
    if other is not None:
        both = np.exp(np.clip(logsumexp(peak_logits(np.vstack((pp.reshape(-1,d),p)),other),
                                        axis=1),-745.,700.))
        v0 = v-both[-1]
        subtract = vals-both[:-1].reshape(vals.shape)
        usable = np.sum(subtract > vals*1e-6,axis=1)
        if v0 > v*.003 and np.min(usable) >= 2:
            vals = subtract
            v = v0
    v = max(v,1e-290)
    L = np.log(v)
    b, a = np.zeros(d), np.ones(d)*10.
    for j in range(d):
        t = pp[j,:,j]-p[j]
        keep = (vals[j]>1e-290)&(t!=0)
        if np.sum(keep) >= 2:
            tj = t[keep]
            z = np.log(vals[j,keep])-L
            design = np.column_stack([tj,tj*tj])
            weight = 1/np.maximum(.025,np.abs(tj))**.7
            c = solve_ls(design*weight[:,None],z*weight)
            b[j] = c[0]
            a[j] = max(.04,-c[1])
    m = np.clip(p+b/(2*a),-.55,1.55)
    r = np.clip(a,.02,1e9)
    A = min(690.,L+float(np.sum(r*(p-m)**2)))
    return np.r_[A,m,np.log(r)]


def _run_fit(x,y,q,maxfev,objective="log",power=None):
    d = x.shape[1]
    k = len(q)
    ys = np.log(y)
    if objective == 'log':
        def fun(par):
            qq = par.reshape(k,-1)
            return logsumexp(peak_logits(x,qq,power),axis=1)-ys
    else:
        if objective == "balanced":
            norm = np.maximum(y,1e-8)**.55
        else:
            norm = np.maximum(y,1e-12)**.18
        def fun(par):
            qq = par.reshape(k,-1)
            return (np.exp(np.clip(logsumexp(peak_logits(x,qq,power),axis=1),-745.,690.))-y)/norm
    def jac(par):
        qq = par.reshape(k,-1)
        z = x[:,None,:]-qq[None,:,1:1+d]
        r = np.exp(qq[:,1+d:])
        w = r*z*z
        if power is None:
            la = qq[None,:,0]-w.sum(axis=2)
            drv = 1.
        else:
            la = qq[None,:,0]-power*np.sum(np.log1p(w),axis=2)
            drv = power/(1.+w)
        if objective == "log":
            co = np.exp(np.clip(la-logsumexp(la,axis=1)[:,None],-745,0))
        else:
            co = np.exp(np.clip(la,-745,690))/norm[:,None]
        return np.concatenate([co[:,:,None],2*co[:,:,None]*z*r*drv,
                               -co[:,:,None]*w*drv],axis=2).reshape(len(x),-1)
    lo = np.tile(np.r_[-1200.,np.full(d,-1.5),np.full(d,-5.5)],k)
    hi = np.tile(np.r_[690.,np.full(d,2.5),np.full(d,23.)],k)
    sol = least_squares(fun,np.clip(q.ravel(),lo+1e-12,hi-1e-12),
                        jac=jac,bounds=(lo,hi),max_nfev=maxfev,method="trf",
                        ftol=1e-11,xtol=1e-12,gtol=3e-13,x_scale="jac")
    return sol.x.reshape(k,-1)


def _model(q, sign, name):
    d = (q.shape[1]-1)//2
    m,r = q[:,1:1+d]-.5,np.exp(q[:,1+d:])
    ints = [qq[0]-np.sum(rr*mm*mm)+sum(log_gaussian_1d(-a,b)
                                      for a,b in zip(rr,2*rr*mm))
            for qq,rr,mm in zip(q,r,m)]
    volume = logsumexp(ints)
    if volume > 690:
        return None
    def predict(x):
        return sign*np.exp(np.clip(logsumexp(peak_logits(x,q),axis=1),-745.,709.))
    model = Model(predict, sign*math.exp(volume), name)
    model.peak_params = q
    return model


def _peak_error(q,x,y,power=None):
    v = logsumexp(peak_logits(x,q,power),axis=1)
    val = np.exp(np.clip(v,-745,690))
    er = np.max(np.abs(val-y))/max(1.,np.max(np.abs(y)))
    active = y>1e-265
    le = np.max(np.abs(v[active]-np.log(y[active]))) if np.any(active) else np.inf
    return er,le


def gaussian_mixture(orc,x,y,scale,max_components=5):
    d = x.shape[1]
    if d > 10 or orc.left < 180:
        return None
    ref = float(np.max(np.abs(y)))
    sign = 1. if np.max(y) >= -np.min(y) else -1.
    u = sign*y
    if np.min(u) < -1e-14 or np.sum(u>1e-280)<max(16,3*d):
        return None
    if np.min(u) > .87*ref or np.ptp(u)<1e-13:
        return None
    n = int(.82*len(x))
    # Using extremes from the training partition avoids interpolation-only
    # acceptance of a fitted spot.
    pos = int(np.argmax(u[:n]))
    logspan = np.log(ref)-np.log(max(1e-290,np.quantile(u,.21)))
    local = logspan > 41.
    starts = [init_peak(orc,x[pos],u[pos],sign,scale,local=local)]
    best = None
    qlast = None
    for k in range(1,max_components+1):
        if k>1 and (orc.left < 110+6*d or orc.elapsed > 11):
            break
        xx,raw = orc.data()
        xx = np.vstack([x[:n],xx[len(x):]])
        uu = np.r_[u[:n],sign*raw[len(x):]/scale]
        keep=(uu>max(1e-287,2.3e-308/scale))&np.isfinite(uu)
        xx,uu=xx[keep],uu[keep]
        if k>1:
            prev = np.asarray(starts)
            zz0 = logsumexp(peak_logits(xx,prev),axis=1)
            zz1 = zz0 if qlast is None else logsumexp(peak_logits(xx,qlast),axis=1)
            pred0 = np.exp(np.clip(zz0,-745,690))
            pred1 = np.exp(np.clip(zz1,-745,690))
            miss = np.maximum(uu-np.minimum(pred0,pred1),0.)
            nearest = np.min(np.sum((xx[:,None,:]-np.array(starts)[None,:,1:d+1])**2,axis=2),
                             axis=1)
            if local:
                logs = np.log(uu)
                score = np.maximum(logs-zz0,0.) * np.maximum(.035,np.exp((logs-max(logs))/72.))
                score += .035*np.maximum(logs-zz1,0.)
                score *= np.minimum(nearest/.07,1.)
            else:
                score = (miss + .004*uu)*np.minimum(nearest/.017,1.)
            j = np.argmax(score)
            if nearest[j] < .0008:
                j = int(np.argmax(uu*np.minimum(nearest/.05,1.)**2))
            p, val = xx[j],uu[j]
            starts.append(init_peak(orc,p,val,sign,scale,prev,local))
            xx1,raw1 = orc.data()
            j0 = len(raw)
            ok = (sign*raw1[j0:]/scale > 1e-287)
            xx = np.vstack((xx,xx1[j0:][ok]))
            uu = np.r_[uu,(sign*raw1[j0:]/scale)[ok]]
        options = [np.array(starts)]
        if k>1 and qlast is not None:
            options.append(np.vstack((qlast,starts[-1])))
        direct = []
        for i, st in enumerate(options):
            if i and direct and direct[0][0]<2e-8:
                break
            if local:
                e, le = _peak_error(st,x[n:],u[n:])
                ep, lep = _peak_error(st,xx,uu)
                if e < 5e-9 and max(le,lep) < 8e-7:
                    cand = _model(st,sign,"Gaussian mixture")
                    if cand is not None:
                        cand.error,cand.logerror=e,le
                        return cand
            st_fit = _run_fit(xx,uu,st,60 if k<=2 else 75,"log")
            e,le = _peak_error(st_fit,x[n:],u[n:])
            eps,le_fit = _peak_error(st_fit,xx,uu)
            score = max(e,eps*.3) + (.035 if local else .001)*ref*min(le,40.)
            direct.append((score, st_fit, le,e))
            if best is None or e<best.error:
                if e < .19*ref:
                    cand = _model(st_fit,sign,"Gaussian mixture")
                    if cand is not None:
                        cand.error = e
                        cand.logerror = le
                        best = cand
            if e < 5e-10 and max(le,le_fit) < 8e-7 and eps < 2e-9:
                cand = _model(st_fit,sign,"Gaussian mixture")
                if cand is not None:
                    cand.error,cand.logerror=e,le
                    return cand
            if e < .23*ref and orc.elapsed < 12 and (not local or le < 4.):
                st_fit2 = _run_fit(xx,uu,st_fit,38,"balanced")
                e2,le2 = _peak_error(st_fit2,x[n:],u[n:])
                eps2 = _peak_error(st_fit2,xx,uu)[0]
                if eps2 < .13*ref and (best is None or e2<best.error):
                    cand = _model(st_fit2,sign,"Gaussian mixture")
                    if cand is not None:
                        cand.error = e2
                        cand.logerror = le2
                        best = cand
                if e2 < 5e-10 and le2 < 1e-6 and eps2 < 2e-9:
                    cand = _model(st_fit2,sign,"Gaussian mixture")
                    if cand is not None:
                        cand.error,cand.logerror=e2,le2
                        return cand
        qlast = min(direct,key=lambda t:t[0])[1]
        if k==1 and direct[0][2]>16. and direct[0][3]>.6*ref and np.min(u)>.04*ref:
            break
    return best


def peak_audit(orc, model, scale):
    q = model.peak_params
    d = (q.shape[1]-1)//2
    mu = np.clip(q[:,1:d+1],0,1)
    s = np.exp(-.5*q[:,d+1:])
    probes = [mu]
    for a in (.5,1.6):
        probes.extend([np.clip(mu + a*v*s,0.,1.) for v in
                       orc.rng.choice([-1.,1.],size=(4,d))])
    x = np.vstack(probes)
    y = orc.sample(x)/scale
    pred = model.predict(x)
    diff = np.abs(y-pred)
    amp = np.max(np.abs(y))
    floor = 2e-11*amp if model.name=="signed Gaussian mixture" else 0.
    return bool(np.max(diff) <= 1e-8*max(1e-290,amp) and
                np.all(diff <= floor+4e-7*np.maximum(1e-266,np.maximum(np.abs(y),np.abs(pred)))))


def signed_gaussian(orc,x,y,scale):
    d=orc.d
    if d>9 or orc.left<210 or orc.elapsed>9.5:
        return None
    mid=np.median(y)
    off=mid
    r0=y-off
    a,b=r0.min(),r0.max()
    mag=max(-a,b)
    if mag<1e-14 or np.median(abs(r0))>.20*mag:
        return None
    idx=np.argsort(abs(r0))[len(y)*92//100:]
    if np.sum(r0[idx]**2)<.50*np.sum(r0**2):
        return None
    if np.min(y)<-.04*mag and np.max(y)>.04*mag and abs(off)<.12*mag:
        off=0.
    elif abs(off)<max(8e-14,2e-5*mag):
        return None
    else:
        sg=1. if b>=-a else -1.
        # Far corners often give a much better estimate of a flat
        # background than the middle of a small but nonzero tail.
        if orc.left >= 280:
            if d<=5:
                corners=((np.arange(2**d)[:,None]>>np.arange(d))&1)
            else:
                corners=orc.rng.integers(0,2,(28,d))
            v=sg*orc.sample(corners)/scale
            vv=min(np.min(sg*y),np.min(v))
            if np.isfinite(vv) and np.abs(vv-sg*mid)<max(.32*mag,2e-13):
                off=sg*vv
    n=int(.82*len(y))
    r=y[:n]-off
    signs=[]
    seeds=[]
    for s in (1.,-1.):
        j=int(np.argmax(s*r))
        if s*r[j]>.065*mag:
            signs.append(s)
            seeds.append(init_peak(orc,x[j],s*r[j],s,scale,baseline=off))
    if not seeds:
        return None
    best=None
    for _ in range(5-len(seeds)):
        k=len(seeds)
        xx,yy=orc.data()
        xx,yy=np.vstack((x[:n],xx[len(x):])),np.r_[y[:n],yy[len(x):]/scale]
        sg=np.array(signs)
        norm=np.maximum(abs(yy-off),.0003*mag)**.49
        def vals(t,p):
            qq=p[1:].reshape(k,-1)
            z=t[:,None,:]-qq[None,:,1:d+1]
            R=np.exp(qq[:,d+1:])
            q=R*z*z
            a=sg*np.exp(np.clip(qq[:,0]-q.sum(axis=2),-745,100.))
            return a,z,R,q
        def fun(p):
            return (p[0]+np.sum(vals(xx,p)[0],axis=1)-yy)/norm
        def jac(p):
            v,z,r,q=vals(xx,p)
            J=np.concatenate((v[:,:,None],2*v[:,:,None]*r*z,-v[:,:,None]*q),axis=2)
            return np.column_stack((np.ones(len(xx)),J.reshape(len(xx),-1)))/norm[:,None]
        lower=np.r_[-np.inf,np.tile(np.r_[-90.,np.full(d,-1.5),np.full(d,-5.1)],k)]
        upper=np.r_[np.inf,np.tile(np.r_[75.,np.full(d,2.5),np.full(d,22.3)],k)]
        theta=np.r_[off,np.array(seeds).ravel()]
        theta=np.clip(theta,lower+1e-12,upper-1e-12)
        sol=least_squares(fun,theta,jac=jac,bounds=(lower,upper),max_nfev=64,
                        method="trf",ftol=4e-11,xtol=3e-12,gtol=2e-13,x_scale="jac")
        theta=sol.x.copy()
        def predict(p, theta=theta,sg=sg):
            qq=theta[1:].reshape(len(sg),-1)
            v=np.exp(np.clip(peak_logits(p,qq),-745,100.))
            return theta[0]+v@sg
        err=np.max(np.abs(predict(x[n:])-y[n:]))
        err_train=np.max(np.abs(predict(xx)-yy))
        q=theta[1:].reshape(k,-1)
        if err < .16*mag and (best is None or err<best.error):
            mu,R=q[:,1:d+1]-.5,np.exp(q[:,d+1:])
            moment=[]
            for v,mm,rr,s in zip(q,mu,R,sg):
                l=v[0]-np.sum(rr*mm*mm)+sum(log_gaussian_1d(-aa,bb)
                                               for aa,bb in zip(rr,2*rr*mm))
                moment.append(s*math.exp(l))
            best=Model(predict,math.fsum([theta[0]]+moment),"signed Gaussian mixture",float(err))
            best.peak_params=q.copy()
            if max(err,err_train)<2e-9*max(1.,mag):
                return best
        if orc.left<100+6*d or orc.elapsed>11:
            break
        diff=yy-predict(xx)
        dist=np.linalg.norm(xx[:,None,:]-q[None,:,1:d+1],axis=2).min(axis=1)
        i=np.argmax(abs(diff)*np.minimum(dist/.11,1.))
        vv=diff[i]
        if abs(vv)<1e-14:
            break
        s=np.sign(vv)
        seeds=list(q)+[init_peak(orc,xx[i],abs(vv),s,scale,baseline=predict)]
        signs.append(s)
        off=theta[0]
    return best


def _rational_seed(orc, p, pv, scale, sign, power, other=None):
    d = len(p)
    offsets = np.array([-.195,-.115,-.043,-.016,.016,.043,.115,.195])
    pts = np.clip(p+np.eye(d)[:,None,:]*offsets[None,:,None],0.,1.)
    all_points = np.vstack((pts.reshape(-1,d),p))
    yy = sign*orc.sample(all_points)/scale
    if other is not None:
        est = np.exp(logsumexp(peak_logits(all_points,other,power),axis=1))
        delta = yy-est
        if delta[-1] > .01*yy[-1] and (delta[:-1].reshape(d,-1)>1e-12).sum(axis=1).min()>=3:
            yy = delta
    vv = yy[-1]
    v = yy[:-1].reshape(d,-1)
    mu, rho = [], []
    goodness = []
    for j in range(d):
        t = pts[j,:,j]-p[j]
        u = v[j]/vv
        use = (u>1e-12) & (t!=0)
        uu,tt = u[use],t[use]
        if len(tt) < 3:
            mu.append(p[j])
            rho.append(.12)
            goodness.append(3.)
            continue
        ll = -np.log(uu)/power
        z = np.exp(ll)
        options = []
        for w in (mult for mult in (np.ones(len(z)),np.sqrt(uu),uu)):
            design = np.column_stack((tt,tt*tt))
            cc = solve_ls(design*w[:,None],np.expm1(ll)*w)
            a,b = cc[1],cc[0]
            if a <= 0.:
                continue
            m = -b/(2*a)
            s2 = 1./a-m*m
            if s2 <= 1e-14 or abs(m) > 1.2:
                continue
            slope = (s2+(tt-m)**2)/(s2+m*m)
            err = np.sqrt(np.mean((np.log(slope)-ll)**2))
            options.append((err,m+p[j],math.sqrt(s2)))
        if options:
            e,m,s = min(options,key=lambda a:a[0])
        else:
            # A mixed slice may have a poor rational seed. Its largest
            # value still gives a useful center for a subsequent joint fit.
            e,m,s=1.5,p[j]+tt[np.argmax(uu)],.083
        mu.append(m)
        rho.append(s)
        goodness.append(e)
    mu,rho = np.asarray(mu),np.asarray(rho)
    lw = np.clip(-2*np.log(rho),-5.,21.7)
    A = np.log(vv)+power*np.sum(np.log1p((p-mu)**2*np.exp(lw)))
    return np.r_[A,mu,lw],float(np.median(goodness))


def rational_mixture(orc,x,y,scale):
    d = orc.d
    if d>9 or orc.left < 230 or orc.elapsed>10:
        return None
    sign = 1. if np.max(y)>=-np.min(y) else -1.
    u = sign*y
    if np.min(u)<=0 or np.median(u)>.25*np.max(u):
        return None
    n = int(len(x)*.82)
    # Reuse the strongest anchor for both candidate tail powers, so
    # classification costs only a single set of line probes.
    j = np.argmax(u[:n])
    seeds = [_rational_seed(orc,x[j],u[j],scale,sign,power)+(power,) for power in (1.,2.,.5)]
    q0,quality,power = min(seeds,key=lambda c:c[1]*(1. if c[2]==1. else 1.15))
    if quality>.39:
        return None
    qs = [q0]
    best = None
    previous = None
    for k in range(1,4):
        xx,yy = orc.data()
        xx,yy = np.vstack((x[:n],xx[len(x):])),np.r_[u[:n],sign*yy[len(x):]/scale]
        use = (yy>1e-280)&np.isfinite(yy)
        xx,yy=xx[use],yy[use]
        if len(yy)<3*d or orc.elapsed>12:
            break
        if k>1:
            if orc.left < 8*d+90:
                break
            other = np.asarray(qs)
            b = logsumexp(peak_logits(xx,other,power),axis=1)
            err = np.log(yy)-b
            if previous is not None:
                err = np.maximum(err,np.log(yy)-logsumexp(peak_logits(xx,previous,power),axis=1))
            dist = np.min(np.linalg.norm(xx[:,None,:]-other[None,:,1:d+1],axis=2),axis=1)
            importance = np.maximum(err,0.)*np.minimum(dist/.11,1.)*np.maximum(2e-5,(yy/yy.max())**.21)
            j = np.argmax(importance)
            q,qu = _rational_seed(orc,xx[j],yy[j],scale,sign,power,other)
            qs.append(q)
            all_x,all_y=orc.data()
            xx,yy = np.vstack((x[:n],all_x[len(x):])),np.r_[u[:n],sign*all_y[len(x):]/scale]
            use = (yy>1e-280)&np.isfinite(yy)
            xx,yy=xx[use],yy[use]
        initial = np.asarray(qs)
        opts = [initial]
        if previous is not None:
            opts.append(np.vstack((previous,qs[-1])))
        qchoice = (np.inf,None)
        for guess in opts:
            qfit = _run_fit(xx,yy,guess,60 if k<=2 else 45,power=power)
            e,le = _peak_error(qfit,x[n:],u[n:],power)
            et,lt = _peak_error(qfit,xx,yy,power)
            score = e + .04*max(le,lt)*np.max(u)+.3*et
            if score<qchoice[0]:
                qchoice = score,qfit
            if e<.12*np.max(u) and (best is None or e<best.error):
                best = _rational_model(qfit,sign,power)
                best.error,best.logerror=e,le
            if max(e,et)<3e-9 and max(le,lt)<8e-7:
                model = _rational_model(qfit,sign,power)
                model.error,model.logerror=e,le
                return model
        previous = qchoice[1]
    return best


def _rational_model(q,sign,power):
    d = (q.shape[1]-1)//2
    x0, sig = q[:,1:1+d],np.exp(-.5*q[:,1+d:])
    a,b = -x0/sig,(1.-x0)/sig
    if power==1.:
        int1 = sig*(np.arctan(b)-np.arctan(a))
    elif power==2.:
        int1 = .5*sig*((np.arctan(b)-np.arctan(a))+b/(1+b*b)-a/(1+a*a))
    else:
        int1 = sig*(np.arcsinh(b)-np.arcsinh(a))
    vol = logsumexp(q[:,0]+np.log(int1).sum(axis=1))
    def predict(p):
        return sign*np.exp(np.clip(logsumexp(peak_logits(p,q,power),axis=1),-745,709))
    model = Model(predict, sign*math.exp(min(695.,vol)), "rational peak mixture")
    model.peak_params = q
    return model
