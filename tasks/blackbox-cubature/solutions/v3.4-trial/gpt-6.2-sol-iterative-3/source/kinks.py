"""Recover candidate quadrics from locations and normals of continuous kinks."""

import math

import numpy as np
from scipy import linalg, special

from numerics import local_gradient, quadratic_features, residual_average, ridge, rotation
import smooth_cv


def _line_candidates(w, survey):
    d = w.d
    n = 48 if d < 32 else 64
    v = w.normal(n)
    v *= math.sqrt(d)/np.linalg.norm(v,axis=1)[:,None]
    cen = w.normal(n)*.39
    cen[:n//2] = 0.
    step = .16
    t = step*np.arange(-10,11)
    y = w.query((cen[:,None,:]+v[:,None,:]*t[None,:,None]).reshape(-1,d))
    y = y.reshape(n,len(t))
    a, b, ts, strength = [], [], [], []
    chunks = [(cen,v,t,y)]
    if survey is not None:
        vv, tt, yy = survey
        chunks.append((np.zeros_like(vv),vv,tt,yy))
    for cc, vv, tt, yy in chunks:
        dd = yy[:,2:]-2.*yy[:,1:-1]+yy[:,:-2]
        # Compare to more distant second differences: smooth quadratics
        # should not use most of the limited normal-observation budget.
        subl = np.column_stack([dd[:,:2],dd[:,:-2]])
        subr = np.column_stack([dd[:,2:],dd[:,-2:]])
        gain = np.abs(dd)-.72*np.minimum(np.abs(subl),np.abs(subr))
        i0 = np.arange(len(vv))
        j = np.argmax(gain,axis=1)
        for repeat in range(2):
            if repeat:
                gain[np.abs(np.arange(gain.shape[1])-j[:,None]) <= 2] = -1.
                j = np.argmax(gain,axis=1)
            use = gain[i0,j] > 3.e-5
            a.extend(cc[use])
            b.extend(vv[use])
            ts.extend(tt[j[use]+1])
            strength.extend(gain[i0,j][use])
    if not a:
        return (np.empty((0,d)),)*2+(np.empty(0),)*2
    order = np.argsort(strength)[::-1][:128 if d<64 else 148]
    return np.asarray(a)[order],np.asarray(b)[order],np.asarray(ts)[order],np.asarray(strength)[order]


def _refine(w, a, v, t, h):
    loc = t[:,None] + h*np.array([-2.,-1.,1.,2.])
    xs = a[:,None,:]+loc[:,:,None]*v[:,None,:]
    f0,f1,f2,f3 = w.query(xs.reshape(-1,w.d)).reshape(-1,4).T
    gl, gr = (f1-f0)/h, (f3-f2)/h
    g = gr-gl
    offset = ((2.*f1-f0)-(2.*f2-f3))/np.where(abs(g)>1.e-12,g,1.e-12)
    return t+np.clip(offset,-h,h),g,offset,xs[:,[1,2]],np.column_stack([f1,f2])


def _normals(w, survey, scale):
    d = w.d
    a,v,t,_ = _line_candidates(w,survey)
    if len(t)<4:
        return None
    old = None
    for h in [.162, .071, .017, .0031]:
        tnew, g, off, xs, ys = _refine(w,a,v,t,h)
        keep = (np.abs(g) > max(3.e-5,.0018*scale)) & (np.abs(off)<.96*h)
        if old is not None:
            keep &= (np.sign(old)==np.sign(g)) & (np.abs(g)>.42*np.abs(old))
        if np.sum(keep) < 4:
            return None
        a,v,t,old = a[keep],v[keep],tnew[keep],g[keep]
    limit = min(len(t), 76 if d<=16 else (63 if d<=32 else 46),
                (w.remaining-5248)//(2*d+8))
    if limit < 5:
        return None
    # Spread the selected roots through the two slope-sign groups, with
    # preference for a well-resolved jump and a nonduplicate location.
    order = np.argsort(np.abs(old))[::-1]
    take = []
    space = a + t[:,None]*v
    for i in order:
        if len(take)>=limit:
            break
        if not take or np.min(np.linalg.norm(space[take]-space[i],axis=1))>.004:
            take.append(i)
    rest = np.ones(len(space),dtype=bool)
    rest[take] = False
    other = space[rest],old[rest]
    a,v,t,old = a[take],v[take],t[take],old[take]
    for h in [.00057,.000052]:
        t0,g,off,xs,ys = _refine(w,a,v,t,h)
        keep = (np.abs(off) < h*.80) & (g*old > 0.) & (np.abs(g)>.82*np.abs(old))
        a,v,t,old = a[keep],v[keep],t0[keep],g[keep]
        xs,ys = xs[keep],ys[keep]
        if len(t)<4:
            return None
    grads = local_gradient(w,xs.reshape(-1,d),ys.ravel(),.000004).reshape(-1,2,d)
    jump = grads[:,1]-grads[:,0]
    slope = np.sum(jump*v,axis=1)
    good = (abs(slope-old) < .06*abs(old)+2.e-5*scale)
    return (a+t[:,None]*v)[good],jump[good],old[good],other


def _cluster(q, g, sg):
    n = len(q)
    if n < 5:
        return []
    lengths = np.maximum(np.linalg.norm(g,axis=1),1.e-100)
    g = g / lengths[:,None]
    z = g@q.T
    d = z - np.diag(z)[:,None]
    raw = np.abs(d)*lengths[:,None]
    graph = np.abs(raw-raw.T) < .0032*(raw+raw.T)+1.4e-4
    graph &= sg[:,None]*sg[None,:]>0.
    graph &= ((raw+raw.T)>.015) | np.eye(n,dtype=bool)
    active = np.ones(n,bool)
    clusters = []

    def synchronize(jj):
        for _ in range(8):
            if len(jj)<max(5,min(10,n//7)):
                break
            K = d[np.ix_(jj,jj)]
            weight = 1./(K*K+K.T*K.T+1.e-4)
            M = -K*K.T*weight
            M[np.diag_indices(len(jj))] = np.sum(K*K*weight,axis=1)
            eig, vecs = linalg.eigh(M,subset_by_index=[0,1],check_finite=False)
            u = vecs[:,0]
            rel = u[:,None]*K
            r = np.abs(rel-rel.T)/(np.abs(rel)+np.abs(rel.T)+2.e-4)
            point_badness = np.median(r,axis=1)
            if eig[0]<max(.006*eig[1],.0006) and np.median(point_badness)<.023:
                us = np.abs(u)
                good = (us>np.median(us)*.06) & (point_badness<.10)
                if np.sum(good)<5:
                    break
                beta = np.median(lengths[jj[good]]/us[good])
                return jj[good],g[jj[good]]*(u[good]*beta)[:,None]
            worst = np.argsort(point_badness)[-max(1,len(jj)//16):]
            ii = np.ones(len(jj),dtype=bool)
            ii[worst] = False
            jj = jj[ii]
        return None

    for _ in range(3):
        k = np.flatnonzero(active)
        if len(k)<6:
            break
        i,j = np.triu_indices(len(k),1)
        i,j = k[i],k[j]
        keep = (abs(d[i,j])>.018) & (abs(d[j,i])>.018) & (sg[i]*sg[j]>0)
        i,j = i[keep],j[keep]
        if not len(i):
            break
        # A single quadric makes all three-chord products agree.
        a = d[i,j,None]*d[j,:]*d[:,i].T
        b = d[j,i,None]*d[:,j].T*d[i,:]
        graph1 = graph[k,:] & active[None,:]
        count = graph1.sum(axis=1)
        cand = [np.flatnonzero(graph1[r0]) for r0 in np.argsort(count)[-7:][::-1]]
        for tol in [.0025,.012,.043]:
            passes = np.abs(a-b) < tol*(np.abs(a)+np.abs(b)) + 1.e-8
            passes &= active[None,:] & (sg[i,None]*sg[None,:]>0)
            passes &= ((np.abs(d[i,:])+np.abs(d[:,i].T))>.010)
            passes[np.arange(len(i)),i] = True
            passes[np.arange(len(j)),j] = True
            count = passes.sum(axis=1)
            cand.extend(np.flatnonzero(passes[r0]) for r0 in np.argsort(count)[-6:][::-1])
        cand.sort(key=len,reverse=True)
        best = None
        for j0 in cand:
            if best is not None and len(j0)<=len(best[0]):
                break
            if len(j0)<max(6,min(10,n//7)):
                break
            trygroup = synchronize(j0)
            if trygroup is not None and (best is None or len(trygroup[0])>len(best[0])):
                best = trygroup
        if best is None:
            break
        clusters.append(best)
        active[best[0]] = False
    return clusters


class Kink:
    def __init__(self, A, b, c, quality):
        self.A, self.b, self.c = A,b,c
        self.quality = quality
        self.lam,self.q = linalg.eigh(A,check_finite=False)
        self.h = self.q.T @ b

    def __call__(self,x):
        return np.sum((x@self.A)*x,axis=1) + x@self.b + self.c

    def abs_mean(self):
        # E |Q| = 2/pi times the integral of (1-Re cf(t))/t^2.
        a,b,c = self.lam,self.h,self.c
        h, w = special.roots_legendre(64)
        out = 0.
        for j in range(14):
            hi = 2.**j
            lo = 0. if j==0 else .5*hi
            x = (hi+lo)*.5 + (hi-lo)*.5*h
            t = 1j*x[:,None]
            v = 1.-2.*t*a
            log = c*t[:,0] + .5*np.sum((t*b)**2/v-np.log(v),axis=1)
            out += (hi-lo)*.5*(w@(-np.expm1(log).real/(x*x)))
            if j>=4 and np.max(log.real) < -22.:
                break
        return float(2./math.pi * (out+1./hi))


def _envelope_mean(w, first, second, rotation0, weights):
    cs = []
    for q,c in zip((first,second),weights):
        cs.append((c*(rotation0.T@q.A@rotation0), c*(rotation0.T@q.b), c*q.c))

    def cuts(a,b,c):
        if abs(a)<1.e-11:
            f = np.where(abs(b)>1.e-11, -c/np.where(abs(b)>1.e-11,b,1.),9.)
            return f,np.full(len(b),9.)
        ds = b*b-4.*a*c
        p = -.5*(b+np.copysign(np.sqrt(np.maximum(ds,0.)),b))
        r1 = np.where(ds>0.,p/a,9.)
        r2 = np.where(ds>0.,c/np.where(abs(p)>1.e-20,p,1.),9.)
        return r1,r2

    n = 1<<17
    xx = w.normal(n,w.d-1)
    total = 0.
    for x in np.array_split(xx,n//4096):
        pars = []
        for A,b,c in cs:
            f = np.sum((x@A[1:,1:])*x,axis=1)+x@b[1:]+c
            pars.append((A[0,0],2.*x@A[1:,0]+b[0],f))
        (a,b,c),(aa,bb,cc) = pars
        nodes = np.column_stack([np.full(len(x),-9.),np.full(len(x),9.),*cuts(a,b,c),
                                  *cuts(aa,bb,cc),*cuts(a-aa,b-bb,c-cc)])
        nodes = np.sort(np.clip(nodes,-9.,9.),axis=1)
        l,h = nodes[:,:-1],nodes[:,1:]
        t = .5*(l+h)
        v = a*t*t+b[:,None]*t+c[:,None]
        vv = aa*t*t+bb[:,None]*t+cc[:,None]
        f1 = (v>=vv)&(v>0.)
        f2 = (vv>v)&(vv>0.)
        mu0 = np.where(l>0.,special.ndtr(-l)-special.ndtr(-h),
                       special.ndtr(h)-special.ndtr(l))
        p1,p2 = np.exp(-.5*l*l)/math.sqrt(2.*math.pi),np.exp(-.5*h*h)/math.sqrt(2.*math.pi)
        mu1 = p1-p2
        mu2 = mu0+l*p1-h*p2
        s1 = a*mu2+b[:,None]*mu1+c[:,None]*mu0
        s2 = aa*mu2+bb[:,None]*mu1+cc[:,None]*mu0
        total += np.sum(f1*s1+f2*s2)
    return float(total/n)


def _interaction(w,kk,near,sign,amp,resid,basis,hinge_mean):
    if len(kk)<2:
        return None,0.
    scale = np.maximum(np.abs(amp),.06)
    tr = np.arange(len(resid))%5 != 0
    rt = resid[tr]
    rr = resid[~tr]
    best = None
    target = .86*np.var(rr)
    for i in range(len(kk)):
        for j in range(i+1,len(kk)):
            f = np.minimum(near[:,i]*scale[i],near[:,j]*scale[j])
            if np.count_nonzero(f)<.04*len(f):
                continue
            v = f[tr]
            m = float(v.mean())
            h = v-m
            c = np.clip(np.mean(h*rt)/max(np.mean(h*h),1.e-12),-2.0,2.0)
            mse = np.var(rr - c*(f[~tr]-m))
            if mse<target:
                target = mse
                best = i,j,c,m
    if best is None:
        return None,0.
    i,j,c,m = best
    ds = _envelope_mean(w,kk[i],kk[j],basis,(scale[i]*sign[i],scale[j]*sign[j]))
    mean = c*(scale[i]*hinge_mean[i]+scale[j]*hinge_mean[j]-ds-m)
    def feature(z):
        return c*(np.minimum(scale[i]*np.maximum(kk[i](z)*sign[i],0.),
                             scale[j]*np.maximum(kk[j](z)*sign[j],0.))-m)
    return feature,mean


def _fit_kink(x,g,other,full=False):
    xm,gm = x.mean(axis=0),g.mean(axis=0)
    xc = x-xm
    u0,ls,U0 = linalg.svd(g,full_matrices=False,check_finite=False)
    ratio = ls[:-1]/np.maximum(ls[1:],1.e-20)
    good = (ratio>12.)&(ls[:-1]>ls[0]*.016)
    guided = False
    if np.any(good):
        rank = int(np.flatnonzero(good)[-1]+1)
        if rank<len(x)-2:
            U = U0[:rank].T
            D = xc@U
            B = linalg.lstsq(D,(g-gm)@U,cond=.023,check_finite=False)[0]
            A = .25*U@(B+B.T)@U.T
            guided = True
    if not guided:
        u,s,v = linalg.svd(xc,full_matrices=True,check_finite=False)
        take = np.flatnonzero(s > max(.11, s[0]*.022))
        if len(take) < 2:
            return None
        V = v[take].T
        G = (u[:,take].T @ (g-gm))/s[take,None]
        proj = V.T @ G.T
        proj = .5*(proj+proj.T)
        W = V @ G
        A = .5*(W+W.T - V@proj@V.T)
    b = gm - 2.*A@xm
    vv = np.sum((x@A)*x,axis=1)+x@b
    c = -np.median(vv)
    scale = .8*np.sqrt(2.*np.sum(A*A)+b@b)+.2*abs(np.trace(A)+c)
    if scale < 1.e-8:
        return None
    quality = np.sqrt(np.mean((vv+c)**2))/scale
    if quality > .27:
        return None
    null = None
    if not guided and full and v.shape[0]>len(take):
        reject = np.ones(v.shape[0],dtype=bool)
        reject[take] = False
        ns = v[reject].T
        null = ns
    if null is not None and len(other)>max(16,len(x)//3):
        xn = (other-xm)@ns
        tar = -(np.sum((other@A)*other,axis=1) + other@b + c)
        ok = abs(tar)<max(1.05,4.*quality)*scale
        p = ns.shape[1]
        if np.sum(ok)>15 and p<38:
            j,k = np.triu_indices(p,1)
            ws = np.column_stack([xn*xn, 2.*xn[:,j]*xn[:,k]])
            coef,_ = ridge(ws[ok],tar[ok],.21,center=False)
            dif = np.diag(coef[:p])
            dif[j,k] = dif[k,j] = coef[p:]
            dif = ns@dif@ns.T
            A += dif
            b -= 2.*dif@xm
            c += xm@dif@xm
    result = Kink(A/scale,b/scale,c/scale,quality)
    result.null,result.anchor = null,xm
    return result


def _complete_quadric(w,k,roots,grad,orient):
    W = k.null
    if W is None or k.quality>.024 or w.remaining<4950:
        return k
    p = W.shape[1]
    n_cross = p*(p-1)//2
    if p<2 or p>36:
        return k
    budget = min(2304,w.remaining-4080)
    n = min(p+n_cross,budget//8)
    if n < min(p+n_cross,2*p):
        return k
    g = grad-(grad@W)@W.T
    j = int(np.argmax(np.linalg.norm(g,axis=1)))
    x0 = roots[j]
    e = g[j]/max(np.linalg.norm(g[j]),1.e-15)
    if np.linalg.norm(e)<.9:
        return k
    ia,ib = np.triu_indices(p,1)
    idx = np.arange(n_cross)
    if n < p+n_cross:
        idx = w.rng.choice(idx,n-p,replace=False)
    U = np.vstack([np.eye(p),(np.eye(p)[ia[idx]]+np.eye(p)[ib[idx]])*2.**-.5])
    a = x0+.76*(U@W.T)
    c = k(a)
    g0 = 2.*a@(k.A@e)+k.b@e
    g1 = e@k.A@e
    dd = g0*g0-4.*g1*c
    r = -2.*c/np.where(abs(g0)>.015,g0+np.copysign(np.sqrt(np.maximum(dd,0.)),g0),np.inf)
    ok = (dd>0.) & (abs(g0)>.045) & (abs(r)<.55)
    v = np.broadcast_to(e,(sum(ok),w.d))
    a,r,g0 = a[ok],r[ok],g0[ok]
    if len(r)<p:
        return k
    rr,g2,o2,_,_ = _refine(w,a,v,r,.105)
    keep = (abs(o2)<.83*.105) & (g2*orient>0.)
    a,v,rr,g2 = a[keep],v[keep],rr[keep],g2[keep]
    if len(rr)<p:
        return k
    rr,g3,o3,_,_ = _refine(w,a,v,rr,.0185)
    keep = (abs(o3)<.88*.0185) & (g3*g2>0.) & (abs(g3)>.65*abs(g2))
    if np.sum(keep)<max(p,.58*n):
        return k
    z = a[keep]+rr[keep,None]*e
    xx = (z-k.anchor)@W
    gshape = k(z)
    F = np.column_stack([xx*xx, 2.*xx[:,ia]*xx[:,ib]])
    c,_ = ridge(F,-gshape,.00085,center=False)
    if np.std(gshape+F@c)>.55*max(np.std(gshape),1.e-7):
        return k
    H = np.diag(c[:p])
    H[ia,ib] = H[ib,ia] = c[p:]
    H = W@H@W.T
    return Kink(k.A+H,k.b-2.*H@k.anchor,k.c+k.anchor@H@k.anchor,k.quality)


def estimate(w,x,y,survey):
    d = w.d
    res = _normals(w,survey,max(.01,float(np.std(y))))
    if res is None:
        return None
    roots, g, sg, other = res
    if len(roots)<5:
        return None
    w.kink_cov = g.T@g/len(g)
    groups = _cluster(roots,g,sg)
    kk = []
    for j, grad in groups:
        if len(j)<5:
            continue
        osites, oslope = other
        matched = oslope * sg[j[0]]>0.
        k = _fit_kink(roots[j],grad,osites[matched],len(j)>.77*len(roots))
        if k is not None:
            if len(groups)==1 and len(j)>=.80*len(roots) and d>=32:
                k = _complete_quadric(w,k,roots[j],grad,np.sign(np.median(sg[j])))
            ng = np.linalg.norm(2.*(roots[j]@k.A)+k.b,axis=1)
            amp = np.linalg.norm(g[j],axis=1)/np.maximum(ng,1.e-15)
            k.amp = np.median(amp)*np.sign(np.median(sg[j]))
            err = np.abs(amp - abs(k.amp))/max(abs(k.amp),1.e-15)
            k.amp_error = np.quantile(err,.8)
            kk.append(k)
    if not kk:
        return None
    # Coordinates for the surrogate, ordered using the recovered surfaces.
    cm = sum(k.A@k.A + .22*np.outer(k.b,k.b) for k in kk)
    for k in kk:
        cm += .08*np.outer(k.b,k.b)
    b0 = rotation(cm)
    n_fit = min(1536,w.remaining//4)
    raw_train = w.normal(n_fit)
    ztr = raw_train@b0.T
    y = np.r_[y,w.query(ztr)]
    ztr = np.vstack([x,ztr])
    xr = ztr@b0
    keep = min(d,12)
    base = quadratic_features(xr,width=keep)

    absmean = np.array([k.abs_mean() for k in kk])
    qmean = np.array([float(np.trace(k.A)+k.c) for k in kk])
    amp = np.array([k.amp for k in kk])
    penalty = np.array([.55 if k.amp_error<.011 and k.quality<.004 else .0018 for k in kk])
    qt = np.column_stack([k(ztr) for k in kk])
    vals = np.array([1. if not (s>>j)&1 else -1. for s in range(2**len(kk)) for j in range(len(kk))])
    vals = vals.reshape(-1,len(kk))
    scores = [np.std(y-np.maximum(qt*v,0.)@amp) for v in vals]
    sign = vals[np.argmin(scores)]
    near = np.maximum(qt*sign,0.)
    if min(scores) > .66*max(np.std(y),1.e-10):
        penalty *= .016
    # Include centered quadratic values themselves so branch orientation
    # does not restrict the smooth part of the regression.
    def features(z, base):
        qp = np.column_stack([k(z) for k in kk])
        return np.column_stack([base,qp-qmean,np.maximum(qp*sign,0.)])

    feat = np.column_stack([base,qt-qmean,near])
    adj = np.where(penalty >= .01,amp,0.)
    c,m = ridge(feat,y-near@adj,np.r_[np.full(base.shape[1],.0032),
                                     np.full(len(kk),.0025),penalty])
    c[-len(kk):] += adj
    hinge_mean = .5*(absmean+sign*qmean)
    exp = m + c[-len(kk):] @ hinge_mean
    cv = feat@c+m
    interact,ex_interact = _interaction(w,kk,near,sign,amp,y-cv,b0,hinge_mean)
    if interact is not None:
        cv += interact(ztr)
        exp += ex_interact
    ids = np.linspace(0,len(ztr)-1,min(15,len(ztr))).astype(int)
    dx = .0015
    z = ztr[ids]
    g = local_gradient(w,z,y[ids],dx)
    probes = (z[:,None,:]+np.eye(d)[None,:,:]*dx).reshape(-1,d)
    pv = features(probes,quadratic_features(probes@b0,width=keep))@c+m
    if interact is not None:
        pv += interact(probes)
    diff = g-(pv.reshape(-1,d)-cv[ids,None])/dx
    maxn = np.median(np.linalg.norm(diff,axis=1))*4.+1.e-7
    diff /= np.maximum(1.,np.linalg.norm(diff,axis=1)/maxn)[:,None]
    a = diff.T@diff/len(diff)
    cov = a + .16*np.trace(a)*cm/max(np.trace(cm),1.e-15)
    basis = rotation(cov)
    booster = smooth_cv.fit(ztr,y-cv,a)
    if booster is not None:
        exp += booster.mean

    def resid(z,observed):
        r = observed-(features(z,quadratic_features(z@b0,width=keep))@c+m)
        if interact is not None:
            r -= interact(z)
        if booster is not None:
            r -= booster(z)
        return r
    if booster is not None:
        return float(exp + smooth_cv.crossfit(w,basis,resid,a,max(k.quality for k in kk)<.012))
    return float(exp + residual_average(w,basis,resid,max(k.quality for k in kk)<.012))
