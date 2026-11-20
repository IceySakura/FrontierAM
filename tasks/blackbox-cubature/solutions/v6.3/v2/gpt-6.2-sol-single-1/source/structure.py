import math

import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_ndtr, logsumexp

from common import (AxisQuad, Model, NoBudget, gl_interval, legmat, multiindices,
                    residual, solve_ls)


def polynomial(x, y):
    d = x.shape[1]
    n = int(.8 * len(x))
    for degree in range(1, min(8, 4 + (d < 4)*3)):
        ids = multiindices(d, degree, min(degree, 3))
        if ids.shape[1] > n * .82:
            break
        a = legmat(x, ids)
        c = solve_ls(a[:n], y[:n])
        err = np.max(np.abs(a[n:] @ c - y[n:]))
        if err < 5e-11:
            c = solve_ls(a, y)
            model=Model(lambda p: legmat(p, ids) @ c, float(c[0]), "polynomial")
            model.degree=degree
            return model
    return None


def log_gaussian_1d(a, b):
    """log integral of exp(a*t*t + b*t), -1/2 <= t <= 1/2."""
    if a < -2.:
        w = np.sqrt(-a)
        m = b / (-2.*a)
        hi, lo = (0.5-m)*w*np.sqrt(2.), (-0.5-m)*w*np.sqrt(2.)
        if hi + lo > 0.:
            hi, lo = -lo, -hi
        lhi, llo = log_ndtr(hi), log_ndtr(lo)
        gap = -np.expm1(llo-lhi)
        if gap > 0.:
            return float(lhi + np.log(gap) + np.log(np.sqrt(np.pi)/w) + b*m/2)
    x, wt = gl_interval(96, -.5, .5)
    return float(logsumexp(a*x*x+b*x, b=wt))


def transformed(x, y):
    d = x.shape[1]
    sign = 1 if abs(np.max(y)) >= abs(np.min(y)) else -1
    u = sign * y
    pos = (u > 1e-290)
    if np.sum(pos) < d+2 or np.any(u < -4e-14):
        return None
    z = x-.5
    aa = np.column_stack([np.ones(len(x)), z, z*z])
    zz = aa[pos]
    v = np.log(u[pos])
    if len(v) >= zz.shape[1]:
        c = solve_ls(zz, v)
        log_pred = aa@c
        if (np.max(np.abs(zz@c - v)) < 3e-7 and
                np.max(np.abs(np.exp(np.clip(log_pred,-745,709))-u)) < 1e-11):
            def predict(p):
                pp = p-.5
                return sign*np.exp(np.clip(c[0] + pp@c[1:d+1]+(pp*pp)@c[d+1:],-745,709))
            integ = c[0] + sum(log_gaussian_1d(a,b) for a,b in zip(c[d+1:], c[1:d+1]))
            if integ < 700:
                return Model(predict, sign*math.exp(integ), "exponential quadratic")
    # Affine inverse-power functions are integrated using their 1-D density.
    if not np.all(pos):
        return None
    for power in (1., 2., 3., 4., d + 1., d + 2., .5):
        v = np.exp(-np.log(u)/power)
        if not np.all(np.isfinite(v)) or np.max(v) > 1e100:
            continue
        aa = np.column_stack([np.ones(len(x)), z])
        cc = solve_ls(aa, v)
        if np.max(np.abs(aa@cc-v)) > 2e-10*np.max(v):
            continue
        t = np.sum(np.abs(cc[1:]))*.5
        if 0 < t < cc[0]:
            from projection import Projection
            projection = Projection(cc[1:])
            def predict(p, cc=cc, power=power):
                return sign*(cc[0]+(p-.5)@cc[1:])**(-power)
            quad = AxisQuad(projection)
            low = cc[0]-t
            br = (low/(2*t))*np.array([1.,4.,16.,64.,256.,1024.])
            integral, err = quad.adaptive(
                lambda s: sign*(low+2*t*s)**(-power), lambda: 2000,
                br[(br < .5)], maxpts=1500)
            return Model(predict, integral, "inverse power", err)
    return None


def kink_lin(t, v, tol=3e-10):
    """Recover knots of a continuous piecewise linear profile."""
    s = np.diff(v)/np.diff(t)
    diff = np.diff(s)
    active = np.flatnonzero(np.abs(diff) > tol*max(1., float(np.max(np.abs(s)))))
    groups = np.split(active, np.where(np.diff(active) > 1)[0]+1)
    if not len(active):
        return []
    out = []
    for g in groups:
        if len(g) > 2:
            continue
        i,j = g[0], g[-1]+1
        if i < 1 or j >= len(t)-2:
            continue
        sl,sr = s[i-1:i+1].mean(), s[j:j+2].mean()
        if abs(sl-sr) < tol:
            continue
        root = t[i]+(v[j]-v[i]-sr*(t[j]-t[i]))/(sl-sr)
        if t[i]-1e-8 <= root <= t[j]+1e-8:
            out.append((root,i,j,sr-sl))
    return out


def _absolute_profile(t, vals, mode):
    """Fit a cusp with a learned order and affine background."""
    width = np.ptp(vals)
    if width < 1e-13 or len(t)<13:
        return None
    u = (vals - np.mean(vals))/width
    diff = np.diff(u,2)
    ind = np.argsort(np.abs(diff))[-3:]+1
    # Without a local change of curvature, there is little basis for an
    # inferred cusp. This also avoids unnecessary nonlinear fits.
    di = np.max(np.abs(diff))
    if di < 3.4*np.median(np.abs(diff))+1e-8:
        return None
    options = []
    for j in ind:
        for p in (.45,.85,1.45,2.65):
            A = np.column_stack((np.ones(len(t)),t-.5,np.abs(t-t[j])**p))
            coef = solve_ls(A,u)
            err = np.sqrt(np.mean((A@coef-u)**2))
            options.append((err,np.r_[coef,t[j],p]))
    options.sort(key=lambda a:a[0])
    for a,st in options[:4]:
        if a > .071:
            break
        def fun(q):
            b,c,A,mu,p = q
            return b+c*(t-.5)+A*np.abs(t-mu)**p-u
        def jac(q):
            b,c,A,mu,p = q
            vv = t-mu
            xx = np.maximum(4e-14,np.abs(vv))
            xp = xx**p
            return np.column_stack((np.ones(len(t)),t-.5,xp,
                                    -A*p*np.sign(vv)*xp/xx,A*xp*np.log(xx)))
        sol = least_squares(fun,st,jac=jac,max_nfev=36,
                            bounds=([-np.inf,-np.inf,-np.inf,0.,.12],
                                    [np.inf,np.inf,np.inf,1.,5.5]),
                            ftol=1e-11,xtol=3e-14,gtol=2e-13,x_scale="jac")
        if np.max(np.abs(sol.fun)) > 4e-11:
            continue
        B,C,A,m,p = sol.x
        B=B*width+np.mean(vals)
        C=C*width
        A=A*width
        if mode==0:
            return float(B+A*(m**(p+1.)+(1.-m)**(p+1.))/(1.+p)), m
        vals_i = []
        mp = max(1.,np.ceil(6./p))
        for h,sg in ((m,-1.),(1.-m,1.)):
            if h<=0:
                continue
            if A<0. and abs(A)*h**p>20:
                w0 = np.exp(-np.log(-A)/p)
                br = w0*np.array([1.,4.,16.,64.,256.,1024.])
                cuts = np.r_[0.,br[br<h],h]
            else:
                cuts = [0,h]
            for j,(aa,bb) in enumerate(zip(cuts[:-1],cuts[1:])):
                tr = mp if j==0 else 1.
                t0,wt = gl_interval(104)
                t1 = aa+(bb-aa)*t0**tr
                l = B+C*(m-.5+sg*t1)+A*t1**p
                logw = np.log(wt)+np.log(bb-aa)+np.log(tr)+(tr-1.)*np.log(t0)
                vals_i.append(logsumexp(l+logw))
        return math.exp(logsumexp(vals_i)),m
    return None


def line_model(t, y):
    """Return (integral, estimated breakpoints) for easily resolved profiles."""
    sc = np.max(np.abs(y))
    if sc == 0.:
        return 0., []
    v = y/sc
    z, w = gl_interval(128)
    if np.ptp(v) < 3e-14:
        return float(np.mean(y)), []
    for vv, mode in [(v,0),(np.log(np.maximum(np.abs(v),1e-300)),1)]:
        if mode and (np.min(np.abs(v)) < 1e-280 or np.any(np.sign(v) != np.sign(v[0]))):
            continue
        for degree in (2,4):
            c = np.polynomial.polynomial.polyfit(t-.5, vv, degree)
            err = np.max(np.abs(vv - np.polynomial.polynomial.polyval(t-.5,c)))
            if err < 3e-11:
                if mode and degree == 2:
                    integ = c[0]+log_gaussian_1d(c[2],c[1])
                    return float(sc*np.sign(v[0])*math.exp(integ)), []
                vals = np.polynomial.polynomial.polyval(z-.5,c)
                if mode:
                    vals = np.sign(v[0])*np.exp(vals)
                return float(sc*(w@vals)), []
        br = [a[0] for a in kink_lin(t, vv)]
        if br and len(br) <= 7:
            def f(p):
                return np.column_stack([np.ones(len(p)),p,np.maximum(p[:,None]-br,0.)])
            a = f(t)
            c = solve_ls(a,vv)
            if np.max(np.abs(a@c-vv)) < 1e-10:
                if not mode:
                    return float(sc*np.dot(np.r_[1,.5,.5*(1-np.array(br))**2],c)), br
                u = np.r_[0.,br,1.]
                v0 = f(u)@c
                hi = np.maximum(v0[:-1],v0[1:])
                diff = -np.abs(np.diff(v0))
                ratio = np.ones(len(diff))
                np.divide(np.expm1(diff),diff,out=ratio,where=diff != 0)
                return float(sc*np.sign(v[0])*np.dot(np.diff(u),np.exp(hi)*ratio)), br
        cusp = _absolute_profile(t,vv,mode)
        if cusp is not None:
            integ,knot = cusp
            return float(sc*np.sign(v[0])*integ if mode else sc*integ), [knot]
    if np.min(np.abs(v)) > 1e-30 and np.all(np.sign(v)==np.sign(v[0])):
        for powr in (1., .5):
            vv = np.abs(v)**(-powr)
            c = np.polynomial.polynomial.polyfit(t-.5,vv,2)
            if np.max(np.abs(vv - np.polynomial.polynomial.polyval(t-.5,c))) < 5e-13*np.max(vv):
                a,b,e = c[::-1]
                if a > 0:
                    mid = -b/(2*a)
                    den = e - b*b/(4*a)
                    if den > 0.:
                        width = math.sqrt(den/a)
                        if powr == 1.:
                            diff = math.atan2(width, width*width+(.5-mid)*(-.5-mid))
                            return float((sc*np.sign(v[0])/(a*width))*diff), []
                        else:
                            ints = lambda z: z/(z*z+1) + math.atan(z)
                            return float(sc*np.sign(v[0])*width/(2*den**2)*
                                         (ints((.5-mid)/width)-ints((-.5-mid)/width))), []
    return None, []


def section_points(anchor, t):
    d = len(anchor)
    pts = np.broadcast_to(anchor, (d,len(t),d)).copy()
    for j in range(d):
        pts[j,:,j] = t
    return pts


def axes_separable(orc, x, y, scale):
    """Cross-check multiplicative or additive separability on scattered points."""
    d = orc.d
    if d == 1:
        return (np.array([.5]), 0., "add", 0.)
    count = min(24, len(x) // 9)
    if count < 4 or orc.left < 85+count*d:
        return None
    spread = np.ptp(y)
    mid = np.median(y)
    narrow = np.min(np.abs(y)) < 5e-5*np.max(np.abs(y)) or np.max(np.abs(y)) > 80*np.median(np.abs(y))
    a = np.abs(y) if spread > .27 or narrow else np.abs(y-mid)
    ind = np.argsort(a)
    pivot = int(ind[-1])
    mx = a[pivot]
    if mx > 0:
        close = ind[a[ind] >= mx*.85]
        pivot = close[np.argmin(np.linalg.norm(x[close]-.5,axis=1))]
    anchor = x[pivot]
    f0 = y[pivot]
    if narrow:
        tail = ind[np.abs(y[ind]) > 1e-274]
        ii = tail[np.linspace(0,len(tail)-1,min(10,len(tail)),dtype=int)]
        ids = np.unique(np.r_[np.arange(len(x)-count+len(ii)+4,len(x)),ii,ind[-4:]])
    else:
        first = np.arange(count-4)+len(x)-count+4
        ids = np.unique(np.r_[first, ind[-4:]])
    xx = x[ids]
    vals = np.broadcast_to(anchor,(len(xx),d,d)).copy()
    for j in range(d):
        vals[:,j,j] = xx[:,j]
    u = orc.sample(vals.reshape(-1,d)).reshape(len(xx),d)/scale
    v = y[ids]
    err_add = np.max(np.abs(u.sum(axis=1)-(d-1)*f0-v))
    tol = max(3e-15,min(7e-11,spread*4e-9))
    if np.isfinite(err_add) and err_add < tol and not (narrow and np.median(np.abs(y))<1e-8*abs(f0)):
        return anchor, f0, "add", 0.
    if f0 == 0.:
        return None
    def pred(c):
        if abs(f0-c) < 5e-16:
            return np.ones(len(v))*f0
        uu = (u-c)/(f0-c)
        if not narrow:
            return c + (f0-c)*np.prod(uu,axis=1)
        logprod = np.sum(np.log(np.abs(uu)),axis=1) + np.log(abs(f0-c))
        return c + np.sign(f0-c)*np.prod(np.sign(uu),axis=1)*np.exp(np.clip(logprod,-745,709))
    direct = pred(0.)
    err_prod = np.max(np.abs(direct-v))
    if err_prod < tol and (not narrow or np.all(np.abs(direct-v) <
                       2e-7*np.maximum(1e-270,np.maximum(np.abs(direct),np.abs(v))))):
        return anchor, f0, "prod", 0.
    if orc.left < 110 or not np.isfinite(err_prod):
        return None
    if narrow and np.median(np.abs(y)) < 1e-8*np.max(np.abs(y)):
        return None
    i,j = np.triu_indices(d,1)
    influence = np.abs(u[:,i]-f0)*np.abs(u[:,j]-f0)
    irank = np.argsort(influence.ravel())[-min(30,len(xx)*len(i)):]
    pairs = [(ii//len(i),i[ii%len(i)],j[ii%len(i)]) for ii in irank]
    pp = np.broadcast_to(anchor, (len(pairs),d)).copy()
    for n,(k,i,j) in enumerate(pairs):
        pp[n,[i,j]] = xx[k,[i,j]]
    fy = orc.sample(pp)/scale
    f1 = np.array([u[k,i] for k,i,j in pairs])
    f2 = np.array([u[k,j] for k,i,j in pairs])
    xx1,xx2=f1-f0,f2-f0
    den = (fy-f0)-xx1-xx2
    good = np.abs(den) > max(4e-15,1e-6*spread)
    if np.any(good):
        cond = np.abs(den[good])
        cs = f0-xx1[good]*xx2[good]/den[good]
        top = np.argsort(cond)[-7:]
        for c in (np.median(cs[top]),cs[top[-1]]):
            if abs(c) < 20. and abs(f0-c) > 6e-15 and np.max(np.abs(pred(c)-v)) < tol:
                return anchor, f0, "prod", float(c)
    return None


def _step_brackets(t, y):
    dv = np.diff(y)
    amp = max(np.ptp(y),1e-290)
    score = np.abs(dv)
    pad = np.r_[0.,score,0.]
    ids = np.flatnonzero((score > .045*amp) &
                         (score > 3.0*np.maximum(pad[:-2],pad[2:]) + 1e-12*amp))
    return ids


def _steps_profile(probes, t, y, fn, maxpts):
    ids = _step_brackets(t,y)
    if not len(ids) or len(ids) > 5 or maxpts < 12*len(ids):
        return [], 0
    a,b = t[ids].copy(),t[ids+1].copy()
    ya,yb = y[ids].copy(),y[ids+1].copy()
    observed = np.abs(yb-ya)
    n = len(a)
    for k in range(min(46,maxpts//n)):
        m = (a+b)*.5
        ym = fn(m)
        move = np.abs(ym-ya) <= np.abs(ym-yb)
        a[move],ya[move] = m[move],ym[move]
        b[~move],yb[~move] = m[~move],ym[~move]
        if k == 9:
            good = np.abs(ya-yb) > .18*observed
            a,b,ya,yb = a[good],b[good],ya[good],yb[good]
            if not len(a):
                break
    return list((a+b)*.5), n*(k+1)


def integrate_axes(orc, args, scale):
    anchor, f0, mode, offset = args
    d = orc.d
    if orc.left < 25*d:
        raise NoBudget
    t = np.linspace(0.,1.,25 if orc.left < 100*d else 33)
    p = section_points(anchor,t)
    v = orc.sample(p.reshape(-1,d)).reshape(d,len(t))/scale - offset
    integrals = np.zeros(d)
    errors = np.zeros(d)
    q = AxisQuad()
    profiles = []
    for j in range(d):
        integ, br = line_model(t,v[j])
        profiles.append((integ,br))
    remaining = sum(vv is None for vv,_ in profiles)
    for j,(integ,br) in enumerate(profiles):
        if integ is not None:
            integrals[j] = integ
            continue
        def f(t, j=j):
            pts = np.broadcast_to(anchor,(len(t),d)).copy()
            pts[:,j] = t
            return orc.sample(pts)/scale-offset
        limit = orc.left//remaining
        remaining -= 1
        if limit < 15:
            raise NoBudget
        if limit > 90:
            steps, used = _steps_profile(orc,t,v[j],f,min(limit//2,200))
            br += steps
            limit -= used
        # Kinks broaden high-order finite differences, while smooth regions do not.
        if len(br) < 4:
            vj = v[j]
            trans = [vj]
            if np.min(np.abs(vj)) > 1e-100 and np.all(np.sign(vj)==np.sign(vj[0])):
                trans.append(np.log(np.abs(vj)))
            candidates = sorted({a[0] for yy in trans for a in kink_lin(t,yy)})
            br += [bp for bp in candidates if all(abs(bp-bq) > .008 for bq in br)]
        if not br and limit >= 65:
            im = np.argmax(np.abs(v[j]))
            if np.max(np.abs(v[j])) > 10.*np.median(np.abs(v[j])):
                br = [t[im-1]] if im > 0 else []
                br += [t[im+1]] if im+1 < len(t) else []
        integrals[j], errors[j] = q.adaptive(f, lambda: orc.left, br, maxpts=limit)
    if mode == "add":
        return float(integrals.sum() - (d-1)*f0)
    return float(offset + (f0-offset)*np.prod(integrals/(f0-offset)))
