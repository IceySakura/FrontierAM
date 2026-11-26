"""Geometry and conditional controls for isolated quadratic jumps."""

import math

import numpy as np
from scipy import linalg, special

from numerics import local_gradient, quadratic_features, quantile, residual_average, ridge, rotation
import smooth_cv


def find_jump(w, x, y):
    """Look for a difference that does not decay under dyadic refinement."""
    d = w.d
    v = w.normal(32)
    v *= math.sqrt(d) / np.linalg.norm(v, axis=1)[:, None]
    grid = np.r_[0., np.arange(.15, 2.71, .15)]
    yr = w.query((v[:, None, :] * grid[None, :, None]).reshape(-1, d))
    yr = yr.reshape(len(v), len(grid))
    w.line_survey = (v,grid,yr)
    dr = np.diff(yr, axis=1)
    prev = np.column_stack([dr[:, 1], dr[:, :-1]])
    nxt = np.column_stack([dr[:, 1:], dr[:, -2]])
    score = np.abs(dr) - .65*np.minimum(np.abs(prev), np.abs(nxt))
    j = np.argsort(score, axis=1)[:, -2:].reshape(-1)
    vv = np.repeat(v, 2, axis=0)
    rows = np.repeat(np.arange(len(v)), 2)
    lo, hi = grid[j].copy(), grid[j+1].copy()
    yl, yh = yr[rows,j].copy(), yr[rows,j+1].copy()
    first = yh - yl
    for _ in range(7):
        mid = (lo + hi) * .5
        ym = w.query(mid[:, None]*vv)
        in_left = np.abs(ym-yl) >= np.abs(yh-ym)
        hi = np.where(in_left, mid, hi)
        yh = np.where(in_left, ym, yh)
        lo = np.where(in_left, lo, mid)
        yl = np.where(in_left, yl, ym)
    jump = yh - yl
    med = np.median(y)
    big = np.abs(jump) >= max(5., 5.0*abs(med))
    small = np.abs(jump) >= max(.035, .08*np.std(np.minimum(y,np.quantile(y,.98))))
    persist = small & (np.abs(jump) > .47*np.abs(first))
    o = float(yr[0, 0])
    # A rare island can sit between two points of the coarse scan. If the
    # main pilot found such an island, test short rays starting from its hit.
    high = max(np.max(y), np.max(yr))
    rare = high > med + max(7., 9.*max(np.quantile(y,.8)-np.quantile(y,.2),.25))
    pool_x = np.concatenate([x, (v[:,None,:]*grid[None,:,None]).reshape(-1,d)])
    pool_y = np.r_[y,yr.ravel()]
    if not rare and np.sum(persist) < 4:
        # Contracted and expanded probes, also useful for off-center events.
        for sc in [.65, 1.38]:
            xx = w.normal(192)*sc
            yy = w.query(xx)
            pool_x, pool_y = np.vstack([pool_x,xx]), np.r_[pool_y,yy]
        high = np.max(pool_y)
        rare = high > med + max(10., 10.*max(np.quantile(y,.8)-np.quantile(y,.2),.25))
    if rare:
        thresh = .45*high + .55*med
        keep = pool_y > thresh
        if not np.any(keep):
            return None
        if o > thresh:
            anchor = np.zeros(d)
        else:
            # A convex average is typically much deeper than the closest hit.
            z_hit = pool_x[keep]
            rad = np.linalg.norm(z_hit,axis=1)
            wt = np.exp(-.12*(rad*rad - np.min(rad)**2))
            anchor = (wt @ z_hit) / wt.sum()
            if w.query(anchor[None,:])[0] <= thresh:
                anchor = z_hit[np.argmin(rad)]
            else:
                probe = w.normal(96)*.36 + anchor
                obs = w.query(probe)
                ok = obs > thresh
                if np.any(ok):
                    anchor = .45*anchor + .55*probe[ok].mean(axis=0)
        if not np.any(big & persist):
            xx = w.normal(16)
            probe = anchor + xx * .8
            pval = w.query(probe)
            ii = pval < thresh
            if np.sum(ii)==0:
                return None
            yy = w.query(anchor[None,:])[0]
            ll, rr = np.zeros(sum(ii)), np.ones(sum(ii))
            p_l, p_r = np.full(sum(ii),yy), pval[ii]
            dd = (probe-anchor)[ii]
            for _ in range(8):
                mid = (ll+rr)*.5
                ym = w.query(anchor + mid[:,None]*dd)
                k = ym > thresh
                ll, rr = np.where(k,mid,ll), np.where(k,rr,mid)
                p_l, p_r = np.where(k,ym,p_l), np.where(k,p_r,ym)
            a0 = np.median(p_l-p_r)
            if a0 < .35*(high-med):
                return None
            jumps = p_l-p_r
        else:
            jumps = np.abs(jump[big & persist])
        return anchor, float(thresh), 1., np.median(jumps), True

    if np.sum(persist) < 4:
        return None
    lz, hz = yl[persist], yh[persist]
    typical = np.median(hz-lz)
    polarity = -np.sign(typical)
    if polarity == 0.:
        return None
    h = polarity * (lz-hz) > 0.
    if np.mean(h) < .72:
        return None
    # Centers of the intervals need not be typical: a weak jump still shares
    # the same exterior branch. Pick a threshold that separates the endpoint
    # pairs with a margin on both sides.
    mn = polarity*hz[h]; mx = polarity*lz[h]
    cut = (np.sort(np.r_[mn,mx])[:-1]+np.sort(np.r_[mn,mx])[1:])*.5
    margins = (cut[:,None]-mn[None,:]) / np.maximum(mx-mn,1.e-9)
    merit = (np.clip(np.minimum(margins, 1.-margins), -.35, .28)).mean(axis=1)
    thresh = polarity*cut[np.argmax(merit)]
    if polarity*(o-thresh) <= 0:
        keep = polarity*(pool_y-thresh) > 0.
        if np.sum(keep) == 0:
            return None
        anchor = pool_x[keep][np.argmin(np.sum(pool_x[keep]**2,axis=1))]
    else:
        anchor = np.zeros(d)
    xb = vv[persist][h]*hi[persist][h,None]
    yb = hz[h]
    xgb, ygb = [xb], [yb]
    for ii, row in enumerate(rows[persist][h]):
        tr = (grid > 1.01*hi[persist][h][ii]) & (grid < 1.8*hi[persist][h][ii])
        if np.any(tr):
            xgb.append(v[row]*grid[tr,None])
            ygb.append(yr[row,tr])
    xgb, ygb = np.vstack(xgb), np.concatenate(ygb)
    p = ridge(xgb, ygb, .30)
    margin = np.quantile(np.abs(lz[h]-hz[h]), .10)*.40
    c1, m1 = p
    scale = max(.1,np.std(yb))
    base_th = float(np.clip(polarity*(thresh-m1),.5*margin,2.5*margin))
    def classifier(y,z):
        t = m1 + np.clip(z@c1,-2.3*scale,2.3*scale) + polarity*base_th
        return polarity*(y-t)>0.
    return anchor, classifier, polarity, -typical, False


def _ellipsoid_radius(aa, bb):
    aa = np.maximum(aa, 1.e-14)
    discr = np.sqrt(bb*bb + 4.*aa)
    return np.where(bb >= 0., 2./np.maximum(bb+discr,1.e-150), (discr-bb)/(2.*aa))


def _limit(a, v):
    return np.min(np.where(np.abs(v)<1.e-12, 1.e12,
                           (39.85 - np.sign(v)*a) / np.maximum(np.abs(v),1.e-12)), axis=1)


def _rays(w, a, v, inside, steps, scale):
    n = len(v)
    mx = _limit(a, v)
    hi = np.minimum(np.broadcast_to(scale,(n,)), mx)
    lo = np.zeros(n)
    active = np.arange(n)
    for _ in range(7):
        if not len(active):
            break
        arg = a+hi[active,None]*v[active]
        active = active[inside(w.query(arg),arg)]
        lo[active] = hi[active]
        hi[active] = np.minimum(hi[active]*1.85, mx[active])
        active = active[hi[active] > lo[active]]
    # Discard only upper endpoints not yet validated as outside.
    hi[active] = lo[active]
    good = hi > lo
    for _ in range(steps):
        m = (hi+lo)*.5
        arg = a+m[good,None]*v[good]
        hit = inside(w.query(arg),arg)
        lo[good] = np.where(hit,m[good],lo[good])
        hi[good] = np.where(hit,hi[good],m[good])
    return .5*(lo + hi), good


def _contrast_rays(w, a, v, polarity, height):
    n, d = v.shape
    cap = _limit(a,v)
    t = math.sqrt(d)*np.array([0.,.30,.48,.61,.73,.85,.97,1.09,1.22,1.36,1.52,
                            1.71,1.95,2.23,2.58,3.03,3.63,4.50,5.7,7.5])
    if d > 32:
        t = t[[0,2,4,5,6,7,8,9,11,13,15,17,19]]
    s = np.minimum(cap[:,None], t)
    pt = a + s[:,:,None]*v[:,None,:]
    y = w.query(pt.reshape(-1,d)).reshape(s.shape)
    dd = -polarity*np.diff(y,axis=1)
    slope = dd/np.maximum(np.diff(s,axis=1),.1)
    u = np.column_stack([slope[:,:1],slope[:,:-1]])
    z = np.column_stack([slope[:,1:],slope[:,-1:]])
    score = dd - .65*np.maximum(0.,np.minimum(u,z))*np.diff(s,axis=1)

    def bracket(k, alt):
        j = np.argsort(score[k],axis=1)[:,-alt]
        ll, rr = s[k,j].copy(), s[k,j+1].copy()
        fl, fr = y[k,j].copy(), y[k,j+1].copy()
        for _ in range((17 if alt==1 else 15) if d<=32 else (12 if alt==1 else 10)):
            mm = .5*(ll+rr)
            fm = w.query(a+mm[:,None]*v[k])
            left = polarity*(fl+fr-2.*fm) > 0.
            ll, rr = np.where(left,ll,mm), np.where(left,mm,rr)
            fl, fr = np.where(left,fl,fm), np.where(left,fm,fr)
        return .5*(ll+rr), fl, fr
    rr, il, ol = bracket(np.arange(n),1)
    bad = np.flatnonzero(polarity*(il-ol)<max(.047,.12*abs(height)))
    if len(bad):
        rt, ilt, olt = bracket(bad,2)
        rr[bad], il[bad], ol[bad] = rt, ilt, olt
    return rr, polarity*(il-ol)>max(.047,.12*abs(height)), il, ol


class Shape:
    def __init__(self, anchor, base, matrix, linear):
        lm, ev = linalg.eigh((matrix+matrix.T)*.5, check_finite=False)
        self.lm = np.maximum(lm[::-1], 0.)
        self.basis = (base @ ev[:,::-1]).copy()
        self.qb = (linear @ ev[:,::-1]).copy()
        low = self.lm < max(1.e-12, self.lm[0]*.00012)
        self.qb[low] = 0.
        self.lm[low] = 0.
        self.anchor = anchor
        self.az = anchor @ self.basis
        self.k = int(np.sum(~low))
        self.height = 1. + .25*np.sum(self.qb[:self.k]**2 / self.lm[:self.k])
        self.center = self.az.copy()
        self.center[:self.k] -= .5*self.qb[:self.k]/self.lm[:self.k]

    def score(self, x):
        xc = x-self.center
        return (xc*xc)@(self.lm/self.height) - 1.

    def radius(self, v):
        vz = v @ self.basis
        return _ellipsoid_radius((vz*vz)@self.lm, vz@self.qb)

    def integral(self, c, b, matrix):
        k, d = self.k, len(self.lm)
        if not k:
            return c+np.trace(matrix)
        if k <= 4:
            return self._tensor_integral(c,b,matrix)
        lam = self.lm[:k]/self.height
        qb = -2.*lam*self.center[:k]
        cc = np.sum(lam*self.center[:k]**2) - 1.
        bm = b[:k]
        mm = matrix[:k,:k]
        dd = np.diag(mm)
        c += np.trace(matrix[k:,k:])
        # Shift the Fourier inversion towards the saddlepoint of the
        # exponential tilt. Periodization error is negligible at exp(-28).
        def tilted_mean(s):
            inv = 1./(1.+2.*s*lam)
            mn = -s*qb*inv
            return cc + lam@(inv+mn*mn) + qb@mn
        l, h = 2., 32.
        for _ in range(18):
            m=(l+h)*.5
            if tilted_mean(m)>0.:
                l = m
            else:
                h = m
        shift = .5*(l+h)
        step = 2.*math.pi / max(1.7, 29./shift)

        def summand(t):
            w = (-shift+1j*t)[:,None]
            v = 1. - 2.*w*lam
            inv = 1./v
            mn = w*qb*inv
            e = w[:,0]*cc + np.sum((w*qb)*mn - np.log(v),axis=1)*.5
            poly = c + mn @ bm + inv @ dd + np.sum((mn @ mm)*mn,axis=1)
            return (np.exp(e)*poly/(shift-1j*t)).real
        total = .5*summand(np.zeros(1))[0]
        block = 256
        for j in range(1,12289,block):
            ts = step*np.arange(j,j+block)
            vals = summand(ts)
            total += vals.sum()
            if j >= 2*block and np.max(np.abs(vals)) < 5.e-15:
                break
        return float(total*step/math.pi)

    def _tensor_integral(self,c,b,m):
        k = self.k
        nd, wt = special.roots_legendre(56)
        r = np.sqrt(self.height/self.lm[:k])
        cen = self.center[:k]
        dens = np.ones(1)
        fac = np.ones(1)
        xx = np.zeros((1,k-1))
        for i in range(k-1):
            # Truncate only beyond the numerically relevant normal tails.
            rez = r[i]*fac
            th_l = np.arcsin(np.clip((-9.-cen[i])/rez,-1.,1.))
            th_r = np.arcsin(np.clip((9.-cen[i])/rez,-1.,1.))
            th = (th_l+th_r)[:,None]*.5 + ((th_r-th_l)*.5)[:,None]*nd
            ch = np.cos(th)
            t = cen[i] + rez[:,None]*np.sin(th)
            f0 = np.exp(-.5*t*t)/math.sqrt(2.*math.pi)
            dens = (dens[:,None]*f0*rez[:,None]*ch* ((th_r-th_l)*.5)[:,None]*wt).ravel()
            fac = (fac[:,None]*ch).ravel()
            xx = np.repeat(xx, len(nd),axis=0)
            xx[:,i] = t.ravel()
        low = cen[-1]-r[-1]*fac
        high = cen[-1]+r[-1]*fac
        p = special.ndtr(high) - special.ndtr(low)
        alt = special.ndtr(-low) - special.ndtr(-high)
        p = np.where(low > 0., alt,p)
        fl = np.exp(-.5*low*low)/math.sqrt(2.*math.pi)
        fh = np.exp(-.5*high*high)/math.sqrt(2.*math.pi)
        t1, t2 = fl-fh, p+low*fl-high*fh
        o = c + np.trace(m[k:,k:]) + xx @ b[:k-1] + np.sum((xx @ m[:k-1,:k-1])*xx,axis=1)
        return float(dens @ (p*o+(b[k-1]+2.*(xx @ m[:k-1,k-1]))*t1+m[k-1,k-1]*t2))


def learn_shape(w, a, inside, contrast=None):
    d = w.d
    # Orthogonal chords yield the diagonal and linear terms from reciprocal
    # lengths. Subsequent chords measure individual cross terms.
    base = linalg.qr(w.rng.normal(size=(d,d)), check_finite=False)[0]
    v = np.vstack([base.T,-base.T])
    sheared = False
    edges = []
    if contrast is None:
        r, good = _rays(w,a,v,inside,19 if d<=16 else 15, math.sqrt(d)*1.3)
        local_level = None
    else:
        polarity, height = contrast
        r, good, yl, yh = _contrast_rays(w,a,v,polarity,height)
        local_level = .46*yl+.54*yh
    if np.any(~good) and np.sum(good) >= 1.5*d and w.remaining > 9200:
        fail = np.flatnonzero(~(good[:d]&good[d:]))
        allgood = np.flatnonzero(good[:d]&good[d:])
        bp = allgood[np.argmin((r[:d]*r[d:])[allgood])]
        base[:,fail] += .89*base[:,bp,None]
        vv = np.vstack([base[:,fail].T,-base[:,fail].T])
        idx = np.r_[fail,fail+d]
        if contrast is None:
            r[idx],good[idx] = _rays(w,a,vv,inside,18,math.sqrt(d))
        else:
            r[idx],good[idx],yl[idx],yh[idx] = _contrast_rays(w,a,vv,polarity,height)
            local_level = .46*yl+.54*yh
        v[idx] = vv
        sheared = True
    if np.sum(good) < len(good) or np.any(r<1.e-5):
        return None
    inv = 1./r
    diag = inv[:d]*inv[d:]
    b = inv[:d]-inv[d:]
    matrix = np.diag(diag)
    remaining = np.ones((d,d),bool)
    np.fill_diagonal(remaining, False)
    residual = diag.copy()
    cols = []
    norms = []

    def finish(mat):
        if sheared:
            qi = linalg.inv(base,check_finite=False)
            shape = Shape(a,np.eye(d),qi.T@mat@qi,b@qi)
        else:
            shape = Shape(a,base,mat,b)
        if contrast is not None:
            edges.insert(0,(a+r[:,None]*v,yl-yh,np.ones(len(r))))
            shape.edges = tuple(np.concatenate([e[j] for e in edges]) for j in range(3))
        return shape

    def cross(i,j,bits, guess, sig=None):
        # Shifted chords can stay close to a known crossing, so a local
        # jump level is enough even if the global branches overlap.
        if local_level is not None:
            first, second = np.where(diag[i]>=diag[j],i,j), np.where(diag[i]>=diag[j],j,i)
            tau = .31
            cross_factor = 2.*tau
            mid0 = diag[first] + tau*tau*diag[second]
            bv = b[first]+tau*b[second]
            sign = np.where(bv<0.,-1.,1.)
            v = ((base[:,first]+tau*base[:,second])*sign).T.copy()
            thresh = local_level[first + d*(sign<0.)]
        else:
            cross_factor = 1.
            mid0 = .5*(diag[i]+diag[j])
            bv = (b[i]+b[j]) * (2.**-.5)
            sign = np.where(bv<0.,-1.,1.)
            v = ((base[:,i]+base[:,j])*(2.**-.5)*sign).T.copy()
        bv *= sign
        maxr = _limit(a,v)
        bound = np.sqrt(diag[i]*diag[j]) * 1.001 + 1.e-9
        lo = -bound
        hi = bound
        if sig is not None:
            ls = special.ndtr((lo-guess)/sig)
            hs = special.ndtr((hi-guess)/sig)
            # Shift each coarse coefficient grid independently. In particular,
            # coefficients near zero should not all round to the same side
            # when the axial chord estimates have a small common error.
            shift = (w.rng.random(len(i))-.5)*(hs-ls)/(2.**bits)
        if local_level is not None:
            xin = np.full(len(i),np.nan)
            xout = np.full(len(i),np.nan)
            yin = np.zeros(len(i))
            yout = np.zeros(len(i))
        for _ in range(bits):
            if sig is not None:
                mids = (ls+hs)*.5
                mid = np.clip(guess + sig*quantile(mids+shift),lo,hi)
            else:
                mid = (lo+hi)*.5
            root = np.minimum(_ellipsoid_radius(mid0+cross_factor*mid,bv),maxr)
            arg = a+v*root[:,None]
            val = w.query(arg)
            pred = inside(val,arg) if local_level is None else polarity*(val-thresh)>0.
            if local_level is not None:
                xin = np.where(pred,root,xin)
                xout = np.where(pred,xout,root)
                yin = np.where(pred,val,yin)
                yout = np.where(pred,yout,val)
            hi = np.where(pred,mid,hi)
            lo = np.where(pred,lo,mid)
            if sig is not None:
                hs = np.where(pred,mids,hs)
                ls = np.where(pred,ls,mids)
        if local_level is not None:
            jj = np.isfinite(xin)&np.isfinite(xout)&(polarity*(yin-yout)>.075*abs(height))
            if np.any(jj):
                dz = np.abs(xin-xout)[jj]*np.linalg.norm(v[jj],axis=1)
                edges.append((a + (.5*(xin+xout))[jj,None]*v[jj],
                              (yin-yout)[jj],np.minimum(1.,.029/np.maximum(dz,1.e-8))))
        if sig is None:
            return (hi+lo)*.5
        # Shrink unresolved tail cells toward their conditional means.
        ls = special.ndtr((lo-guess)/sig)
        hs = special.ndtr((hi-guess)/sig)
        den = np.maximum(hs-ls,1.e-10)
        g1 = np.exp(-.5*quantile(ls)**2)
        g2 = np.exp(-.5*quantile(hs)**2)
        return np.clip(guess + sig*(g1-g2)/(math.sqrt(2.*math.pi)*den),lo,hi)

    if d > 16:
        for step in range(7):
            p = np.argmax(residual)
            if residual[p] < np.mean(diag)*.00055:
                return finish(np.asarray(cols).T@np.asarray(cols))
            prev = np.asarray(cols)
            rs = np.sum(np.maximum(residual,0.))/np.sum(diag)
            if step >= 2 and rs > max(.48, .91 - .14*step):
                break
            js = np.flatnonzero(remaining[p])
            if not len(js):
                break
            g = prev[:,p] @ prev[:,js] if step else np.zeros(len(js))
            val = cross(np.full(len(js),p),js,15,g)
            matrix[p,js] = matrix[js,p] = val
            remaining[p,js] = remaining[js,p] = False
            residual_vec = matrix[p] - (prev[:,p] @ prev if step else 0.)
            row = residual_vec/math.sqrt(max(residual[p],1.e-20))
            cols.append(row)
            residual = residual-row*row
            dd = np.zeros(d, dtype=bool)
            dd[js] = True
            norms.extend((residual_vec[js]/np.sqrt(diag[p]*diag[js])).tolist())
            if np.sum(np.maximum(residual,0.)) < .0012*np.sum(diag):
                return finish(np.asarray(cols).T@np.asarray(cols))
    i,j = np.where(np.triu(remaining, 1))
    n = len(i)
    if n:
        bits = min(19 if d<=16 else (10 if d<=32 else 5), max(0,(w.remaining-4450)//n))
        if bits < 2:
            return None
        if bits <= 7:
            guess = np.asarray(cols).T@np.asarray(cols) if cols else np.zeros((d,d))
            sg = np.clip(np.sqrt(np.mean(np.asarray(norms)**2)) if norms else .20,.086,.39)
            mid = cross(i,j,bits,guess[i,j],sg*np.sqrt(diag[i]*diag[j]))
        else:
            mid = cross(i,j,bits,np.zeros(n))
        matrix[i,j] = matrix[j,i] = mid
    return finish(matrix)


def _poly(d, keep, coef, intercept):
    i,j = np.triu_indices(keep,1)
    m = np.diag(coef[d+len(i):]*2.**-.5)
    m[i,j] = m[j,i] = .5*coef[d:d+len(i)]
    return intercept-np.trace(m), coef[:d], m


def _poly_jets(x, keep):
    n,d = x.shape
    i,j = np.triu_indices(keep,1)
    p = np.arange(len(i))+d
    idx = np.arange(d)
    grad = np.zeros((n,d,2*d+len(i)))
    grad[:,idx,idx] = 1.
    grad[:,i,p] = x[:,j]
    grad[:,j,p] = x[:,i]
    grad[:,idx,idx+d+len(i)] = math.sqrt(2.)*x
    return grad.reshape(n*d,-1)


def _shell_points(w, shape, delta, n, stretch=.21):
    """Angular change of variables for a radial band about the fitted surface."""
    d = w.d
    uu = w.unit(n, d+1)
    # Draw directions from an angular central Gaussian matched approximately
    # to a tilted Gaussian on the fitted level set.
    eff = shape.lm/shape.height
    scales = (1.+max(0.,stretch*d)*eff)**-.46
    ww = quantile(uu[:,:d])*scales
    ww /= np.linalg.norm(ww, axis=1)[:,None]
    vx = ww @ shape.basis.T
    nr = shape.radius(vx)
    mx = _limit(shape.anchor,vx)
    lo = np.minimum(np.maximum(0., nr*(1.-delta)),mx)
    hi = np.minimum(nr*(1.+delta),mx)
    hi = np.maximum(hi,lo)
    u = uu[:,d]
    radial = lo + (hi-lo)*u
    ay = vx @ shape.anchor
    sz2 = shape.anchor @ shape.anchor
    # Upper-half chi density times the translation and angular corrections.
    logchi = (d-1.)*np.log(np.maximum(radial,1.e-100)) - .5*radial*radial
    logchi -= (d/2.-1.)*math.log(2.) + special.gammaln(d/2.)
    ratio = np.sum(np.log(scales)) + d/2.*np.log(np.sum((ww/scales)**2,axis=1))
    density = np.exp(np.clip(logchi - radial*ay - sz2*.5 + ratio,-200.,60.))
    return shape.anchor + radial[:,None]*vx, density*(hi-lo), radial < nr


def _band_mask(shape,z,delta):
    xx = z-shape.anchor
    rad = np.linalg.norm(xx,axis=1)
    v = xx/np.maximum(rad[:,None],1.e-100)
    rr = shape.radius(v)
    hi = np.minimum(rr*(1.+delta),_limit(shape.anchor,v))
    lo = np.minimum(rr*(1.-delta),hi)
    return (rad >= lo) & (rad <= hi)


def _shell_correction(w, shape, inside, delta, n):
    if n == 0:
        return 0.
    zz, wt, approx = _shell_points(w,shape,delta,n)
    exact = inside(w.query(zz),zz)
    return float(np.mean((exact.astype(float)-approx)*wt))


def _ordinary_estimate(w,x0,y0,shape,amp,polarity):
    d = w.d
    # Validate against actual local contrasts. The midpoint of a global
    # value gap is not reliable when the amplitude depends on location.
    n = min(96,max(40,w.remaining//52))
    v = w.normal(n)
    v /= np.linalg.norm(v,axis=1)[:,None]
    r = shape.radius(v)
    limit = _limit(shape.anchor,v)
    idx = np.arange(n)
    al,ah,fl,fh = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
    valid = np.zeros(n,dtype=bool)
    width = np.ones(n)
    for fac in [.018,.065,.19,.42]:
        j = idx[~valid]
        if not len(j):
            break
        lo,hi = np.minimum(r[j]*(1.-fac),limit[j]),np.minimum(r[j]*(1.+fac),limit[j])
        y1,y2 = np.split(w.query(np.vstack([shape.anchor+lo[:,None]*v[j],
                                          shape.anchor+hi[:,None]*v[j]])),2)
        ok = polarity*(y1-y2) > max(.045,.18*abs(amp))
        k = j[ok]
        al[k],ah[k],fl[k],fh[k] = lo[ok],hi[ok],y1[ok],y2[ok]
        valid[k] = True
        width[k] = fac
    good = np.flatnonzero(valid)
    if len(good)<n*.32 or w.remaining<2450:
        return None
    j = good[:min(52,len(good))]
    vv = v[j]
    al,ah,fl,fh = al[j],ah[j],fl[j],fh[j]
    for _ in range(8):
        mm = .5*(al+ah)
        fm = w.query(shape.anchor+mm[:,None]*vv)
        side = polarity*(fl+fh-2.*fm)>0.
        al,ah = np.where(side,al,mm),np.where(side,mm,ah)
        fl,fh = np.where(side,fl,fm),np.where(side,fm,fh)
    act = polarity*(fl-fh)>max(.025,.12*abs(amp))
    err = np.abs(.5*(al+ah)/r[j]-1.)
    al,ah,fl,fh,vv = al[act],ah[act],fl[act],fh[act],vv[act]
    if len(al)<6:
        return None
    npairs = min(14, max(4,w.remaining//(11*d)),len(al))
    sub = np.linspace(0,len(al)-1,npairs).astype(int)
    lo,hi = shape.anchor+al[sub,None]*vv[sub],shape.anchor+ah[sub,None]*vv[sub]
    sep = np.maximum(.85*(ah-al)[sub],.0007)
    lgrad = lo-sep[:,None]*vv[sub]
    hgrad = hi+sep[:,None]*vv[sub]
    sites = np.vstack([lgrad,hgrad])
    vals = w.query(sites)
    gi,go = np.split(local_gradient(w,sites,vals,.00014),2)
    ag = gi-go
    scale = max(np.median(np.linalg.norm(ag,axis=1)),1.e-6)
    ag /= np.maximum(1.,np.linalg.norm(ag,axis=1)/(4.*scale))[:,None]
    A = (ag.T@ag)/len(ag)
    C = (go.T@go)/len(go)
    sp = (shape.basis*(shape.lm**2))@shape.basis.T
    R = rotation(A + .045*np.trace(A)*sp/max(np.trace(sp),1.e-12))
    W = rotation(C + .075*np.trace(C)*sp/max(np.trace(sp),1.e-12))
    runrot = rotation(.49*A/max(np.trace(A),1.e-13) + .42*C/max(np.trace(C),1.e-13) +
                      .22*sp/max(np.trace(sp),1.e-13))
    nex = min(1536,w.remaining//5)
    x = w.normal(nex) @ runrot.T
    y = np.r_[y0,w.query(x)]
    x = np.vstack([x0,x])
    p = min(12,d)
    q = min(12,d)
    bx = quadratic_features(x@W,width=p)
    hx = quadratic_features(x@R,width=q)
    jy = shape.score(x@shape.basis)<0.
    br = np.column_stack([bx,jy,jy[:,None]*hx])
    ex,ej,ew = shape.edges
    ex = np.vstack([ex,shape.anchor+.5*(al+ah)[:,None]*vv])
    ej,ew = np.r_[ej,fl-fh],np.r_[ew,np.full(len(al),2.5)]
    ew *= np.exp(-.022*np.maximum(0.,np.sum(ex*ex,axis=1)-1.7*d))
    ej = np.clip(ej, -8.*abs(amp), 8.*abs(amp))
    er = np.column_stack([np.zeros((len(ex),bx.shape[1])), np.ones(len(ex)),
                          quadratic_features(ex@R,width=q)])
    # The intercept belongs only to the observations of f itself, not the
    # observations of its one-sided differences.
    full = np.column_stack([np.r_[np.ones(len(br)),np.zeros(len(er))],np.vstack([br,er])])
    ew *= min(4.,.44*len(br)/max(1.,ew.sum()))
    data = np.r_[y,ej]
    weights = np.r_[np.ones(len(y)),ew]
    reg = np.r_[.000035,np.full(bx.shape[1],.0035),.0010,np.full(hx.shape[1],.004)]
    use_jet = False
    if len(ag)>=6:
        dj = _poly_jets((.5*(lo+hi))@R,q)
        base_der = _poly_jets(hgrad@W,p)
        f0 = np.column_stack([np.zeros(len(dj)),base_der,np.zeros((len(dj),1+hx.shape[1]))])
        f1 = np.column_stack([np.zeros((len(dj),2+bx.shape[1])),dj])
        jets = np.vstack([f0,f1])
        targets = np.r_[(go@W).ravel(),(ag@R).ravel()]
        wj = np.r_[np.full(len(dj),.22),np.full(len(dj),.92)]
        tv = np.arange(len(full))%6==0
        zz,yz,wz = full[~tv],data[~tv],weights[~tv]
        c0,_ = ridge(zz,yz,reg,wz,center=False)
        c1,_ = ridge(np.vstack([zz,jets]),np.r_[yz,targets],reg,np.r_[wz,wj],center=False)
        risk0 = np.sum(weights[tv]*(data[tv]-full[tv]@c0)**2)
        risk1 = np.sum(weights[tv]*(data[tv]-full[tv]@c1)**2)
        use_jet = risk1 < .966*risk0
    if use_jet:
        c,_ = ridge(np.vstack([full,jets]),np.r_[data,targets],reg,
                    weights=np.r_[weights,wj],center=False)
    else:
        c,_ = ridge(full,data,reg,weights=weights,center=False)
    base,bc = c[0],c[1:1+bx.shape[1]]
    j0,jc = c[1+bx.shape[1]],c[2+bx.shape[1]:]
    cc,bb,mm = _poly(d,q,jc,j0)
    H = shape.basis.T @ R
    jump_int = shape.integral(cc,H@bb,H@mm@H.T)
    if not np.isfinite(jump_int) or abs(jump_int)>5.*max(1.,np.max(np.abs(y))):
        return None
    def prediction(z):
        return (quadratic_features(z@W,width=p)@bc + base,
                quadratic_features(z@R,width=q)@jc + j0)

    def jump_resid(y,bg,h,ix):
        threshold = bg + polarity*.43*np.maximum(polarity*h,.23*abs(amp))
        yes = polarity*(y-threshold)>0.
        return h*(yes.astype(float)-ix)

    shell = 0.
    delta = max(.002,min(.38,1.8*np.quantile(err[act],.93)))
    use_shell = delta>.003 and w.remaining>2900
    booster = None
    train_resid = y-base-bx@bc-jy*(j0+hx@jc)
    if use_jet and delta<.012:
        gp = (_poly_jets(sites@W,p)@bc).reshape(len(sites),d)@W.T
        rr = np.vstack([gi,go])-gp
        rr[:len(gi)] -= ((_poly_jets(lgrad@R,q)@jc).reshape(len(gi),d))@R.T
        RC = rr.T@rr/len(rr)
        mix = .62*RC/max(np.trace(RC),1.e-13)+.38*(A+C)/max(np.trace(A+C),1.e-13)
        runrot = rotation(mix)
    if delta<.055:
        booster = smooth_cv.fit(x,train_resid,C+.15*A)
    if use_shell:
        ns = int(.35*w.remaining)
        zz,wt,ix = _shell_points(w,shape,delta,ns,stretch=.055)
        y1 = w.query(zz)
        g,h = prediction(zz)
        shell = np.mean(wt*jump_resid(y1,g,h,ix))
    def resid(z,y):
        ix = shape.score(z@shape.basis)<0.
        bg,h = prediction(z)
        ans = y-bg-ix*h
        if use_shell:
            ans -= _band_mask(shape,z,delta)*jump_resid(y,bg,h,ix)
        if booster is not None:
            ans -= booster(z)
        return ans
    if booster is not None and delta<.016:
        correction = smooth_cv.crossfit(w,runrot,resid,C+.15*A)
    else:
        correction = residual_average(w,runrot,resid,delta<.016)
    return float(base + jump_int + shell + (booster.mean if booster is not None else 0.) + correction)


def jump_estimate(w,x0,y0,det):
    a, th, sign, amp, rare = det
    d = w.d
    inside = th if not rare else lambda y,z: sign*(y-th) > 0.
    shape = learn_shape(w,a,inside,None if rare else (sign,amp))
    if shape is None:
        return None
    if not rare:
        return _ordinary_estimate(w,x0,y0,shape,amp,sign)
    valn = 96 if d<=32 else 128
    v = w.normal(valn)
    v /= np.linalg.norm(v,axis=1)[:,None]
    rhat = shape.radius(v)
    pmax = _limit(a,v)
    r = np.minimum(rhat,pmax*.999)
    x_at = a+r[:,None]*v
    dy0 = w.query(x_at)
    rels = np.array([.0006,.003,.014,.045,.115,.26])
    errors = np.full(len(r), rels[-1]*1.6)
    ih = inside(dy0,x_at)
    for frac in rels:
        j = np.flatnonzero(errors > rels[-1])
        if not len(j):
            break
        rl = np.minimum(r[j]*(1.+frac*np.where(ih[j],1.,-1.)),pmax[j])
        arg = a+rl[:,None]*v[j]
        answer = inside(w.query(arg),arg)
        errors[j[answer != ih[j]]] = frac
    qerr = np.quantile(errors,[.5,.9,.98])
    if qerr[0] > .15 or qerr[1] > .35:
        return None
    vgood = errors <= .115
    # Estimate the jump height using opposite sides of resolved crossings.
    if np.sum(vgood)>=12:
        jg = np.flatnonzero(vgood)
        if len(jg)>32:
            vgood[jg[32:]] = False
        r1, gr = _rays(w,a,v[vgood],inside,12,r[vgood]*(1.+2.2*errors[vgood]))
        if gr.sum()>=8:
            jj = np.flatnonzero(vgood)[gr]
            r1 = r1[gr]
            vv = v[jj]
            delta = np.maximum(2.2e-4*r1,1.e-5)
            yl, yh = np.split(w.query(np.vstack([a+(r1-delta)[:,None]*vv,
                                                a+(r1+delta)[:,None]*vv])),2)
            dh = yl-yh
            ok = (yl>th)&(yh<=th)
            if sum(ok)>=8:
                amp = float(np.median(dh[ok]))

    nextra = min(1024, w.remaining//6)
    xx = w.normal(nextra) @ shape.basis.T
    y = np.r_[y0,w.query(xx)]
    x = np.vstack([x0,xx])
    keep = min(10,d)
    zz = x @ shape.basis
    features = quadratic_features(zz,width=keep)
    yy = y-amp*inside(y,x)
    c,m = ridge(features,yy,.012)
    pred = yy-features@c-m

    # Avoid the jump when estimating the background gradient covariance.
    select = np.argsort(np.abs(shape.score(zz)))[::-1]
    select = select[:max(len(x)//2,16)]
    sub = select[np.linspace(0,len(select)-1,min(20, len(select))).astype(int)]
    g = local_gradient(w,x[sub],y[sub],.0012)
    gn = np.linalg.norm(g,axis=1)
    g /= np.maximum(1., gn/max(np.median(gn)*4.,1.e-7))[:,None]
    cov = g.T @ g / max(len(g),1)
    basis = rotation(cov)
    p = shape.integral(1.,np.zeros(d),np.zeros((d,d)))
    if not -1.e-5 <= p <= 1.:
        return None
    delta = min(.60,max(.0016,1.4*qerr[2], 2.4*qerr[1]))
    precise = qerr[2] <= .0006
    n_shell = min(4096, w.remaining//(4 if precise else 2))
    p += _shell_correction(w,shape,inside,delta,n_shell)
    booster = smooth_cv.fit(x,pred,cov)
    z = w.normal(w.remaining) @ basis.T
    yy = w.query(z)
    flag = inside(yy,z)
    ihat = shape.score(z@shape.basis)<0.
    a0 = yy - amp*flag + amp*(~_band_mask(shape,z,delta))*(flag.astype(float)-ihat)
    cf = quadratic_features(z@shape.basis,width=keep)
    r0 = a0-(cf@c+m)
    if booster is not None:
        m += booster.mean
        r0 -= booster(z)
    return float(amp*np.clip(p,0.,1.) + m + np.mean(r0))
