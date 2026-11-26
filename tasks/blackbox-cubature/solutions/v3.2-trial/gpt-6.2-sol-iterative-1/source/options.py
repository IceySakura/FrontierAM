"""Slicing and defensive importance sampling for nonnegative payoffs."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.special import ndtr, roots_legendre

from common import first_basis, gauss, gradient, moment_rotation, perp, report, ridge, rqm, unit


def crossing(w, yy, v, lo, hi, steps, level, budget_all=None):
    """Locate upward crossings, retaining signs on both endpoints."""
    n = len(yy)
    lo = np.broadcast_to(lo, (n,)).astype(float).copy()
    hi = np.broadcast_to(hi, (n,)).astype(float).copy()
    X = np.concatenate([yy+lo[:, None]*v, yy+hi[:, None]*v])
    f_lo, f_hi = w.ask(X).reshape(2, -1)
    for _ in range(steps):
        mid = (lo+hi)*.5
        f_mid = w.ask(yy+mid[:, None]*v)
        low = f_mid <= level
        lo[low], f_lo[low] = mid[low], f_mid[low]
        hi[~low], f_hi[~low] = mid[~low], f_mid[~low]
    return lo, hi, f_lo, f_hi


def digital(w, x, y, levels):
    low, high = levels
    gap = high-low
    level = .5*(low+high)
    v = unit((y-y.mean()) @ x)
    D = w.d
    if w.left < 4800:
        return rqm(w, first_basis(v))
    lo, hi, fa, fb = crossing(w, np.zeros((1,D)), v, -6., 6., 14, level)
    if not fa[0] < level < fb[0]:
        return rqm(w, first_basis(v))
    t_origin = float(.5*(lo[0]+hi[0]))
    if D > 1:
        E = perp(v)
        h = .271
        offsets = h * np.concatenate([E.T, -E.T])
        a, b, fa, fb = crossing(w, offsets, v, t_origin-.31, t_origin+.31, 8, level)
        if np.all((fa <= level) & (fb > level)):
            r_plus, r_minus = (.5*(a+b)).reshape(2,-1)
            vnew = unit(v-E@((r_plus-r_minus)/(2*h)))
            t_origin *= np.dot(v, vnew)
            v = vnew
    E = perp(v)
    nfit = 192
    if D < 3:
        nfit = 96
    yf = w.normal(nfit, D-1)
    a,b,fa,fb = crossing(w, yf@E.T, v, t_origin-.68, t_origin+.68, 7, level)
    tf = .5*(a+b)
    valid = (fa <= level) & (fb > level)
    if valid.mean() < .91:
        return rqm(w, first_basis(v))
    yf = yf[valid]
    tf = tf[valid]
    d = D-1
    rad = (np.sum(yf*yf, axis=1)-d)/math.sqrt(max(1,2*d))
    A = np.column_stack([yf, rad])
    c, coeff = ridge(A, tf, 1.7 + d*.03, center=True)
    r = tf-c-A@coeff
    # Use the curvature visible in the training residual to order the tangent coordinates.
    H = (yf.T*r)@yf / len(yf)
    he, V = eigh(H, check_finite=False)
    V = V[:, np.argsort(-np.abs(he))]
    yr = yf@V
    k = min(12, d)
    C = np.column_stack([A, yr[:, :k]**2-1])
    pp = np.concatenate([np.full(d+1, 1.8+d*.03), np.full(k, 95.+d)])
    c1, coeff1 = ridge(C, tf, pp, center=True)
    res = tf-c1-C@coeff1
    sigma = max(.010, np.sqrt(np.mean(res**2))*(1.15 + .45*d/nfit))
    width = min(.57, max(.055, 3.7*sigma))
    n = 2048 if w.left >= 12450 else 1024
    if d <= 14 and w.left >= 12540:
        n = 2048
    if w.left < n*5+64:
        return rqm(w, first_basis(v))
    steps = min(9, (w.left-200)//n - 3)
    U = w.uniforms(n, d+1)
    Z = gauss(U[:,:d])
    original = Z @ V.T
    yy = original @ E.T
    t_hat = (c1 + original@coeff1[:d] + coeff1[d]*(np.sum(Z*Z,axis=1)-d)/math.sqrt(max(1,2*d))
             + (Z[:,:k]**2-1)@coeff1[d+1:])
    a = t_hat-width
    b = t_hat+width
    fa,fb = w.ask(np.concatenate([yy+a[:,None]*v, yy+b[:,None]*v])).reshape(2,-1)
    for m in (3., 9., 25.):
        below = fa > level
        above = fb <= level
        total = int(below.sum()+above.sum())
        if total == 0:
            break
        if total > w.left-(steps+1)*n:
            break
        a[below] = t_hat[below]-m*width
        b[above] = t_hat[above]+m*width
        raw = w.ask(np.concatenate([yy[below]+a[below,None]*v, yy[above]+b[above,None]*v]))
        fa[below], fb[above] = raw[:below.sum()], raw[below.sum():]
    ok = (fa <= level) & (fb > level)
    for _ in range(steps):
        mid = .5*(a+b)
        below = w.ask(yy+mid[:,None]*v) <= level
        a[below] = mid[below]
        b[~below] = mid[~below]
    A0, B0 = ndtr(a), ndtr(b)
    rand_t = gauss(A0+(B0-A0)*U[:,-1])
    res_y = w.ask(yy+rand_t[:,None]*v)
    p = 1-B0+(B0-A0)*(res_y > level)
    if not np.all(ok):
        p[~ok] = ndtr(-.5*(a[~ok]+b[~ok]))
    return float(low+gap*p.mean())


def payoff_pilot(w, x, y, level):
    X, Y = [x], [y-level]
    f = y-level
    if np.mean(f > 0) < .15:
        x2 = w.normal(1024)
        X.append(x2)
        Y.append(w.ask(x2)-level)
    x, f = np.concatenate(X), np.concatenate(Y)
    pos = np.flatnonzero(f > max(0., f.max()*1e-10))
    if len(pos) < 5:
        return x, f, None, None
    sorted_pos = pos[np.argsort(f[pos])]
    n_grads = min(25, len(pos))
    ii = sorted_pos[np.linspace(.16*(len(pos)-1), .95*(len(pos)-1), n_grads).astype(int)]
    _, G = gradient(w, x[ii], h=.001)
    norms = np.linalg.norm(G, axis=1)
    ggood = (norms > .02*np.median(norms)) & (norms < max(1e-5, 5.*np.median(norms)))
    if not np.any(ggood):
        return x, f, unit((f-f.mean())@x), None
    G = G[ggood]
    norms = norms[ggood]
    g = unit(np.mean(G/norms[:, None], axis=0))
    return x, f, g, G


def truncated_rule(m=8):
    """Gaussian quadrature for a lower-truncated standard normal, tabulated in the cut."""
    grid = np.linspace(-6., 6., 241)
    nodes, weights = roots_legendre(120)
    rule_x = np.empty((len(grid),m))
    rule_w = rule_x.copy()
    for j,a in enumerate(grid):
        ts = a+(10.-a)*.5*(1+nodes)
        mass = weights*(10.-a)*.5*np.exp(-ts**2*.5)/math.sqrt(2.*math.pi)
        B = np.sqrt(mass/mass.sum())
        B0 = np.zeros_like(B)
        diag, sub = [], []
        ant = 0.
        for k in range(m):
            v = ts*B-ant*B0
            ak = np.dot(v,B)
            v -= ak*B
            bk = np.linalg.norm(v)
            diag.append(ak)
            if k < m-1:
                sub.append(bk)
            B0,B = B,v/bk
            ant = bk
        t,P = eigh_tridiagonal(diag,sub)
        rule_x[j] = t
        rule_w[j] = P[0]**2
    return CubicSpline(grid, rule_x), CubicSpline(grid, rule_w)


def anchored_cv(z, y, groups):
    """Cross-fitted centered Hermite controls on independent designs."""
    n, d = z.shape
    k = min(10, d)
    ii,jj = np.triu_indices(k,1)
    Z = np.column_stack([z, z**2-1, z[:,ii]*z[:,jj],
                         z[:,:k]*(z[:,:k]**2-3), z[:,:k]**4-6*z[:,:k]**2+3])
    scale = np.r_[np.ones(d), np.full(d, math.sqrt(2.)), np.ones(len(ii)),
                  np.full(k, math.sqrt(6.)), np.full(k, math.sqrt(24.))]
    Z /= scale
    ans = 0.
    for a in range(groups):
        use = np.arange(n)//(n//groups) == a
        c,b = ridge(Z[~use],y[~use],16.,center=True)
        ans += np.mean(y[use]-Z[use]@b)/groups
    return ans


def sliced_call(w, v, basis, level, tau, width):
    m = 8 if w.left >= 11288 else 7
    n = 1024 if w.left >= 1024*(m+4) else 768
    if w.left < n*(m+4):
        m = 6
    if w.left < n*(m+4):
        n = 512
    if w.left < n*(m+4):
        return rqm(w, np.column_stack([v,basis]))
    # Equal independent outer nets let the Hermite coefficients be cross-fitted.
    groups = 2
    z = np.concatenate([w.normal(n//groups, w.d-1) for _ in range(groups)])
    y = z@basis.T
    if np.ndim(tau):
        t0 = tau[0]+y@tau[1:]
    else:
        t0 = np.full(n,tau)
    b = np.clip(t0+width,-4.,5.)
    c = b+.26
    f1,f2 = w.ask(np.concatenate([y+b[:,None]*v, y+c[:,None]*v])).reshape(2,-1)-level
    dp = np.maximum(1.e-9,(f2-f1)/.26)
    pred = np.clip(b-np.maximum(0.,f1)/dp,-6.,b)
    h = .04
    r = np.maximum(pred+h,b-2.7*width)
    r[f1<=0.] = c[f1<=0.]+width
    fr = w.ask(y+r[:,None]*v)-level
    new_dp = np.where((f1>fr) & (b>r+.015), (f1-fr)/np.maximum(.015,b-r), dp)
    new_dp = np.maximum(new_dp, .2*dp)
    rnew = np.clip(r-fr/new_dp,-6.,6.)
    rnew[fr<=0.] = r[fr<=0.]+.025
    bracketed = (fr<=0.) & (f1>0.) & (b>r)
    rnew[bracketed] = np.clip(pred[bracketed]+.03, r[bracketed], b[bracketed])
    t4 = np.where(fr<=0, .55*rnew+.45*b, rnew+.007)
    t4 = np.where((f1<=0)&(fr<=0), r+.65, t4)
    f4 = w.ask(y+t4[:,None]*v)-level
    tmin = np.where(f4>0, t4-f4/np.maximum(1.e-9,new_dp), rnew)
    good = (fr>0)&(f4>0)&(r>t4+.005)&(fr>f4)
    tmin[good] = t4[good] - f4[good]*(r[good]-t4[good])/(fr[good]-f4[good])
    tmin = np.clip(tmin, -6., 6.)
    rule_x, rule_w = truncated_rule(m)
    ts = rule_x(tmin)
    weights = rule_w(tmin)*ndtr(-tmin)[:,None]
    raw = w.ask((y[:,None,:]+ts[:,:,None]*v).reshape(-1,w.d)).reshape(n,m)-level
    values = np.sum(raw*weights,axis=1)
    ans = float(anchored_cv(z,values,groups) + level)
    w.safe = ans
    return ans


def _log_weight(x, mu, R, scales):
    z = (x-mu)@R
    return .5*(np.sum(x*x,axis=1)-np.sum((z/scales)**2,axis=1))-np.log(scales).sum()


def _fit_proposal(X, y, ratio, v, rare):
    n,d = X.shape
    yy = np.maximum(y,0)
    p = (yy>0)*np.power(yy/max(1.e-30,yy.max()),.62)*ratio
    W = p.sum()
    if W <= 0:
        return np.zeros(d),first_basis(v),np.ones(d)
    p /= W
    ess = 1./np.dot(p,p)
    E = perp(v)
    coordinates = X@np.column_stack([v,E])
    mean = p@coordinates
    centered = coordinates-mean
    C = (centered.T*p)@centered
    ev,V = eigh(C[1:,1:]-np.eye(d-1),check_finite=False)
    ind = np.argsort(np.abs(ev))[::-1]
    ev,V = ev[ind],V[:,ind]
    noise = math.sqrt(d/max(10.,ess))
    adj = np.sign(ev)*np.maximum(np.abs(ev)-(.29 if rare else .43)*noise,0)
    iso = np.clip(np.average(np.diag(C)[1:])-1.,-.3,.2)
    nu = 1. + .80*adj + .27*iso
    mvec = mean[1:]@V
    mvec *= (np.abs(mvec) + 1.e-20) / (np.abs(mvec)+np.sqrt(1./max(ess,1.))*.50+1.e-20)
    R = np.column_stack([v,E@V])
    mean = mean[0]*v + (E@V)@mvec
    av = np.clip(.18+.95*C[0,0], .20 if rare else .38, 1.4)
    return mean,R,np.sqrt(np.r_[av, np.clip(nu, .45 if rare else .64, 1.55)])


class WarpedProposal:
    """Piecewise-constant density in one Gaussian probability coordinate."""
    def __init__(self, X,y,iw,mu,R,S):
        Z=X@R
        self.mu=mu@R
        self.R,self.S=R,S
        t=Z[:,0]
        self.n=24
        self.pgrid=np.r_[0., .03, (.06+.89*np.linspace(0.,1.,23)), .98, 1.]
        self.n=len(self.pgrid)-1
        wgt=iw*np.power(np.maximum(y,0.) / max(1.e-12,np.max(y)),.77)
        wgt/=np.sum(wgt)
        self.ess=1./np.dot(wgt,wgt)
        ma=wgt@t
        Dt=t-ma
        v=wgt@(Dt**2)
        coeff=(wgt*Dt)@Z[:,1:]/max(.09,v)
        sh=np.sqrt(1./max(1.,self.ess*max(v,.12)))
        coeff*=np.abs(coeff)/(.45*sh+np.abs(coeff))
        norm=np.linalg.norm(coeff)
        self.corr=coeff*min(.83, .7/(norm+1.e-12))
        self.center=ma
        q=ndtr(t)
        ix=np.clip(np.searchsorted(self.pgrid,q,side='right')-1,0,self.n-1)
        empirical=np.bincount(ix,wgt,minlength=self.n)
        empirical=gaussian_filter1d(empirical, .82, mode='nearest')
        empirical/=empirical.sum()
        baseline=np.diff(ndtr((gauss(self.pgrid)-self.mu[0])/S[0]))
        baseline/=baseline.sum()
        self.prob = .78*empirical+.20*baseline+.02*np.diff(self.pgrid)
        self.prob/=self.prob.sum()
        self.cdf=np.r_[0.,np.cumsum(self.prob)]
        self.density=self.prob/np.diff(self.pgrid)

    def generate(self,w,n):
        U=w.uniforms(n,w.d)
        p=np.interp(U[:,0],self.cdf,self.pgrid)
        t=gauss(p)
        Y=self.mu[1:] + gauss(U[:,1:])*self.S[1:]+(t-self.center)[:,None]*self.corr
        return np.column_stack([t,Y])@self.R.T

    def log_density(self,X):
        z=X@self.R
        t=z[:,0]
        y=z[:,1:]
        p=ndtr(t)
        bins=np.clip(np.searchsorted(self.pgrid,p,side='right')-1,0,self.n-1)
        y0=y-self.mu[1:]-(t-self.center)[:,None]*self.corr
        log0=.5*np.sum(y*y-(y0/self.S[1:])**2,axis=1)-np.log(self.S[1:]).sum()
        return np.log(self.density[bins])+log0


def importance_payoff(w, x, y, v, G, level):
    rare = np.mean(y>0) < .18
    N = 2048 if rare else 1024
    mu,R,S = _fit_proposal(x,y,np.ones(len(y)),v,rare)
    warp=None
    if rare and w.left > N+4096:
        X2 = mu+(w.normal(N)*S)@R.T
        y2 = w.ask(X2)-level
        trains=np.concatenate([x,X2])
        val=np.concatenate([y,y2])
        ratio = np.exp(np.clip(_log_weight(trains,mu,R,S),-60.,60.))
        weight = 1./((len(x)+len(X2)*ratio)/(len(x)+len(X2)))
        mu,R,S = _fit_proposal(trains,val,weight,v,rare)
        if np.sum(val>0)>140:
            valid=np.arange(len(val))%3==0
            H=WarpedProposal(trains[~valid],val[~valid],weight[~valid],mu,R,S)
            Xv=trains[valid]
            base=np.exp(np.clip(_log_weight(Xv,mu,R,S),-60.,60.))
            alt=np.exp(np.clip(H.log_density(Xv),-60.,60.))
            wts=weight[valid]*np.maximum(0.,val[valid])**2
            if H.ess>50 and np.sum(wts/(.18+.82*alt))<.96*np.sum(wts/(.18+.82*base)):
                warp=WarpedProposal(trains,val,weight,mu,R,S)
    if G is not None and not rare:
        # The covariance change from survival is relatively weak here; retain sensitive payoff axes.
        basis = first_basis(v, (G.T@G)/len(G))
        if np.max(np.abs(S[1:]-1)) < .22:
            R, S = basis, np.r_[S[0], np.ones(w.d-1)]
    n = w.left
    n0 = 2048 if n >= 9000 else max(256,n//6)
    n1 = n-n0
    if n1 < 256:
        return rqm(w, R)
    X0 = w.normal(n0) @ R.T
    X1 = mu+(w.normal(n1)*S)@R.T if warp is None else warp.generate(w,n1)
    X = np.concatenate([X0,X1])
    lratio=_log_weight(X,mu,R,S) if warp is None else warp.log_density(X)
    ratio = np.exp(np.clip(lratio, -60.,60.))
    out = w.ask(X)-level
    return float(level+np.mean(out/(n0/n+(n1/n)*ratio)))


def payoff(w, x, yf, level):
    x, f, v, G = payoff_pilot(w,x,yf,level)
    if v is None:
        return rqm(w, moment_rotation(x,f))
    cov = G.T@G/len(G) if G is not None else None
    E = first_basis(v,cov)[:,1:]
    yy0 = w.normal(48,w.d-1)@E.T
    probe_t = np.array([-3.1,-1.6,-.6,.3,1.1,2.1,3.5,5.15])
    py = w.ask((yy0[:,None,:]+probe_t[None,:,None]*v).reshape(-1,w.d)).reshape(len(yy0),-1)-level
    on = py>0.
    close = on[:,:-1] > on[:,1:]
    vanilla = on[:,-1].mean() >= .96 and not np.any(close)
    if vanilla:
        k = on.shape[1]-1-np.argmax(on[:,::-1]==False,axis=1)
        none = np.all(on,axis=1)
        k[none] = 0
        ok = (~none)&(k<on.shape[1]-1)
        if np.sum(ok)>=24:
            ya = yy0[ok]
            a,b,_,fb=crossing(w,ya,v,probe_t[k[ok]],probe_t[k[ok]+1],8,level)
            th = .5*(a+b)
            jump = np.maximum(fb-level,0)
            if np.max(jump) < max(.13,.015*np.quantile(f[f>0],.75)):
                beta0,beta = ridge(ya,th,20.+w.d,center=True)
                noise = np.sqrt(np.mean((th-beta0-ya@beta)**2))
                width = float(np.clip(.15+3.4*noise,.24,.8))
                return sliced_call(w,v,E,level,np.r_[beta0,beta],width)
    return importance_payoff(w,x,f,v,G,level)
