"""Membership tomography and Markov integration for terminal barrier calls."""
import numpy as np
from scipy.special import ndtr, roots_legendre

from common import log, normalize, sobol, orth


def any_bisect(s, base, dirs, lo, hi, iters):
    lo, hi = np.broadcast_arrays(lo, hi)
    lo, hi = lo.copy(), hi.copy()
    for _ in range(iters):
        if s.remaining < len(lo):
            break
        mid = (lo+hi)*.5
        y = s.query(base + mid[:,None]*dirs)
        inside = y>0
        lo[inside] = mid[inside]
        hi[~inside] = mid[~inside]
    return (lo+hi)*.5


def facet_candidates(s, anchor, v, rays):
    bases = anchor[None,:]+np.zeros_like(rays)
    big = np.full(len(rays),13.)
    yh = s.query(bases+big[:,None]*rays)
    use = (yh==0.)
    if not np.any(use):
        return []
    rays, bases, big = rays[use], bases[use], big[use]
    r = any_bisect(s, bases, rays, np.zeros(len(rays)), big, 17)
    # Move along the terminal direction to read the slope of the crossing.
    delta = .008
    x = bases+r[:,None]*rays
    c = np.r_[x-delta*v, x+delta*v]
    w = np.tile(rays,(2,1))
    eps = np.r_[r,r]
    span = np.minimum(.45, eps*.7)
    yl, yh = np.split(s.query(np.concatenate((c-span[:,None]*w,c+span[:,None]*w))),2)
    ok = (yl>0)&(yh==0)
    root = np.full(len(c),np.nan)
    if np.any(ok):
        root[ok] = any_bisect(s,c[ok],w[ok],-span[ok],span[ok],14)
    t = anchor@v
    a,b = np.split(root,2)
    slope = -(b-a)/(2*delta)
    C = t + r / np.where(np.abs(slope)>1e-5,slope,np.nan)
    good = np.isfinite(C) & (np.abs(a+b) < .002+0.025*np.abs(a-b))
    return [(float(C[i]),float(r[i]),float(slope[i]),rays[i],anchor) for i in np.flatnonzero(good)]


def inspect_facet(s, v, cand, sigma):
    _, r, gv, w, anchor = cand
    d = s.d
    if s.remaining < (d+7)*18+200:
        return None
    r0 = any_bisect(s, anchor, w, np.array([max(0.,r-.00012)]), np.array([r+.00012]), 17)[0]
    x = anchor+r0*w
    # Exact crossing law on a flat face: shift in the ray coordinate
    # equals -h (n·e)/(n·w).  Only the angle is needed to recover the time.
    crank = s.rng.standard_normal((4,d))
    crank = normalize(crank)
    e = np.r_[np.eye(d), v[None,:], crank]
    for h in (.0009,.00018):
        limit = h * min(150., max(5., abs(gv)*8., r*5.))
        base = x+h*e
        yl, yh = np.split(s.query(np.r_[base-limit*w,base+limit*w]),2)
        if np.any(yl==0.) or np.any(yh>0):
            continue
        tt = any_bisect(s,base,w,np.full(len(e),-limit),np.full(len(e),limit),15)
        lam = -tt[:d]/h
        ghat = -tt[d]/h
        if np.max(np.abs(e[d:]@lam+tt[d:]/h)) > .002+1e-3*np.linalg.norm(lam):
            continue
        if abs(lam@w-1.)>.004 or abs(ghat-gv)>.008+.02*abs(gv):
            continue
        tfrac = (ghat/np.linalg.norm(lam))**2
        day = int(round(d*tfrac))
        if not (1 <= day < d) or abs(day-d*tfrac)>.14:
            continue
        # Use a wider two-sided terminal shift to improve the intercept.
        # Reject the shift if a third face has become active.
        for a in (.10,.025,.006,.0015):
            aa = np.array([a,-a])
            center = x+aa[:,None]*v
            pred = -aa*ghat
            span = min(.10,.004+abs(ghat*a)*.15)
            if np.min(r0+pred-span)<=0.:
                continue
            lo = center+(pred-span)[:,None]*w
            hi = center+(pred+span)[:,None]*w
            yl,yh = np.split(s.query(np.r_[lo,hi]),2)
            if np.any(yl==0.) or np.any(yh>0):
                continue
            ss = any_bisect(s,center,w,pred-span,pred+span,22)
            g = -(ss[0]-ss[1])/(2*a)
            if abs(ss[0]+ss[1])>3e-7 or abs(g-ghat)>.003+.003*abs(ghat):
                continue
            # Remove the tiny r0 crossing error used to compute the shifts.
            x = x + .5*(ss[0]+ss[1])*w
            ghat = g
            break
        # At an axial anchor the unknown orthogonal components drop out.
        if np.linalg.norm(anchor-(anchor@v)*v)<1e-8:
            C = (x@v) + (x@w)/ghat
        else:
            C = (x@lam)/ghat
        log("barrier face", day, "upper" if ghat>0 else "lower", sigma*C)
        return day, float(sigma*C), bool(ghat>0), normalize(lam)
    return None


def cdf_interval(a,b):
    return np.where(a > 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))


def killed_walk(m, sigma, mu, lower, upper, a, k):
    sd = sigma/np.sqrt(m)
    drift = mu/m
    strike = np.log(k/a)+mu
    low = max(lower,strike)
    high = upper if np.isfinite(upper) else max(mu+11*sigma+sigma*sigma, low+8.5*sigma)
    lo = lower if np.isfinite(lower) else min(mu-11*sigma, -9*sigma)
    if high <= low or high <= lo or sd <= 0.:
        return 0.0
    n = min(760,max(115,int(3.6*(high-lo)/sd)+35))
    nodes, w = roots_legendre(n)
    h = .5*(high-lo)
    x = (lo+h)+h*nodes
    w = h*w
    mean = x+drift
    A=(low-mean)/sd
    B=(upper-mean)/sd
    cash=a*np.exp(mean-mu+.5*sd*sd)
    val = np.maximum(cash*cdf_interval(A-sd, B-sd)-k*cdf_interval(A,B),0.)
    z = (x[None,:]-mean[:,None])/sd
    mat = np.exp(-.5*z*z)*(w/(sd*np.sqrt(2*np.pi)))
    for _ in range(m-2):
        val = mat@val
    return float((np.exp(-.5*((x-drift)/sd)**2)*(w/(sd*np.sqrt(2*np.pi))))@val)


def price_barrier(s, x, y, p, terminal):
    b = terminal.B[0]
    sigma = np.linalg.norm(b)
    a = terminal.a[0]
    k = terminal.k
    v = b/sigma
    strk = np.log(k/a)/sigma
    T = p@v
    if abs(T)>20 or abs(strk)>20:
        return None
    axis_test = s.query(np.array([T,T+.12,T+.45,T+1.,T+2.,T+4.,T+8.])[:,None]*v)
    has_axis = axis_test[0]>0
    if not has_axis:
        return None
    hit = np.flatnonzero(axis_test==0.)
    upper_t = np.inf
    if len(hit):
        j=hit[0]
        r = np.array([0.,.12,.45,1.,2.,4.,8.])
        upper_t = T+any_bisect(s,T*v,v,r[j-1:j],r[j:j+1],29)[0]
    low_t = strk
    if np.isfinite(upper_t):
        trial = (strk + min(T,upper_t))*.5
    else:
        trial = min(T, strk+0.4)
    if s.query(trial*v)[0] == 0.:
        lower_a = any_bisect(s,T*v,-v,np.array([0.]),np.array([max(.01,T-strk+1e-6)]),27)[0]
        low_t=T-lower_a
    if np.isfinite(upper_t):
        anchors = (low_t+(upper_t-low_t)*.45, low_t+(upper_t-low_t)*.74, low_t+(upper_t-low_t)*.23)
    else:
        anchors = (low_t+.6,low_t+1.6,low_t+2.6)
    D = {}
    U = {s.d: upper_t*sigma} if np.isfinite(upper_t) else {}
    if low_t>strk+2e-5:
        D[s.d] = low_t*sigma
    tried = {True:[],False:[]}
    normals = []
    inspected = 0
    directed = set()

    def params():
        for dd, uu in [(D,U),(U,D)]:
            if len(dd)>=2 and len(D)>0:
                ij,C = np.array(list(dd.items())).T
                xx = s.d/ij
                if np.ptp(xx)<1e-3:
                    continue
                q,t = np.polyfit(xx,C,1)
                if np.max(np.abs(q*xx+t-C)) > 1e-4:
                    continue
                mu = -t
                def cal(table):
                    return np.mean([(i/s.d)*(ci+mu) for i,ci in table.items()])
                lower = cal(D)
                upper = cal(U) if U else np.inf
                errs = [abs(c-(u*s.d/i-mu)) for tbl,u in [(D,lower),(U,upper)] for i,c in tbl.items()]
                if abs(mu)<3. and lower<.01 and upper>-.01 and max(errs)<.001:
                    return lower,upper,mu
        return None

    result = None
    for at, nr in zip(anchors,(44,62,70)):
        if s.remaining<2500 or s.elapsed>14.:
            break
        rays = s.rng.standard_normal((nr,s.d))
        rays -= (rays@v)[:,None]*v
        if normals:
            Q = np.linalg.svd(np.stack(normals),full_matrices=False)[2]
            for j in range(nr//3):
                rays[j] -= (rays[j]@Q.T)@Q
        rays = normalize(rays)
        cand = facet_candidates(s, at*v, v, rays)
        cand.sort(key=lambda q: q[1])
        while cand:
            c=cand.pop(0)
            C, r, g, w, _ = c
            up = g>0
            if (up and not U) or (up and len(U)>=2):
                continue
            if any(abs(C-old)<.012 for old in tried[up]):
                continue
            if inspected>=8 or s.remaining<1700 or s.elapsed>17.:
                break
            tried[up].append(C)
            inspected+=1
            ff = inspect_facet(s,v,c,sigma)
            if ff is None:
                # Another ray to approximately the same face can make a better angle.
                if inspected < 6:
                    tried[up].pop()
                continue
            day, val, up2, lam = ff
            if up != up2:
                continue
            vv = lam-(lam@v)*v
            normals.append(normalize(vv))
            if up:
                U.setdefault(day,val)
            else:
                D.setdefault(day,val)
            result = params()
            if result is not None:
                break
            if U and s.remaining>2100:
                # A projected Brownian loading has nonnegative inner
                # products with all other projected loadings. Its
                # negative ray therefore exposes a lower facet only;
                # its positive ray exposes an upper facet only.
                want_up = len(U)==1 and len(D)>0
                want_low = len(D)==0 and len(U)>=2
                if want_up or want_low:
                    tag=(day,want_up)
                    if tag not in directed:
                        directed.add(tag)
                        u=normalize(vv)*(1. if lam@v>0. else -1.)
                        u = u if want_up else -u
                        jitter=s.rng.normal(size=(6,s.d))
                        jitter-= (jitter@v)[:,None]*v
                        steer=normalize(u+np.r_[0.,.04,.06,.09,.14,.18][:,None]*normalize(jitter))
                        tc=facet_candidates(s,at*v,v,steer)
                        tc=[q for q in tc if (q[2]>0)==want_up]
                        tc.sort(key=lambda q:q[1])
                        cand=tc+cand
        if result is not None:
            break
    if result is not None:
        L,U,mu = result
        val=killed_walk(s.d,sigma,mu,L,U,a,k)
        log("barrier model",sigma,mu,L,U,"value",val)
        if val>0 and np.isfinite(val):
            return val
    log("barrier unresolved", D,U, inspected)
    # Fallback: sample conditional on terminal log-price being between
    # the observed axial bounds.  For a walk from inside these are necessary.
    p0=ndtr(low_t)
    p1=ndtr(upper_t)
    if p1<=p0 or s.remaining<32:
        return None
    from scipy.special import ndtri
    basis=orth(v)
    ys=[]
    ws=[]
    n=s.remaining
    while n>=32 and s.elapsed<21.:
        b=min(4096,1<<(int(n).bit_length()-1))
        u=sobol(b,s.d,s.rng)
        u[:,0]=ndtri(np.clip(p0+(p1-p0)*ndtr(u[:,0]),1e-15,1.-1e-15))
        ys.append(s.query(u@basis.T).sum())
        ws.append(b)
        n-=b
    if ws:
        return float((p1-p0)*sum(ys)/sum(ws))
    return None
