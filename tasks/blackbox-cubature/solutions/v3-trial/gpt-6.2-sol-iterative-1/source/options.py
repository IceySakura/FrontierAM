"""Identification of the analytic pieces of the path-dependent payoffs."""
import numpy as np
from scipy import linalg

from common import log, normalize, orth, sobol


def stencil(d, h):
    e = np.eye(d) * h
    i, j = np.triu_indices(d)
    return np.concatenate((np.zeros((1,d)), e, e[i]+e[j])), i, j


def hess_values(y, d, i, j):
    f = y[0]
    g = y[1:1+d] - f
    H = np.empty((d,d))
    H[i,j] = H[j,i] = (y[1+d:]-f) - g[i] - g[j]
    return H, g, f


class ExpPiece:
    def __init__(self, B, a, k):
        self.B, self.a, self.k = B, a, k

    def __call__(self, points):
        return np.exp(np.clip(points @ self.B.T, -300., 300.)) @ self.a - self.k


def fit_terminal(s, p, f0, small=False):
    """Fit a single exponential without differencing a second full stencil."""
    d = s.d
    h = .0022 if small else .01
    e = np.eye(d)*h
    p1 = p + e
    yp = s.query(p1)
    if np.any(yp <= 0.0):
        return None
    g = yp-f0
    if np.max(np.abs(g)) < 1e-10*max(1., f0):
        return None
    v = normalize(g)
    h1 = .0027 if small else .008
    zz = p + (h1*np.array([-2.,-1.,1.,2.]))[:,None]*v
    fm2, fm1, fp1, fp2 = s.query(zz)
    if min(fm2, fm1, fp1, fp2) <= 0 or fp1 <= f0:
        return None
    ratio = (fp2-fp1)/(fp1-f0)
    if ratio <= 1+1e-6 or abs(ratio - (f0-fm1)/(fm1-fm2)) > 3e-9:
        return None
    a0 = (fp1 - f0)/(ratio-1)
    b = np.log1p(g/a0)/h
    k = a0 - f0
    if k <= 0 or not (0.02 < np.linalg.norm(b) < 5):
        return None
    model = ExpPiece(b[None,:], np.array([a0 * np.exp(-p @ b)]), k)
    t = s.rng.normal(size=(12,d))*(.0035 if small else .012)
    obs = s.query(p+t)
    expect = model(p+t)
    if np.count_nonzero(obs>0)<8 or np.max(np.abs(expect[obs>0]-obs[obs>0]))>2e-7*max(1.,f0):
        return None
    return model


def exp_pencil(s, p, fp, shift, h, yy=None):
    """Two finite-difference matrices recover an identifiable exponential sum."""
    d = s.d
    st, i, j = stencil(d,h)
    if yy is None:
        yy = s.query(p+st)
    H, g, f0 = hess_values(yy,d,i,j)
    eig, U = linalg.eigh(H, check_finite=False)
    keep = eig > max(2e-11, eig[-1]*2e-8)
    R = int(np.count_nonzero(keep))
    if R not in (d, d//2) or np.any(yy <= 0) or eig[0] < -1e-7*eig[-1]:
        return None, R
    U, eig = U[:,keep], eig[keep]
    if R<d:
        _,_,perm = linalg.qr(U.T,pivoting=True,mode='economic',check_finite=False)
        ix = perm[:min(d,R+6)]
        sub = H[np.ix_(ix,ix)]
        le, evec = linalg.eigh(sub,check_finite=False)
        if le[-R]<=2e-12:
            return None,R
        Y = evec[:,-R:]/np.sqrt(le[-R:])
        W = H[:,ix]@Y
        w0 = Y.T@g[ix]
        u,ii,jj = stencil(len(ix),h)
        st2 = np.zeros((len(u),d))
        st2[:,ix] = u
    else:
        Y = U/np.sqrt(eig)
        W = U*np.sqrt(eig)
        w0 = Y.T @ g
        st2,ii,jj=st,i,j
        ix=np.arange(d)
    best = None
    tries=np.atleast_2d(shift)
    if R==d:
        tries=tries[np.argsort(-np.linalg.norm(tries,axis=1))][:2]
    else:
        tries=tries[:3]
    for t in tries:
        if s.remaining < len(st2)+320:
            break
        y2 = s.query(p+t+st2)
        if np.any(y2 <= 0):
            continue
        H2, g2, f2 = hess_values(y2,len(ix),ii,jj)
        rr, P = linalg.eigh(Y.T @ H2 @ Y, check_finite=False)
        sw = P.T @ w0
        if np.min(np.abs(sw)) < 0.04 or np.any(rr <= 0):
            continue
        A = (W @ P / sw).T
        if np.min(A) <= -0.95:
            continue
        B = np.log1p(A)/h
        aloc = sw**2
        a = aloc*np.exp(-B@p)
        K = aloc.sum()-f0
        if not np.all(np.isfinite(a)) or K <= 0.0 or np.max(a)>1000*K:
            continue
        model = ExpPiece(B, a, K)
        err = max(np.max(np.abs(np.log(rr)-B@t)), abs(model((p+t)[None,:])[0]-f2)/max(1.,abs(f2)))
        log("exp fit", R, "K", K, "pencil", err)
        if best is None or err < best[0]:
            best = (err, model)
        if err < 3e-6:
            break
    if best is None or best[0] > 2e-4:
        return None, R
    return best[1], R


def fit_exp_piece(s, p, fp, rank=None, down=None):
    d = s.d
    def steps():
        yield .37
        yield .14
        yield .061
    ns = (d+1)*(d+2)//2
    for h in steps():
        if s.remaining < 2*ns+640:
            break
        st, i, j = stencil(d,h)
        yp = s.query(p+np.r_[np.zeros((1,d)), np.eye(d)*h])
        if np.any(yp <= 0):
            continue
        v = normalize(yp[1:]-yp[0])
        cross = s.rng.standard_normal((10,d))
        cross = (cross-(cross@v)[:,None]*v)
        cross = normalize(cross)
        trial = p+np.r_[.19*v+.38*cross[:2], -.14*v+.45*cross[2:4],
                       .40*v+.57*cross[4:6],.73*v+.91*cross[6:8],
                       1.20*v+1.27*cross[8:]]
        ytr = s.query(trial)
        sel = (ytr > .45 * fp)
        if down is not None:
            gap = down(trial)-ytr
            sel &= gap > max(.1, .10*fp)
            trust = np.minimum(1.7*ytr,gap)/(1.+.30*np.linalg.norm(trial-p,axis=1))
            sel = np.flatnonzero(sel)
            sel = sel[np.argsort(trust[sel])[::-1]]
        trial = trial[sel]
        if len(trial) == 0:
            continue
        # A min-branch splice often produces negative finite-difference
        # curvature on a much smaller stencil than the full pencil.
        sd,si,sj=stencil(min(7,d),h)
        co=s.rng.choice(d,sd.shape[1],replace=False)
        E=np.zeros((len(sd),d))
        E[:,co]=sd
        U=np.r_[p[None,:],trial]
        ff=s.query((U[:,None]+E).reshape(-1,d)).reshape(len(U),len(E))
        ok=np.zeros(len(U),dtype=bool)
        for j,u in enumerate(ff):
            H,_,_=hess_values(u,len(co),si,sj)
            l=linalg.eigvalsh(H,check_finite=False)
            ok[j]=np.min(u)>0 and l[0]>-max(1e-9,1e-6*l[-1])
        if not ok[0] or not np.any(ok[1:]):
            continue
        tt = trial[ok[1:]]-p
        path = p+tt[:,None]*np.array([0.,.2,.4,.6,.8,1.])[None,:,None]
        tests = s.query(path.reshape(-1,d)).reshape(len(tt),6)
        good = np.min(np.diff(tests,2,axis=1),axis=1)>=-1e-9*max(1.,fp)
        tt=tt[good]
        if not len(tt):
            continue
        model, r = exp_pencil(s, p, fp,tt, h)
        log("local rank", r, "h", h)
        if model is not None and (rank is None or r == rank):
            if r == d or r == d//2:
                return model
        if r < d//2-1:
            break
    return None


def spacious_bases(s,x,y):
    """Prefer positive centers with a clear smooth neighborhood."""
    d=s.d
    positive=np.flatnonzero(y > 0)
    if len(positive)==0:
        return x,y,np.array([],dtype=int)
    ids=positive[np.argsort(y[positive])[-12:][::-1]]
    if len(ids)<3:
        return x,y,ids
    p=x[ids]
    small=.019*np.eye(d)
    f=s.query((p[:,None]+small).reshape(-1,d)).reshape(-1,d)
    g=normalize(f-y[ids,None])
    agree=g@g[0]
    j=int(np.argmin(agree))
    if .07<np.min(agree)<.895 or (np.min(agree)<=.07 and (g[j]@g[j]>.9)):
        # Move each candidate in directions that raise both empirical
        # branches, but raise the potential competitor first.
        u=normalize(g[0]+g[j])
        w=normalize(g[0]-g[j])
        E=np.r_[u[None,:],normalize(u+.37*w)[None,:],normalize(u-.37*w)[None,:]]
        aa=np.r_[p,
                 (p[:8,None,None,:]+ E[None,:,None,:]*np.array([1.4,3.0,5.2])[None,None,:,None]).reshape(-1,d)]
        yy=np.r_[y[ids],s.query(aa[len(ids):])]
        h=.5
        Yp=s.query(np.r_[aa-h*w,aa+h*w])
        ym,yp=np.split(Yp,2)
        signal=(ym+yp-2.*yy)/np.maximum(1.,yy)
        ok=(np.minimum(ym,yp)>0) & (signal>-.001)
        rating=np.where(ok,yy,-np.inf)
        k=np.argsort(rating)[::-1]
        if not np.isfinite(rating[k[0]]):
            return x,y,ids
        k=k[np.isfinite(rating[k])]
        return np.r_[x,aa],np.r_[y,yy],np.r_[len(y)+k,ids]
    return x,y,ids


def deepen_other_piece(s, a, x, y):
    av = a(x)
    quality = np.minimum(2.0*y, av-y)
    js = np.argsort(quality)[-5:][::-1]
    base = x[js]
    vals = y[js]
    E = np.eye(s.d)*0.02
    grads = (s.query((base[:,None,:]+E).reshape(-1,s.d)).reshape(len(base),s.d)-vals[:,None])/.02
    dirs = []
    tests = []
    lengths = np.array([.65,1.3,2.1,3.3,4.8,6.5,8.])
    for p,g in zip(base,grads):
        v = normalize(g)
        u = normalize((a.a*np.exp(a.B@p))@a.B)
        for alpha in (.6,1.,1.55,2.5):
            dirs.append(normalize(v+alpha*u))
            tests.append(p+lengths[:,None]*dirs[-1])
    q = np.concatenate(tests)
    qy = s.query(q)
    return np.r_[x,q], np.r_[y,qy]


def expose_other_piece(s, a, x, y):
    """Vary the unknown branch while holding the recovered one constant."""
    d=s.d
    pred=a(x)
    target=np.minimum(1.2*y,pred-y)
    if np.max(target)>max(3.,.15*a.k):
        return x,y
    v=normalize(np.average(a.B,axis=0,weights=a.a))
    ids=np.argsort(y)[-7:][::-1]
    p=x[ids]
    new=(p[:,None,:]+np.array([.9,2.0,3.5,5.5])[None,:,None]*v).reshape(-1,d)
    vals=s.query(new)
    x,y=np.r_[x,new],np.r_[y,vals]
    pred=a(x)
    score=np.minimum(1.2*y,pred-y)
    if np.max(score)>max(3.,.15*a.k):
        return x,y
    # Select several substantial anchors where the known branch is the
    # active one. On a kernel ray its payoff cannot change.
    good=np.flatnonzero((y>0)&(abs(y-pred)<2e-5*np.maximum(1.,y)))
    if len(good)==0:
        return x,y
    good=good[np.argsort(y[good])[::-1]][:6]
    V=linalg.null_space(a.B,rcond=2e-7)
    w=normalize(s.rng.normal(size=(len(good)*15,V.shape[1]))@V.T)
    base=np.repeat(x[good],15,axis=0)
    goals=np.repeat(y[good]*.57,15)
    left=np.zeros(len(base))
    right=np.zeros(len(base))
    active=np.arange(len(base))
    for t in (2.2,4.8,8.6,14.1,20.):
        if not len(active):
            break
        val=s.query(base[active]+t*w[active])
        hit=val<goals[active]
        right[active[hit]]=t
        left[active[~hit]]=t
        active=active[~hit]
    ix=np.flatnonzero(right>0)
    if not len(ix):
        return x,y
    rating=goals[ix]/(1.+.10*right[ix])
    ix=ix[np.argsort(rating)[-20:]]
    b,u,lo,hi,tar=base[ix],w[ix],left[ix],right[ix],goals[ix]
    for _ in range(11):
        tt=(lo+hi)*.5
        f=s.query(b+tt[:,None]*u)
        lower=f>=tar
        lo,hi=np.where(lower,tt,lo),np.where(lower,hi,tt)
    q=b+(.5*(lo+hi))[:,None]*u
    f=s.query(q)
    log("kernel-ray exposures",len(q),"best second clearance",np.max(np.minimum(1.2*f,a(q)-f)))
    return np.r_[x,q],np.r_[y,f]


def explore_positive(s,x,y):
    n=min(2400,s.remaining//4)
    d=s.d
    p=s.rng.standard_normal((n,d))*np.resize([.055,.115,.23,.37,.54,.75,
                                              1.03,1.35,1.72,2.35,3.10],n)[:,None]
    yp=s.query(p)
    x,y=np.r_[x,p],np.r_[y,yp]
    for i in range(4):
        if np.any(y>0.) or s.remaining<8000 or s.elapsed>11.5:
            break
        p=s.rng.standard_normal((890,d))*np.resize([.17,.22,.27,.32,.39,.47,.57,.69,1.95,2.9],890)[:,None]
        yp=s.query(p)
        x,y=np.r_[x,p],np.r_[y,yp]
    if np.count_nonzero(y)>8 or not np.any(y>0.):
        return x,y
    b=x[np.argmax(y)]
    spreads=s.rng.standard_normal((240,d))*np.resize([.022,.08,.19,.35,.64],240)[:,None]
    p=b+spreads
    yp=s.query(p)
    return np.r_[x,p],np.r_[y,yp]


def one_piece_control(s, a, x, y):
    """When only one min branch is known, condition on its exercise set."""
    pred=np.maximum(a(x),0.)
    if np.max((y-pred)/np.maximum(5.,y))>8e-5:
        return None
    walk=fit_walk([a],len(a.a))
    if walk is None or len(walk[1])!=1:
        return None
    m,vol,drift,scale,rho,k=walk
    from perpetuity import single_price
    known=single_price(m,vol[0],drift[0],scale[0],k)
    v=normalize(np.average(a.B,axis=0,weights=a.a))
    av=a.B@v
    if np.min(av)<1e-5 or s.remaining<128:
        return None
    E=orth(v)
    from scipy.special import ndtr, ndtri
    zz=[]
    weights=[]
    while s.remaining>=64 and s.elapsed<21.:
        n=min(4096,1<<(s.remaining.bit_length()-1))
        z=sobol(n,s.d,s.rng)
        p=z[:,1:]@E[:,1:].T
        q=np.exp(np.clip(p@a.B.T,-120.,120.))*a.a
        lo=np.full(n,-18.)
        hi=np.full(n,18.)
        for _ in range(26):
            M=.5*(lo+hi)
            hit=np.sum(q*np.exp(M[:,None]*av),axis=1)>a.k
            lo,hi=np.where(hit,lo,M),np.where(hit,M,hi)
        mass=ndtr(-.5*(lo+hi))
        t=ndtri(np.clip(mass*ndtr(z[:,0]),1e-100,1.-1e-15))
        p-=t[:,None]*v
        resid=s.query(p)-np.maximum(0.,a(p))
        zz.append(float(np.mean(mass*resid)))
        weights.append(n)
    if not weights:
        return None
    value=known+np.average(zz,weights=weights)
    log("one-piece conditional control",known,value)
    return float(np.clip(value,0.,known))


def fit_walk(pieces, d):
    """Find equal-time Brownian paths in the recovered exponent Gram matrix."""
    B = np.concatenate([p.B for p in pieces], axis=0)
    logs = np.log(np.concatenate([p.a for p in pieces]))
    G = B @ B.T
    n2 = np.diag(G)
    if len(B) != d:
        return None
    modes = (1, 2) if len(pieces)==1 else (2,)
    for L in modes:
        m = d//L
        # Each endpoint is the longest unused Brownian loading in its asset.
        unused = np.ones(d, dtype=bool)
        groups = []
        norms = []
        if len(pieces)==2:
            for j in range(2):
                iii=np.arange(j*m,(j+1)*m)
                groups.append(iii[np.argsort(n2[iii])])
        else:
            for _ in range(L):
                end = np.argmax(np.where(unused, n2, -np.inf))
                inds = np.flatnonzero(unused & (np.abs(G[end]-n2) < 3e-4 * n2[end]))
                if len(inds) != m:
                    break
                ids = inds[np.argsort(n2[inds])]
                unused[ids] = False
                groups.append(ids)
                norms.append(n2[end])
        if len(groups) != L or (len(pieces)==2 and np.any(groups[0]<m) and np.any(groups[0]>=m)):
            continue
        day = np.arange(1,m+1,dtype=float)
        mins = np.minimum.outer(day,day)
        stride = []
        drift = []
        start = []
        for ids in groups:
            gg = G[np.ix_(ids,ids)]
            v = np.sum(gg*mins)/np.sum(mins**2)
            if not (1e-7 < v < 0.6) or np.max(np.abs(gg-v*mins)) > 2e-4*m*v:
                break
            ell = logs[ids]
            b,a = np.polyfit(day,ell,1)
            if np.max(np.abs(ell-a-b*day))>1e-3:
                break
            stride.append(np.sqrt(v))
            drift.append(b)
            start.append(np.exp(a))
        if len(stride) != L:
            continue
        rho = 0.0
        if L == 2:
            gg = G[np.ix_(*groups)]
            v = np.sum(gg*mins)/np.sum(mins**2)
            rho = v/(stride[0]*stride[1])
            if abs(rho)>=.998 or np.max(np.abs(gg-v*mins)) > 3e-4*m*stride[0]*stride[1]:
                continue
        k = np.mean([p.k for p in pieces])
        return m, np.array(stride), np.array(drift), np.array(start), rho, k
    return None


def continuous_option(s, x, y):
    d = s.d
    if np.count_nonzero(y > 0) < 12:
        # Narrow convex survival regions can be much easier to enter near zero.
        x,y=explore_positive(s,x,y)
        if np.max(y) == 0:
            return None
        f = y[y>0.]
        if len(f)>=2 and np.ptp(f)<1e-12*f[0]:
            # Train the direction only on sites of a comparable scale.
            return digital_option(s,x,y)
    ix = np.argsort(y)[-12:][::-1]
    terminal = None
    for j in ix[:4]:
        if y[j] <= 0:
            break
        terminal = fit_terminal(s,x[j],y[j])
        if terminal is not None:
            hit=y>0.
            if np.max(np.abs(terminal(x[hit])-y[hit])/np.maximum(1.,y[hit]))<3e-6:
                from barrier import price_barrier
                return price_barrier(s,x,y,x[j],terminal)
        if j == ix[1] and np.count_nonzero(y>0)>50:
            break
    if np.count_nonzero(y>0)<110:
        hit=np.flatnonzero(y>0.)
        for p in (np.mean(x[hit],axis=0),x[hit[np.argsort(y[hit])[len(hit)//2]]]):
            val=s.query(p)[0]
            if val<=0.:
                continue
            a=fit_terminal(s,p,val,small=True)
            if a is not None and np.max(np.abs(a(x[hit])-y[hit])/np.maximum(1.,y[hit]))<3e-6:
                from barrier import price_barrier
                return price_barrier(s,x,y,p,a)
    if y[ix[0]] <= 0:
        return None
    x,y,bc=spacious_bases(s,x,y)
    a=None
    for j in bc[:3]:
        if s.remaining<max((d+1)*(d+2)+650,1600) or s.elapsed>13.:
            break
        a = fit_exp_piece(s, x[j], y[j])
        if a is not None:
            break
    if a is None:
        return None
    pieces = [a]
    if len(a.a) != d:
        aa = a(x)
        # First bring more points onto the other smooth branch, then move
        # away from the two nearby surfaces using its local gradient.
        v = normalize(np.average(a.B,axis=0,weights=a.a))
        qs = x[ix[:8]][:,None,:] + np.linspace(.45,5.1,10)[None,:,None]*v
        extra = qs.reshape(-1,d)
        x = np.r_[x,extra]
        y = np.r_[y,s.query(extra)]
        if s.remaining>(d+1)*(d+2)+1100:
            x,y=expose_other_piece(s,a,x,y)
        aa = a(x)
        sel = np.minimum(2.*y,aa-y)/np.sqrt(1.+(y/(6*a.k))**2)
        if np.max(sel) <= .1:
            return one_piece_control(s,a,x,y)
        x, y = deepen_other_piece(s,a,x,y)
        aa = a(x)
        sel = np.minimum(2.*y,aa-y)
        log("other branch clearance",float(np.max(sel)))
        js = np.argsort(sel)[::-1]
        for j in js[:2]:
            if not np.isfinite(sel[j]):
                break
            b = fit_exp_piece(s, x[j], y[j], rank=d//2, down=a)
            if b is not None and abs(b.k-a.k)<3e-4*a.k:
                pieces.append(b)
                break
        if len(pieces) != 2:
            return one_piece_control(s,a,x,y)
    walk = fit_walk(pieces,d)
    log("walk", walk)
    adj = np.minimum(*(p(x) for p in pieces)) if len(pieces)==2 else a(x)
    err = np.max(np.abs(np.maximum(0., adj)-y)/np.maximum(5., y))
    log("option pilot error",err)
    if walk is None or err > 8e-5:
        return None
    m, vol, drift, scale, rho, k = walk
    from perpetuity import single_price, pair_price
    if len(vol)==1:
        val = single_price(m,vol[0],drift[0],scale[0],k)
    else:
        val = pair_price(m,vol,drift,scale,rho,k,len(pieces)==2)
    if np.isfinite(val) and val > 0:
        log("identified Asian",val)
        return val
    return None


def digital_option(s,x,y):
    from digital import digital_price
    return digital_price(s,x,y)
