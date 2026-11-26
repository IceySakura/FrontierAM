"""Detect discontinuities and build exact-integral or preintegrated controls."""

import itertools

import numpy as np
from scipy.linalg import eigh

from common import first_basis, gradient, jet, moment_rotation, report, rqm, unit
from models import Control, Quadratic


def scanner(w, x, y):
    d = w.d
    rng = w.rng
    ids = np.argsort(y)
    n = 24
    dirs = rng.normal(size=(n,d))
    dirs /= np.linalg.norm(dirs,axis=1,keepdims=True)
    centers = np.zeros_like(dirs)
    radii = np.full(n, max(6.5, 1.5*np.sqrt(d)))
    for i in range(6,14):
        k = ids[min(len(ids)-1, int((i-6)*.12*len(ids)))]
        dirs[i] = unit(x[k])
    for i in range(14,24):
        a = x[ids[rng.integers(len(ids)//5, len(ids)*4//5)]]
        b = x[ids[rng.integers(len(ids))]]
        if np.linalg.norm(a-b)<.02:
            b=-a
        centers[i] = (a+b)*.5
        dirs[i] = unit(b-a)
        radii[i] = max(4.5,np.linalg.norm(a-b)*.78)
    # A far outlier can be the only discovered point in a displaced rare ellipsoid.
    if y.max()-np.median(y) > 12*max(.0001,np.quantile(y,.75)-np.quantile(y,.25)):
        centers[-2:] = x[np.argmax(y)]
    T = radii[:,None]*np.linspace(-1.,1.,37)
    X = centers[:,None,:]+T[:,:,None]*dirs[:,None,:]
    Y = w.ask(X.reshape(-1,d)).reshape(T.shape)
    return centers, dirs, T, X, Y


def rescan(w, scan, ctrl=None, n=9):
    centers, dirs, T, X, Y = scan
    R = Y if ctrl is None else Y-ctrl(X.reshape(-1,w.d)).reshape(Y.shape)
    dd = np.abs(np.diff(R, n=2, axis=1))
    dd /= (T[:,1]-T[:,0])[:,None]
    cand = []
    for flat in np.argsort(dd.ravel())[::-1]:
        i,j = np.unravel_index(flat,dd.shape)
        if any(i==v and abs(j-k)<3 for v,k in cand):
            continue
        cand.append((i,j))
        if len(cand)==n:
            break
    li, lj = np.array(cand).T
    C, D = centers[li], dirs[li]
    a,b = T[li,np.maximum(0,lj-1)], T[li,np.minimum(T.shape[1]-1,lj+3)]
    multi = len(li)
    for _ in range(5):
        g = a[:,None]+(b-a)[:,None]*np.linspace(0.,1.,9)
        X1 = C[:,None,:]+g[:,:,None]*D[:,None,:]
        raw = w.ask(X1.reshape(-1, w.d)).reshape(g.shape)
        if ctrl is not None:
            raw -= ctrl(X1.reshape(-1,w.d)).reshape(g.shape)
        h = g[:,1]-g[:,0]
        curv = np.abs(np.diff(raw,n=2,axis=1))
        mt = np.argmax(curv,axis=1)+1
        r = np.arange(multi)
        a, b = g[r,mt-1],g[r,mt+1]
    ts = .5*(a+b)
    t = ts[:,None]+np.array([-4.,-3.,-2.,2.,3.,4.])*h[:,None]
    X1 = C[:,None,:]+t[:,:,None]*D[:,None,:]
    raw = w.ask(X1.reshape(-1,w.d)).reshape(t.shape)
    rr = raw if ctrl is None else raw-ctrl(X1.reshape(-1,w.d)).reshape(t.shape)
    sl = (rr[:,2]-rr[:,0])/(2*h)
    sr = (rr[:,5]-rr[:,3])/(2*h)
    yl = rr[:,1]+3*h*sl
    yr = rr[:,4]-3*h*sr
    delta = yr-yl
    slope = sr-sl
    scale = max(.1, np.quantile(Y,.7)-np.quantile(Y,.3), np.median(np.abs(Y))*.17)
    edges = []
    for k in range(multi):
        if abs(delta[k]) > max(.012*scale,abs(slope[k])*h[k]*2.4):
            mode = 'jump'
            tt = ts[k]
        elif abs(slope[k]) > .015*scale and curv[k].max()/h[k]>.20*abs(slope[k]):
            mode = 'kink'
            tt = ts[k]-np.clip(delta[k]/slope[k],-1.3*h[k],1.3*h[k])
        else:
            continue
        pos = C[k]+tt*D[k]
        rank = abs(delta[k])*10 if mode=='jump' else abs(slope[k])
        edges.append((mode,rank,pos,D[k],float(slope[k]),float(delta[k]),
                      C[k]+t[k,2]*D[k],C[k]+t[k,3]*D[k],raw[k,2],raw[k,3]))
    return sorted(edges,key=lambda v:v[1],reverse=True)


def recover_kink(w, edge):
    _,_,x, u, slope, _,_,_,_,_ = edge
    _, G = gradient(w,np.stack([x-u*.00076,x+u*.00076]), h=.000014)
    g = G[1]-G[0]
    if abs(g@u)<.0001 or np.linalg.norm(g)<.0001:
        return None
    v = unit(g)*np.sign(g@u)
    R = first_basis(v)
    h = .00068
    p, q = x-2.8*h*v, x+2.8*h*v
    _,gp,Hp = jet(w,p,h=h,frame=R)
    _,gq,Hq = jet(w,q,h=h,frame=R)
    A = (Hq-Hp)*.5
    b = (gq+Hq@(x-q))-(gp+Hp@(x-p))-2*A@x
    F = Quadratic(-x@A@x-b@x,b,A,drop=.000035)
    # Preserve the located boundary point after spectral roundoff cleanup.
    F.c -= F(x)
    gproj = (F.b+2*F.A@x)@u
    if (not np.isfinite(F.std()) or abs(gproj-slope) > .30*abs(slope)
            or F.std() > max(1.,abs(slope))*140.):
        return None
    return F


def choose_kink_model(w, fs, scan):
    d = w.d
    Y = scan[-1]
    X = scan[-2].reshape(-1,d)
    Q = [q(X).reshape(Y.shape) for q,s in fs]
    a = np.diff(Y,n=3,axis=1)
    if a.size==0:
        return None, 0.
    n = len(Q)
    penalty = max(1.e-14,np.mean(a**2))
    baseline = np.mean(a**2)
    best_val = baseline*.99
    best = None

    def check(C, Z, penalty):
        nonlocal best_val,best
        err = a-np.diff(Z,n=3,axis=1)
        val = np.mean(err**2)+penalty
        if val < best_val:
            best_val,best = val,C

    for mask in range(1,2**n):
        use = [j for j in range(n) if mask&(1<<j)]
        check(Control([fs[j][0] for j in use],signs=[fs[j][1] for j in use]),
              sum(fs[j][1]*np.maximum(0.,Q[j]) for j in use),.000002*len(use)*penalty)
    z = np.zeros_like(Y)
    zero = Quadratic(0.,np.zeros(d),np.zeros((d,d)))
    rules = [((0,), (1,), (2,)), ((0,1,2),)]
    for k in range(3):
        i,j=[p for p in range(3) if p!=k]
        rules.extend([((k,), (i,j)), ((k,i),(k,j))])
    rules.append(((0,1),(0,2),(1,2)))
    for i in range(n):
        for j in range(i):
            for s,t in itertools.product([-1.,1.],repeat=2):
                vals = [z,s*Q[i],t*Q[j]]
                forms = [zero,fs[i][0].scale(s),fs[j][0].scale(t)]
                for sets in rules:
                    ctrl=Control(forms,'lattice',sets=sets)
                    check(ctrl,ctrl.select(vals),.000006*penalty)
    return best, best_val/max(baseline,1.e-30)


def continuous(w, x, y, scan, edges):
    p = (w.d+1)*(w.d+2)//2
    models = []
    ctrl = None
    for j in range(3):
        cand = None
        for edge in edges:
            if edge[0]!='kink':
                continue
            if any(abs(q(edge[2]))<.002*q.std() for q,s in models):
                continue
            cand = edge
            break
        if cand is None or w.left < 2*p+2*(w.d+1)+2300:
            break
        f = recover_kink(w,cand)
        if f is not None:
            models.append((f,float(np.sign(cand[4]))))
            ctrl,rel = choose_kink_model(w,models,scan)
            if rel < .00035:
                break
        if w.left < 2*p+2800 or j==2:
            break
        edges = rescan(w,scan,ctrl,n=7)
    if ctrl is None:
        return rqm(w,moment_rotation(x,y))
    # Independent pilot gradients for the smooth residual, followed by a centered Hermite control.
    estimate = ctrl.mean(w)
    zt = w.normal(min(512,w.left//5))
    yt = w.ask(zt)-ctrl(zt)
    return estimate+smooth_tail(w,zt,yt,ctrl)


def smooth_tail(w, x, f, ctrl):
    """RQMC remainder, rotated and with an integrated smooth control."""
    from smooth import smooth_integrate
    return smooth_integrate(w,x,f,ctrl)


def synthetic(w, x, y):
    if w.left<3700:
        return rqm(w,moment_rotation(x,y))
    scan=scanner(w,x,y)
    edges=rescan(w,scan)
    if not edges:
        return smooth_tail(w,x,y,lambda z: 0.)
    if edges[0][0]=='jump':
        from jump import discontinuous
        return discontinuous(w,x,y,scan,edges)
    return continuous(w,x,y,scan,edges)
