"""Localizes branch transitions and fits Gaussian-integrable control variates."""
import math

import numpy as np
from scipy import linalg, special

from numerics import (Quad, balanced_blocks, balanced_rule, fd_gradient, hessian,
                      log, max_mean, normal, rect_gaussian_regression, sym)


def no_model(x):
    return np.zeros(len(x))


class SignedQueries:
    def __init__(self, ctx, sign):
        self.ctx,self.sign=ctx,sign

    def __getattr__(self, name):
        return getattr(self.ctx,name)

    def query(self, points):
        return self.sign*self.ctx.query(points)


def jump_orientation(edges):
    votes=[]
    for i,(p,v,h,l,r) in enumerate(edges):
        for q,u,k,a,b in edges[i+1:]:
            if v@u<1.-5e-10 or np.sign(r-l)==np.sign(b-a):
                continue
            sep=(q-p)@v
            if abs(sep)<max(.027,8*(h+k)):
                continue
            score=min(abs(r-l),abs(b-a))
            if score>1e-6:
                votes.append((np.sign((r-l)*sep),score))
    if len(votes)>=2:
        s,w=np.asarray(votes).T
        if np.dot(s,w)<-.75*np.sum(w):
            return -1.
    return 1.


def jump_branch(ctx, x, y, edge, z, w):
    from jumps import jumps
    sign=jump_orientation(edge)
    if sign<0:
        e0=[(p,v,h,-l,-r) for p,v,h,l,r in edge]
        out=jumps(SignedQueries(ctx,sign),x,-y,e0,z,-w)
        return None if out is None else -out
    return jumps(ctx,x,y,edge,z,w)


def radial_search(ctx, x, y):
    d = ctx.d
    R = ctx.rng.standard_normal((16, d))
    a = np.zeros_like(R)
    scale = max(1e-3, np.median(np.abs(y)))
    if (np.max(y) > 2.5*scale or np.max(y)-np.min(y) > .5*scale):
        # In addition to the origin, inspect lines through a few high values.
        a[-4:] = x[np.argsort(y)[-4:]]
    t = np.linspace(-3., 3., 41)
    z = a[:, None] + t[None, :, None] * R[:, None]
    w = ctx.query(z.reshape(-1, d)).reshape(len(R), -1)
    score = np.abs(np.diff(w, n=4, axis=1))
    order = np.argsort(score.ravel())[::-1]
    picks = []
    for item in order:
        i, j = np.unravel_index(item, score.shape)
        j += 2
        if any(ii == i and abs(j-jj) <= 2 for ii, jj in picks):
            continue
        if score[i, j-2] < max(1e-5*scale, .022*score.flat[order[0]]):
            break
        picks.append((i, j))
        if len(picks) >= (14 if d < 40 else 10):
            break
    if not picks:
        return [], z.reshape(-1, d), w.ravel()
    i, j = np.array(picks).T
    V = R[i]
    c = a[i] + t[j, None]*V
    length = np.linalg.norm(V, axis=1)
    V /= length[:, None]
    h = (t[1]-t[0])*length
    u = np.arange(-7, 8) * .25
    for _ in range(9):
        w0 = ctx.query((c[:, None] + h[:, None, None]*u[None, :, None]*
                        V[:, None]).reshape(-1, d)).reshape(len(c), -1)
        ii = np.abs(np.diff(w0, n=4, axis=1)).argmax(axis=1)+2
        c += (u[ii]*h)[:, None]*V
        h *= .25
    h = np.maximum(4.1*h, 3e-6)
    pm = ctx.query(np.vstack([c - V*h[:, None], c + V*h[:, None]]))
    l, r = np.split(pm, 2)
    return list(zip(c, V, h, l, r)), z.reshape(-1, d), w.ravel()


def jet_pair(ctx, x, direction, G, full=True):
    """Difference of the two local Hessians, with low rank interpolation if safe."""
    d = ctx.d
    E = np.eye(d)
    u = G / np.linalg.norm(G)
    if direction @ u < 0:
        u = -u
    h = .00019
    rho = (2.5*np.max(np.abs(u))+.1)*h + 8e-6
    def difference(P):
        Y=ctx.query(np.concatenate([x+rho*u+P,x-rho*u+P])).reshape(2,-1)
        base=Y[0]-Y[1]
        if d<=16:
            Y2=ctx.query(np.concatenate([x+2*rho*u+P,x-2*rho*u+P])).reshape(2,-1)
            base=2*base-(Y2[0]-Y2[1])
        return base
    P = np.vstack([np.zeros((1, d)), h*E, -h*E])
    D = difference(P)
    H = np.diag((D[1:1+d]+D[1+d:]-2*D[0])/h**2)
    plus = D[1:1+d]
    old = 0
    i_all, j_all = np.triu_indices(d, 1)

    def extend(k):
        nonlocal old
        take = (i_all >= old) & (i_all < k)
        i, j = i_all[take], j_all[take]
        P = h*(E[i]+E[j])
        if i.size:
            Y = difference(P)
            H[i, j] = H[j, i] = (Y-plus[i]-plus[j]+D[0])/h**2
        old = k

    k = min(d, 12)
    while True:
        extend(k)
        if k >= d:
            return sym(H)
        ev, R = np.linalg.eigh(H[:k, :k])
        tol = max(2e-6, np.max(np.abs(ev))*4e-5)
        keep = np.abs(ev) > tol
        if 0 < keep.sum() <= k-3:
            projected = H[:, :k] @ R[:, keep]
            Q = sym((projected/ev[keep]) @ projected.T)
            if np.linalg.norm(Q, ord="fro") < 12*np.linalg.norm(H[:, :k], ord="fro"):
                # Validate the missing block via a small random sample of entries.
                pairs = np.flatnonzero(i_all >= k)
                pairs = ctx.rng.choice(pairs, size=min(18, len(pairs)), replace=False)
                i, j = i_all[pairs], j_all[pairs]
                if len(i):
                    P = h*(E[i]+E[j])
                    Y = difference(P)
                    v = (Y-plus[i]-plus[j]+D[0])/h**2
                    err = np.max(np.abs(v-Q[i, j]))
                else:
                    err = 0.
                if err < max(2e-5, 2e-4*np.max(np.abs(ev))):
                    return Q
        if full:
            k = min(d, max(k+12, k*2))
        else:
            max_k = min(d, max(k, (ctx.remaining-3500)//(2*max(d-k, 1))+k))
            if max_k <= k:
                raise ValueError("insufficient budget for this branch jet")
            k = min(max_k, max(k+8, k*2))


def clean_quad(q):
    e, V = np.linalg.eigh(q.A)
    j = np.abs(e) > max(np.max(np.abs(e))*5e-5, 1.5e-6)
    b = q.b
    if j.sum() < len(e) and np.linalg.norm(V[:, ~j].T @ b) < max(5e-5, 8e-4*np.linalg.norm(b)):
        b = V[:, j] @ (V[:, j].T @ b)
    return Quad((V[:, j] * e[j]) @ V[:, j].T, b, q.c), e, j


def fit_kinks(ctx, edges, x, y):
    d = ctx.d
    quads = []
    bumps = []
    eps = max(4e-5, 2e-4 * np.std(y))
    for p, v, h, left, right in edges:
        if ctx.remaining < 3800 + 6*d:
            break
        if any(abs(q(p)) < max(.002, .0005*q.sd()) for q in quads):
            continue
        k = max(3.5*h, 1.8e-5)
        G = fd_gradient(ctx.query, np.vstack([p-k*v, p+k*v]), h=max(3e-7, h*.11))
        g0 = G[1]-G[0]
        if np.linalg.norm(g0) < eps or g0@v < eps*.15:
            continue
        G2 = fd_gradient(ctx.query, np.vstack([p-2*k*v, p+2*k*v]), h=max(3e-7, h*.11))
        g = 2*g0-(G2[1]-G2[0])
        if np.linalg.norm(g-g0) > .045*np.linalg.norm(g0):
            continue
        if any(abs(q(p)) < .015 and np.linalg.norm(q.grad(p)-g) < .005 for q in quads):
            continue
        if ctx.remaining < 3500+d*(d+1)+8*d and len(quads) and not bumps:
            break
        try:
            J = jet_pair(ctx, p, v, g, full=(d < 40 or len(quads) == 0))
        except ValueError:
            break
        q0 = Quad.at(.5*J, g, p, 0.)
        q, eig, keep = clean_quad(q0)
        # Keep the reconstructed surface through its observed crossing.
        q = Quad(q.A, q.b, q.c - q(p))
        log("kink jet", len(quads), keep.sum(), eig[0], eig[-1], "anchor", np.linalg.norm(p))
        if not np.any(keep) or q.sd() > 1e4*max(1., np.std(y)):
            continue
        quads.append(q)
        if np.all(eig[keep] < 0):
            bumps.append(q)
        elif np.all(eig[keep] > 0):
            bumps.append(q.scaled(-1.))

        if len(bumps) >= 2 or len(quads) >= 3:
            break
        if len(quads) == 1:
            resid = y - np.maximum(0, q(x))
            if np.std(resid) < .19*max(np.std(y), 1e-7) or (
                not bumps and np.std(resid) < .40*max(np.std(y), 1e-7)
            ):
                break
    if not quads:
        return [], no_model, 0.
    if len(bumps) == 1 and len(quads) > 1:
        a = bumps[0]
        for q in quads:
            if q is a or abs(abs(np.sum(a.A*q.A))-np.linalg.norm(a.A)*np.linalg.norm(q.A)) < 1e-9:
                continue
            for ans in [a.plus(q), a.plus(q.scaled(-1))]:
                cand, eig, keep = clean_quad(ans)
                if keep.any() and np.all(eig[keep] < 0):
                    bumps.append(cand)
                    break
            if len(bumps) > 1:
                break

    opts = [[q] for q in quads]+[[q.scaled(-1.)] for q in quads]
    if len(bumps) > 1:
        opts.append(bumps)
    opts.append([])
    rot = np.eye(d)
    if quads:
        _, rot = np.linalg.eigh(sum(q.A @ q.A for q in quads))
        rot = rot[:, ::-1]
    tr = np.arange(len(y)) % 4 != 0
    best = None
    for opts0 in opts:
        def m(z, qs=opts0):
            if not qs:
                return no_model(z)
            return np.maximum(0., np.max([q(z) for q in qs], axis=0))
        r = y-m(x)
        _, pred = rect_gaussian_regression(x[tr], r[tr], rank_quad=min(5, d), R=rot, ridge=.02)
        err = float(np.std(r[~tr]-pred(x[~tr]))+.014*np.std(r))
        if best is None or err < best[0]:
            best = err, opts0, m
    err, qs, m = best
    log("kink control", len(qs), err, "raw", np.std(y))
    return qs, m, max_mean(qs, ctx.rng)


def hermite_ridge_control(ctx, model, mean, x, y, spend_hessian=True, eigen_only=False):
    """Learn centered low-order additive Gaussian polynomials of the remainder."""
    d = ctx.d
    r = y-model(x)
    eye = np.eye(d)
    R = eye
    if spend_hessian and ctx.remaining > (d+1)*(d+2)//2+2600 and np.std(r)>2e-8:
        def diff(z):
            return ctx.query(z)-model(z)
        try:
            c, b, H = hessian(diff, np.zeros(d), .015)
            ev, R0 = np.linalg.eigh(H if eigen_only else H@H + .1*np.outer(b,b))
            if np.linalg.norm(H) < 15*max(1e-6,np.std(r)):
                R = R0[:, np.argsort(-np.abs(ev))].copy()
        except (ValueError, np.linalg.LinAlgError):
            pass
    if ctx.remaining < 11*d+160:
        a, poly = rect_gaussian_regression(x, r, min(d, 3), R, .025)
        return mean+a, lambda z: model(z)+poly(z), R
    t, w = special.roots_hermitenorm(9)
    w /= np.sqrt(2*np.pi)
    Z = (t[None, :, None]*R.T[:, None, :]).reshape(-1, d)
    f = (ctx.query(Z)-model(Z)).reshape(d, -1)
    p = np.vstack([special.eval_hermitenorm(j, t)/math.sqrt(math.factorial(j)) for j in range(1, 8)])
    coef = (f*w[None]) @ p.T
    # Strong shrinkage on a high-order univariate term if its scale signals a missed jump.
    raw = np.std(r)
    deg = np.arange(1, 8)
    trunc = np.array([2.5, 2., .9, .65, .45, .22, .14])*max(raw, 1e-9)
    coef = np.clip(coef, -trunc, trunc)

    def pol(z):
        u = z @ R
        out = u @ coef[:, 0]
        g0, g1 = np.ones_like(u), u
        for j in range(2, 8):
            g0, g1 = g1, (u*g1 - np.sqrt(j-1)*g0)/np.sqrt(j)
            out += g1 @ coef[:, j-1]
        return out

    rr = r-pol(x)
    # Fit low-degree cross terms on pilot points, with an independent fit check.
    tr = np.arange(len(y)) % 4 != 0
    _, aux = rect_gaussian_regression(x[tr], rr[tr], min(7, d), R, .06)
    if np.std(rr[~tr]-aux(x[~tr])) < .99*np.std(rr[~tr]):
        a, aux = rect_gaussian_regression(x, rr, min(7, d), R, .06)
    else:
        a = float(np.mean(rr))
        def aux(z):
            return a
    if np.std(rr[~tr]-aux(x[~tr])) > np.std(r[~tr])*1.10:
        a, aux = rect_gaussian_regression(x, r, min(5, d), R, .06)
        return mean+a, lambda z: model(z)+aux(z), R
    return mean+a, lambda z: model(z)+pol(z)+aux(z), R


def folded_residual(blocks, yy, d):
    n=sum(map(len,blocks))
    if n<1900 or len(blocks)<2:
        return float(np.concatenate(yy).mean())
    a=[];b=[]
    c1=c2=0
    for k,u in enumerate(blocks):
        if c1<=c2:
            a.append(k)
            c1+=len(u)
        else:
            b.append(k)
            c2+=len(u)
    if min(c1,c2)<660:
        return float(np.concatenate(yy).mean())
    rk=min(10,max(3,int(np.sqrt(min(c1,c2)/85))))
    i,j=np.triu_indices(d,1)
    use=i<min(rk,max(2,(min(c1,c2)//5)//d))
    i,j=i[use],j[use]
    q,k=np.triu_indices(min(rk+3,d),1)

    def features(U):
        S=U*U-1
        return np.column_stack([np.ones(len(U)),S/np.sqrt(2),U[:,i]*U[:,j],
              (S[:,:rk]**2-4*S[:,:rk]-2)/np.sqrt(24), .5*S[:,q]*S[:,k],
              (U[:,q]**3-3*U[:,q])*U[:,k]/np.sqrt(6),
              (U[:,k]**3-3*U[:,k])*U[:,q]/np.sqrt(6)])
    Xa=features(np.concatenate([blocks[k] for k in a]))
    Ya=np.concatenate([yy[k] for k in a])
    Xb=features(np.concatenate([blocks[k] for k in b]))
    Yb=np.concatenate([yy[k] for k in b])

    def fit(X,Y):
        G=X.T@X
        penalty=np.full(len(G), .09*len(X))
        penalty[0]=1e-9
        penalty[1:1+d]=.028*len(X)
        penalty[1+d:1+d+len(i)]=.053*len(X)
        G.flat[::len(G)+1]+=penalty
        return linalg.solve(G, X.T@Y, assume_a="pos")
    try:
        pa = fit(Xa,Ya)
        pb = fit(Xb,Yb)
    except (ValueError,np.linalg.LinAlgError):
        return float((Ya.sum()+Yb.sum())/n)
    ma = (Ya-Xa[:,1:]@pb[1:]).mean()
    mb = (Yb-Xb[:,1:]@pa[1:]).mean()
    log("folded residual",ma,mb,"raw",Ya.mean(),Yb.mean(),"feature count",Xa.shape[1])
    return float((c1*ma+c2*mb)/n)


def integrate_remainder(ctx, m, mean, x, y, antithetic=True, spend_hessian=True,
                        continuous=False):
    if ctx.d<=16 and ctx.remaining>=9300 and antithetic and spend_hessian and continuous:
        rel=np.std(y-m(x))/max(np.std(y),1e-10)
        if rel<.47:
            from sparse import sparse_residual
            q=sparse_residual(ctx,m,mean,x,y)
            if q is not None and np.isfinite(q):
                return q
    a, cv, rot = hermite_ridge_control(ctx, m, mean, x, y, spend_hessian=spend_hessian,
                                       eigen_only=continuous)
    if ctx.remaining < 16:
        return float(a)
    if antithetic and ctx.d>=20 and ctx.remaining>=4600 and continuous:
        blocks=balanced_blocks(ctx.remaining//2,ctx.d,ctx.rng)
        smp=np.concatenate(blocks)
        obs=smp@rot.T
        z=np.vstack([obs,-obs])
        f=ctx.query(z)-cv(z)
        half=.5*(f[:len(obs)]+f[len(obs):])
        return float(a+folded_residual(blocks,np.split(half,np.cumsum([len(s) for s in blocks])[:-1]),ctx.d))
    if antithetic:
        obs = balanced_rule(ctx.remaining//2, ctx.d, ctx.rng) @ rot.T
        z = np.vstack([obs, -obs])
    else:
        z = balanced_rule(ctx.remaining, ctx.d, ctx.rng) @ rot.T
    err = ctx.query(z)-cv(z)
    log("hard residual", np.mean(err), np.std(err), len(err))
    return float(a+np.mean(err))


def hard(ctx, x, y):
    if ctx.remaining < 3900:
        return ctx.query(normal(ctx.remaining, ctx.d, ctx.rng)).mean()
    # More pilot points on concentrated Gaussians also look for small central
    # events. These points are for design only, never directly averaged.
    edge, z, w = radial_search(ctx, x, y)
    scale = max(1e-3, np.median(np.abs(y)))
    gap = np.array([abs(e[4]-e[3]) for e in edge])
    looks_jump = len(gap) > 0 and np.max(gap) > max(.0015*scale, 100*np.median(gap) if np.median(gap) < 1e-6 else .0)
    if looks_jump:
        try:
            ans = jump_branch(ctx, x, y, edge, z, w)
            if ans is not None and np.isfinite(ans):
                return ans
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            log("jump fit declined", repr(exc))
        return integrate_remainder(ctx, no_model, 0., x, y, antithetic=False)
    qs, m, mean = fit_kinks(ctx, edge, x, y)
    if not qs and ctx.remaining>10600:
        # Looking at two extra scales helps when a small closed component
        # contributed no successful observations to the first pilot.
        P=normal(768,ctx.d,ctx.rng)
        P[:256]*=.68
        P[512:]*=1.53
        pp=ctx.query(P)
        if np.max(pp)>5.4*scale:
            xx,yy=np.vstack([x,P]),np.r_[y,pp]
            edge0,z0,w0=radial_search(ctx,xx,yy)
            if edge0 and max(abs(e[3]-e[4]) for e in edge0)>1.1*scale:
                try:
                    ans=jump_branch(ctx,xx,yy,edge0,z0,w0)
                    if ans is not None and np.isfinite(ans):
                        return ans
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    log("rare component rescue declined",repr(exc))
    return integrate_remainder(ctx, m, mean, x, y, antithetic=True, continuous=bool(qs))
