"""Value-only recovery and shell sampling for closed jump surfaces."""
import math

import numpy as np
from scipy import linalg, optimize, special

from hard import hermite_ridge_control, no_model
from numerics import (Quad, balanced_rule, ellipsoid_prob, fd_gradient, log,
                      normal, orthogonal_to, rect_gaussian_regression, sobol)


def jump_amplitude(ctx, edges):
    p, v, h, lo, hi = map(np.asarray, zip(*edges))
    order = np.argsort(-np.abs(lo-hi))
    k = order[:min(len(order), 5)]
    p, v, h, lo, hi = p[k], v[k], h[k], lo[k], hi[k]
    vals=np.abs(hi-lo)
    if len(k)>=4 and np.max(np.abs(vals/np.median(vals)-1))<1.8e-6 and (
            np.max(np.linalg.norm(p-p[0],axis=1))>.45):
        return float(np.log(np.median(vals))), np.zeros(ctx.d)
    if ctx.d>=48 and len(k)>3:
        j=max(range(1,len(k)),key=lambda i:np.linalg.norm(p[i]-p[0]))
        sel=[0,next(i for i in range(1,len(k)) if i!=j),j]
        p,v,h,lo,hi=p[sel],v[sel],h[sel],lo[sel],hi[sel]
        k=k[sel]
    hh = np.maximum(2.6*h, 1.6e-5)
    xx = np.vstack([p-hh[:, None]*v, p+hh[:, None]*v])
    G = fd_gradient(ctx.query, xx, h=max(4e-7, min(2e-6, .1*np.min(h))))
    dm = hi-lo
    estimate = (G[len(k):] - G[:len(k)]) / dm[:, None]
    vhat = np.median(estimate, axis=0)
    if np.linalg.norm(vhat) < 3e-5:
        vhat *= 0.
    aa = np.log(np.abs(dm))-p@vhat
    if np.linalg.norm(vhat) > 2. or np.median(np.abs(aa-np.median(aa))) > .18:
        raise ValueError("not a consistent exponential jump")
    loga = float(np.median(aa))
    return loga, vhat


class Phase:
    def __init__(self, ctx, x, y, edge, z, w, c, tilt):
        self.c = c
        self.tilt = tilt
        self.constant = float(np.median(y))
        self.d = x.shape[1]
        self.low = float(min(y.min(), w.min()))
        nrm = np.linalg.norm(z, axis=1)
        q = (nrm < 1.65*math.sqrt(ctx.d))
        xx, yy = np.vstack([x, z[q]]), np.r_[y, w[q]]
        amp = self.amplitude(xx)
        edge_bg = np.array([min(e[3],e[4]) for e in edge])
        if np.max(amp) < 12*np.std(np.minimum(yy, np.median(yy)+.5)):
            qq = np.isfinite(edge_bg)
            a = np.array([e[0] for e in edge])[qq]
            bb = edge_bg[qq]
            xx = np.vstack([xx, np.repeat(a, 6, axis=0)])
            yy = np.r_[yy, np.repeat(bb, 6)]
            amp = self.amplitude(xx)
        base = float(np.median(edge_bg))
        self.bg = lambda x: base
        for _ in range(3):
            t = yy-self.bg(xx)
            take = t > .5*amp
            yy0 = yy-take*amp
            cut = np.quantile(np.abs(yy0-base), .89)+1e-6
            yy0 = np.clip(yy0, base-2.4*cut, base+2.4*cut)
            _, self.bg = rect_gaussian_regression(xx, yy0, 0, ridge=.055)
        bg = self.bg
        qq = yy-self.amplitude(xx)*(yy > bg(xx)+.5*self.amplitude(xx))
        q0, q1 = np.quantile(qq, [.015,.985])
        self.bg = lambda x: np.clip(bg(x), q0-.1*(q1-q0), q1+.18*(q1-q0))
        vv = yy-self.bg(xx)-self.label(xx, yy)*amp
        self.background_sd = np.std(vv)

    def amplitude(self, z):
        return np.exp(np.clip(self.c+z @ self.tilt, -80, 45))

    def label(self, z, y):
        m = self.bg(z)
        return y > m + .5*self.amplitude(z)

    def remainder(self, z, y):
        return y - self.label(z, y)*self.amplitude(z)


def choose_anchor(ctx, p, x, y, z, w, edges):
    d = ctx.d
    zero = np.zeros((1, d))
    if p.label(zero, ctx.query(zero))[0]:
        return zero[0]
    ws = np.r_[y, w]
    xs = np.vstack([x,z])
    pos = xs[p.label(xs, ws)]
    if len(pos) < 1:
        e = max(edges, key=lambda e: abs(e[4]-e[3]))
        return e[0]+2*e[2]*np.sign(e[4]-e[3])*e[1]
    o = np.argsort(np.linalg.norm(pos, axis=1))
    a = pos[o[:min(30,len(o))]].mean(axis=0)
    good = ctx.query(a)
    if p.label(a[None], good)[0]:
        return a
    return pos[o[0]]


def learn_quad(u, l, h, a, ridge, width=None):
    n, d = u.shape
    i,j = np.triu_indices(d,1)
    t = .5*(l+h)
    t = np.maximum(t, .018)
    norm2 = np.sum(u*u, axis=1)
    diag = u*u-norm2[:, None]/d
    X = np.column_stack([norm2/d, u/t[:, None], diag, u[:, i]*u[:, j]*np.sqrt(2.)])
    yy = 1/t**2
    if width is None:
        wt = np.minimum(1., (t/(np.median(t)*.75))**2)
    else:
        wt = np.minimum(1., (t/(np.median(t)*.75))**2) * width
    scale = max(1e-12,np.sqrt(np.mean(yy**2)))
    M = X.T @ (X*wt[:, None])
    g = X.T @ (yy*wt)
    rd = np.full(len(g), ridge*n)
    rd[0] = 1e-8*n
    rd[1:1+d] *= .3
    M.flat[::len(M)+1] += rd
    W = linalg.solve(M, g, assume_a="pos")
    Q = np.diag(W[0]/d + W[d+1:2*d+1] - W[d+1:2*d+1].mean())
    Q[i,j] = Q[j,i] = W[2*d+1:]/np.sqrt(2.)
    e, V = np.linalg.eigh(Q)
    largest = max(float(e[-1]), 1e-6*scale)
    if ridge >= .002 and np.min(e) < -0.05*largest:
        # Preserve the directional mean when projecting a noisy matrix to the PSD cone.
        floor = min(.18*largest, -np.quantile(e, .16)*.5)
    else:
        floor = 0.
    keep = e > max(floor, 2e-5 * largest)
    ep = np.where(keep, e, 0.)
    if not np.any(keep):
        raise ValueError("no convex interior")
    if min(e) < -.025*largest and sum(e)>0:
        ep *= sum(e)/sum(ep)
    Q = (V*ep)@V.T
    b = W[1:1+d]
    if np.linalg.norm(b - V[:, keep] @ (V[:, keep].T @ b)) > .03*max(1e-12,np.linalg.norm(b)):
        # A large center component in a nearly unobserved direction is poorly determined.
        b = V[:, keep] @ (V[:, keep].T @ b)
    else:
        b = V[:, keep] @ (V[:, keep].T @ b)
    q = Quad.at(Q, b, a, -1.)
    log("ellipse fit", d, n, len(g), "rank", keep.sum(), "eig", largest, min(e), "shift", np.linalg.norm(a))
    return q


def relative_root(q, a, u):
    aa = np.maximum(np.sum(u*(u@q.A), axis=1), 1e-18)
    bb = u@q.grad(a)
    c = min(-1e-12, q(a))
    disc = np.sqrt(bb**2-4*aa*c)
    return np.where(bb >= 0, (-2*c)/(disc+bb), (disc-bb)/(2*aa))


def train_geometry(ctx, phase, a, edges, x, y, z, w):
    d = ctx.d
    anchor_dist = np.array([np.linalg.norm(e[0]-a) for e in edges])
    r0 = float(np.clip(np.median(anchor_dist)/math.sqrt(d), .22, 3.))
    p = d*(d+3)//2
    if d <= 9:
        n, it = max(100,2*p+14), 25
    elif d <= 20:
        n, it = int(1.85*p)+26, 18
    elif d <= 40:
        n, it = int(1.37*p), 10
    else:
        n, it = int(1.0*p), 4
    n = min(n, (ctx.remaining-4550)//(it+1))
    if n < max(25, p*.65):
        n = int(max(25, min(n*1.2, (ctx.remaining-4000)//(it+1))))
    if n*it > ctx.remaining-2600:
        raise ValueError("too little budget for ray design")
    u = normal(n, d, ctx.rng)
    if np.linalg.norm(phase.tilt) > .02:
        vv = phase.tilt/np.linalg.norm(phase.tilt)
        tu = u@vv
        u -= (2*np.minimum(0.,tu+.40))[:,None]*vv
    u *= (np.sqrt(d)/np.linalg.norm(u, axis=1))[:, None]
    large = min(max(1.65*r0, 1.22), 4.)
    maxcoord = np.maximum(1e-3,np.max(np.abs(u), axis=1))
    hi = np.minimum(large, (38-np.max(np.abs(a)))/maxcoord)
    lo = np.zeros(len(u))
    zz = a+hi[:, None]*u
    yh = ctx.query(zz)
    ix = ~phase.label(zz, yh)
    if ix.mean() < .97 and ctx.remaining > 4250+it*len(u):
        missed = ~ix
        hi[missed] = np.minimum(1.80*hi[missed], (38-np.max(np.abs(a)))/maxcoord[missed])
        z2 = a+hi[missed,None]*u[missed]
        yh[missed] = ctx.query(z2)
        ix[missed] = ~phase.label(z2,yh[missed])
    if ix.sum() < max(24, .52*n):
        raise ValueError("too few rays leave the inferred component")
    hi, lo, u, yh = hi[ix], lo[ix], u[ix], yh[ix]
    amp_lo = np.repeat(phase.amplitude(a[None]),len(u))
    yl = np.repeat(ctx.query(a),len(u))
    for _ in range(it):
        t = .5*(hi+lo)
        zz = a+t[:, None]*u
        ym = ctx.query(zz)
        amp = phase.amplitude(zz)
        local = .5*(yl-amp_lo+yh)
        blend = np.clip((hi-lo)*1.1-.12,0,.50)
        bg = (1-blend)*local + blend*phase.bg(zz)
        ok = ym > bg+.5*amp
        lo = np.where(ok, t, lo)
        hi = np.where(ok, hi, t)
        amp_lo = np.where(ok,amp,amp_lo)
        yl = np.where(ok,ym,yl)
        yh = np.where(ok,yh,ym)
    discrepancy = np.abs(yl-yh-amp_lo)
    good_jump = discrepancy < .23*amp_lo + .005
    u,lo,hi = u[good_jump],lo[good_jump],hi[good_jump]
    # The small set located directly from jump differences also anchors the fit.
    other = np.array([e[0]-a for e in edges])
    size = np.linalg.norm(other, axis=1)/math.sqrt(d)
    take = (size > .05) & (size < large)
    uu = other[take]/size[take,None]
    tt = size[take]
    U = np.vstack([u, uu])
    L = np.r_[lo, tt]
    H = np.r_[hi, tt]
    ok = (L > 1e-4) & (H > .022)
    if np.count_nonzero(ok) < .55*p:
        raise ValueError("insufficient consistent ray crossings")
    ridge = .000000001 if d <= 20 else (.000002 if d <= 40 else .012)
    q = learn_quad(U[ok], L[ok], H[ok], a, ridge)
    radial_error = np.abs(relative_root(q, a, u)-.5*(lo+hi))
    log("ray radius residual", np.quantile(radial_error,[.50,.8,.95]), "steps", it)
    noise = max(np.quantile(radial_error,.88), np.median(hi-lo)*.5)*math.sqrt(d)
    return q, noise


def phase_bracket(ctx, phase, a, u, lo, hi, y0, rounds, extend=True):
    amp0 = phase.amplitude(a[None])[0]
    cap = (38-np.max(np.abs(a)))/np.maximum(1e-4,np.max(np.abs(u),axis=1))
    hi = np.minimum(hi,cap)
    zh = a+hi[:,None]*u
    yh = ctx.query(zh)
    outside = ~phase.label(zh,yh)
    if extend:
        for _ in range(2):
            ix = ~outside & (hi < cap*.98)
            if not ix.any():
                break
            hi[ix] = np.minimum(1.8*hi[ix],cap[ix])
            zh = a+hi[ix,None]*u[ix]
            yh[ix] = ctx.query(zh)
            outside[ix] = ~phase.label(zh,yh[ix])
    if np.all(lo==0):
        yl = np.full(len(u),y0)
        amp_l = np.full(len(u),amp0)
        inside = np.ones(len(u),dtype=bool)
    else:
        zl = a+lo[:,None]*u
        amp_l = phase.amplitude(zl)
        yl = ctx.query(zl)
        inside = phase.label(zl,yl)
    for _ in range(rounds):
        t = .5*(lo+hi)
        zz = a+t[:,None]*u
        ym = ctx.query(zz)
        amp = phase.amplitude(zz)
        blend = np.clip((hi-lo)*.8-.05,0,.55)
        bg = (1-blend)*.5*(yl-amp_l+yh)+blend*phase.bg(zz)
        ok = ym > bg+.5*amp
        lo = np.where(ok,t,lo)
        hi = np.where(ok,hi,t)
        yl = np.where(ok,ym,yl)
        yh = np.where(ok,yh,ym)
        amp_l = np.where(ok,amp,amp_l)
    good = outside & inside & (np.abs(yl-yh-amp_l) < .25*amp_l+.005)
    return .5*(lo+hi), good


def scan_bracket(ctx, phase, a, u, r0, rounds):
    """Detect a local drop of the discovered size; distant labels are unnecessary."""
    d = ctx.d
    n = len(u)
    cap = (38-np.max(np.abs(a)))/np.maximum(1e-4,np.max(np.abs(u),axis=1))
    hi = np.minimum(cap, 2.18*r0)
    t = hi[:,None]*np.linspace(0.,1.,17)
    z = a+(t[:,:,None]*u[:,None]).reshape(-1,d)
    vals = ctx.query(z).reshape(n,-1)
    aa = phase.amplitude(z).reshape(n,-1)
    err = abs(vals[:,:-1]-vals[:,1:]-aa[:,:-1])/np.maximum(1e-10,aa[:,:-1])
    j = np.argmin(err,axis=1)
    row = np.arange(n)
    L,H=t[row,j],t[row,j+1]
    yL,yH=vals[row,j],vals[row,j+1]
    aL=aa[row,j]
    for _ in range(rounds):
        t=.5*(L+H)
        zz=a+t[:,None]*u
        ym=ctx.query(zz)
        am=phase.amplitude(zz)
        inside=ym>.5*(yL-aL+yH+am)
        L=np.where(inside,t,L)
        H=np.where(inside,H,t)
        yL=np.where(inside,ym,yL)
        yH=np.where(inside,yH,ym)
        aL=np.where(inside,am,aL)
    good=np.abs(yL-yH-aL) < .085*aL + .0004
    return .5*(L+H),good


def pair_geometry(ctx, phase, a, edges, x, y, z, w):
    """One-dimensional binary information for the diagonal and cross entries."""
    d = ctx.d
    y0 = ctx.query(a)[0]
    ell = np.array([np.linalg.norm(e[0]-a) for e in edges])
    r0 = np.clip(np.median(ell)/np.sqrt(d), .24, 2.7)
    amp = phase.amplitude(a[None])[0]
    if np.linalg.norm(phase.tilt) > .004:
        R0 = orthogonal_to(phase.tilt/np.linalg.norm(phase.tilt))
        V,_ = np.linalg.qr(ctx.rng.standard_normal((d-1,d-1)))
        eq = orthogonal_to(np.full(d,1/np.sqrt(d)))
        R = np.column_stack([R0[:,0],R0[:,1:]@V]) @ eq.T
    else:
        R,_ = np.linalg.qr(ctx.rng.standard_normal((d,d)))
    # A rotation also avoids large individual query coordinates on long rays.
    dirs = np.sqrt(d)*R.T
    U = np.vstack([dirs,-dirs])
    if np.linalg.norm(phase.tilt)>.025:
        rp,op=phase_bracket(ctx,phase,a,dirs,np.zeros(d),np.full(d,r0*1.57),
                           y0,16 if d<=40 else 13)
        rm,om=scan_bracket(ctx,phase,a,-dirs,r0,14 if d<=40 else 12)
        bad=np.flatnonzero(~op)
        if len(bad):
            rp[bad],op[bad]=scan_bracket(ctx,phase,a,dirs[bad],r0,13)
        roots,ok=np.r_[rp,rm],np.r_[op,om]
    else:
        roots, ok = phase_bracket(ctx,phase,a,U,np.zeros(len(U)),np.full(len(U),r0*1.85),
                                  y0, 16 if d<=40 else 13)
    rp,rm=np.split(roots,2)
    valid=ok[:d]&ok[d:]
    log("axis probe validity",np.count_nonzero(ok[:d]),np.count_nonzero(ok[d:]),"of",d)
    if valid.sum()<d-(1 if np.linalg.norm(phase.tilt)>.05 else 0):
        raise ValueError("pair design axis was not bracketed")
    ai=1/(rp*rm)
    bi=1/rp-1/rm
    if not valid.all():
        # The linear term of the poorly conditioned axis is constrained by
        # the closest available detected crossings.
        take=np.flatnonzero(~valid)
        ai[take]=np.median(ai[valid])
        bi[take]=1/rp[take]-ai[take]*rp[take]
    if (np.max(ai) > 25*max(1e-6,np.median(ai)) or np.min(ai)<.006*np.median(ai)):
        raise ValueError("axis design is poorly conditioned")
    if d>=24 and np.std(ai)>.53*np.mean(ai):
        raise ValueError("axis variation suggests low-dimensional shape")
    i,j=np.triu_indices(d,1)
    k=len(i)
    bd = (bi[i]+bi[j])/np.sqrt(2)
    ad = .5*(ai[i]+ai[j])
    uv = (dirs[i]+dirs[j])/np.sqrt(2)
    bound = np.sqrt(ai[i]*ai[j])
    md = float(np.median(ai))
    sd = max(.055*md, min(.84*md,np.std(ai)/np.sqrt(2)))
    p0 = special.ndtr(-bound/sd)
    p1 = special.ndtr(bound/sd)
    steps = min(14 if d < 40 else 7, (ctx.remaining-3500)//max(1,k))
    if steps < 3:
        raise ValueError("pair design is out of budget")
    maxr = (38-np.max(np.abs(a)))/np.maximum(1e-4,np.max(np.abs(uv),axis=1))
    def advance(idx):
        pm = .5*(p0[idx]+p1[idx])
        av = np.maximum(ad[idx] + sd*special.ndtri(pm),1e-10)
        cb=bd[idx]
        disc = np.sqrt(cb*cb+4*av)
        r = np.where(cb>=0, 2/(disc+cb), (disc-cb)/(2*av))
        xx = a+np.minimum(r,maxr[idx])[:,None]*uv[idx]
        val=ctx.query(xx)
        inside=phase.label(xx,val)
        p1[idx]=np.where(inside,pm,p1[idx])
        p0[idx]=np.where(inside,p0[idx],pm)
    for it in range(steps):
        advance(slice(None))
    if d>=48 and steps<8:
        for _ in range(2):
            nref=min(k//2,max(0,ctx.remaining-4750))
            if nref<112:
                break
            u0=special.ndtri(np.clip(.17*p1+.83*p0,1e-15,1-1e-15))
            u1=special.ndtri(np.clip(.83*p1+.17*p0,1e-15,1-1e-15))
            sel=np.argpartition(u1-u0,len(i)-nref)[-nref:]
            advance(sel)
    vals=sd*special.ndtri(.5*(p0+p1))
    # Use the posterior mean in the extreme one-sided cells.
    ind=(p0 < .025)|(p1 > .975)
    ga = special.ndtri(np.maximum(p0[ind],1e-15))
    gb = special.ndtri(np.minimum(p1[ind],1-1e-15))
    vals[ind]= sd*(np.exp(-.5*ga**2)-np.exp(-.5*gb**2)) / (
        np.sqrt(2*np.pi)*np.maximum(1e-15,(p1-p0)[ind]))
    Q=np.diag(ai)
    Q[i,j]=Q[j,i]=vals
    # If a strongly tilted ray is ambiguous on the negative side of the
    # anchor, recover its missing second equation from that row's known hits.
    for ii in np.flatnonzero(~valid):
        jj=np.flatnonzero(valid)
        sel=np.abs(Q[ii,jj]) > max(1e-7,.015*md)
        if sel.sum()>3:
            j2=jj[sel]
            u2=(dirs[ii]-dirs[j2])/np.sqrt(2)
            rr,oo=phase_bracket(ctx,phase,a,u2,np.zeros(len(j2)),np.full(len(j2),r0*1.7),
                                y0, min(10,ctx.remaining//max(1,5*len(j2))))
            if oo.sum()>2:
                # Avoid an aggressive extrapolation from a quantized row.
                test=(1/rr**2-.5*(ai[ii]+ai[j2])+Q[ii,j2]) * (np.sqrt(2)*rr)+bi[j2]
                if np.linalg.norm(test[oo]-bi[ii]) < .30*md*np.sqrt(oo.sum()):
                    bi[ii]=np.median(test[oo])
    Q = Q/d
    b=bi/np.sqrt(d)
    e,V=np.linalg.eigh(Q)
    ep=np.where(e > max(1.e-6, .01*e[-1] if e[0]<-.025*e[-1] else .0001*e[-1]),e,0.)
    if not np.any(ep>0):
        raise ValueError("pair design did not give an ellipsoid")
    ep *= np.sum(e)/np.sum(ep)
    Q = R@(V*ep)@V.T@R.T
    vv = R@V[:,ep>0]
    b = vv@(vv.T@(R@b))
    q=Quad.at(Q,b,a,-1.)
    q.pair_design=True
    p=np.array([e0[0]-a for e0 in edges])
    r=np.linalg.norm(p,axis=1)
    rr=relative_root(q,a,p/np.maximum(r,1e-20)[:,None])
    err=np.abs(rr-r)
    noise=max(3e-6,np.quantile(err,.65), .00008 if d<40 else .003)
    log("pair ellipse",steps,"rank",np.count_nonzero(ep),"mean diag",md,"off scale",sd,
        "root noise",noise,np.quantile(err,[.1,.5,.9]),"axis confidence",valid.mean())
    return q,noise


def measured_normal(ctx, phase, edge, steps):
    p,v,h,yl,yr=edge
    outward=v*np.sign(yl-yr)
    d=ctx.d
    # Only a small neighborhood of an observed jump is needed for these tests.
    # The two farthest points used below are along the short local bracket.
    dh=.045
    E=dh*np.eye(d)
    X=p+np.vstack([E,-E])
    ya=min(yl,yr)
    width=.22
    if abs(yl-yr)<.12:
        width=.11
        X=p+(X-p)*.60
        dh*=.60
    def inside(Z):
        q=ctx.query(Z)
        return q>ya+.5*phase.amplitude(Z)
    if abs(yl-yr) > 8:
        width=.29
    bracket=inside(np.vstack([X-width*outward,X+width*outward])).reshape(2,-1)
    ok=bracket[0]&~bracket[1]
    if ok.mean()<.96 and ok.any() and ctx.remaining>2*d*(steps+1)+2200:
        miss=np.flatnonzero(~ok)
        B=inside(np.vstack([X[miss]-2.0*width*outward,X[miss]+2.0*width*outward])).reshape(2,-1)
        ok[miss]=B[0]&~B[1]
    else:
        miss=np.empty(0,dtype=int)
    if ok.mean()<.89 or (not ok.all() and np.count_nonzero(~ok)>3):
        return None
    L=np.full(len(X),-width)
    H=np.full(len(X),width)
    L[miss]*=2.0
    H[miss]*=2.0
    for _ in range(steps):
        t=.5*(L+H)
        inside0=inside(X+t[:,None]*outward)
        L=np.where(inside0,t,L)
        H=np.where(inside0,H,t)
    t=.5*(L+H)
    G=(t[d:]-t[:d])/(2*dh)
    if not np.isfinite(G).all() or np.linalg.norm(G)<.47:
        return None
    # The directional derivative along the probing direction is positive.
    return G/np.linalg.norm(G)


def normal_geometry(ctx, phase, a, edges):
    d=ctx.d
    budget=ctx.remaining
    rounds=11 if d>=48 else 12
    max_normals=min(6,max(0,(budget-4400)//((rounds+2)*2*d)))
    if max_normals<3:
        raise ValueError("too little budget for normal sketch")
    good_edges=[e for e in edges if abs(e[3]-e[4])>.075*phase.amplitude(a[None])[0]]
    if phase.amplitude(a[None])[0] > 7 and ctx.remaining>6700:
        nsc=12
        ns=normal(nsc,d,ctx.rng)
        ns*=np.sqrt(d)/np.linalg.norm(ns,axis=1)[:,None]
        dist=np.array([np.linalg.norm(e[0]-a) for e in good_edges])
        r0=np.clip(np.quantile(dist,.57)/np.sqrt(d),.08,3.5)
        t,ok=scan_bracket(ctx,phase,a,ns,r0,16)
        p=a+t[ok,None]*ns[ok]
        v=ns[ok]/np.sqrt(d)
        h=max(3e-5,3e-5*r0*np.sqrt(d))
        yl,yr=ctx.query(np.vstack([p-h*v,p+h*v])).reshape(2,-1)
        cand=[(p0,v0,h,l,r) for p0,v0,l,r in zip(p,v,yl,yr) if l-r>.5*phase.amplitude(a[None])[0]]
        good_edges=cand+good_edges
    # Spread the probes across different line directions and signs.
    good_edges.sort(key=lambda e:np.linalg.norm(e[0]-a)*(
        1+.055*np.linalg.norm(e[0])))
    used=[]
    ev=[]
    batch=[]
    for _ in range(min(2*max_normals+2,len(good_edges))):
        if len(ev)>=max_normals or ctx.remaining<2*d*(rounds+3)+3350:
            break
        rem=[i for i in range(len(good_edges)) if i not in used]
        if not rem:
            break
        if batch:
            P=np.asarray([e[0]-a for e in batch])
            P=P/np.maximum(1e-14,np.linalg.norm(P,axis=1))[:,None]
            def spread(i):
                e=good_edges[i]
                D=e[0]-a
                return np.max(np.abs(P@(D/max(1e-14,np.linalg.norm(D)))) ) + .018*np.linalg.norm(D)
            j=min(rem,key=spread)
        else:
            j=rem[0]
        used.append(j)
        e=good_edges[j]
        g=measured_normal(ctx,phase,e,rounds)
        if g is None:
            continue
        ev.append(g)
        batch.append(e)
        if len(ev)>=4:
            _, sv, V = np.linalg.svd(np.array(ev),full_matrices=False)
            keep=sv>max(.022,.026*sv[0])
            if not keep[-1]:
                break
    if len(ev)<2:
        raise ValueError("insufficient boundary normal directions")
    _, sv, V = np.linalg.svd(np.array(ev),full_matrices=False)
    keep=sv>max(.021,.028*sv[0])
    V=V[keep].T
    k=V.shape[1]
    if k<2 or ctx.remaining<3500:
        raise ValueError("normal sketch did not reveal a useful projection")
    xc=V@(V.T@a)+phase.tilt-V@(V.T@phase.tilt)
    if phase.label(xc[None],ctx.query(xc))[0]:
        a=xc
    last_points=np.array([e[0] for e in edges])
    rad=np.linalg.norm((last_points-a)@V,axis=1)
    r0=float(np.clip(np.quantile(rad,.55)/np.sqrt(k),.045,6.))
    n=max(42,2*k*(k+3)//2+24)
    if 33*n>ctx.remaining-1950:
        raise ValueError("too few queries left for reduced geometry")
    U=normal(n,k,ctx.rng)
    U*=np.sqrt(k)/np.linalg.norm(U,axis=1)[:,None]
    t,valid=scan_bracket(ctx,phase,a,U@V.T,r0,16)
    if valid.sum()<max(16,k*(k+3)//2+5):
        raise ValueError("projection did not bracket enough points")
    q=learn_quad(U[valid],t[valid],t[valid],V.T@a,1e-9)
    q=Quad(V@q.A@V.T,V@q.b,q.c)
    level=np.abs(q(last_points))
    if np.quantile(level,.7)>.048 or np.quantile(level,.9)>.14:
        raise ValueError("projection disagrees with independent crossings")
    g=np.maximum(1e-10,np.linalg.norm(q.grad(last_points),axis=1))
    eq=np.abs(q(last_points))/g
    noise=float(np.clip(np.quantile(eq,.77),.00022,1.1))
    log("shape within normal span",k,"singular values",sv,"normalized deviations",
        np.quantile(level,[.3,.6,.9]),"est. radius noise",noise)
    return q,noise


def probability_curvature_basis(q, a, tilt, e, V):
    if np.linalg.norm(tilt)<.06 or np.sum(e>.015*e[-1])<22:
        return None
    m=V.T@(tilt-a)
    l=V.T@q.grad(tilt)
    s=max(1e-8,np.sqrt(2*(e@e)+l@l))
    g,w=special.roots_legendre(360)
    t=(g+1)*27/s
    w*=27/s
    A=1.-2j*t[:,None]*e[None]
    v=l[None]/A
    exponent=1j*t*q(tilt)-.5*np.log(A).sum(axis=1)-.5*t*t*(v@l)
    cf=np.exp(exponent)
    if abs(cf[-1])>.000004:
        return None
    p=m[None]+1j*t[:,None]*v
    wc=w*cf
    G=(p.T@(p*wc[:,None])+np.diag(np.sum(wc[:,None]/A,axis=0))).real
    w2,R=np.linalg.eigh(.5*(G+G.T))
    if w2[-1]<=0 or w2[0]<-.013*w2[-1] or np.std(w2)<.02*np.mean(w2):
        return None
    return V@R


def calibrate_spectrum(ctx, phase, q, a, edges, noise):
    d=ctx.d
    e,V=np.linalg.eigh(q.A)
    off=None
    Qr=probability_curvature_basis(q,a,phase.tilt,e,V)
    if Qr is not None:
        base=Qr.T@q.A@Qr
        diag=np.diag(base)
        cross=base-np.diag(diag)
        if np.linalg.norm(cross) <= .27*np.linalg.norm(base):
            V=Qr
            e=np.maximum(0.,diag)
            off=cross
    keep=(e>max(1e-10,.14*np.max(e)))
    if np.count_nonzero(keep)<min(24,int(.50*d)):
        return q,noise
    J=np.flatnonzero(keep)
    U=np.sqrt(d)*V[:,J].T
    n=len(U)
    Vd=np.vstack([U,-U])
    guess=relative_root(q,a,Vd)
    span=np.clip((.19+.23*min(noise,.9))*guess,.023,.8)
    cap=(38-np.max(np.abs(a)))/np.maximum(1e-4,np.max(np.abs(Vd),axis=1))
    L=np.maximum(.03,guess-span)
    H=np.minimum(guess+span,cap)
    H=np.maximum(H,L)
    def paired(L,H,dirs):
        xL=a+L[:,None]*dirs
        xH=a+H[:,None]*dirs
        yl,yh=ctx.query(np.vstack([xL,xH])).reshape(2,-1)
        al=phase.amplitude(xL)
        ok=(np.abs(yl-yh-al)<.25*al+.007)&(H>L)
        return yl,yh,al,ok
    yl,yh,al,ok=paired(L,H,Vd)
    if ok.sum()<.72*len(ok):
        return q,noise
    bad=np.flatnonzero(~ok)
    if len(bad) and ctx.remaining>3200:
        L[bad]=np.maximum(0.,guess[bad]-2.7*span[bad])
        H[bad]=np.maximum(L[bad],np.minimum(cap[bad],guess[bad]+2.7*span[bad]))
        yl[bad],yh[bad],al[bad],ok[bad]=paired(L[bad],H[bad],Vd[bad])
    ids=np.flatnonzero(ok)
    it=min(11,max(0,(ctx.remaining-2650)//max(1,len(ids))))
    if it<6:
        return q,noise
    l,h=L[ids],H[ids]
    fl,fh,amp=yl[ids],yh[ids],al[ids]
    for _ in range(it):
        t=.5*(l+h)
        z=a+t[:,None]*Vd[ids]
        fv=ctx.query(z)
        at=phase.amplitude(z)
        b=fv>.5*(fl-amp+fh+at)
        l=np.where(b,t,l)
        h=np.where(b,h,t)
        fl=np.where(b,fv,fl)
        fh=np.where(b,fh,fv)
        amp=np.where(b,at,amp)
    rr=guess.copy()
    rr[ids]=.5*(l+h)
    ir=1/np.maximum(rr,1e-6)
    complete=ok[:n]&ok[n:]
    if complete.mean()<.70:
        return q,noise
    lab=J[complete]
    e_new=e.copy()
    v1=ir[:n][complete]
    v2=ir[n:][complete]
    e_new[lab]=np.clip(v1*v2/d,.50*e[lab],2.1*e[lab])
    e_new*=np.sum(e)/np.sum(e_new)
    b=(V.T@q.grad(a))
    b1=(v1-v2)/np.sqrt(d)
    b[lab] += np.clip(b1-b[lab],-.075,.075)*.86
    pp=np.asarray([ed[0] for ed in edges])
    nr=np.linalg.norm(pp-a,axis=1)
    uz=(pp-a)/np.maximum(nr,1e-12)[:,None]
    r_old=np.abs(relative_root(q,a,uz)-nr)
    if off is None:
        new=(V*e_new)@V.T
    else:
        t,U0=np.linalg.eigh(off+np.diag(e_new))
        t=np.maximum(0.,t)
        t *= np.sum(e)/max(1e-15,np.sum(t))
        V0=V@U0
        new=(V0*t)@V0.T
    q0=Quad.at(new,V@b,a,-1.)
    r_new=np.abs(relative_root(q0,a,uz)-nr)
    cv=np.quantile(r_new,.72)
    cv0=np.quantile(r_old,.72)
    log("spectrum check",complete.mean(),it,"radius old/new",cv0,cv,
        "eig change",np.linalg.norm(e-e_new),"sensitivity basis",off is not None)
    if cv>cv0*1.12+.018:
        return q,noise
    return q0,max(.00012,cv)


def tilted_proposal(q, tilt):
    e, V = np.linalg.eigh(q.A)
    e = np.maximum(e,0.)
    g = V.T@q.grad(tilt)
    c0 = q(tilt)

    def make(t):
        s = 1/(1+2*t*e)
        z = -t*g*s
        err = (e*s).sum() + c0 + z@g + (e*z*z).sum()
        return err, tilt+V@z, s

    t = 0.
    if make(0.)[0] > 0.:
        u = 1.
        while make(u)[0] > 0. and u < 2.**18:
            u *= 2.
        if make(u)[0] < 0.:
            t = optimize.brentq(lambda t: make(t)[0],0.,u)
    _, center, s = make(t)
    # Place the most distorted coordinate at the head of the digital net.
    order = np.argsort(-np.abs(g)*.15 - e)
    return center, V[:,order]*np.sqrt(s[order]), .5*np.log(s).sum()


def angular_geometry(q, S):
    # Split a cylinder into its active radial coordinates and its exactly
    # Gaussian inactive coordinates. Keep the ordinary sphere for full rank.
    lens=np.linalg.norm(q.A@S,axis=0)
    good=lens>max(1e-11,2e-6*np.max(lens))
    if 1<good.sum() and good.sum() <= .72*len(lens):
        return S[:,good],S[:,~good]
    return S,S[:,:0]


def shell_calibration(ctx, phase, q, center, S, a, noise):
    d = len(a)
    if q(center) >= -1e-5:
        center = a
    n = min(64, max(20, ctx.remaining//140))
    sa,si=angular_geometry(q,S)
    ds=sa.shape[1]
    qn = normal(n,d,ctx.rng)
    vv=qn[:,:ds]
    vv/=np.linalg.norm(vv,axis=1)[:,None]
    v=vv@sa.T
    ca=center+qn[:,ds:]@si.T
    r = relative_root(q, center, v)
    f = np.minimum(r*.65, min(max(.042, 9*noise),max(.46, .10*np.sqrt(d))))
    rr = np.stack([np.maximum(0.,r-f), r+f])
    xx = (ca[None] + rr[:, :, None]*v[None]).reshape(-1,d)
    values = ctx.query(xx)
    inside, outside = phase.label(xx, values).reshape(2,-1)
    ok = inside & ~outside
    if not np.any(ok):
        return max(.75,.2*np.sqrt(d)), center, 1.
    rr, v, r, ca = rr[:,ok],v[ok],r[ok],ca[ok]
    for _ in range(9 if d <= 20 else 8):
        new = .5*(rr[0]+rr[1])
        x = ca + new[:,None]*v
        inside = phase.label(x,ctx.query(x))
        rr[0] = np.where(inside,new,rr[0])
        rr[1] = np.where(inside,rr[1],new)
    err = np.abs(r-.5*(rr[0]+rr[1]))
    fail = 1-ok.mean()
    width = max(2e-5, np.quantile(err,.97)*3 + 1.5*np.median(rr[1]-rr[0]))
    if fail > .07:
        width = max(width, min(1.7, .40+2*fail))
    log("shell width",width,"fail",fail,"radius error",np.quantile(err,[.50,.9]))
    return width, center, fail


def shell_correction(ctx, phase, q, n, center, S, logdet, width):
    d = ctx.d
    u = sobol(n, d+2, ctx.rng)
    gauss = special.ndtri(np.clip(u[:,:d],1e-14,1.-1e-14))
    sa,si=angular_geometry(q,S)
    ds=sa.shape[1]
    s=gauss[:,:ds]
    v = (s/np.linalg.norm(s,axis=1)[:,None])@sa.T
    tr=gauss[:,ds:]
    edge = relative_root(q,center,v)
    window = np.minimum(width, 3.+.08*edge)
    low = np.maximum(0.,edge-window)
    high = edge+window
    Fa = special.gammainc(ds/2, low*low*.5)
    Fb = special.gammainc(ds/2, high*high*.5)
    spread = np.maximum(0., Fb-Fa)
    shell = (u[:,d] < .94) & (spread > 1e-14)
    quantile = np.where(shell, Fa+spread*u[:,d+1], u[:,d+1])
    r = np.sqrt(2*special.gammaincinv(ds/2,np.minimum(1.-1e-15,quantile)))
    x = center+r[:,None]*v+tr@si.T
    vals = ctx.query(x)
    delta = phase.label(x,vals).astype(float) - (q(x)<0)
    p = np.where(spread > 1e-14, .06 + .94*((r>=low)&(r<=high))/np.maximum(1e-40,spread), 1.)
    zz = x-phase.tilt
    wp = np.exp(np.clip(logdet + .5*(r*r+np.sum(tr*tr,axis=1)-np.sum(zz*zz,axis=1)), -650, 600))/p
    vals = wp*delta
    log("shell update", vals.mean(), vals.std(), n, "mismatches",np.count_nonzero(delta))
    return vals.mean()


def bracketed_shell(ctx, phase, q, center, S, logdet, width):
    """Conditional weighted Gaussian mass of the discrepancy on each ray."""
    d=ctx.d
    sa,si=angular_geometry(q,S)
    ds=sa.shape[1]
    count=ctx.remaining//5
    if count<192:
        return shell_correction(ctx,phase,q,ctx.remaining,center,S,logdet,width)
    Z=balanced_rule(count,d+2,ctx.rng)
    n=len(Z)
    ga=Z[:,:ds]
    v=(ga/np.linalg.norm(ga,axis=1)[:,None])@sa.T
    zp=Z[:,ds:d]
    w=center+zp@si.T
    r0=relative_root(q,center,v)
    c0=logdet+.5*(np.sum(zp*zp,axis=1)-np.sum((w-phase.tilt)**2,axis=1))
    c1=-np.sum(v*(w-phase.tilt),axis=1)
    c2=.5*(1-np.sum(v*v,axis=1))
    def read(t,subset=None):
        if subset is None:
            x=w+t[:,None]*v
        else:
            x=w[subset]+t[:,None]*v[subset]
        return phase.label(x,ctx.query(x))
    c=read(r0)
    sign=np.where(c,1.,-1.)
    exp=np.clip(width*1.14,.00015,1.60)
    edge=np.maximum(0.,r0+exp*sign)
    other=read(edge)
    good=c!=other
    unresolved=np.flatnonzero(~good)
    if len(unresolved) and ctx.remaining>=3*n + len(unresolved):
        idx=unresolved
        edge[idx]=np.maximum(0.,r0[idx]+min(3.8,4*exp+.06)*sign[idx])
        other[idx]=read(edge[idx],idx)
        good=c!=other
    l=np.minimum(r0,edge)
    h=np.maximum(r0,edge)
    for _ in range(min(2,max(0,ctx.remaining//max(1,n)-1))):
        tt=.5*(l+h)
        mid=read(tt)
        l=np.where(mid,tt,l)
        h=np.where(mid,h,tt)
    F0=special.gammainc(ds/2,r0*r0*.5)
    Flo=special.gammainc(ds/2,l*l*.5)
    Fhi=special.gammainc(ds/2,h*h*.5)
    uu=special.ndtr(Z[:,d+1])
    def rad(u):
        return np.sqrt(2*special.gammaincinv(ds/2,np.clip(u,1e-15,1-1e-15)))
    a=np.maximum(0.,r0-exp)
    b=r0+exp
    Fa=special.gammainc(ds/2,a*a*.5)
    Fb=special.gammainc(ds/2,b*b*.5)
    dith=(special.ndtr(Z[:,d])<.91)&(Fb-Fa>1e-14)
    t=np.where(good,rad(Flo+uu*(Fhi-Flo)),rad(np.where(dith,Fa+uu*(Fb-Fa),uu)))
    g=read(t)
    wt=np.exp(np.clip(c0+c1*t+c2*t*t,-600,550))
    den=np.where(Fb-Fa>1e-14,.09+.91*((t>=a)&(t<=b))/np.maximum(1e-30,Fb-Fa),1.)
    miss_delta=(g.astype(float)-(t<r0))*wt/den
    # Integrate the determined interval in the original Gaussian law.  The
    # interpolation is in chi probability, not in length along the ray.
    u,weights=special.roots_legendre(10)
    U=F0[:,None]+((Flo-F0)/2)[:,None]*(u[None,:]+1)
    r=rad(U)
    T=np.exp(np.clip(c0[:,None]+c1[:,None]*r+c2[:,None]*r*r,-600,550))
    known=((Flo-F0)/2)*(T@weights)
    diff=np.where(good,known+wt*(Fhi-Flo)*g,miss_delta)
    log("bracketed shell",n,"confirmed",np.mean(good),"correction",diff.mean(),"sd",diff.std())
    return float(diff.mean())


class Background:
    def __init__(self, ctx, phase):
        self.ctx = ctx
        self.phase = phase

    def __getattr__(self, key):
        return getattr(self.ctx,key)

    def query(self, z):
        return self.phase.remainder(z,self.ctx.query(z))


def radial_weight_integral(Fa, Fb, k, bb, shift):
    """Signed radial integral of exp(b*r+shift) d P(chi_k <= r)."""
    if np.max(np.abs(bb)) < 1e-12:
        return (Fb-Fa)*np.exp(shift)
    t,w = special.roots_legendre(12)
    u = Fa[:,None] + ((Fb-Fa)*.5)[:,None]*(t[None]+1)
    r = np.sqrt(2*special.gammaincinv(k/2,np.clip(u,1e-15,1-1e-15)))
    P = np.exp(np.clip(bb[:,None]*r+shift,-80,40))
    return ((Fb-Fa)*.5)*(P@w)


def conditional_jump(ctx, phase, q, a, noise, model_mass, av, x, y):
    """Bracket a transition using nearby values rather than global phase labels."""
    from hard import integrate_remainder
    d = ctx.d
    amp = phase.amplitude
    f0 = ctx.query(phase.tilt)[0]
    center = phase.tilt if q(phase.tilt) < -.04 and phase.label(phase.tilt[None],np.array([f0]))[0] else a
    if np.linalg.norm(center) > 2.6:
        raise ValueError("shift too large for the ordinary-jump conditional rule")
    A_e, A_V = np.linalg.eigh(q.A@q.A + .04*np.outer(q.b,q.b) + .14*np.outer(phase.tilt,phase.tilt))
    rot0 = A_V[:,::-1].copy()
    phase_fit=lambda z: amp(z)*(q(z)<0)
    acv, cv, Rb = hermite_ridge_control(ctx,phase_fit,0.,x,y,
                                       spend_hessian=ctx.remaining>5250+d*(d+1)//2)
    def bcv(z):
        return cv(z)-phase_fit(z)
    # A small independent design check calibrates the initial radial interval.
    n0 = min(72,max(24,ctx.remaining//130))
    zz = normal(n0,d,ctx.rng)@rot0.T
    u0 = zz/np.linalg.norm(zz,axis=1)[:,None]
    r0 = relative_root(q,center,u0)
    width = min(1.9,max(.045,noise*4.4))
    ybg = ctx.query(center)[0]-amp(center[None])[0]

    def paired(U, r, width):
        l = np.maximum(0.,r-width)
        h = np.minimum(r+width,np.maximum(8.,r))
        yL, yH = ctx.query(np.vstack([center+l[:,None]*U, center+h[:,None]*U])).reshape(2,-1)
        aL = amp(center+l[:,None]*U)
        aH = amp(center+h[:,None]*U)
        # In a small interval the expected jump dominates the background drift.
        ok = (np.abs(yL-yH-aL) < .29*aL+.006)
        return l,h,yL,yH,aL,aH,ok

    L,H,yL,yH,aL,aH,g0=paired(u0,r0,width)
    if g0.mean()<.97:
        coarse=min(2.75,2.4*width+.08)
        subset=~g0
        g1=paired(u0[subset],r0[subset],coarse)[-1]
        if np.mean(g0)+np.count_nonzero(g1)/n0 < .93:
            return integrate_remainder(ctx,phase_fit,av*model_mass,x,y,
                                       antithetic=True,spend_hessian=False)
        if g0.mean()<.88:
            width=max(width,min(coarse,.2+width*1.6))
    it=3 if width>.095 else 2
    n=int(ctx.remaining/(5.30+it))
    if n<200:
        return integrate_remainder(ctx,phase_fit,av*model_mass,x,y,
                                   antithetic=True,spend_hessian=False)
    X = balanced_rule(n,d+2,ctx.rng)
    n=len(X)
    U=X[:,:d]@rot0.T
    U/=np.linalg.norm(U,axis=1)[:,None]
    r0=relative_root(q,center,U)
    L,H,yL,yH,aL,aH,good=paired(U,r0,width)
    for ep in range(2):
        ids=np.flatnonzero(~good)
        lim=max(0,(ctx.remaining-(2+it)*n-25)//2)
        if len(ids)>lim:
            ids=ids[:lim]
        if not len(ids):
            break
        size=min(5.2, (width+.10)*(3.1 if ep==0 else 7.8))
        L[ids],H[ids],yL[ids],yH[ids],aL[ids],aH[ids],good[ids]=paired(U[ids],r0[ids],size)
    if np.mean(good)<.942:
        return integrate_remainder(ctx,phase_fit,av*model_mass,x,y,
                                   antithetic=False,spend_hessian=False)
    for _ in range(it):
        t=.5*(L+H)
        uu=center+t[:,None]*U
        yy=ctx.query(uu)
        a=amp(uu)
        inside=yy>.5*(yL-aL+yH+a)
        L=np.where(inside,t,L)
        H=np.where(inside,H,t)
        yL=np.where(inside,yy,yL)
        yH=np.where(inside,yH,yy)
        aL=np.where(inside,a,aL)
    F0=special.gammainc(d/2,r0*r0*.5)
    FL=special.gammainc(d/2,L*L*.5)
    FH=special.gammainc(d/2,H*H*.5)
    ps=special.ndtr(X[:,d])
    r=np.sqrt(2*special.gammaincinv(d/2,np.clip(FL+(FH-FL)*ps,1e-15,1-1e-15)))
    zi=center+r[:,None]*U
    yi=ctx.query(zi)
    tfrac=np.clip((r-L)/np.maximum(1e-12,H-L),0,1)
    yi0=(1-tfrac)*(yL-aL)+tfrac*yH
    si=yi>yi0+.5*amp(zi)
    v=phase.tilt-center
    b=U@v
    shift=-.5*(v@v)
    diff=radial_weight_integral(F0,FL,d,b,shift)+np.exp(b*r+shift)*si*(FH-FL)
    # Separate off a radial sample for the smooth remainder.  The same line's
    # interval supplies the phase label and a local prediction in its interior.
    rB=np.sqrt(2*special.gammaincinv(d/2,np.clip(special.ndtr(X[:,d+1]),1e-15,1-1e-15)))
    zB=center+rB[:,None]*U
    yB=ctx.query(zB)
    tB=np.clip((rB-L)/np.maximum(1e-12,H-L),0,1)
    yB0=(1-tB)*(yL-aL)+tB*yH
    chi=(rB<L) | ((rB<=H)&(yB>yB0+.5*amp(zB)))
    chi=np.where(good,chi,phase.label(zB,yB))
    w=np.exp(np.clip(-rB*(U@center)-.5*np.sum(center*center),-80,40))
    smooth=(yB-amp(zB)*chi-bcv(zB))*w
    # When a bracket was not confirmed, estimate the branch correction from
    # the ordinary radial sample; do not assume an interval contains a jump.
    fallback=(chi.astype(float)-(q(zB)<0))*np.exp(np.clip(b*rB+shift,-80,40))
    diff=np.where(good,diff,fallback)
    est=acv + av*(model_mass+np.mean(diff))+np.mean(smooth)
    log("conditional closed jump",n,"width",width,"valid",np.mean(good),
        "rms",np.std(diff),np.std(smooth),"components",acv,model_mass,np.mean(diff),np.mean(smooth))
    return est


def jumps(ctx, x, y, edges, z, w):
    loga, tilt = jump_amplitude(ctx,edges)
    p = Phase(ctx,x,y,edges,z,w,loga,tilt)
    log("jump amplitude", loga, "tilt",np.linalg.norm(tilt),"background scatter",p.background_sd)
    a = choose_anchor(ctx,p,x,y,z,w,edges)
    if ctx.d>=24 and ctx.remaining>11200:
        try:
            q,noise=pair_geometry(ctx,p,a,edges,x,y,z,w)
        except (ValueError,np.linalg.LinAlgError) as exc:
            log("pair geometry declined",repr(exc))
            try:
                q,noise=normal_geometry(ctx,p,a,edges)
            except (ValueError,np.linalg.LinAlgError) as exc:
                log("normal sketch declined",repr(exc))
                q,noise=train_geometry(ctx,p,a,edges,x,y,z,w)
    else:
        q,noise=train_geometry(ctx,p,a,edges,x,y,z,w)
    if getattr(q,"pair_design",False) and ctx.d>=48 and ctx.remaining>=4000 and noise>.0015:
        q,noise=calibrate_spectrum(ctx,p,q,a,edges,noise)
    estimate = ellipsoid_prob(q.A,q.b,q.c,tilt)
    if not .000000003 < estimate < .999999999:
        raise ValueError("implausible ellipsoid probability")
    log("ellipse probability", estimate)
    av = math.exp(loga+.5*(tilt@tilt))
    if (noise < 1.2e-4 and ctx.d <= 24) or (
            noise < .012 and loga < np.log(10*max(.1,np.median(np.abs(y))))):
        from hard import integrate_remainder
        def fit(z):
            return p.amplitude(z)*(q(z)<0)
        return integrate_remainder(ctx,fit,av*estimate,x,y,antithetic=True)
    if loga < np.log(10*max(.1,np.median(np.abs(y)))) and ctx.remaining>=2650:
        try:
            return conditional_jump(ctx,p,q,a,noise,estimate,av,x,y)
        except (ValueError,np.linalg.LinAlgError) as exc:
            log("conditional jump declined",repr(exc))
    center, S, logdet = tilted_proposal(q,tilt)
    if q(center) >= -1e-5 or not p.label(center[None], ctx.query(center))[0]:
        # Centering at the observed interior point preserves a one-sided ray.
        center = a
    width, center, fail = shell_calibration(ctx,p,q,center,S,a,noise)
    d = ctx.d
    bg = Background(ctx,p)
    ys = p.remainder(x,y)
    spend = ctx.remaining > (d+1)*(d+2)//2+3550
    acv, cv, R = hermite_ridge_control(bg,no_model,0.,x,ys,spend_hessian=spend)
    # Use a common baseline even when a fitted phase is imperfect: the two
    # integrals sum to the original payoff pointwise.
    available = ctx.remaining
    bw = min(4096, max(640, int(available*.52 if width < .04 else available*.38)))
    if bw+128 > available:
        bw = max(16,available//2)
    nn = balanced_rule(bw//2,d,ctx.rng) @ R.T
    bk = np.vstack([nn,-nn])
    b = acv + np.mean(bg.query(bk)-cv(bk))
    if (loga>np.log(7*max(.1,np.median(np.abs(y)))) and np.linalg.norm(tilt)<.02
        and ctx.d>=24 and .002<width<1.55 and fail<.12 and ctx.remaining>=1880):
        corr = bracketed_shell(ctx,p,q,center,S,logdet,width)
    else:
        corr = shell_correction(ctx,p,q,ctx.remaining,center,S,logdet,width)
    log("jump components", b, av, estimate, corr)
    return b+av*(estimate+corr)
