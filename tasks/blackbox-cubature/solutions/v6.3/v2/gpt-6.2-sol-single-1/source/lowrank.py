"""Low-rank tensor reconstruction by simultaneous diagonalization of slices."""
import math

import numpy as np
from scipy.linalg import qr, solve, svd
from numpy.polynomial.legendre import legvander

from common import Model, legmat, multiindices, residual, solve_ls
from sparse import _cardinal


def _cheb(n):
    x = (1.-np.cos(np.pi*np.arange(n)/(n-1)))*.5
    bw = (-1.)**np.arange(n)
    bw[[0,-1]] *= .5
    moments = np.zeros(n)
    moments[0] = 1.
    w = np.linalg.solve(legvander(2*x-1,n-1).T,moments)
    return x,bw,w


def _select_axes(x,y):
    d = x.shape[1]
    ids = multiindices(d,4,2)
    if ids.shape[1] > len(x)*.6:
        ids = multiindices(d,3,2)
    a = legmat(x,ids)
    c = solve_ls(a,y)
    c *= np.prod(1./np.sqrt(2*ids+1),axis=0)
    pairs = []
    for i in range(d):
        for j in range(i+1,d):
            mask = (ids[i]>0)&(ids[j]>0)
            vv = abs(c[mask])
            ss = float(np.sqrt(vv@vv))
            pairs.append((ss,i,j))
    _,i,j = max(pairs)
    num = (ids>0).sum(axis=0)
    v = np.array([np.linalg.norm(c[(ids[l]>0)&(num<=2)]) for l in range(d)])
    v[[i,j]]=-1
    k = int(np.argmax(v))
    return i,j,k


def _decode(slices,rank):
    a = np.concatenate(slices,axis=1)
    b = np.concatenate(slices,axis=0)
    u,s,_ = svd(a,full_matrices=False,check_finite=False)
    _,sv,v = svd(b,full_matrices=False,check_finite=False)
    u,v=u[:,:rank],v[:rank].T
    m0 = np.array([u.T@c@v for c in slices])
    options = []
    for beta in (.48,0.):
        vl = np.maximum(1e-18,s[:rank]/s[0])**beta
        vr = np.maximum(1e-18,sv[:rank]/sv[0])**beta
        m = m0 / vl[:,None] / vr[None,:]
        for j in (0,1,2):
            B = m[j]
            if np.linalg.cond(B)>3e11:
                continue
            for a in (.3,.82,1.61):
                D = m[(j+1)%3]+a*m[(j+2)%3]
                J = solve(B.T,D.T,check_finite=False).T
                e,g = np.linalg.eig(J)
                if np.any(e.imag != 0.):
                    continue
                g = g.real
                if rank>1:
                    gap = np.min(np.diff(np.sort(e.real)))
                else:
                    gap = 1.
                cond = np.linalg.cond(g)
                if cond > 8e8 or gap < 2e-8:
                    continue
                A = u@(vl[:,None]*g)
                C = solve(g,B,check_finite=False)*vr
                V = v@C.T
                ax,bx = np.max(np.abs(A),axis=0),np.max(np.abs(V),axis=0)
                A, V = A/ax, V/bx
                gains = ax*bx
                # All observed slices should be diagonal in the same basis.
                core = (A[:,None,:]*V[None,:,:]).reshape(-1,rank)
                coef = solve_ls(core,np.array(slices).reshape(3,-1).T)
                err = np.max(np.abs(core@coef-np.array(slices).reshape(3,-1).T))
                q = max(err,5e-15)/(min(gap,1.)**.2)*max(1.,cond/60.)**.14
                options.append((q,err,j,A,V,gains))
    if not options:
        return None
    return min(options,key=lambda s:s[0])


def tensor_model(orc, scale, x, y):
    d = orc.d
    if d < 3 or d > 9 or orc.left < 580:
        return None
    i,j,k = _select_axes(x,y)
    n = 9
    z,bw,w = _cheb(n)
    anchor = np.ones(d)*.5
    if np.max(np.abs(y)) > 6.*max(1e-200,np.median(np.abs(y))):
        anchor = x[np.argmax(np.abs(y))].copy()
    # Use three transverse shifts, not repetitions of the same slice.
    levels = np.array([anchor[k],.183,.817])
    if np.min(np.abs(np.diff(np.sort(levels)))) < .08:
        levels[0] = .5
    pts = np.broadcast_to(anchor,(3,n,n,d)).copy()
    pts[:,:,:,i] = z[:,None]
    pts[:,:,:,j] = z[None,:]
    pts[:,:,:,k] = levels[:,None,None]
    v = orc.sample(pts.reshape(-1,d)).reshape(3,n,n)/scale
    ss = svd(np.concatenate(v,axis=1),full_matrices=False,check_finite=False,compute_uv=False)
    if ss[0]<1e-280:
        return None
    expected = max(1e-20,np.ptp(v),.12*np.ptp(y))
    max_rank = min(8,(orc.left-65)//(d*11))
    ranks = sorted(set(int(np.sum(ss > thr*ss[0])) for thr in (1e-12,2e-11,8e-9,3e-6,4e-5)
                       if 1 <= np.sum(ss > thr*ss[0]) <= max_rank))
    if not ranks:
        return None
    opts=[]
    for rank in ranks:
        opt = _decode(v,rank)
        if opt is not None and opt[1] < .008*expected:
            opts.append((opt,rank))
    if not opts:
        return None
    op,r = min(opts,key=lambda o:o[0][0]*(1.+o[1]*.15))
    _,_,base,A,B,c = op
    if np.min(c) < 1e-12*np.max(c):
        return None
    anchor[k] = levels[base]
    def select(F):
        # Pivot the transpose to find well-conditioned evaluation rows.
        _,_,pp = qr(F.T,mode="economic",pivoting=True,check_finite=False)
        return pp[:r]
    rows,cols = select(A),select(B)
    U = A[rows]*c
    V = B[cols]*c
    phi = (A[:,None,:]*B[None,:,:]).reshape(n*n,r)
    if np.linalg.cond(U) > 6e10 or np.linalg.cond(V) > 6e10:
        return None
    act = select(phi)
    D = phi[act]*c
    if np.linalg.cond(D) > 6e10:
        return None
    ii,jj=np.unravel_index(act,(n,n))
    s = min(51,(orc.left-55)//(d*r))
    if s<9:
        return None
    if s%2==0:
        s-=1
    t,bw2,weights = _cheb(s)
    pp = np.broadcast_to(anchor,(d,s,r,d)).copy()
    for a in range(d):
        if a==i:
            pp[a,:,:,j] = z[cols]
        elif a==j:
            pp[a,:,:,i] = z[rows]
        else:
            pp[a,:,:,i] = z[ii]
            pp[a,:,:,j] = z[jj]
        pp[a,:,:,a] = t[:,None]
    g = orc.sample(pp.reshape(-1,d)).reshape(d,s,r)/scale
    q = np.array([solve(V if a==i else U if a==j else D,g[a].T,
                         check_finite=False).T for a in range(d)])
    if np.max(np.abs(q))>1e11:
        return None
    def predict(p,coef=c):
        result = np.ones((len(p),r))
        for a in range(d):
            result *= _cardinal(p[:,a],t,bw2)@q[a]
        return result@coef
    ix = np.arange(len(x)) % 4 != 3
    vv0 = (weights@q).prod(axis=0)
    err = np.sqrt(np.mean((predict(x[~ix])-y[~ix])**2))
    if err > .35*max(1e-15,np.std(y[~ix])):
        return None
    coef = c
    integ = vv0@coef
    # Mixing factors can lose precision if the slices are nearly dependent.
    # Rescaling the already learned factors at scattered samples is cheap.
    ds = np.ones((int(ix.sum()),r))
    for a in range(d):
        ds *= _cardinal(x[ix,a],t,bw2)@q[a]
    new = solve_ls(ds,y[ix])
    if not np.all(np.isfinite(new)):
        return None
    alt = np.sqrt(np.mean((predict(x[~ix],new)-y[~ix])**2))
    if alt < .85*err and np.max(np.abs(new)) < 9.*np.max(np.abs(coef)):
        coef,err = new,alt
        integ = vv0@coef
    return Model(lambda p:predict(p,coef),float(integ),"low-rank tensor",float(err))
