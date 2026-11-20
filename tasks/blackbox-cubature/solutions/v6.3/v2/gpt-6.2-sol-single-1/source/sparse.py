"""Dimension-adaptive nested interpolation and integration."""
from functools import lru_cache
import heapq
import math

import numpy as np
from numpy.polynomial.legendre import legvander

from common import Model, residual


def _cardinal(x, nodes, bw):
    w = x[:,None]-nodes
    hit = w==0.
    with np.errstate(divide="ignore",invalid="ignore"):
        v = bw / w
        v /= np.sum(v,axis=1)[:,None]
    has = np.any(hit,axis=1)
    v[has] = hit[has]
    return v


@lru_cache(None)
def rule(l):
    if l == 0:
        return (np.array([.5]), np.ones(1), np.ones(1),
                np.array([0]), np.ones((1,1)))
    n = 1 << l
    x = (1.-np.cos(np.pi*np.arange(n+1)/n))*.5
    if l == 1:
        new,old = np.array([0,2]),np.array([1])
    else:
        new,old = np.arange(1,n,2),np.arange(0,n+1,2)
    px,_,pb,_,_ = rule(l-1)
    x[old]=px
    bw = (-1.)**np.arange(n+1)
    bw[[0,-1]] *= .5
    coeff = np.zeros(n+1)
    coeff[0] = 1.
    wi = np.linalg.solve(legvander(2*x-1,n).T,coeff)
    op = np.zeros((len(new),n+1))
    op[np.arange(len(new)),new] = 1.
    op[:,old] = -_cardinal(x[new],px,pb)
    return x,wi,bw,new,op


def _grid_shape(ix, new_only=False):
    coords = [rule(l)[0][rule(l)[3]] if new_only else rule(l)[0] for l in ix]
    return np.stack(np.meshgrid(*coords,indexing="ij"),axis=-1).reshape(-1,len(ix))


def _count(ix):
    return math.prod(len(rule(l)[3]) for l in ix)


def _parents(ix):
    for j,l in enumerate(ix):
        if l:
            yield ix[:j]+(l-1,)+ix[j+1:],j


def _predict(p,terms):
    cards = {}
    out = np.zeros(len(p))
    for ix,r,_,_ in terms:
        vals = np.ones((len(p),1))
        for j,k in enumerate(ix):
            if not k:
                continue
            key = j,k
            mat = cards.get(key)
            if mat is None:
                x,_,bw,ind,_ = rule(k)
                mat = _cardinal(p[:,j],x,bw)[:,ind]
                cards[key] = mat
            vals = (vals[:,:,None]*mat[:,None,:]).reshape(len(p),-1)
        out += vals@r.ravel()
    return out


def _surplus(ix, vals):
    pr = [rule(l) for l in ix]
    q = []
    arr = vals.reshape([len(r[0]) for r in pr])
    for j,(x,w,b,n,op) in enumerate(pr):
        q.append(w[n])
        if ix[j]:
            arr = np.moveaxis(np.tensordot(op,arr,axes=(1,j)),0,j)
    w = q[0]
    for z in q[1:]:
        w = np.multiply.outer(w,z)
    contrib = float(np.sum(arr*w))
    importance = float(np.sqrt(np.sum(arr*arr*w)*max(np.sum(w),0.)))
    return ix,arr,contrib,importance


def adaptive_sparse(orc,scale,x,y,base=None,maxeval=None):
    d = orc.d
    if not (2 <= d <= 9) or orc.left < max(160,24*d):
        return None
    maxeval = min(orc.left,orc.left-64 if maxeval is None else maxeval)
    start = orc.used
    z = (0,)*d
    unit = [z[:j]+(1,)+z[j+1:] for j in range(d)]
    terms = []
    level = {}
    active = set()
    x0 = (x[:256] if len(x)>256 else x)
    y0 = y[:len(x0)]
    target = y0 if base is None else y0-base.predict(x0)
    initial_err = max(1e-20,float(np.std(target)))
    def add(rows):
        if not rows:
            return
        # Parents have been processed; only the new tensor nodes cost points.
        orc.sample(np.vstack([_grid_shape(ix,True) for ix in rows]))
        for ix in rows:
            pts = _grid_shape(ix)
            yy = orc.sample(pts)/scale
            if base is not None:
                yy -= base.predict(pts)
            item = _surplus(ix,yy)
            terms.append(item)
            level[ix] = item[3]
            active.discard(ix)
            for j,l in enumerate(ix):
                if l>=6:
                    continue
                k = ix[:j]+(l+1,)+ix[j+1:]
                if k in level or k in active:
                    continue
                if all(p in level for p,_ in _parents(k)):
                    active.add(k)
    add([z])
    add(unit)
    low = sorted(k for k in active if sum(k)==2)
    add(low)
    axis = np.array([level[ix] for ix in unit])
    for j in range(d):
        tj = z[:j]+(2,)+z[j+1:]
        if tj in level:
            axis[j] = max(axis[j],level[tj])
    axis = np.clip(axis/initial_err,.05,1.)
    checkpoints = [max(135,20*d), max(255,34*d), max(425,55*d)]
    best = (1.0,None,0.)
    last_checkpoint = 0
    def state():
        ims = _predict(x0,terms)
        rr = ims-target
        err = float(np.sqrt(np.mean(rr**2)))/initial_err
        return err, math.fsum(s[2] for s in terms)
    count = orc.used-start
    while maxeval-count >= 2 and active and orc.elapsed < 18:
        caps = maxeval-count
        queue = []
        for k in sorted(active):
            cost = _count(k)
            if cost > caps:
                continue
            m = sum(j>0 for j in k)
            candidates = []
            for p,j in _parents(k):
                if k[j]==1:
                    rate = .45*axis[j]+.075
                elif k[j]==2:
                    rate = .56
                else:
                    rate = .38
                    pp = p[:j]+(p[j]-1,)+p[j+1:]
                    if pp in level:
                        rate = np.clip((level[p]+1e-25)/(level[pp]+1e-25),.08,.68)
                candidates.append(level[p]*rate)
            candidate = min(candidates)+.045*max(candidates)
            priority = (candidate + initial_err*1e-13) / (cost**.63*(1.+.18*m))
            heapq.heappush(queue,(-priority,cost,k))
        if not queue:
            break
        calc = []
        s = 0
        top = queue[0][0]
        while queue and (not calc or s < 38) and len(calc)<10:
            a,c,k = heapq.heappop(queue)
            if s+c<=caps and (not calc or a <= top*.13):
                calc.append(k)
                s += c
        add(calc)
        count = orc.used-start
        if last_checkpoint < len(checkpoints) and count >= checkpoints[last_checkpoint]:
            curr,integ = state()
            if curr < best[0]:
                best = curr,len(terms),integ
            last_checkpoint += 1
            if curr > (.60 if d>=4 else .47):
                break
            if last_checkpoint>1 and curr > best[0]*1.7:
                break
            if curr < 2e-12:
                break
    cv,iv = state()
    if best[1] is not None and cv > best[0]*1.25:
        cv,n,iv = best
        terms = terms[:n]
    if cv >= .65:
        return None
    def predict(q):
        out = _predict(q,terms)
        if base is not None:
            out += base.predict(q)
        return out
    if base is not None:
        iv += base.integral
    return Model(predict,iv,"adaptive sparse grid",cv*initial_err)
