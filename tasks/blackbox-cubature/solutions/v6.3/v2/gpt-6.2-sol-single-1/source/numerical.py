"""Surrogate-assisted and adaptive numerical estimates for the residual."""
import itertools
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf
from scipy.spatial.distance import cdist

from common import gl_interval, legmat, multiindices, residual, solve_ls


def _axis_kernel(r, a, kind):
    r = np.abs(r) / a
    if kind == "se":
        return np.exp(-r*r*.5)
    if kind == "m32":
        return np.exp(-r)*(1.+r)
    return np.exp(-r)*(1.+r+r*r/3)


def _kernel_primitive(x,a,kind):
    z = x/a
    if kind == "se":
        return a*np.sqrt(np.pi/2)*erf(z / np.sqrt(2.))
    if kind == "m32":
        return a*(2.-(2.+z)*np.exp(-z))
    return a*(8./3.-(8.+5*z+z*z)/3.*np.exp(-z))


class Kernel:
    def __init__(self, ell, rho, kind):
        self.ell = ell
        self.rho = rho
        self.kind = kind
        t,w = gl_interval(48)
        val = _kernel_primitive(t[:,None],ell,kind)
        self.mean = 2.*(w@val)

    def mat(self,x,y):
        d = x.shape[1]
        mx = _kernel_primitive(x,self.ell,self.kind)+_kernel_primitive(1.-x,self.ell,self.kind)
        my = _kernel_primitive(y,self.ell,self.kind)+_kernel_primitive(1.-y,self.ell,self.kind)
        out = np.ones((len(x),len(y)))
        for j in range(d):
            s = _axis_kernel(x[:,j,None]-y[None,:,j],self.ell[j],self.kind)
            s -= mx[:,j,None]
            s -= my[None,:,j]
            s += self.mean[j]
            out *= 1. + (self.rho[j]/(1.-self.mean[j]))*s
        return out

    def fit(self,x,y,ridge):
        K = self.mat(x,x)
        K.flat[::len(K)+1] += ridge
        factor = cho_factor(K,lower=True,check_finite=False,overwrite_a=True)
        return cho_solve(factor,y,check_finite=False)


def _lengths(x,y):
    d = x.shape[1]
    z = (x-.5)*2
    a = np.column_stack([np.ones(len(x)),z,1.5*z*z-.5])
    c = solve_ls(a,y)
    v = c[1:1+d]**2/3+c[1+d:]**2/5
    v += max(.07*np.var(y)/d,1e-25)
    sc = np.clip(v/np.mean(v),.14,3.)
    return sc


def _kernel_models(x,y,d):
    r = _lengths(x,y)
    inv = np.power(r,-.18)
    if d <= 3:
        options = [(.28,.93,"se",2e-9),(.13,.95,"se",2e-8),(.17,1.05,"m52",1e-8),
                   (.20,1.18,"m32",1e-7),(.4,.8,"se",1e-7)]
    else:
        options = [(.28,.46,"se",5e-9),(.15,.66,"se",2e-8),(.21,.8,"m52",2e-8),
                   (.25,.95,"m32",5e-7),(.38,.17,"se",3e-8),(.38,.39,"m52",2e-8)]
    for ell,rho,kind,ridge in options:
        yield Kernel(ell*inv,rho*r**.45,kind),ridge


def _choose_surrogate(base, x0,y0, val_x,val_y):
    d = x0.shape[1]
    yy = y0 if base is None else y0-base.predict(x0)
    vv = val_y if base is None else val_y-base.predict(val_x)
    ref = np.sqrt(np.mean(vv**2))
    best = (ref,None,0.)
    for ke,rr in _kernel_models(x0,yy,d):
        w = ke.fit(x0,yy,rr)
        delta = (ke.mat(val_x,x0)@w-vv)
        cv = np.sqrt(np.mean(delta**2))
        if cv < best[0]:
            best = cv,ke,rr
    return best


def weighted_qmc(orc, scale, models=()):
    """Keep independent scrambled nets for applying the surrogate correction."""
    d = orc.d
    if d > 20:
        orc.uniform(orc.left)
        return orc.fallback()/scale
    old = orc.global_n
    if orc.left >= 92:
        n = min(orc.left//4,256)
        n = 1 << ((n).bit_length()-1)
    else:
        n = 0
    # The training set is an extensible net; the held-out set is a new net.
    nx = orc.left-n
    want = (1 << ((old+nx).bit_length()-1)) if old+nx else 0
    if want-old >= 0 and nx-(want-old) >= 32:
        nx = want-old
        n = orc.left-nx
    orc.uniform(nx)
    x = np.concatenate(orc.uniform_x)
    y = np.concatenate(orc.uniform_y)/scale
    if n:
        z = orc.new_sobol(n)
        rr = orc.sample(z)/scale
    else:
        z,rr = x[-min(64,len(x)//4):], y[-min(64,len(x)//4):]
    na = min(340,len(x)-min(96,len(x)//4))
    ia = np.unique(np.r_[np.arange(min(na,256)), np.linspace(256,len(x)-min(96,len(x)//4)-1,
                                                max(0,na-256),dtype=int)])
    ia = ia[ia>=0]
    a,b = x[ia],y[ia]
    if n:
        nval = min(144,len(z))
        va,vb = z[:nval],rr[:nval]
    else:
        nval = min(96,len(x)//4)
        va,vb = x[-nval:],y[-nval:]
    if len(a) < 40 or len(va) < 16 or not np.all(np.isfinite(y)):
        return float(np.mean(np.r_[y,rr] if n else y))
    ref = max(1e-200,np.std(vb))
    best_base = None
    err_base = ref
    for m in models:
        e, mx = residual(m,va,vb)
        if e < err_base and abs(m.integral) < max(10.,3*np.max(np.abs(y))):
            best_base,err_base = m,e
    if best_base is not None and err_base < ref*3e-3:
        c1 = y-best_base.predict(x)
        c2 = rr-best_base.predict(z) if n else np.empty(0)
        return float(best_base.integral+np.mean(np.r_[c1,c2]))
    best = _choose_surrogate(best_base,a,b,va,vb)
    if best_base is not None and best[0] > ref*.3:
        alt = _choose_surrogate(None,a,b,va,vb)
        if alt[0] < best[0]*.85:
            best_base,best = None,alt
    cv,kernel,ridge = best
    mean = 0. if best_base is None else best_base.integral
    yy = y if best_base is None else y-best_base.predict(x)
    val = rr if best_base is None or not n else rr-best_base.predict(z)
    if n:
        # All dyadic prefix blocks contribute; avoid giving a small tail of the
        # new net as much leverage as a complete net.
        nn = 1 << (n.bit_length()-1)
        tn = len(x)
        t0 = 1 << (tn.bit_length()-1)
        tq = np.r_[yy[:t0],val[:nn]]
        q = float(tq.mean())
    else:
        q = float(yy.mean())
    if kernel is None or cv > .94*max(err_base if best_base is not None else ref,1e-200):
        return mean+q
    w = kernel.fit(x,yy,ridge)
    if n:
        e = val-kernel.mat(z,x)@w
        er = float(np.mean(e[:nn]))
        if np.isnan(er) or np.std(e) > 1.5*max(ref,cv):
            return mean+q
        # Refit on the full data, then use the holdout for a conservative
        # correction of smooth extrapolation error.
        xx = np.vstack((x,z))
        data = np.r_[yy,val]
        v = kernel.fit(xx,data,ridge).sum()
        r = float(np.clip(cv / max(err_base if best_base is not None else ref,1e-200),0.,1.))
        v = (1.-r)*v + r*(w.sum()+.7*er)
        if not np.isfinite(v) or abs(v-q)>7.*max(ref,cv):
            return mean+q
        return mean+float(v)
    return mean+float(w.sum())


def _box_rule(d):
    if d == 2:
        r = np.array([-.9604912687080203,-.7745966692414834,-.4342437493468026,0.,
                      .4342437493468026,.7745966692414834,.9604912687080203])
        w = np.array([.1046562260264673,.2684880898683334,.4013974147759622,.4509165386584741,
                      .4013974147759622,.2684880898683334,.1046562260264673])*.5
        v = np.array([0.,5/18,0.,4/9,0.,5/18,0.])
        grid = np.asarray(list(itertools.product(r, repeat=2)))
        ww = np.kron(w,w)
        err_w = [ww-np.kron(v,w),ww-np.kron(w,v)]
        return grid,ww,np.asarray(err_w)
    # Embedded fully symmetric Genz-Malik rule, normalized by box volume.
    n = d
    grid = [np.zeros(d)]
    qw = [(12824.-9120*n+400*n*n)/19683.]
    lw = [(729.-950*n+50*n*n)/729.]
    for a, wt, low in [(math.sqrt(9./70),980./6561.,245./486.),
                         (math.sqrt(9./10),(1820.-400*n)/19683.,(265.-100*n)/1458.)]:
        for v in np.eye(d):
            grid.extend((-a*v,a*v))
            qw.extend([wt]*2)
            lw.extend([low]*2)
    for i,j in itertools.combinations(range(d),2):
        for u,v in itertools.product((-1.,1.),repeat=2):
            grid.append(math.sqrt(.9)*(u*np.eye(d)[i]+v*np.eye(d)[j]))
            qw.append(200./19683.)
            lw.append(25./729.)
    for signs in itertools.product((-1.,1.),repeat=d):
        grid.append(np.array(signs)*math.sqrt(9./19))
        qw.append(6859./(19683.*2**d))
        lw.append(0.)
    w = np.asarray(qw)
    a = np.zeros((d,len(w)))
    for j in range(d):
        a[j,0] = -2*(1.-(9./70)/.9)
        a[j,1+2*j:3+2*j] = 1.
        a[j,1+2*j+2*d:3+2*j+2*d] = -(9./70)/.9
    return np.asarray(grid),w,(w-lw,a)


def adaptive_box(orc, scale, model=None, x=None, y=None):
    d = orc.d
    grid,w,err = _box_rule(d)
    p = len(w)
    if orc.left < p:
        return weighted_qmc(orc,scale,() if model is None else (model,))
    def evaluate(ranges):
        a = np.array([v[0] for v in ranges])
        b = np.array([v[1] for v in ranges])
        centers = (a+b)*.5
        vol = np.prod(b-a,axis=1)
        pts = (centers[:,None,:]+grid*(b-a)[:,None,:]*.5).reshape(-1,d)
        g = orc.sample(pts)/scale
        if model is not None:
            g -= model.predict(pts)
        v = g.reshape(len(a),p)
        mean = v@w
        if d==2:
            ae = np.abs(v@err.T)
            ee = np.sum(ae,axis=1)
        else:
            low,aniso = err
            ee = np.abs(v@low)
            ae = np.abs(v@aniso.T)
        direc = np.argmax(ae + np.max(ae,axis=1)[:,None]*.07*(b-a),axis=1)
        return [(aa,bb,val,er,int(j)) for aa,bb,val,er,j in
                zip(a,b,vol*mean,vol*ee,direc)]
    state = evaluate([(np.zeros(d),np.ones(d))])
    while orc.left >= 2*p:
        state.sort(key=lambda v:v[3])
        maxerr = state[-1][3]
        spl = min(len(state), orc.left//(2*p), 4)
        ranges = []
        for k in range(spl):
            if k and state[-1][3] < .4*maxerr:
                break
            a,b,_,_,j = state.pop()
            mid = (a[j]+b[j])*.5
            bl,ar = b.copy(), a.copy()
            bl[j],ar[j] = mid,mid
            ranges.extend([(a,bl),(ar,b)])
        state.extend(evaluate(ranges))
    base = model.integral if model else 0.
    return base+math.fsum(r[2] for r in state)
