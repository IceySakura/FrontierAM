"""A validated Gaussian-kernel residual control with an exact normal mean."""

import numpy as np
from scipy import linalg


def _kernel(x,y):
    dx = np.sum(x*x,axis=1)
    dy = np.sum(y*y,axis=1)
    ds = dx[:,None]+dy[None,:]-2.*(x@y.T)
    np.maximum(ds,0.,out=ds)
    np.multiply(ds,-.5,out=ds)
    np.exp(ds,out=ds)
    return ds


class KernelControl:
    def __init__(self, basis, scale, centers, coef, intercept):
        self.transform = basis*np.sqrt(scale)
        self.centers = centers
        self.coef = coef
        self.intercept = intercept
        normalizer = np.exp(-.5*np.sum(np.log1p(scale)))
        m = normalizer*np.exp(-.5*(centers*centers)@(1./(1.+scale)))
        self.mean = float(intercept + m @ coef)

    def __call__(self,z):
        return np.concatenate([_kernel(zz@self.transform,self.centers)@self.coef+self.intercept
                                for zz in np.array_split(z,max(1,(len(z)+1023)//1024))])


def fit(z, resid, cov):
    d = z.shape[1]
    if len(z)<160:
        return None
    s = float(np.std(resid))
    if not 1.e-9<s<1.e7:
        return None
    active = np.minimum(np.abs(resid-np.median(resid)),5.*s)
    v = max(float(np.mean(active**2)),s*s*.10)
    ls,basis = linalg.eigh(.5*(cov+cov.T),check_finite=False)
    ls = np.maximum(ls[::-1],0.)
    basis = basis[:,::-1].copy()
    sc = np.clip(.26*ls/max(v,1.e-22),.014,1.9)
    sc = sc * min(1., 5.5/max(sc.sum(),.01))

    val = np.arange(len(z))%7 == 0
    trn = np.flatnonzero(~val)
    if len(trn)>875:
        trn = trn[np.linspace(0,len(trn)-1,875).astype(int)]
    yz = z[trn]@basis
    yv = z[val]@basis
    r = resid[trn]
    rt = resid[val]
    score0 = float(np.mean(rt*rt))
    model = None
    score = score0*.91
    shift = float(np.mean(r))
    for h,reg in [(.52,.010),(1.25,.0045),(2.45,.017)]:
        scl = h*sc
        a = yz*np.sqrt(scl)
        b = yv*np.sqrt(scl)
        gram = _kernel(a,a)
        gram.flat[::len(gram)+1] += reg
        c = linalg.solve(gram,r-shift,assume_a="pos",check_finite=False)
        rhat = _kernel(b,a)@c+shift
        mse = np.mean((rt-rhat)**2)
        if mse < score:
            model = KernelControl(basis,scl,a,c,shift)
            score = mse
    return model


def crossfit(w, basis, residual, cov, test_symmetry=True):
    """Two independently scrambled folds, each held out from its own fit."""
    from numerics import residual_average, ridge

    if w.remaining < 4608 or not np.isfinite(cov).all() or np.trace(cov)<1.e-22:
        return residual_average(w,basis,residual,test_symmetry)
    pilot = w.normal(128)@basis.T
    sites = np.vstack([pilot,-pilot])
    score = residual(sites,w.query(sites))
    var = float(np.var(score))
    if not 1.e-16 < var < 1.e5:
        return residual_average(w,basis,residual,False)
    paired = test_symmetry and np.var((score[:128]+score[128:])*.5)<.29*var
    d = w.d
    ls,u = linalg.eigh(cov,check_finite=False)
    u = u[:,::-1].copy()
    sc = np.clip(.38*ls[::-1]/var,.008,2.)
    sc *= min(1.,4.8/sc.sum())
    landmark = w.normal(384,d)
    idx = [np.arange(224),np.arange(224,384)]
    scales = [sc*.48,sc*1.65]
    xs = [landmark[i]*np.sqrt(s) for i,s in zip(idx,scales)]
    mean = np.concatenate([
        np.exp(-.5*np.sum(np.log1p(s))-.5*(x*x)@(1./(1.+s)))
        for x,s in zip(xs,scales)])
    def fs(z):
        x = z@u
        feat = [_kernel(x*np.sqrt(s),c) for c,s in zip(xs,scales)]
        q = x[:,:min(18,d)]
        feat += [q,(q*q-1.)*2.**-.5,
                 q[:,:10]*(q[:,:10]**2-3.)*.40824829]
        return np.column_stack(feat)

    n = w.remaining
    n1 = 1 << (int(n*.67).bit_length()-1)
    values = []
    for k in (n1,n-n1):
        x = w.normal((k+1)//2 if paired else k) @ basis.T
        if paired:
            x = np.vstack([x,-x[:k-len(x)]])
        ax = residual(x,w.query(x))
        values.append((fs(x),ax))
    nf = values[0][0].shape[1]
    mean = np.r_[mean,np.zeros(nf-len(mean))]
    ans = float(sum(np.sum(yy) for _,yy in values)/n)
    for fold in [0,1]:
        f,y = values[fold]
        hold = np.arange(len(y))%9 == 0
        f0,y0 = f[~hold],y[~hold]
        v = np.var(f0,axis=0)
        c,b = ridge(f0,y0,np.maximum(v,1.e-4)*.0013 + 5.e-6)
        pred = f[hold]@c + b
        yv = y[hold]
        pc = pred-pred.mean()
        yc = yv-yv.mean()
        pv = np.var(pc)
        cv = np.mean(pc*yc)
        if pv > 1.e-20 and cv > 0.:
            beta = float(np.clip(cv/pv,0.,1.))
            if 2.*beta*cv-beta*beta*pv > .125*max(np.var(yv),1.e-20):
                op = values[1-fold][0]
                ans += float((len(op)/n)*beta * ((mean-op.mean(axis=0))@c))
    wt = 76.8/(n+256.)
    return float((1.-wt)*ans + wt*np.mean(score))
