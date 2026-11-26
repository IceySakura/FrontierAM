"""A conditional control for a smooth positive payoff with exercise clipping."""

import math

import numpy as np
from scipy import special

from numerics import quantile, ridge
from payoff_spline import SplinePayoff


def _cdf_diff(a,b):
    return np.where(a>=0.,special.ndtr(-a)-special.ndtr(-b),special.ndtr(b)-special.ndtr(a))


def useful_tilt(x, residual):
    e = np.asarray(residual)*residual
    e = np.minimum(e, max(1.e-20,np.quantile(e,.965)*5.))
    grid = np.array([0.,-.21,.20,.41,.63,.84,1.02])
    risk = e @ np.exp(-x[:,0,None]*grid+.5*grid*grid)
    j = int(np.argmin(risk))
    return float(grid[j]) if risk[j]<.89*risk[0] else 0.


class Exercise:
    def __init__(self,d,mean_t):
        self.d = d
        self.mean_t = mean_t
        k = d-1
        i,j = np.triu_indices(k,1)
        choose = i<min(4,k)
        self.p1,self.p2 = i[choose],j[choose]
        self.r1,self.r2 = np.triu_indices(min(5,k),1)

    def columns(self,x):
        sq = (x*x-1.) * 2.**-.5
        a = np.column_stack([x,sq,x[:,self.p1]*x[:,self.p2],
                             x[:,:8]*(x[:,:8]**2-3.)*.40824829])
        b = np.column_stack([np.ones(len(x)),x,sq[:,:7],x[:,self.r1]*x[:,self.r2]])
        c = np.column_stack([np.ones(len(x)),x[:,:5]])
        return a,b,c

    def rows(self,z):
        t = z[:,0] - self.mean_t
        a,b,c = self.columns(z[:,1:])
        return np.column_stack([a,t[:,None]*b,(t*t)[:,None]*c]),(a.shape[1],b.shape[1])

    def initialize(self, fit, val, jets=None):
        zx,yx = fit
        zv,yv = val
        use = yx>0.
        yy = yx[use]
        if len(yy)<210:
            return False
        raw,sizes = self.rows(zx[use])
        p,q = sizes
        test,_ = self.rows(zv)
        self.mags = np.quantile(yy,[.5,.99])
        med = max(self.mags[0],1.e-9)
        pen = np.r_[np.full(p,.010),np.full(q,.012),np.full(raw.shape[1]-p-q,.015)]
        pen[p] = .0016
        pen[p+q] = .0035
        ys = np.maximum(yv,0.)
        positive = ys>0.
        if np.sum(positive)<60:
            return False
        if jets is not None:
            zg, yg, g0 = jets
            if len(zg)>2:
                eps = 2.e-5
                delta = np.eye(self.d)[None,:,:]*eps
                orig,_ = self.rows(zg)
                d1,_ = self.rows((zg[:,None,:]+delta).reshape(-1,self.d))
                df = (d1.reshape(len(zg),self.d,-1)-orig[:,None,:]).reshape(len(d1),-1)/eps
                ar = np.column_stack([np.r_[np.ones(len(raw)),np.zeros(len(df))],
                                      np.vstack([raw,df])])
            else:
                jets = None
        self.cv = None
        for strike in med*np.array([2.5,5.,9.,16.]):
            ylog = np.log1p(yy/strike)
            weight = np.minimum(5.,1.+yy/strike)
            if jets is not None:
                deriv = (g0/(strike+yg[:,None])).ravel()
                wg = np.broadcast_to(.52 + .48*np.eye(1,self.d),(len(zg),self.d)).ravel()
                cc,_ = ridge(ar,np.r_[ylog,deriv],np.r_[.00009,pen],np.r_[weight,wg],center=False)
                m,c = cc[0],cc[1:]
            else:
                c,m = ridge(raw,ylog,pen,weight)
            yh = strike*np.expm1(np.clip(test@c+m,0.,3.5))
            e = yh[positive]-yv[positive]
            risk = np.mean(np.minimum(e**2,(3.*self.mags[1])**2) /
                           (1.+.35*yv[positive]/med))
            if self.cv is None or risk<self.cv:
                self.cv = risk
                self.coef,self.off,self.K,self.sizes = c,m,strike,sizes
        self.cap = float(np.log1p(max(12.*self.mags[1],55.*med)/self.K))
        self.cap = min(5.7,self.cap)
        yh = self.value(zv)
        missed = np.sum((~positive)&(yh>.30*med))/len(zv)
        relative_sd = np.sqrt(np.mean((yh-yv)**2))/max(np.std(yv),med)
        self.validation_error = relative_sd**2
        return missed<.017 and relative_sd<.30

    def parameters(self,x):
        a,b,c = self.columns(x)
        p,q = self.sizes
        u = self.off + a@self.coef[:p]
        v = b@self.coef[p:p+q]
        w = c@self.coef[p+q:]
        u,v = u-self.mean_t*v+self.mean_t**2*w,v-2.*self.mean_t*w
        return np.clip(u,-4.,3.),np.clip(v,-1.7,1.7),np.clip(w,-.4,.26)

    def _pay(self,g):
        return self.K*np.expm1(np.clip(g,0.,self.cap))

    def value(self,z):
        a,b,c = self.parameters(z[:,1:])
        t = np.clip(z[:,0],-6.,6.)
        return self._pay(a+t*(b+t*c))

    def root(self,x):
        a,b,c = self.parameters(x)
        dis = np.maximum(b*b-4.*c*a,.005)
        root = -2.*a/np.maximum(.1,b+np.sqrt(dis))
        return np.clip(root-.5,-2.3,1.15)

    def conditional(self,x):
        a,b,c = self.parameters(x)
        T = 6.
        c_safe = np.where(abs(c)>1.e-8,c,1.)
        b_safe = np.where(abs(b)>1.e-8,b,1.)
        points = [np.full(len(x),-T),np.full(len(x),T)]
        for level in (0.,self.cap):
            dis = b*b-4*c*(a-level)
            g = -.5*(b + np.where(b<0.,-1.,1.)*np.sqrt(np.maximum(dis,0.)))
            g_safe = np.where(abs(g)>1.e-12,g,1.)
            rr = np.where((abs(b)>1.e-8)&(abs(c)<=1.e-8),(level-a)/b_safe,T)
            f = (abs(c)>1.e-8)&(dis>0.)
            points.extend([np.clip(np.where(f,g/c_safe,rr),-T,T),
                           np.clip(np.where(f,(a-level)/g_safe,T),-T,T)])
        knots = np.sort(np.column_stack(points),axis=1)
        lo,hi = knots[:,:-1],knots[:,1:]
        mid = (lo+hi)*.5
        y0 = a[:,None]+mid*(b[:,None]+mid*c[:,None])
        z0 = _cdf_diff(lo,hi)
        ss = 1.-2.*c
        mn = (b/ss)[:,None]
        rt = np.sqrt(ss)[:,None]
        moment = _cdf_diff((lo-mn)*rt,(hi-mn)*rt)
        mul = (np.exp(a+.5*b*b/ss)/np.sqrt(ss))[:,None]
        integ = np.where(y0<=0.,0.,np.where(y0>=self.cap,np.expm1(self.cap)*z0,
                                           np.maximum(0.,mul*moment-z0)))
        tails = (self._pay(a-T*b+T*T*c)+self._pay(a+T*b+T*T*c))*special.ndtr(-T)
        return integ.sum(axis=1)*self.K + tails


def trial(w, basis, x0, y0, threshold, jets=None):
    """Trains and checks the smooth clipped branch; retains sample data on failure."""
    d = w.d
    n = min(2048,w.remaining//6)
    u = w.unit(n)
    x = quantile(u)
    lo = special.ndtr(np.clip(threshold-.41,-1.35,.35))
    hi = special.ndtr(3.65)
    x[:,0] = quantile(lo+(hi-lo)*u[:,0])
    y = w.query(x@basis.T)
    prior = x0@basis
    val = (np.vstack([prior[-256:],x[::8]]),np.r_[y0[-256:],y[::8]])
    ii = np.arange(n)%8 != 0
    fit = (np.vstack([prior[:-256],x[ii]]),np.r_[y0[:-256],y[ii]])
    model = Exercise(d,.66)
    checked = model.initialize(fit,val,jets)
    sp = SplinePayoff(d,threshold)
    spline_checked = sp.initialize(fit,val,jets)
    if not checked and not spline_checked:
        spline_checked = sp.calibrate(fit,val)
    if not checked and not spline_checked:
        return None,x,y
    use_spline = spline_checked and (not checked or sp.validation_error<1.12*model.validation_error)
    if use_spline:
        model = sp
        mean = model.mean()
    else:
        target = 1<<17
        v = w.normal(target,d-1)
        total = 0.
        for k0 in range(0,target,4096):
            total += model.conditional(v[k0:k0+4096]).sum()
        mean = total/target
    va = prior[-256:]
    tilt = useful_tilt(va, y0[-256:]-model.value(va))
    xmain = w.normal(w.remaining)
    xmain[:,0] += tilt
    obs = w.query(xmain@basis.T)
    wt = np.exp(-tilt*xmain[:,0]+.5*tilt*tilt)
    return float(mean + np.mean(wt*(obs-model.value(xmain)))),None,None
