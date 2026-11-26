"""Spline in one factor, Gaussian-linear log payoff in the other factors."""

import math
import numpy as np
from scipy import interpolate, special

from numerics import ridge


class SplinePayoff:
    def __init__(self,d,th):
        self.d = d
        self.th = th
        self.breakpoints = np.r_[-4., -1.55, -.82, -.20, .36, 1.05, 1.84, 2.70, 4.8] + th
        self.knots = np.r_[np.repeat(self.breakpoints[0],3),self.breakpoints,
                           np.repeat(self.breakpoints[-1],3)]
        self.na = len(self.knots)-4

    def parts(self,t):
        t = np.clip(t,self.breakpoints[0],self.breakpoints[-1])
        A = interpolate.BSpline.design_matrix(t,self.knots,3,extrapolate=False).toarray()
        tt = (t-self.th-.55)/1.55
        B = np.column_stack([np.ones_like(t),tt,(tt*tt-.5)*.65,
                              .4*np.tanh(1.65*tt)])
        return A,B

    def rows(self,z):
        t,v = z[:,0],z[:,1:]
        a,b = self.parts(t)
        return np.column_stack([a,(b[:,:,None]*v[:,None,:]).reshape(len(z),4*(self.d-1))])

    def initialize(self,fit,val,jets):
        zone,y = fit
        use = y>0.
        if np.sum(use)<230:
            return False
        zy = zone[use]
        ycal = y[use]
        med,hi = np.quantile(ycal,[.5,.99])
        h = self.rows(zy)
        xv,yv = val
        vf = self.rows(xv)
        wg = np.empty(0)
        fgrad = np.empty((0,h.shape[1]))
        gobs = np.empty((0,self.d))
        gg_y = np.empty(0)
        if jets is not None and len(jets[0])>=4:
            zz,gg_y,gobs = jets
            e = 2.e-5
            fgrad = (self.rows((zz[:,None,:]+e*np.eye(self.d)[None,:,:]).reshape(-1,self.d))
                     - np.repeat(self.rows(zz),self.d,axis=0))/e
            w0 = np.full(self.d,.61)
            w0[0] = 1.5
            wg = np.tile(w0,len(zz))
        fp = np.row_stack([h,fgrad])
        a = np.linspace(.028,.036,self.na)
        a[3:-3] = .0013
        pen = np.r_[a, np.repeat([.006,.011,.027,.044],self.d-1)]
        best = None
        npos = yv>0.
        sd = max(np.std(yv),med,1.e-9)
        for k in med*np.array([4.,8.,14.]):
            k = max(k,1.e-9)
            tgt = np.r_[np.log1p(ycal/k),(gobs/(k+gg_y[:,None])).ravel()]
            w = np.r_[np.minimum(5.,1.+ycal/k),wg]
            c,_ = ridge(fp,tgt,pen,w,center=False)
            top = np.clip(np.log1p(max(13.*hi,50.*med)/k),1.,5.5)
            test = k*np.expm1(np.clip(vf@c,0.,top))
            # Use the same validation values to reject barrier-shaped
            # payoffs and excessive extrapolation among the positives.
            mse = np.mean((yv-test)**2)
            bad = np.mean((~npos)&(test>.30*med))
            if best is None or mse < best[0]:
                best = mse,bad,c,k,top
        mse,bad,self.coef,self.K,self.cap = best
        self.validation_error = mse/(sd*sd)
        return bad<.016 and self.validation_error<.083

    def calibrate(self,fit,val):
        if not hasattr(self,"coef"):
            return False
        x,y = fit
        v,b = val
        mod = self.value(x)
        scale = max(np.mean(y*y),1.e-20)
        S = mod[:,None]*self.parts(x[:,0])[0]/math.sqrt(scale)
        c,_ = ridge(S,(y-mod)/math.sqrt(scale),.017,center=False)
        self.calibration = np.clip(1.+c,0.,1.7)
        loss = np.mean((b-self.value(v))**2)
        self.validation_error = loss/max(np.var(b),1.e-20)
        if self.validation_error>.78:
            del self.calibration
            return False
        return True

    def params(self,t):
        a,b = self.parts(np.asarray(t))
        m = np.clip(a@self.coef[:self.na],-5.2,4.5)
        v = b@self.coef[self.na:].reshape(4,self.d-1)
        ns = np.linalg.norm(v,axis=1)
        v *= np.minimum(1.,1.15/np.maximum(ns,1.e-30))[:,None]
        return m,v

    def logvalue(self,t,v):
        a,b = self.params(t)
        return a+np.sum(b*v,axis=1)

    def value(self,z):
        val = self.K*np.expm1(np.clip(self.logvalue(z[:,0],z[:,1:]),0.,self.cap))
        if hasattr(self,"calibration"):
            val *= self.parts(z[:,0])[0]@self.calibration
        return val

    def root(self,v):
        # One interpolation step near the pilot exercise level is enough:
        # this threshold only chooses an importance distribution.
        t = np.full(len(v),self.th)
        g = self.logvalue(t,v)
        h = self.logvalue(t+.48,v)
        return np.clip(t-.48*g/np.maximum(.055,h-g)-.53,-2.4,1.6)

    def marginal(self,t):
        t = np.atleast_1d(t)
        m,v = self.params(t)
        s = np.maximum(np.linalg.norm(v,axis=1),1.e-12)
        u = -m/s
        r = (self.cap-m)/s
        a,b = u-s,r-s
        dif = np.where(a>=0.,special.ndtr(-a)-special.ndtr(-b),special.ndtr(b)-special.ndtr(a))
        out = np.exp(m+.5*s*s)*dif + np.exp(self.cap)*special.ndtr(-r) - special.ndtr(-u)
        if hasattr(self,"calibration"):
            out *= self.parts(t)[0]@self.calibration
        return self.K*np.maximum(out,0.)

    def mean(self):
        a = np.unique(np.r_[-9.,self.breakpoints,9.])
        xx, ww = special.roots_legendre(58)
        t = ((a[:-1]+a[1:])[:,None]+np.diff(a)[:,None]*xx)*.5
        mass = np.diff(a)[:,None]*.5*ww[None,:]*np.exp(-.5*t*t)/math.sqrt(2.*math.pi)
        return float(np.sum(self.marginal(t.ravel()).reshape(t.shape)*mass))
