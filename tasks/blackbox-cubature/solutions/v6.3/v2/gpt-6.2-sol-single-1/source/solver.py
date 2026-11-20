import json
import itertools
import math
import signal
import sys
import time
import warnings

STARTED = time.monotonic()

import numpy as np
from scipy.linalg import LinAlgError

from common import AxisQuad, Deadline, NoBudget, Oracle, gl_interval, residual, stable_mean
from concentrated import focused_residual
from directions import directional_spectrum, periodic_ridges
from envelope import linear_envelope
from lowrank import tensor_model
from modulated import modulated_spectrum
from numerical import adaptive_box, weighted_qmc
from peaks import gaussian_mixture, peak_audit, rational_mixture, signed_gaussian
from periods import local_periodic_peak
from pieces import halfspace_cut, jump_trace, piecewise_grid, support_box
from ridges import (hinges, make_line, rank_one, spectral, spectral_quality,
                    _poly_breaks, _spectral_candidates)
from sparse import adaptive_sparse
from structure import axes_separable, integrate_axes, polynomial, transformed


NUMERIC_FAILURES = (NoBudget, ArithmeticError, ValueError, LinAlgError, RuntimeError)


def optional(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except NUMERIC_FAILURES:
        return None


def polynomial_cubature(orc,m,scale):
    d=orc.d
    if m.degree==1:
        return float(orc.sample(np.full((1,d),.5))[0]/scale)
    if d<=3:
        n=(m.degree+2)//2
        z,w=gl_interval(n)
        p=np.array(list(itertools.product(z,repeat=d)))
        wei=np.prod(np.array(list(itertools.product(w,repeat=d))),axis=1)
        return math.fsum(wei*(orc.sample(p)/scale))
    if m.degree<=3:
        h=np.ones((1,1))
        while len(h)<d:
            h=np.block([[h,h],[h,-h]])
        q=h[:,:d]/(2*math.sqrt(3.))
        return stable_mean(orc.sample(np.vstack((.5+q,.5-q)))/scale)
    return m.integral


def integrate(orc):
    d = orc.d
    if orc.budget < 192 or d > 20:
        orc.uniform(orc.left)
        return orc.fallback()
    x, raw = orc.uniform(min(256,1 << (max(1,orc.budget//6)-1).bit_length()))
    scale = float(np.max(np.abs(raw)))
    if np.all(raw==raw[0]):
        plateau = raw[0]
        aa = np.empty(0)
        if d<=8 and orc.left >= 2**d+90:
            aa = orc.sample(((np.arange(2**d)[:,None] >> np.arange(d)) & 1))
        if not np.any(aa!=plateau):
            a = orc.new_sobol(min(64,orc.left))
            aa = orc.sample(np.where(a > .5, 1.-(1.-a)*.15, a*.15))
            # Pause between blocks, so that a small support discovered in a
            # late block can still be localized with the remaining budget.
            if not np.any(aa!=plateau):
                reserve = min(370,70+38*d)
                if plateau!=0.:
                    reserve = max(reserve,orc.left//2)
                while orc.left > reserve+32:
                    _, vv = orc.uniform(min(192,orc.left-reserve))
                    if np.any(vv!=plateau):
                        break
                x = np.concatenate(orc.uniform_x)
                raw = np.concatenate(orc.uniform_y)
                if np.all(raw==plateau):
                    orc.uniform(orc.left)
                    return orc.fallback()
        scale = float(max(np.max(np.abs(raw)),np.max(np.abs(aa)) if len(aa) else 0.))
        if scale == 0. or orc.left < 150:
            orc.uniform(orc.left)
            return orc.fallback()
    y = raw/scale
    models = []
    def done(v,name):
        if orc.note is None:
            orc.note = name
        return v*scale
    def try_model(m, tight=2e-10, corners=False):
        if m is None:
            return None
        e, mx = residual(m,x,y)
        if e > .22*np.std(y)+1e-14:
            return None
        models.append(m)
        vr = max(1e-200,float(np.std(y)))
        if e < .045*vr and abs(m.integral) < max(3.0,4.*np.max(np.abs(y))):
            q = e/vr
            est = float(scale*(m.integral+np.mean(y-m.predict(x))))
            if math.isfinite(est) and (orc.provisional is None or q<orc.provisional[0]):
                orc.provisional=(q,est)
        if mx > tight or orc.left < 26 or (m.name=="piecewise ridge" and m.error>tight):
            return None
        # For exceptionally small integrals, require correspondingly tight
        # prediction rather than extrapolating from a low absolute RMSE.
        if mx > max(2e-13,abs(m.integral)*8e-8):
            return None
        want = 42
        if abs(m.integral) < .003 or abs(m.integral) > 2.:
            want = 90
        u = orc.new_sobol(min(want,orc.left))
        if corners and 2**d <= 256 and orc.left >= len(u)+2**d:
            co = np.arange(2**d)[:,None]
            u = np.vstack((u, ((co >> np.arange(d)) & 1),
                           np.full((1,d),.5))) if orc.left > len(u)+2**d else u
        v = orc.sample(u)/scale
        _, ee = residual(m,u,v)
        if hasattr(m,"peak_params") or m.name=="exponential quadratic":
            vv=m.predict(u)
            e_rel=np.max(np.abs(v-vv)/np.maximum(1.,np.abs(v)))
            if e_rel<max(5e-11,8*tight):
                if m.name!="exponential quadratic":
                    return m.integral
        if m.name == "exponential quadratic" and np.ptp(y) > .2:
            vv = m.predict(u)
            clear = np.maximum(np.abs(vv),np.abs(v))>1e-267
            if np.any(np.abs(v[clear]-vv[clear]) > 5e-7*np.maximum(np.abs(vv[clear]),np.abs(v[clear]))):
                return None
        if ee <= max(8*tight,min(5e-8,abs(m.integral)*8e-8)) and ee < 5e-9*max(1.,np.max(np.abs(v))):
            return m.integral
        return None
    def accept_peak(m):
        if m is None:
            return None
        if m.error < 1e-8 and optional(peak_audit,orc,m,scale):
            return optional(try_model,m,tight=2e-9)
        if m.error > 1e-8:
            optional(try_model,m,tight=0.)
        return None
    if d == 1:
        try:
            val = AxisQuad().adaptive(lambda z:orc.sample(z[:,None])/scale,
                                       lambda:orc.left, maxpts=min(orc.left,1200))[0]
            return done(val,"one-dimensional")
        except NUMERIC_FAILURES:
            return orc.fallback()
    if d <= 10:
        y0 = float(np.median(y))
        count_levels = len(np.unique(y))
        if (np.count_nonzero(y==0.) > .15*len(y)
                or np.count_nonzero(y==y0) > .4*len(y)
                or (np.ptp(y)>0. and count_levels<=3)):
            if count_levels==2:
                v = optional(halfspace_cut,orc,x,y,scale)
                if v is not None and np.isfinite(v):
                    return done(v,"halfspace support")
            v = optional(support_box,orc,scale,x,y)
            if v is not None and np.isfinite(v):
                return done(v,"rectangular support")
            if count_levels!=2:
                v = optional(halfspace_cut,orc,x,y,scale)
                if v is not None and np.isfinite(v):
                    return done(v,"halfspace support")
        for m in (optional(polynomial,x,y),optional(transformed,x,y)):
            v = optional(try_model,m,corners=True)
            if v is not None:
                if m.name=="polynomial" and residual(m,x,y)[1]<1e-12:
                    precise=optional(polynomial_cubature,orc,m,scale)
                    if precise is not None and abs(precise-v)<1e-10:
                        v=precise
                return done(v,m.name)
    if orc.left < 290 or d > 10:
        try:
            return done(weighted_qmc(orc,scale,models),"sampled fallback")
        except NUMERIC_FAILURES:
            return orc.fallback()
    line = make_line(orc,size=73)
    line_y = line[3]/scale
    max_line = float(np.max(np.abs(line[3])))
    if max_line/1e5 > scale:
        scale = max_line
        y = raw/scale
        line_y = line[3]/scale
        models.clear()
        orc.provisional = None
    kink_probes, piecewise = _poly_breaks(line[2],line_y,allow_approx=True)
    step = optional(jump_trace,orc,line,scale)
    delta = np.diff(line_y,2)
    stiff = max(1e-13,4e-12*np.max(np.abs(line_y)))
    convexity = 0.
    if (1 <= np.count_nonzero(np.abs(delta)>stiff) < .35*len(delta)
            and (np.min(delta) >= -stiff or np.max(delta) <= stiff)):
        convexity = 1. if np.min(delta) >= -stiff else -1.
    smooth_coeffs = np.polynomial.chebyshev.chebfit(line[2]*2,line_y,17)
    line_fit = np.polynomial.chebyshev.chebval(line[2]*2,smooth_coeffs)
    line_roughness = np.std(line_y-line_fit)/max(1e-15,np.std(line_y))
    flips = int(np.count_nonzero(np.diff(np.sign(np.diff(line_y)))))
    if not step and flips >= 4 and orc.left >= 420:
        m = optional(local_periodic_peak,orc,x,y,scale)
        v = optional(try_model,m)
        if v is not None:
            return done(v,m.name)
    yy = np.abs(y)
    yr = y - np.median(y)
    yh = np.abs(yr)
    ix = np.argsort(yh)[len(yh)*94//100:]
    heavy = bool(np.ptp(y)>8e-15 and np.sum(yh[ix]**2)>.50*np.sum(yh**2))
    concentrated = bool(np.median(yy) < .045*np.max(yy) and
                        (np.min(y) >= -1e-15 or np.max(y) <= 1e-15))
    curve = line_y*np.sign(y[np.argmax(yy)])
    core = np.maximum(np.ptp(curve),1e-260)
    v0=curve[1:-1]
    local_peaks=int(np.sum((v0>curve[:-2])&(v0>=curve[2:])&
                           (v0>np.min(curve)+.10*core)))
    tried = set()
    if heavy and not step and local_peaks<=3 and orc.left>=530:
        options = []
        if concentrated:
            options.extend((gaussian_mixture,rational_mixture))
        if np.ptp(y) < .72*np.max(yy) or (y.min()<0. and y.max()>0.):
            options.append(signed_gaussian)
        for fit in options:
            if orc.left<245 or orc.elapsed>9.:
                break
            tried.add(fit)
            m=optional(fit,orc,x,y,scale)
            v=accept_peak(m)
            if v is not None:
                return done(v,m.name)
    roots = _spectral_candidates(line[2],line_y)
    ridge_first = piecewise and not step and bool(
        kink_probes or (not concentrated and np.ptp(line_y) > 2e-6))
    if roots and not ridge_first and not step:
        m = optional(spectral,orc,line,x,y,scale,roots)
        v = optional(try_model,m)
        if v is not None:
            return done(v,m.name)
        if orc.left >= 400:
            mm = optional(modulated_spectrum,orc,line,x,y,scale,roots)
            vv = optional(try_model,mm)
            if vv is not None:
                return done(vv,mm.name)
        evidence = optional(spectral_quality,line[2],line_y,roots)
        informative = (line_roughness > 2e-5 or flips>=8 or
                       (np.min(y)<-.03*np.max(np.abs(y)) and
                        np.max(y)>.03*np.max(np.abs(y))))
        if evidence is not None and evidence < 7e-6 and informative:
            for j in range(2):
                if orc.left < 620 or orc.elapsed > 6:
                    break
                if j and d >= 4:
                    m = optional(directional_spectrum,orc,x,y,scale)
                    v = optional(try_model,m)
                    if v is not None:
                        return done(v,m.name)
                    if orc.left < 620:
                        break
                sg = (-1.)**np.arange(d) if not j else orc.rng.choice([-1.,1.],d)
                radius = orc.rng.uniform(.335,.407 if j else .435,d)
                ctr = .5 + orc.rng.uniform(-.035,.035,d)
                alt = make_line(orc,endpoints=(ctr-sg*radius,ctr+sg*radius),size=(51 if j else 89))
                m = optional(spectral,orc,alt,x,y,scale,ensemble=bool(j))
                v = optional(try_model,m)
                if v is not None:
                    return done(v,m.name)
    if ridge_first:
        if convexity and orc.left >= 420:
            mm = optional(linear_envelope,orc,x,y,scale,convexity,8 if d<=5 else 6)
            vv = optional(try_model,mm,corners=True)
            if vv is not None:
                return done(vv,mm.name)
        m = optional(hinges,orc,line,x,y,scale)
        v = optional(try_model,m,corners=True)
        if v is not None:
            return done(v,m.name)
    arg = optional(axes_separable,orc,x,y,scale)
    if arg is not None:
        v = optional(integrate_axes,orc,arg,scale)
        if v is not None and np.isfinite(v):
            return done(v,"separable sections")
    if step and orc.left >= 325:
        v = optional(piecewise_grid,orc,x,y,scale)
        if v is not None and np.isfinite(v):
            return done(v,"piecewise grid")
    if orc.left >= 250:
        v = optional(rank_one,orc,x,y,scale)
        if v is not None and np.isfinite(v):
            return done(v,"single ridge")
    if (not step and not piecewise and orc.left>=620 and orc.elapsed<8 and flips>=7
            and np.max(np.abs(y))>3*np.median(np.abs(y))):
        m=optional(periodic_ridges,orc,x,y,scale)
        v=optional(try_model,m,tight=8e-10)
        if v is not None:
            return done(v,m.name)
    if not ridge_first and not step and orc.left >= 340:
        fit = np.polynomial.chebyshev.chebfit(line[2]*2,line_y,3)
        poly_trace = np.max(np.abs(np.polynomial.chebyshev.chebval(line[2]*2,fit)-line_y)) <= 5e-11
        if kink_probes or (piecewise and np.ptp(line_y)>1e-11) or (poly_trace and np.std(y)>1e-9):
            m = optional(hinges,orc,line,x,y,scale)
            v = optional(try_model,m,corners=True)
            if v is not None:
                return done(v,m.name)
    y_peak = (np.min(np.abs(y)) < .52*np.max(np.abs(y)) or
              np.ptp(y) > .5*np.max(np.abs(y)))
    if orc.left >= 250 and orc.elapsed < 12 and (y_peak or heavy) and not step:
        for fitting in (gaussian_mixture,rational_mixture,signed_gaussian):
            if orc.left < 210 or orc.elapsed > 13:
                break
            if fitting in tried or (not y_peak and fitting is not signed_gaussian):
                continue
            m = optional(fitting,orc,x,y,scale)
            mval=accept_peak(m)
            if mval is not None:
                return done(mval,m.name)
    if orc.left >= 580 and d >= 3 and orc.elapsed < 13 and not step and line_roughness < .004:
        m = optional(tensor_model,orc,scale,x,y)
        v = optional(try_model,m,tight=8e-10)
        if v is not None:
            return done(v,m.name)
    if orc.left >= 350 and not step and orc.elapsed < 16:
        model = min(models,key=lambda m:residual(m,x,y)[0],default=None)
        if d == 2 and not kink_probes and line_roughness < .004:
            v = optional(adaptive_box,orc,scale,model,x,y)
            if v is not None:
                return done(v,"adaptive boxes")
        if d <= 8 and (line_roughness < .002 or model is not None):
            spend = orc.left-min(148,orc.left//4)
            m = optional(adaptive_sparse,orc,scale,x,y,model,spend)
            v = optional(try_model,m,tight=3e-10)
            if v is not None:
                return done(v,m.name)
    if orc.left and orc.elapsed < 22:
        if not step and orc.left>=245:
            v = optional(focused_residual,orc,scale,x,y,models)
            if v is not None:
                return done(v,"concentrated residual")
        v = optional(weighted_qmc,orc,scale,models)
        if v is not None:
            return done(v,"residual cubature")
    if orc.left and orc.elapsed < 24:
        orc.uniform(orc.left)
    return orc.best_effort()


def main():
    init = json.loads(sys.stdin.readline())
    if init.get("type") != "init":
        raise ValueError("missing initialization")
    o = Oracle(init,STARTED)
    signal.signal(signal.SIGALRM,o.on_alarm)
    signal.setitimer(signal.ITIMER_REAL,max(.05,24.-o.elapsed))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            try:
                v = integrate(o)
            except Deadline:
                raise
            except Exception as exc:
                # A model failure must not spoil a valid, completed point set.
                print(f"integration fallback: {type(exc).__name__}: {exc}",file=sys.stderr)
                if o.left and o.elapsed < 22:
                    o.uniform(o.left)
                v = o.best_effort()
        except Deadline:
            v = o.best_effort()
        finally:
            signal.setitimer(signal.ITIMER_REAL,0.)
        if not math.isfinite(v):
            v = o.fallback()
        v *= o.volume
        if not math.isfinite(v):
            v = math.copysign(sys.float_info.max,v)
    print(json.dumps({"type":"result","estimate":float(v)},allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
