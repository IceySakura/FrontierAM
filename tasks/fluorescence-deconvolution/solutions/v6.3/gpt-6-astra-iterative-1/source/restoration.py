"""Deadline-aware selection and refinement of fluorescence structure models."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import time
import numpy as np
from fitting import fit, fit_sheets, init_tubes, detect_points, Model
from points import fit_points_fast, average_localization, refine_unresolved
from numerical import Operator, rl, noise, observation_moments


def restore(y,h,deadline=float("inf"),verbose=False):
    start=time.monotonic()
    if not np.any(y):
        return np.zeros_like(y)
    a,s=noise(y)
    marginals=[h.sum(axis=tuple(j for j in range(3) if j!=axis)) for axis in range(3)]
    separable=np.einsum("i,j,k->ijk",*marginals)
    native=(len(set(h.shape))==1 and np.max(np.abs(h-separable)) < 1e-6)
    method="native" if native else "fft"
    op=Operator(h,y.shape,method=method)
    v=sum(np.dot(p,(np.arange(len(p))-len(p)//2)**2) for p in op.ps)/3
    nit=int(np.clip((4 if s<.015 else 2.4)*v*v,5,50))
    initial=rl(y,h,iters=nit,init="blur",method=method,deadline=deadline)
    def finish(estimate):
        if estimate is None:
            return initial
        if time.monotonic()>deadline-.018:
            return estimate
        mu=op(estimate)
        mean,variance,_=observation_moments(mu,a,s)
        active=(mu>s*.5) | (op(initial)>s*.5)
        if active.sum()>128:
            discrepancy=float(np.mean(((y-mean)**2/variance)[active]))
            blend=float(np.clip((discrepancy-1.3)/.35,0,1))
            if blend:
                return (1-blend)*estimate+blend*initial
        return estimate
    if not native or time.monotonic() > deadline - .12:
        return initial
    sheets=fit_sheets(y,h,initial,verbose,deadline=deadline)
    if sheets is not None:
        predicted=op(sheets)
        initial_predicted=op(initial)
        discrepancy=np.sum((predicted-initial_predicted)**2)/max(np.sum(initial_predicted**2),1e-10)
        if discrepancy > .15:
            sheets=None
    if sheets is None:
        geo=rl(y,h,iters=10,init="blur",method="native",deadline=deadline)
        tubes=init_tubes(geo,verbose,deadline=deadline)
        if len(tubes):
            base,tubes=fit(y,h,tubes,initial,maxiter=100,verbose=verbose,deadline=deadline)
            keep=np.exp(tubes[:,7])>.1
            if not np.all(keep):
                tubes=tubes[keep]
                base=Model(len(tubes),y.shape).render(tubes.ravel()).copy() if len(tubes) else None
        else:
            base=None
    else:
        base=sheets
        tubes=np.empty((0,8))
    fixed=base
    if fixed is None:
        fixed=np.zeros_like(y)
    if time.monotonic() > deadline - .09:
        return finish(base)
    points=detect_points(y,h,base=fixed,threshold=6,verbose=verbose)
    if len(points):
        xp,points=fit_points_fast(y,h,points,initial,fixed=fixed,prior_width=20,prior_amp=7,verbose=verbose,deadline=deadline)
        base=fixed+xp
        extra=(detect_points(y,h,base=base,threshold=7,verbose=verbose)
               if time.monotonic() < deadline-.09 else np.empty((0,8)))
        if len(extra):
            ds=((extra[:,None,:3]-points[None,:,:3])**2).sum(axis=-1).min(axis=1)
            extra=extra[ds>2]
            if len(extra):
                points=np.concatenate([points,extra])
                xp,points=fit_points_fast(y,h,points,initial,fixed=fixed,prior_width=20,prior_amp=7,verbose=verbose,deadline=deadline)
                base=fixed+xp
        if not len(tubes) and sheets is None:
            base,points=refine_unresolved(y,h,base,points,initial,deadline=deadline)
        if len(tubes) and time.monotonic() < deadline-.07:
            combined=np.concatenate([tubes,points])
            base,params=fit(y,h,combined,initial,point_prior=20,amp_prior=7,maxiter=45,verbose=verbose,deadline=deadline)
            points=params[len(tubes):]
            tubes=params[:len(tubes)]
            fixed=Model(len(tubes),y.shape).render(tubes.ravel()).copy()
    if len(tubes)>1 and len(tubes)+len(points)<=10 and time.monotonic()<deadline-.16:
        combined=np.concatenate([tubes,points])
        base,params=fit(y,h,combined,initial,point_prior=20,amp_prior=7,maxiter=10,
                        amp_coupling="auto",deadline=deadline)
        points=params[len(tubes):]
        tubes=params[:len(tubes)]
        fixed=Model(len(tubes),y.shape).render(tubes.ravel()).copy()
    if 0<len(points)<=12 and time.monotonic()<deadline-.09:
        base=fixed+average_localization(y,h,points,fixed)
    if base is None:
        base=initial
    if verbose:
        print("pipeline elapsed",time.monotonic()-start)
    return finish(base)
