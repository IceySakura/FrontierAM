"""Gaussian emitter fitting, model selection, and localization uncertainty."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import ctypes
import time
import numpy as np
from fitting import LIB, FP, DP
from numerical import Operator, noise, observation_moments, damped_least_squares

LIB.render_points.argtypes = [DP, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              FP, ctypes.c_int, ctypes.c_int, ctypes.c_int, FP, FP]


class Points:
    def __init__(self, psf, count, shape):
        self.count, self.shape = count, shape
        self.kernels = np.array([psf.sum(axis=tuple(j for j in range(3) if j != a)) for a in range(3)], np.float32)
        self.mu = np.empty(shape, np.float32)
        self.jac = np.empty((count*5, np.prod(shape)), np.float32)

    def render(self, p, forward=True, derivatives=False):
        LIB.render_points(p, self.count, *self.shape, self.kernels, self.kernels.shape[1]//2,
                          forward, derivatives, self.mu, self.jac)
        return (self.mu, self.jac) if derivatives else self.mu


def fit_points_fast(y, h, p0, initial, fixed=None, prior_width=10, prior_amp=5, iterations=20,
                    verbose=False, deadline=float("inf")):
    p = np.asarray(p0[:, [0,1,2,6,7]],dtype=np.float64).ravel().copy()
    count = len(p0)
    if count == 0:
        return np.zeros_like(y), p0
    model = Points(h, count, y.shape)
    op = Operator(h, y.shape, method="native")
    alpha, sigma = noise(y)
    base = 0 if fixed is None else op(fixed)
    weight = 1/np.sqrt(observation_moments(op(initial), alpha, sigma)[1])
    prior = np.tile([0,0,0,prior_width,prior_amp], count)
    target = np.tile([0,0,0,np.log(.9),np.log(.85)], count)
    lo = np.tile([0,0,0,np.log(.45),np.log(.015)], count)
    hi = np.tile([*(np.asarray(y.shape)-1),np.log(1.6),np.log(3)], count)
    p=np.clip(p,lo,hi)
    stepmax = np.tile([1.5,1.5,1.5,.4,.8], count)
    start = time.monotonic()
    def evaluate(p,derivatives):
        if derivatives:
            mu,jac=model.render(p,derivatives=True)
        else:
            mu=model.render(p)
        mean,variance,derivative=observation_moments(mu+base,alpha,sigma)
        residual=((mean-y)*weight).ravel()
        if not derivatives:
            return residual
        jac*=(derivative*weight).ravel()[None]
        return residual,jac
    p=damped_least_squares(evaluate,p,lo,hi,stepmax,prior,target,iterations=iterations,
                           deadline=deadline,max_trials=6,scale_floor=.001)
    out = model.render(p, forward=False).copy()
    pp = p0.copy()
    pp[:, [0,1,2,6,7]] = p.reshape(-1,5)
    if verbose:
        print("point fit time", time.monotonic()-start)
        print(np.column_stack([p.reshape(-1,5)[:,:3], np.exp(p.reshape(-1,5)[:,3:])]).round(3))
    return out, pp


def average_localization(y,h,points,fixed=None):
    count=len(points)
    model=Points(h,count,y.shape)
    p=points[:,[0,1,2,6,7]].copy().ravel()
    mu,jac=model.render(p,derivatives=True)
    if fixed is not None:
        mu=mu+Operator(h,y.shape,method="native")(fixed)
    alpha,sigma=noise(y)
    _,variance,derivative=observation_moments(mu,alpha,sigma)
    jac*=(derivative/np.sqrt(variance)).ravel()[None]
    matrix=(jac@jac.T).astype(np.float64)
    matrix.flat[::len(p)+1]+=np.tile([.0001,.0001,.0001,20,7],count)
    try:
        covariance=np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return model.render(p,forward=False).copy()
    samples=[]
    for k,point in enumerate(p.reshape(-1,5)):
        cov=covariance[5*k:5*k+3,5*k:5*k+3]
        values,vectors=np.linalg.eigh(cov)
        shifts=vectors*np.sqrt(3*np.clip(values,0,2))[None]
        for axis in range(3):
            for sign in [-1,1]:
                q=point.copy()
                q[:3]=np.clip(q[:3]+sign*shifts[:,axis],0,np.asarray(y.shape)-1)
                distance=np.sum((q[:3]-np.round(q[:3]))**2)
                q[4]=min(q[4],distance/(2*np.exp(2*q[3])))-np.log(6)
                samples.append(q)
    samples=np.array(samples,dtype=np.float64)
    return Points(h,len(samples),y.shape).render(samples.ravel(),forward=False).copy()


def split_candidates(y, h, x, pp, initial, verbose=False):
    alpha, sigma = noise(y)
    op = Operator(h, y.shape, method="native")
    weight = 1/observation_moments(op(initial), alpha, sigma)[1]
    mean, var, der = observation_moments(op(x), alpha, sigma)
    residual = (y-mean)*weight*der
    adjoint = op(residual, True)
    coords = np.indices(y.shape).transpose(1,2,3,0).astype(np.float32)
    candidates = []
    for k,p in enumerate(pp):
        s, a = np.exp(p[6]), np.exp(p[7])
        r = coords-p[:3]
        d2 = np.sum(r*r, axis=-1)
        g = (a*np.exp(-.5*d2/(s*s))).astype(np.float32)
        mat = np.einsum("...i,...j,...->ij", r, r, adjoint*g, optimize=True) / s**4
        mat -= np.eye(3)*np.trace(mat)/3
        eigen, vec = np.linalg.eigh(mat)
        direction = vec[:,-1]
        proj = np.sum(r*direction, axis=-1)
        delta = g*(proj*proj-d2/3)/s**4
        dmu = op(delta)*der
        numerator = np.sum((y-mean)*weight*dmu)
        denominator = np.sqrt(np.sum(dmu*dmu*weight))
        score = numerator / max(denominator, 1e-10)
        candidates.append((score,k,direction))
    candidates.sort(key=lambda z:z[0], reverse=True)
    if verbose:
        print("split scores", [(round(s,2),k,np.round(v,2).tolist()) for s,k,v in candidates])
    return candidates


def point_cost(y,h,x,pp,initial,pw=10,pa=5):
    op=Operator(h,y.shape,method="native")
    alpha,sigma=noise(y)
    w=1/observation_moments(op(initial),alpha,sigma)[1]
    r=observation_moments(op(x),alpha,sigma)[0]-y
    return .5*np.sum(r*r*w,dtype=float)+.5*pw*np.sum((pp[:,6]-np.log(.9))**2)+.5*pa*np.sum((pp[:,7]-np.log(.85))**2)


def refine_unresolved(y,h,estimate,points,initial,deadline=float("inf")):
    for attempt in range(2):
        if len(points)>10 or time.monotonic()>deadline-.14:
            break
        candidates=split_candidates(y,h,estimate,points,initial)
        if not candidates or candidates[0][0]<3.5:
            break
        score,k,direction=candidates[0]
        proposal=points.copy()
        extra=proposal[k].copy()
        proposal[k,:3]-=1.5*direction
        extra[:3]+=1.5*direction
        proposal[k,6]=extra[6]=np.log(.8)
        proposal[k,7]=extra[7]=points[k,7]+3*(points[k,6]-np.log(.8))-np.log(2)
        proposal=np.concatenate([proposal,extra[None]])
        before=point_cost(y,h,estimate,points,initial,pw=20,pa=7)
        candidate,parameters=fit_points_fast(y,h,proposal,initial,prior_width=20,prior_amp=7,
                                             iterations=25,deadline=deadline)
        if time.monotonic()>deadline-.035:
            break
        after=point_cost(y,h,candidate,parameters,initial,pw=20,pa=7)
        if before-after<=10:
            break
        estimate,points=candidate,parameters
    return estimate,points
