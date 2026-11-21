"""Matching-pursuit initialization and constrained geometric reconstruction."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import ctypes
import time
import numpy as np
from scipy import ndimage
from numerical import Operator, noise, observation_moments, damped_least_squares

LIB = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "kernels.so"))
FP = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
DP = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
LIB.render_model.argtypes = [DP, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, FP, FP]
LIB.separable_conv.argtypes = [FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, FP, FP, FP]
LIB.score_lines.argtypes = [DP,DP,DP,DP,ctypes.c_int,DP,DP,ctypes.c_int,DP,DP]
LIB.model_jacobian.argtypes = [DP,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,FP,FP]
LIB.batch_convolve.argtypes = [FP,FP,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,FP,FP,FP]


def convolve_separable(x, kernels):
    x = np.ascontiguousarray(x, np.float32)
    kernels = np.ascontiguousarray(kernels, np.float32)
    t1, t2, out = np.empty_like(x), np.empty_like(x), np.empty_like(x)
    LIB.separable_conv(x, kernels, kernels.shape[1]//2, *x.shape, t1, t2, out)
    return out


class Model:
    def __init__(self, count, shape):
        self.count = count
        self.shape = shape
        self.x = np.zeros(shape, dtype=np.float32)
        self.parts = np.zeros((count,) + shape, dtype=np.float32)

    def render(self, params):
        LIB.render_model(params, self.count, *self.shape, self.x, self.parts)
        return self.x

def fit(y, h, p0, initial, point_prior=10, amp_prior=0, maxiter=30, verbose=False,
        fixed=None, deadline=float("inf"), amp_coupling=0):
    alpha, sigma = noise(y)
    op = Operator(h, y.shape, method="native")
    n = len(p0)
    ispoint = np.linalg.norm(p0[:, 3:6], axis=1) < .01
    model = Model(n, y.shape)
    lo = np.tile([-2, -2, -2, -25, -25, -25, np.log(.45), np.log(.01)], (n, 1))
    hi = np.tile([33, 33, 33, 25, 25, 25, np.log(2.5), np.log(3)], (n, 1))
    lo[ispoint, 3:6] = hi[ispoint, 3:6] = 0
    lo[ispoint,:3]=0
    hi[ispoint,:3]=np.asarray(y.shape)-1
    hi[ispoint, 6] = np.log(1.6)
    w2 = 1 / observation_moments(op(initial), alpha, sigma)[1]
    prior = np.where(ispoint, point_prior, 0)
    ap = np.where(ispoint, amp_prior, 0)
    meanwidth = np.where(ispoint, np.log(.9), 0)
    fixed_blur = 0 if fixed is None else op(fixed)
    fullp=np.clip(np.asarray(p0,dtype=np.float64).ravel(),lo.ravel(),hi.ravel())
    active=np.flatnonzero(lo.ravel()!=hi.ravel())
    jac_source=np.empty((n*8,)+y.shape,np.float32)
    jac_forward=np.empty((len(active),)+y.shape,np.float32)
    sqrtw=np.sqrt(w2)
    priors=np.zeros((n,8))
    priors[:,6]=prior
    priors[:,7]=ap
    targets=np.zeros((n,8))
    targets[:,6]=meanwidth
    targets[:,7]=np.log(.85)
    steps=np.tile([2,2,2,2,2,2,.4,.8],n)
    coupling=None
    filament_ids=np.flatnonzero(np.linalg.norm(p0[:,3:6],axis=1)>2)
    if amp_coupling=="auto":
        if len(filament_ids)>1:
            spread=float(np.var(p0[filament_ids,7],ddof=1))
            amp_coupling=min(100,1/(spread+.01))
        else:
            amp_coupling=0
    if amp_coupling and len(filament_ids)>1:
        positions=np.searchsorted(active,8*filament_ids+7)
        coupling=np.zeros((len(active),len(active)))
        coupling[np.ix_(positions,positions)]=amp_coupling*(np.eye(len(positions))-1/len(positions))

    def evaluate(p,derivatives):
        fullp[active]=p
        clean=model.render(fullp)
        mu=np.maximum(op(clean)+fixed_blur,0)
        mean,var,derivative=observation_moments(mu,alpha,sigma)
        residual=((mean-y)*sqrtw).ravel()
        if not derivatives:
            return residual
        LIB.model_jacobian(fullp,n,*y.shape,model.parts,jac_source)
        source=jac_source if len(active)==n*8 else jac_source[active]
        LIB.batch_convolve(source,op.kernels,op.kernels.shape[1]//2,len(active),*y.shape,
                           op.tmp1,op.tmp2,jac_forward)
        jac=jac_forward.reshape(len(active),-1)
        jac*=(derivative*sqrtw).ravel()[None]
        return residual,jac

    start=time.monotonic()
    optimized=damped_least_squares(evaluate,fullp[active],lo.ravel()[active],hi.ravel()[active],
                                  steps[active],priors.ravel()[active],targets.ravel()[active],
                                  iterations=min(30,maxiter),deadline=deadline,extra_precision=coupling)
    fullp[active]=optimized
    if verbose:
        print("geometric fit time",time.monotonic()-start)
        print(np.column_stack([fullp.reshape(-1,8)[:,:6],np.exp(fullp.reshape(-1,8)[:,6:])]).round(3))
    return model.render(fullp).copy(),fullp.reshape(n,8)


def detect_points(y, h, base=None, threshold=5, exclude=None, verbose=False):
    alpha, sigma = noise(y)
    op = Operator(h, y.shape, method="native")
    mu = np.zeros_like(y) if base is None else op(base)
    mean, variance, derivative = observation_moments(mu, alpha, sigma)
    g = np.exp(-.5*(np.arange(-3, 4)/.9)**2)
    kernels = np.array([np.convolve(p, g) for p in op.ps], np.float32)
    w = 1 / variance
    cross = convolve_separable((y - mean)*w, kernels[:, ::-1])
    norm = np.sqrt(np.maximum(convolve_separable(w, kernels[:, ::-1]**2), 1e-6))
    score = cross / norm
    candidates = (score == ndimage.maximum_filter(score, size=5)) & (score > threshold)
    if exclude is not None and len(exclude):
        coords = np.indices(y.shape).transpose(1,2,3,0).astype(np.float32)
        for p in exclude:
            r = coords-p[:3]
            t = np.clip(np.sum(r*p[3:6], axis=-1) / max(1e-8, np.sum(p[3:6]**2)), -1, 1)
            d2 = np.sum((r - t[..., None]*p[3:6])**2, axis=-1)
            candidates[d2 < (3.5*np.exp(p[6]))**2] = False
    peaks = np.argwhere(candidates)
    peaks = peaks[np.argsort(score[tuple(peaks.T)])[::-1]][:20]
    p0 = np.zeros((len(peaks), 8))
    p0[:, :3] = peaks
    p0[:, 6] = np.log(.9)
    amps = np.maximum(cross[tuple(peaks.T)] / norm[tuple(peaks.T)]**2, .1)
    p0[:, 7] = np.log(np.minimum(amps, 2))
    if verbose:
        print("detected points", len(p0), "scores", score[tuple(peaks.T)].round(1), "peaks", peaks.tolist())
    return p0


def init_tubes(initial, verbose=False, deadline=float("inf"), subtraction=True):
    sm = ndimage.gaussian_filter(initial, .65)
    coords = np.indices(initial.shape).reshape(3, -1).T.astype(float)
    fullcoords=coords.copy()
    remaining=sm.ravel().copy()
    val = sm.ravel().copy()
    cutoff = max(.035, sm.max() * .1)
    mask = val > cutoff
    indices=np.flatnonzero(mask)
    coords, val = coords[mask], val[mask]
    hess = np.empty((len(coords), 3, 3))
    for a in range(3):
        for b in range(a, 3):
            order = [0, 0, 0]
            order[a] += 1
            order[b] += 1
            entry = ndimage.gaussian_filter(initial, 1.0, order=order).ravel()[mask]
            hess[:, a, b] = hess[:, b, a] = entry
    eig, vec = np.linalg.eigh(hess)
    directions = vec[:, :, -1]
    confidence = np.clip((eig[:, 2]-eig[:, 1]) / (np.abs(eig[:, 0])+1e-8), 0, 1)
    rng = np.random.default_rng(83)
    objects = []
    for k in range(6):
        if time.monotonic() >= deadline - (.05+.000002*len(coords)):
            break
        # Fit dominant straight ridges, then remove their contribution.
        if len(val) == 0 or val.sum() < 2 or val.max() < cutoff:
            break
        prob = val**2
        prob /= prob.sum()
        ids = rng.choice(len(val), 150, p=prob)
        a = coords[ids]
        b = coords[rng.choice(len(val), 150, p=prob)]
        v = a - b
        length = np.linalg.norm(v, axis=1)
        valid = length > 5
        a, b, v, length = a[valid], b[valid], v[valid], length[valid]
        if len(a) == 0:
            break
        v /= length[:, None]
        a = np.concatenate([a, coords[ids[:60]]])
        v = np.concatenate([v, directions[ids[:60]]])
        scores = np.empty(len(a),np.float64)
        ranges = np.empty((len(a),2),np.float64)
        LIB.score_lines(np.ascontiguousarray(coords),np.ascontiguousarray(val,np.float64),
                        np.ascontiguousarray(directions),np.ascontiguousarray(confidence),len(coords),
                        np.ascontiguousarray(a),np.ascontiguousarray(v),len(a),scores,ranges)
        kbest=int(np.argmax(scores))
        bestscore=scores[kbest]
        if bestscore < 0:
            break
        aa, vv = a[kbest],v[kbest]
        seglo,seghi=ranges[kbest]
        region = ((coords-aa) @ vv > seglo-2) & ((coords-aa) @ vv < seghi+3)
        for _ in range(3):
            r = coords - aa
            t = r @ vv
            d2 = np.maximum(np.sum(r*r, axis=1) - t*t, 0)
            alignment = (directions @ vv)**4
            w = val * np.exp(-d2 / 3) * region * (confidence*alignment + .1)
            aa = np.sum(coords*w[:, None], axis=0) / w.sum()
            r = coords-aa
            cov = (r.T*w) @ r / w.sum()
            eig, vec = np.linalg.eigh(cov)
            vv = vec[:, -1]
        t = (coords-aa) @ vv
        order = np.argsort(t)
        cw = np.cumsum(w[order])
        bounds = np.interp([.05*cw[-1], .95*cw[-1]], cw, t[order])
        center = aa + np.mean(bounds) * vv
        half = max(.1, (bounds[1]-bounds[0])*.5 - .3)
        radius = np.sqrt(max(.4, (eig[0]+eig[1])*.5))
        if half < 1.8 or bestscore < 5:
            break
        amp = np.percentile(val[d2 < 1.5], 75) if np.any(d2 < 1.5) else val.max()
        subtract_radius=max(1,radius)
        subtract_amp=amp
        if subtraction:
            rr=fullcoords-center
            axial=rr@vv
            radial=np.maximum(np.sum(rr*rr,axis=1)-axial*axial,0)
            middle=np.abs(axial)<max(half*.7,1)
            radial_bins=np.floor(radial).astype(int)
            bx,by=[],[]
            for b in range(12):
                samples=remaining[middle & (radial_bins==b)]
                if len(samples)>=5:
                    intensity=np.quantile(samples,.4)
                    if intensity>max(.002,cutoff*.15):
                        bx.append(b+.5)
                        by.append(np.log(intensity))
            if len(bx)>=4:
                slope,intercept=np.polyfit(bx,by,1)
                if slope<-.06:
                    subtract_radius=float(np.clip(np.sqrt(-.5/slope),.7,2.5))
                    subtract_amp=float(np.clip(np.exp(intercept),.03,2))
                    radius=np.sqrt(max(.5,subtract_radius**2-.65**2))
                    amp=subtract_amp*subtract_radius**2/max(.5,radius**2)
        objects.append(np.r_[center, vv*half, np.log(max(.65, radius)), np.log(max(.1, amp))])
        r = coords-center
        t = np.clip((r@vv) / half, -1, 1) * half
        d2 = np.sum((r-t[:, None]*vv)**2, axis=1)
        if subtraction:
            rr=fullcoords-center
            axial=np.clip(rr@vv,-half-.7,half+.7)
            radial=np.sum((rr-axial[:,None]*vv)**2,axis=1)
            remaining=np.maximum(remaining-1.05*subtract_amp*np.exp(-radial/(2*subtract_radius**2)),0)
            val=remaining[indices]
        else:
            val[d2 < (3.0*max(1, radius))**2] = 0
        keep = val > cutoff
        coords, val = coords[keep], val[keep]
        indices=indices[keep]
        directions, confidence = directions[keep], confidence[keep]
        if subtraction and len(indices) and k<4 and time.monotonic()<deadline-.035:
            field=remaining.reshape(initial.shape).astype(np.float32)
            hess=np.empty((len(indices),3,3))
            for a in range(3):
                for b in range(a,3):
                    order=[0,0,0]
                    order[a]+=1
                    order[b]+=1
                    entry=ndimage.gaussian_filter(field,1.0,order=order).ravel()[indices]
                    hess[:,a,b]=hess[:,b,a]=entry
            eig,vec=np.linalg.eigh(hess)
            directions=vec[:,:,-1]
            confidence=np.clip((eig[:,2]-eig[:,1])/(np.abs(eig[:,0])+1e-8),0,1)
    if verbose:
        print("tube initial", np.array(objects).round(3))
    return np.array(objects).reshape(-1, 8)


class SheetModel:
    def __init__(self, h, axes, shape):
        self.axes = axes
        self.shape = shape
        self.ps = [h.sum(axis=tuple(j for j in range(3) if j != a)).astype(float) for a in range(3)]
        self.coords = [np.arange(n, dtype=float) for n in shape]
        self.linear = [2*c/(len(c)-1)-1 for c in self.coords]

    def render(self, params, forward=True, derivatives=False):
        total = np.zeros(self.shape, dtype=float)
        jac = []
        for axis, p in zip(self.axes, params.reshape(-1, 5)):
            c, s, amp, slope1, slope2 = p[0], np.exp(p[1]), np.exp(p[2]), p[3], p[4]
            others = [a for a in range(3) if a != axis]
            d = (self.coords[axis]-c)/s
            g = np.exp(-.5*d*d)
            gg = [g, g*d/s, g*d*d]
            basis = []
            for a in range(3):
                aa = gg if a == axis else [np.ones(self.shape[a]), self.linear[a]]
                if forward:
                    aa = [ndimage.convolve1d(v, self.ps[a], mode="constant") for v in aa]
                basis.append(aa)
            parts = []
            for term in [None] + others:
                v = [basis[a][1 if a == term else 0] for a in range(3)]
                parts.append(amp*np.einsum("i,j,k->ijk", *v))
            current = parts[0]+slope1*parts[1]+slope2*parts[2]
            total += current
            if derivatives:
                for der in [1, 2]:
                    component = np.zeros(self.shape)
                    for term, coef in zip([None]+others, [1, slope1, slope2]):
                        v = [basis[a][der if a == axis else (1 if a == term else 0)] for a in range(3)]
                        component += amp*coef*np.einsum("i,j,k->ijk", *v)
                    jac.append(component)
                jac.extend([current, parts[1], parts[2]])
        return (total, np.array(jac)) if derivatives else total


def fit_sheets(y, h, initial, verbose=False, deadline=float("inf")):
    alpha, sigma = noise(y)
    axes, p0, candidates = [], [], []
    for a in range(3):
        profile = initial.mean(axis=tuple(j for j in range(3) if j != a))
        coord = np.arange(len(profile))
        low = profile <= np.quantile(profile, .65)
        trend = np.polyval(np.polyfit(coord[low], profile[low], 1), coord)
        residual = profile-trend
        peaks = np.flatnonzero((residual==ndimage.maximum_filter1d(residual,5)) & (residual>.01))
        for p in peaks:
            candidates.append((float(residual[p]),a,int(p)))
    if not candidates or max(c[0] for c in candidates) < .22*initial.max():
        return None
    for amp,a,p in sorted(candidates,reverse=True):
        if amp > .055*initial.max() and len(axes)<5:
            axes.append(a)
            p0.append([p,np.log(.9),np.log(amp),0,0])
    p0 = np.array(p0).ravel()
    op = Operator(h, y.shape, method="native")
    model = SheetModel(h, axes, y.shape)
    w = 1 / np.sqrt(observation_moments(op(initial), alpha, sigma)[1])
    lo = np.tile([-2, np.log(.35), np.log(.01), -.65, -.65], len(axes))
    hi = np.tile([33, np.log(3), np.log(3), .65, .65], len(axes))
    def calc(p,derivatives):
        if derivatives:
            mu,jac=model.render(p,derivatives=True)
        else:
            mu=model.render(p)
        mean,variance,deriv=observation_moments(mu,alpha,sigma)
        r=np.asarray((mean-y)*w,np.float32).ravel()
        if not derivatives:
            return r
        j=np.asarray(jac*(deriv*w)[None],np.float32).reshape(-1,y.size)
        return r,j
    t = time.monotonic()
    def project(p):
        values=p.reshape(-1,5)
        slopes=values[:,3:5]
        slopes/=np.maximum(1,np.sum(np.abs(slopes),axis=1)/.95)[:,None]
        return p
    result=damped_least_squares(calc,p0,lo,hi,np.tile([2,.4,.8,.3,.3],len(axes)),
                                np.zeros_like(p0),np.zeros_like(p0),iterations=20,deadline=deadline,project=project)
    if verbose:
        print("sheets", axes, "params", result.reshape(-1,5).round(3), "time", time.monotonic()-t)
    return np.maximum(model.render(result, forward=False), 0).astype(np.float32)
