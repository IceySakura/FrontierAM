"""Zero-extended convolution, noise estimates, and bounded numerical solvers."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
import numpy as np
from scipy import ndimage


class Operator:
    def __init__(self, psf, shape=(32, 32, 32), method="fft"):
        self.shape = shape
        self.method = method
        self.ps = [psf.sum(axis=tuple(j for j in range(3) if j != a)) for a in range(3)]
        if method == "native":
            import ctypes
            self.lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "kernels.so"))
            fp = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
            self.lib.separable_conv.argtypes = [fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, fp, fp, fp]
            self.kernels = np.ascontiguousarray(self.ps, np.float32)
            self.kernels_adj = self.kernels[:, ::-1].copy()
            self.tmp1 = np.empty(shape, np.float32)
            self.tmp2 = np.empty(shape, np.float32)
        elif method != "sep":
            from scipy import fft
            self.fft=fft
            self.full = tuple(fft.next_fast_len(n + m - 1) for n, m in zip(shape, psf.shape))
            self.sl = tuple(slice(0, n) for n in shape)
            self.buf = np.zeros(self.full, np.float32)
            self.buf[tuple(slice(0, n) for n in psf.shape)] = psf
            h = np.roll(self.buf, tuple(-int(m // 2) for m in psf.shape), axis=(0, 1, 2))
            self.hf = fft.rfftn(h)
            self.hfc = self.hf.conj()
        self.norm = np.maximum(self(np.ones(shape, np.float32), True),1e-8)

    def __call__(self, x, adj=False):
        if self.method == "native":
            out = np.empty(self.shape, np.float32)
            self.lib.separable_conv(np.ascontiguousarray(x, np.float32), self.kernels_adj if adj else self.kernels,
                                    self.kernels.shape[1]//2, *self.shape, self.tmp1, self.tmp2, out)
            return out
        if self.method == "sep":
            for a, p in enumerate(self.ps):
                x = ndimage.convolve1d(x, p if not adj else p[::-1], axis=a, mode="constant")
            return x
        self.buf.fill(0)
        self.buf[self.sl] = x
        return self.fft.irfftn(self.fft.rfftn(self.buf) * (self.hfc if adj else self.hf), self.full)[self.sl].copy()


def noise(y):
    c = (1 / np.exp(-.5*(np.arange(-6, 7)/1.4)**2).sum())**3
    norm = ndimage.gaussian_filter(np.ones_like(y), 1.4, mode="constant") - c
    sm = (ndimage.gaussian_filter(y, 1.4, mode="constant") - c*y) / norm
    bg = y[sm < np.quantile(sm, .35)]
    pos = bg[bg > 0]
    sigma = float(np.quantile(pos, .5) / .67448975) if pos.size else 1e-6
    sigma = max(sigma,1e-6)
    # A robust first-pass distinction, independent of spatial structure.
    return (0.0125, sigma) if sigma < .015 else (1 / 22, sigma)


def rl(y, h, iters=50, method="fft", init="blur", deadline=float("inf")):
    op = Operator(h, y.shape, method=method)
    alpha, sigma = noise(y)
    v = sigma**2 / alpha
    b = sigma / np.sqrt(2 * np.pi)
    x = np.maximum(y - b, .00001)
    if init == "blur":
        x = np.maximum(op(x), .00001)
    for _ in range(iters):
        if time.monotonic() >= deadline - .025:
            break
        mu = np.maximum(op(x), 1e-8)
        bias = b * np.exp(-mu / alpha)
        rat = np.maximum(y + v - bias, 0) / (mu + v + 1e-8)
        x = x * np.maximum(op(rat, True),0) / op.norm
    return x


def observation_moments(mu, alpha, sigma):
    """Approximate moments and mean derivative after clipping noisy counts."""
    b = sigma / np.sqrt(2 * np.pi)
    e = np.exp(-np.maximum(mu, 0) / alpha)
    mean = mu + b * e
    var = alpha * np.maximum(mu, 0) + sigma**2 * (1 - .5 * e) - 2 * mu * b * e - b*b*e*e
    derivative = 1 - b / alpha * e
    return mean, np.maximum(var, sigma*sigma*.25), derivative


def nmse(x, truth):
    return np.mean((x - truth)**2) / np.mean(truth**2)


def damped_least_squares(evaluate, initial, lower, upper, maxstep, prior, target,
                        iterations=20, deadline=float("inf"),extra_precision=None,project=None,
                        max_trials=7,scale_floor=1e-4):
    p=np.asarray(initial,dtype=np.float64).copy()
    prior=np.asarray(prior,dtype=np.float64)
    damping=.01
    reserve=.03+.00002*len(p)**2
    for iteration in range(iterations):
        if time.monotonic() >= deadline-reserve:
            break
        residual,jac=evaluate(p,True)
        residual=np.asarray(residual,dtype=np.float32).ravel()
        jac=np.asarray(jac,dtype=np.float32).reshape(len(p),-1)
        grad=(jac@residual).astype(np.float64)+prior*(p-target)
        # Keep the full gradient; omit negligible columns only from curvature.
        relevant=np.max(np.abs(jac),axis=0)>.025
        compact=jac[:,relevant] if relevant.sum()<.85*jac.shape[1] else jac
        matrix=(compact@compact.T).astype(np.float64)
        matrix.flat[::len(p)+1]+=prior
        cost=.5*np.dot(residual,residual)+.5*np.sum(prior*(p-target)**2)
        if extra_precision is not None:
            correction=extra_precision@p
            cost+=.5*np.dot(p,correction)
            grad+=correction
            matrix+=extra_precision
        scale=np.maximum(np.diag(matrix),scale_floor)
        accepted=False
        for trial in range(max_trials):
            if time.monotonic() >= deadline-.025:
                break
            system=matrix.copy()
            system.flat[::len(p)+1]+=damping*scale
            try:
                delta=np.linalg.solve(system,-grad)
            except np.linalg.LinAlgError:
                damping*=10
                continue
            delta/=max(1,np.max(np.abs(delta)/maxstep))
            proposal=np.clip(p+delta,lower,upper)
            if project is not None:
                proposal=project(proposal)
            r=evaluate(proposal,False)
            newcost=.5*np.dot(r,r)+.5*np.sum(prior*(proposal-target)**2)
            if extra_precision is not None:
                newcost+=.5*np.dot(proposal,extra_precision@proposal)
            if newcost < cost:
                p=proposal
                damping=max(damping/3,1e-7)
                accepted=True
                break
            damping*=10
        if not accepted or abs(cost-newcost)<.001:
            break
    return p
