"""A Gaussian random-walk likelihood model using observed rejection steps."""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gamma, ive, ndtr, roots_genlaguerre
from scipy.stats import chi2, ncx2
from rejection_check import CheckTimeout, shape_weight


def rejection_probability(kappa, radius2, dimension, details=False):
    z = np.maximum(radius2, 1e-12)
    argument = z/kappa
    first = ncx2.cdf(argument, dimension, argument)
    log_survival = ncx2.logsf(z*(1+kappa)/kappa, dimension, z/(kappa*(1+kappa)))
    tilted = np.exp(np.minimum(
        0., -.5*dimension*np.log1p(kappa)+z*kappa/(2*(1+kappa))+log_survival
    ))
    # Bessel recurrences give the derivatives of the tilted noncentral-chi-square tail.
    iv = ive(dimension/2-1, argument)
    iv1 = ive(dimension/2, argument)
    derivative_z = (kappa*tilted-iv1)/(2*(1+kappa))
    derivative_k = (
        -z*iv/(2*kappa*(1+kappa))
        +z*iv1/(2*(1+kappa)**2)
        +(-dimension/(2*(1+kappa))+z/(2*(1+kappa)**2))*tilted
    )
    probability = np.clip(1-first-tilted, 1e-12, 1-1e-12)
    if details:
        return probability, -derivative_z, -derivative_k, .5*(tilted+iv1)/(1+kappa)
    return probability, -derivative_z, -derivative_k


def integrated_time(kappa, dimension):
    if kappa < .0025:
        nodes, weights = roots_genlaguerre(64, dimension/2-1)
        weights /= gamma(dimension/2)
        c = np.dot(
            weights, 4*nodes*ndtr(-.5*np.sqrt(2*kappa*nodes))
        )/dimension
        return 4/(kappa*c)-1
    # The coordinate Poisson equation reduces to its first spherical harmonic.
    maximum = max(9., np.sqrt(dimension)+6)
    size = min(512, max(128, int(np.ceil(maximum/min(.05, np.sqrt(kappa)/4)))))
    spacing = maximum/size
    r = (np.arange(size)+.5)*spacing
    nu = dimension/2-1
    argument = r[:, None]*r[None, :]/kappa
    bessel = ive(nu, argument)
    angular = ive(nu+1, argument)/bessel
    kernel = np.exp(
        -.5*(r[:, None]-r[None, :])**2/kappa
        +(nu+1)*np.log(r)[None, :]-nu*np.log(r)[:, None]
        -np.log(kappa)+np.minimum(0., .5*(r[:, None]**2-r[None, :]**2))
    )*bessel*spacing
    operator = np.diag(kernel.sum(axis=1))-kernel*angular
    g = np.linalg.solve(operator, r)
    weight = r**(dimension-1)*np.exp(-.5*r*r)
    weight /= weight.sum()
    return 2*np.dot(weight, r*g)/dimension-1


def normality_weight(radial, h, n, d):
    threshold = chi2.ppf(.5, d)
    score = (
        (radial <= threshold).astype(float)-.5
        +threshold*chi2.pdf(threshold, d)/d*(radial-d)
    )
    excess = score.mean()
    if excess <= 0:
        return 1.
    score -= excess
    cumulative = np.r_[0., np.cumsum(score)]
    bandwidth = min(n//20, max(8, int(np.linalg.eigvalsh(h)[-1])))
    variance = []
    for width in (bandwidth, 2*bandwidth):
        batch = cumulative[width:]-cumulative[:-width]
        variance.append(batch@batch/(width*(n-width+1)*n))
    uncertainty = max(variance[1], 2*variance[1]-variance[0], 1e-12)
    return float(np.clip((2.58-excess/np.sqrt(uncertainty))/.62, 0., 1.))


class IsotropicMetropolis:
    def __init__(self, y, deadline=np.inf):
        self.n, self.d = y.shape
        first, last = y[:-1], y[1:]
        delta = last-first
        hold = np.all(delta == 0, axis=1)
        self.first, self.last, self.delta = first[~hold], last[~hold], delta[~hold]
        self.number = len(self.delta)
        self.scatter = self.delta.T@self.delta
        self.rejected = first[hold]
        self.rows, self.columns = np.tril_indices(self.d)
        self.diagonal = self.rows == self.columns
        fraction = max(1-self.number/(self.n-1), .02)
        self.scale = 1/np.sqrt(fraction)
        initial_q = self.scatter/self.number*(1+fraction/self.d)
        self.proposal_root = np.linalg.cholesky(initial_q)
        self.initial_kappa = np.trace(initial_q)/self.d
        self.temperature = .01*np.sqrt(self.initial_kappa*self.d)
        self.deadline = deadline

    def unpack(self, parameters):
        mean = parameters[:self.d]*self.scale
        kappa = np.exp(parameters[self.d]*self.scale)
        values = parameters[self.d+1:].copy()
        values[self.diagonal] = np.exp(values[self.diagonal])
        factor = np.zeros((self.d, self.d))
        factor[self.rows, self.columns] = values
        return mean, kappa, self.proposal_root@factor

    def objective(self, parameters):
        if time.perf_counter() > self.deadline:
            raise CheckTimeout()
        mean, kappa, factor = self.unpack(parameters)
        q = factor@factor.T
        qinv = np.linalg.inv(q)
        first, last = self.first-mean, self.last-mean
        raw_energy = .5*np.sum((self.delta@qinv)*(first+last), axis=1)
        energy = kappa*raw_energy
        logaccept = self.temperature*np.logaddexp(0., energy/self.temperature)
        uphill = expit(energy/self.temperature)
        likelihood = (
            -self.number*np.log(np.diag(factor)).sum()
            -.5*np.trace(qinv@self.scatter)-logaccept.sum()
        )
        contrast = last.T@(uphill[:, None]*last)-first.T@(uphill[:, None]*first)
        grad_q = .5*(
            qinv@self.scatter@qinv-self.number*qinv+kappa*qinv@contrast@qinv
        )
        grad_k = -np.dot(uphill, raw_energy)
        grad_mean = kappa*(np.sum(uphill[:, None]*self.delta, axis=0)@qinv)
        x = self.rejected-mean
        xx = x@qinv
        radius2 = kappa*np.sum(xx*x, axis=1)
        probability, dz, dk = rejection_probability(kappa, radius2, self.d)
        likelihood += np.log(probability).sum()
        weight = 1/probability
        radial = weight*dz
        grad_q -= kappa*xx.T@(radial[:, None]*xx)
        grad_mean -= 2*kappa*np.sum(radial[:, None]*xx, axis=0)
        grad_k += np.dot(weight, dk+dz*radius2/kappa)
        qq = (self.proposal_root.T@(grad_q+grad_q.T)@factor)[self.rows,self.columns]
        qq[self.diagonal] *= np.diag(factor)/np.diag(self.proposal_root)
        gradient = np.r_[grad_mean*self.scale, grad_k*kappa*self.scale, qq]
        return -likelihood/self.n, -gradient/self.n

    def fit(self):
        initial = np.zeros(self.d+1+len(self.rows))
        initial[self.d] = np.log(self.initial_kappa)/self.scale
        bounds = [(-4/self.scale, 4/self.scale)]*self.d
        bounds += [(
            np.log(max(1e-5, self.initial_kappa/10))/self.scale,
            np.log(min(9., self.initial_kappa*10))/self.scale,
        )]
        bounds += [(-.7, .7) if diagonal else (-.5, .5) for diagonal in self.diagonal]
        result = minimize(
            self.objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 100, "maxls": 25, "ftol": 1e-10, "gtol": 1e-6},
        )
        mean, kappa, factor = self.unpack(result.x)
        if not np.isfinite(result.fun) or np.linalg.norm(result.jac) > .15:
            return None
        return factor@factor.T/kappa, kappa, mean


def estimate_rejections(y, h, pooling):
    deadline = time.perf_counter()+2.5
    n, d = y.shape
    if pooling is None or pooling.get("kind") != "joint":
        return h
    factor = pooling["factor"]
    if factor >= .5 or np.trace(pooling["original"])/d < .6:
        return h
    difference = np.diff(y, axis=0)
    moving = np.any(difference != 0, axis=1)
    fraction = 1-moving.mean()
    if fraction < .03 or fraction > .9 or moving.sum() < 50*d:
        return h
    radial = np.sum(y*y, axis=1)
    if np.mean(radial*radial)/(d*(d+2)) > 1.5:
        return h
    normal_weight = normality_weight(radial, h, n, d)
    if normal_weight == 0:
        return h
    increments = difference[moving]
    increment_covariance = increments.T@increments/len(increments)
    try:
        radius = np.sum(increments*np.linalg.solve(increment_covariance, increments.T).T, axis=1)
    except np.linalg.LinAlgError:
        return h
    if np.mean(radius*radius)/(d*(d+2)) > 1.2:
        return h
    try:
        fit = IsotropicMetropolis(y, deadline=deadline).fit()
        if fit is None:
            return h
        covariance, kappa, mean = fit
        eigen = np.linalg.eigvalsh(covariance)
        if eigen[0] < .05 or eigen[-1] > 20:
            return h
        autocorrelation_time = integrated_time(kappa, d)
        confidence = None
        try:
            confidence = shape_weight(y, fit, deadline, rejection_probability)
        except (CheckTimeout, np.linalg.LinAlgError, ValueError, FloatingPointError):
            confidence = None
        model = covariance*autocorrelation_time
        if not np.isfinite(model).all():
            return h
        eigen_h, vectors = np.linalg.eigh(h)
        inverse = (vectors/np.sqrt(np.maximum(eigen_h, 1e-15)))@vectors.T
        relative = np.linalg.eigvalsh(inverse@model@inverse)
        if relative[0] < .05 or relative[-1] > 20:
            return h
        weight = (1-2*factor)*normal_weight
        if confidence is not None:
            weight = confidence*normal_weight
        result = (1-weight)*h+weight*model
        return (result+result.T)*.5
    except (CheckTimeout, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return h
