"""Analytic Gaussian Metropolis energy via Hermite and Gaussian integrals."""

from functools import lru_cache
from itertools import product
from math import comb, factorial, gamma

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite_e import herme2poly


@lru_cache(maxsize=4)
def rule(n):
    x, w = leggauss(n)
    return (x+1)*np.pi/4, w*np.pi/4


def compositions(n, k):
    if k == 1:
        yield (n,)
    else:
        for j in range(n+1):
            for right in compositions(n-j,k-1):
                yield (j,*right)


@lru_cache(maxsize=8)
def coefficient_table(degree):
    # c[p] weights Phi(-r/2) and phi(r/2)/r^(2j+1) in the even
    # derivatives of the midpoint generating function.
    c = np.zeros((degree,degree))
    c[0,0] = 1.
    for p in range(1,degree):
        l = 2*p
        c[p,0] = 4.0**(-p)
        for k in range(1,l+1):
            h = np.zeros(k)
            h[-1] = 1
            v = herme2poly(h)
            for j, val in enumerate(v):
                exponent = l-k+j
                if val:
                    idx = (2*p-exponent+1)//2
                    c[p,idx] += val*comb(l,k)*(-1)**(l-k)*2.0**(1-exponent)
    weights = np.zeros((degree+1, degree+1, degree))
    for j in range(1,degree+1):
        for k in range(1,degree+1):
            if (j+k)%2:
                continue
            for a in range(1,j+1,2):
                for b in range(1,k+1,2):
                    weights[j,k] += 2*comb(j,a)*comb(k,b)*2.0**(-a-b)*c[(j+k-a-b)//2]
    return weights


@lru_cache(maxsize=24)
def structure(d, deg, coordinate):
    h = []
    for j in range((deg+1)//2):
        for vals in compositions(j,d):
            v = [2*k for k in vals]
            v[coordinate] += 1
            h.append(tuple(v))
    pairs = []
    monomials = {}
    coeff = coefficient_table(deg)
    n = len(h)
    for i, a in enumerate(h):
        for j in range(i,n):
            b = h[j]
            lo = tuple(min(x,y) for x,y in zip(a,b))
            s = []
            for g in product(*(range(k+1) for k in lo)):
                af = sum(a)-sum(g)
                bf = sum(b)-sum(g)
                if af < 1 or bf < 1:
                    continue
                ct = np.array(tuple((x+y)//2-z for x,y,z in zip(a,b,g)), dtype=int)
                norm = 1
                for x,y,z in zip(a,b,g):
                    norm *= comb(x,z)*comb(y,z)*factorial(z)
                key = tuple(ct)
                if key not in monomials:
                    monomials[key] = len(monomials)
                s.append((monomials[key], norm*coeff[af,bf]))
            pairs.append(s)
    idx = np.zeros((len(pairs),max(map(len,pairs))), dtype=int)
    weights = np.zeros((*idx.shape,deg))
    for j,pair in enumerate(pairs):
        for k,(i,w) in enumerate(pair):
            idx[j,k] = i
            weights[j,k] = w
    return n, tuple(monomials), idx, weights


def dirichlet(b, deg, coordinate):
    d = len(b)
    l, powers, idx, weights = structure(d,deg,coordinate)
    pow = np.asarray(powers)
    theta, w = rule(42)
    u = np.tan(theta)
    v0 = 1 + b[None, :] / (4*np.sin(theta[:,None])**2)
    v1 = 1 + (2*u*u+.25)[:, None] * b
    W0 = (2/np.pi)*w*np.prod(v0,axis=1)**(-0.5)
    W1 = 2/np.sqrt(2*np.pi)*w*np.prod(v1,axis=1)**(-0.5)/np.cos(theta)**2
    moments = np.empty((len(powers),deg))
    fac = np.ones(deg+1)
    for j in range(1,len(fac)):
        fac[j] = fac[j-1]*(2*j-1)
    a = np.prod(fac[pow],axis=1)
    moments[:,0] = a*(np.exp(pow @ np.log(b/v0).T) @ W0)
    ex = np.exp(pow @ np.log(b/v1).T)
    for j in range(deg-1):
        moments[:,j+1] = a*(ex @ (W1*u**(2*j)))/gamma(j+0.5)
    values = np.einsum("ijk,ijk->i", weights, moments[idx])
    out = np.empty((l,l))
    i,j = np.triu_indices(l)
    out[i,j] = out[j,i] = values
    return out


def covariance_factor(b, degree=5, isotropic=False):
    d = len(b)
    h = []
    for i in range(1 if isotropic else d):
        D = dirichlet(b, degree, i)
        rhs = np.zeros(len(D))
        rhs[0] = 1
        # Normalize features for the linear solve.
        a = 1/np.sqrt(np.maximum(1e-14,np.diag(D)))
        c = np.linalg.solve((a[:,None]*D)*a, a*rhs)
        h.append(max(0.0, 2*a[0]*c[0]-1))
    return np.array(h)
