"""A small nested Gaussian sparse rule for the eight-dimensional residual."""
import itertools
from functools import lru_cache

import numpy as np
from scipy import linalg
from scipy.special import eval_hermitenorm, factorial

from common import log


@lru_cache(maxsize=3)
def rule8(extra):
    # The nested 1, 3, 9, 19 point Genz--Keister abscissae, converted from
    # exp(-x^2) to a standard normal.  Solve for the Gaussian weights.
    radii = (1.224744871391589, .5240335474869576, 2.0232301911005157,
             2.959210779063838, .8700408953529029, 1.835707975175187,
             2.266513262056788, 3.667774215946338, 4.499599398310388)
    x = np.r_[0.,np.column_stack((-np.array(radii),radii)).ravel()*np.sqrt(2.)]
    inc=[]
    prev=np.empty(0)
    for m in (1,3,9,19):
        pol = np.stack([eval_hermitenorm(j,x[:m])/np.sqrt(factorial(j)) for j in range(m)])
        b=np.zeros(m)
        b[0]=1.
        w=linalg.solve(pol,b)
        w[1:]=w[1:].reshape(-1,2).mean(axis=1).repeat(2)
        dw=w.copy()
        dw[:len(prev)]-=prev
        inc.append(dw)
        prev=w
    C={}
    for k in range(5+extra):
        for dims in itertools.combinations(range(8),k):
            levels=itertools.product((1,2,3),repeat=k) if k<=4 else ((1,)*k,)
            for lev in levels:
                n=sum(lev)
                if k<=4 and n>4:
                    continue
                v=[inc[l] for l in lev]
                for a in itertools.product(*(range(len(w)) for w in v)):
                    key=tuple((j,i) for j,i in zip(dims,a) if i)
                    w=1.
                    for u,i in zip(v,a):
                        w*=u[i]
                    if key not in C:
                        C[key]=[0.,0.,0.]
                    C[key][0]+=w
                    if n<=3:
                        C[key][1]+=w
                    if n<=4:
                        C[key][2]+=w
    pts=np.zeros((len(C),8))
    for i,key in enumerate(C):
        for j,k in key:
            pts[i,j]=x[k]
    return pts,np.asarray(list(C.values())).T


def sparse_mean(s,residual,W,known):
    extra=2 if s.remaining>=12410 else 1 if s.remaining>=10610 else 0
    p,w=rule8(extra)
    if s.remaining<len(p)+160 or s.elapsed>15.:
        return None
    z=p@W.T
    values=residual(z,s.query(z))
    c,d,b=known+w@values
    log("sparse residual levels",d,b,c,"q",len(p))
    if not np.isfinite(b) or abs(b-d)>.025*max(1.,abs(b)):
        return None
    if not np.isfinite(c) or abs(c-b)>max(.00003,1.25*abs(b-d)):
        return float(b)
    return float(c)
