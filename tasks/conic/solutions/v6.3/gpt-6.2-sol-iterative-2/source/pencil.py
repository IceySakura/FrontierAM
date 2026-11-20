"""Exact scalar bounds induced by definite semidefinite pencils."""
import numpy as np
from scipy.linalg import eigh

from model import smat, svec


def solve_pencil(p):
    if p.n!=1 or not p.K.psd or p.K.groups:
        return None
    a = p.A.toarray().reshape(-1)
    b = p.b
    upper,lower = np.inf,-np.inf
    ub,lb = None,None
    eq,lin = p.K.eq,p.K.lpos
    for keep,side in [(np.r_[eq[a[eq]!=0.],lin[a[lin]<0.]],'lo'),
                      (np.r_[eq[a[eq]!=0.],lin[a[lin]>0.]],'hi')]:
        if keep.size:
            ratio = b[keep]/a[keep]
            run = np.argmax(ratio) if side=='lo' else np.argmin(ratio)
            j = keep[run]
            record = j,j+1,np.ones(1),a[j]
            if side=='lo':
                lower,lb = ratio[run],record
            else:
                upper,ub = ratio[run],record
    for k,d,start,stop in p.K.nonlinear:
        H = smat(-a[start:stop],d)
        S = smat(b[start:stop],d)
        if not np.any(H):
            e,V = eigh(S,subset_by_index=(0,0),check_finite=False)
            if e[0]<0.:
                z = np.zeros(p.m)
                z[start:stop] = svec(np.outer(V[:,0],V[:,0]))
                return p.primal_cert(z)
            continue
        dd = np.diag(H)
        sign = 1. if np.all(dd>0.) else (-1. if np.all(dd<0.) else 0.)
        if not sign:
            return None
        try:
            e,V = eigh(S,sign*H,subset_by_index=(0,0),driver='gvx',check_finite=False)
        except (ValueError,np.linalg.LinAlgError):
            return None
        bound = -sign*e[0]
        if (sign>0. and bound>lower) or (sign<0. and bound<upper):
            z = svec(np.outer(V[:,0],V[:,0]))
            az = float(a[start:stop]@z)
            if not az*sign<0.:
                return None
            record = start,stop,z,az
            if sign>0.:
                lower,lb = bound,record
            else:
                upper,ub = bound,record
    if lower>upper:
        z = np.zeros(p.m)
        for r,w in [(lb,-1.),(ub,1.)]:
            j,k,u,f = r
            z[j:k] += (w/f)*u
        return p.primal_cert(z)
    q,h = float(p.q[0]),float(p.P.diagonal()[0])
    x = np.clip(-q/h if h>0. else (lower if q>0. else upper if q<0. else 0.),lower,upper)
    if not np.isfinite(x):
        return p.dual_cert(np.array([-np.sign(q)]))
    x0 = np.array([x])
    g = h*x+q
    z = np.zeros(p.m)
    r = lb if g>0. and x==lower else (ub if g<0. and x==upper else None)
    if r is not None:
        j,k,u,f = r
        z[j:k] = -g/f*u
    return p.optimal(x0,z,polish_slack=False)
