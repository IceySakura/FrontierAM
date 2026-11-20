"""Projected Markov covariance with a nonsymmetric empirical operator."""

import numpy as np

from reversible import feature_groups


def sym(a):
    return (a+a.T)*0.5


def invroot(a):
    v, u = np.linalg.eigh(sym(a))
    return (u / np.sqrt(np.maximum(v, max(1e-10,v[-1]*1e-8)))) @ u.T


def projected(x, diagnostics=None, power=1.0, extra=False):
    n, d = x.shape
    nn = n-1
    dx = np.diff(x,axis=0)
    direct = dx.T @ dx / (2*nn)
    v, u = np.linalg.eigh(direct)
    groups = feature_groups(x,v,u,more=extra)
    sizes = [g.shape[1] for g in groups]
    # Estimate a stationary, pair-averaged inner product.
    z = np.concatenate([x,*groups],axis=1)
    z -= (z.sum(axis=0)-.5*(z[0]+z[-1]))/nn
    z0, z1 = z[:-1], z[1:]
    cs = sym((z0.T @ z0 + z1.T @ z1)/(2*nn))
    cw = np.linalg.solve(cs[:d,:d],cs[:d,d:])
    w = z[:,d:] - z[:,:d] @ cw
    y = z[:,:d]
    wn = np.sqrt(np.maximum(1e-12,np.mean(w*w,axis=0)))
    w /= wn
    z = np.concatenate([y,w],axis=1)
    dz = np.diff(z,axis=0)
    align = z[:-1].T @ z[1:] / nn
    diag = (dz.T @ dz)/(2*nn)
    M = diag+(align.T-align)*0.5
    H0 = cs[:d,:d]
    lin = M[:d,:d]
    il = np.linalg.solve(lin,H0)
    hlin = sym(H0 @ il) * 2 - H0
    dw = M[d:,d:]
    W = w[:-1]
    B = M[d:,:d]
    A = M[:d,d:]
    ide = 1e-6 * np.maximum(np.diag(diag[d:,d:]),1e-8)
    dw = dw+np.diag(ide)

    # Estimate each cross block's uncertainty with predictable residuals.
    fwd = y[1:] - y[:-1] @ np.linalg.solve(H0,align[:d,:d])
    bwd = y[:-1] - y[1:] @ np.linalg.solve(H0,align[:d,:d].T)
    b = min(25,max(8,n//240))
    loc = np.linspace(0,nn,b+1,dtype=int)
    fs, rs = [], []
    for i,j in zip(loc[:-1],loc[1:]):
        fs.append(-W[i:j].T @ fwd[i:j]/(j-i))
        rs.append(-w[i+1:j+1].T @ bwd[i:j]/(j-i))
    fs, rs = np.array(fs), np.array(rs)
    fs -= fs.mean(axis=0)
    rs -= rs.mean(axis=0)
    fh = invroot(hlin)
    il0 = np.linalg.solve(lin.T, H0) @ fh
    il1 = il @ fh
    wby = np.zeros(len(w.T))
    j0 = 0
    richness = 0.0
    for size in sizes:
        j = slice(j0,j0+size)
        Ri = invroot(diag[d+j0:d+j0+size,d+j0:d+j0+size])
        l1 = (Ri @ B[j]) @ il1
        l2 = (Ri @ A[:,j].T) @ il0
        v1 = np.sum(l1*l1)
        v2 = np.sum(l2*l2)
        n1 = np.sum((Ri @ fs[:,j] @ il1)**2)/(b*(b-1))
        n2 = np.sum((Ri @ rs[:,j] @ il0)**2)/(b*(b-1))
        wa = max(0.,1-1.25*n1/max(1e-14,v1))
        wb = max(0.,1-1.25*n2/max(1e-14,v2))
        wby[j] = (wa*wb)**0.30
        richness += wby[j0]**2*np.sqrt(v1*v2)
        j0 += size
    if richness < .009:
        return None
    ww = wby[:,None]
    corr = (A*wby) @ np.linalg.solve(dw,ww*B)
    sw = fh @ il.T @ corr @ il @ fh
    if not np.isfinite(sw).all():
        return None
    # Bound the relative change in the generator when the nonlinear
    # feature span is estimated from only a handful of effective blocks.
    ratio = np.linalg.norm(np.linalg.solve(lin,corr),2)
    shrink = power * min(1.0,0.60/max(0.1,ratio))
    Mnew = lin - shrink*corr
    sol = np.linalg.solve(Mnew,H0)
    h = 2*sym(H0 @ sol)-H0
    if diagnostics is not None:
        diagnostics["nonrev_projection"] = float(richness)
    return h, hlin, richness
