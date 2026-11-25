"""Integrable controls for the smooth residual."""

import numpy as np
from scipy.linalg import solve
from scipy.special import roots_hermitenorm

from cubature import net_sizes, normals, quadrature_control
from kinks import debug


def smooth_control(session, base, rng, reserve=4100, initial=None, allow_curvature=True):
    d = session.d
    eye = np.eye(d)
    A = np.zeros((d, d))
    U = eye
    if initial is not None:
        A = initial
    if allow_curvature and session.remaining >= reserve + (d+1)*(d+2)//2 + 12*d:
        h = .004
        i, j = np.triu_indices(d)
        z = np.vstack([np.zeros((1, d)), h*eye, h*(eye[i]+eye[j])])
        y = session.query(z)
        if base is not None:
            y -= base(z)
        dy = y[1:d+1]-y[0]
        hess = (y[d+1:]-y[0]-dy[i]-dy[j])/(h*h)
        A = np.empty((d,d))
        A[i,j] = hess
        A[j,i] = hess
    elif allow_curvature and session.remaining >= reserve+10*(d+1)+11*d:
        # A cheap range sketch when a dense curvature stencil would leave too
        # small a randomized correction. Append an isotropic component for
        # the residual trace before selecting the directions.
        r=min(d,8)
        V=np.linalg.qr(rng.standard_normal((d,r)),mode="reduced")[0]
        h=.0025
        x=np.r_[np.zeros((1,d)),h*V.T]
        point=np.concatenate([x[:,None,:],x[:,None,:]+h*eye],axis=1)
        y=session.query(point).reshape(len(x),d+1)
        if base is not None:
            y-=base(point.reshape(-1,d)).reshape(y.shape)
        g=(y[:,1:]-y[:,:1])/h
        Hv=((g[1:]-g[:1])/h).T
        M=.5*(V.T@Hv+Hv.T@V)
        e,W=np.linalg.eigh(M)
        good=np.abs(e)>max(1e-9,.045*np.max(abs(e)))
        if np.any(good):
            C=Hv@W[:,good]
            A=(C/e[good])@C.T
            if np.linalg.norm(A)>12*np.linalg.norm(Hv):
                A=Hv@Hv.T
    if np.max(abs(A)) > 0:
        vals, v = np.linalg.eigh(A)
        U = v[:, np.argsort(-abs(vals))]
    if session.remaining < reserve+11*d:
        return U, None, 0.
    r, w = roots_hermitenorm(11)
    w /= np.sqrt(2*np.pi)
    z = (r[None, :, None]*U.T[:, None, :]).reshape(-1,d)
    y = session.query(z)
    if base is not None:
        y -= base(z)
    y = y.reshape(d,-1)
    middle = y[:,len(r)//2].mean()
    # Each axis slice is smooth on the scale of interest. Work in the Hermite
    # basis with a mild high-order rolloff rather than extrapolating a spline.
    def hermite(x, n):
        seq = [np.ones_like(x), x]
        for i in range(2, n):
            seq.append((x*seq[-1]-np.sqrt(i-1)*seq[-2])/np.sqrt(i))
        return seq

    H = np.stack(hermite(r,11))
    coefficients = ((y-middle)*w) @ H.T
    coefficients[:, 7:] *= [0.75, 0.55, 0.25, 0.0]
    residual_quad = U.T @ A @ U
    residual_quad -= np.diag(np.diag(residual_quad))

    def predict(g):
        x = g @ U
        estim = np.full(len(x), middle)
        for h, c in zip(hermite(x, len(r)), coefficients.T):
            estim += h @ c
        estim += np.sum((x @ residual_quad)*x, axis=1)*.5
        return estim

    exact = middle + coefficients[:,0].sum()
    if not np.isfinite(exact):
        return U, None, 0.
    if session.remaining >= reserve+128:
        z = rng.standard_normal((128,d))
        y = session.query(z)
        if base is not None:
            y -= base(z)
        pred = predict(z)
        raw = np.std(y)
        fitted = np.std(y-pred)
        debug("smooth residual", raw, fitted)
        if fitted > 0.94*max(raw,1e-30):
            return U, None, 0.
    return U, predict, exact


def _features(t):
    n,d=t.shape
    p=min(5,d)
    a=(t*t-1)/np.sqrt(2.)
    b=(t**4-6*t*t+3)/np.sqrt(24.)
    i,j=np.triu_indices(d,1)
    keep=i<min(3,d)
    ii,jj=np.triu_indices(p,1)
    cube=(t[:,:p]**3-3*t[:,:p])/np.sqrt(6.)
    r=(cube[:,:,None]*t[:,None,:p]).reshape(n,p*p)
    return np.c_[a,b*.57,t[:,i[keep]]*t[:,j[keep]]*.78,
                 a[:,ii]*a[:,jj]*.65,r[:,~np.eye(p,dtype=bool).ravel()]*.52]


def _regress(features,y):
    n=len(y)
    yy=y-np.median(y)
    sigma=max(1e-12,np.quantile(abs(yy),.68))
    yy=np.clip(yy,-5*sigma,5*sigma)
    x=features-features.mean(axis=0)
    G=x.T@x
    lam=.075+min(.22,.13*features.shape[1]/n)
    G.flat[::len(G)+1]+=n*lam
    return solve(G,x.T@(yy-yy.mean()),assume_a="pos",check_finite=False)


def smooth_correct(session,rng,basis,control=None,exact=0.):
    if session.remaining<2200:
        return quadrature_control(session,rng,basis,control,exact,True)
    n=session.remaining//2
    h=1<<(n.bit_length()-1)
    if n-h<.23*n:
        h//=2

    def block(n):
        xout,yout=[],[]
        for q in net_sizes(n):
            if session.elapsed>session.limit-4.5 and xout:
                break
            x=normals(rng,q,session.d)
            a=x@basis.T
            Z=np.r_[a,-a]
            y=session.query(Z)
            if control is not None:
                y-=control(Z)
            yout.append(.5*(y[:len(a)]+y[len(a):]))
            xout.append(x)
        return np.vstack(xout),np.concatenate(yout)

    x1,y1=block(h)
    if session.elapsed>session.limit-4.5:
        return exact+y1.mean()
    x2,y2=block(n-h)
    x1,x2=_features(x1),_features(x2)
    if len(y1)<128 or len(y2)<128:
        return exact+(y1.sum()+y2.sum())/(len(y1)+len(y2))
    c1,c2=_regress(x1,y1),_regress(x2,y2)
    value=(y1.sum()+y2.sum()-x1.sum(axis=0)@c2-x2.sum(axis=0)@c1)/(len(y1)+len(y2))
    return exact+value
