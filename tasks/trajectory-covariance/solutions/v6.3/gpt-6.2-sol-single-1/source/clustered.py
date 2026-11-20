"""Detect well separated recurrent states with small observation noise."""

import numpy as np

from discrete import finite_covariance


def sym(a):
    return (a+a.T)*0.5


def invroot(a):
    w, v = np.linalg.eigh(sym(a))
    return v / np.sqrt(np.maximum(w,max(1e-16,w[-1]*1e-9)))


def noisy_states(x, delta, diagnostics=None):
    n,d = x.shape
    if n < 900 or d < 2:
        return None
    r = np.sum(delta*delta, axis=1)
    u = np.quantile(r,[.06,.32,.56,.88,.992])
    if not (u[1] < .44*d and (u[-1] > 19*u[1] or u[3] > 50*u[0])):
        return None
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    cut = u[2] if u[2] < 3.0*u[1] else u[1]
    z = delta[r<=cut]
    W = invroot(z.T @ z / (2*len(z)))
    m = min(3100,n)
    indices = np.floor((np.arange(m)+.5)*n/m).astype(int)
    s = x[indices] @ W
    dist, nn = cKDTree(s).query(s,k=min(18,m),workers=1)
    # Non-core points do not act as bridges between groups.
    core = dist[:,5] <= 2.3*np.sqrt(d)/np.sqrt(5)
    rad = 2.7*np.sqrt(d)/np.sqrt(5)
    good = (dist < rad) & core[:,None] & core[nn]
    i, j = np.where(good)
    graph = csr_matrix((np.ones(len(i)),(i,nn[i,j])),shape=(m,m))
    k, label = connected_components(graph,directed=False)
    size = np.bincount(label,minlength=k)
    use = np.flatnonzero(size>=max(12,m//300))
    if not 2 <= len(use) <= 36 or size[use].sum() < .85*m:
        return None
    centers = np.array([x[indices[label==j]].mean(axis=0) for j in use])
    eps = np.concatenate([x[indices[label==j]]-mu for j,mu in zip(use,centers)],axis=0)
    k = len(centers)

    def classify(centers, e):
        w = invroot(e.T @ e / len(e))
        v = x @ w
        f = centers @ w
        s = np.sum(f*f,axis=1) - 2*v@f.T
        g = np.argmin(s,axis=1)
        q = np.partition(s,1,axis=1)[:,:2]
        uncertainty = np.mean(q[:,1]-q[:,0] < 6.0)
        pair = f[:,None,:]-f[None,:,:]
        sep = np.sum(pair*pair,axis=2)
        sep.flat[::k+1] = np.inf
        return g, uncertainty, sep.min()

    for _ in range(2):
        label, uncertainty, separation = classify(centers,eps)
        freq = np.bincount(label,minlength=k)
        if freq.min() < 9 or uncertainty > .066 or separation < 15:
            return None
        centers = np.array([np.bincount(label,x[:,j],minlength=k)/freq for j in range(d)]).T
        eps = x-centers[label]
    noise = eps.T @ eps/n
    q = eps @ invroot(noise)
    score = sum(n*np.sum((q[:-l].T @ q[l:] / (n-l))**2) for l in (1,2,3))
    if score > 3*d*d+5.1*np.sqrt(6*d*d):
        return None
    counts = np.bincount(label[:-1]*k+label[1:],minlength=k*k).reshape(k,k).astype(float)
    h = finite_covariance(centers,counts,regularization=.85,risk=.56,pool=.78)
    if diagnostics is not None:
        diagnostics.update(model="clustered", states=k)
    return sym(h+noise)
