"""Gaussian numerical conditioning for a two-valued payoff."""
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.special import expit, ndtr, ndtri

from common import log, normalize, orth, sobol


def logistic_axis(x, hit):
    scale = .0025
    m, d = x.shape
    x = np.c_[x,np.ones(m)]
    def f(a):
        z = x@a
        return ((np.logaddexp(0.,z)-hit*z).mean()+scale*(a@a),
                ((expit(z)-hit)@x)/m+2*scale*a)
    a = minimize(f,np.zeros(d+1),jac=True,method='L-BFGS-B',options={'maxiter':45}).x
    return normalize(a[:-1])


def brackets(s, x, v, lo, hi, tries=4):
    n = len(x)
    lo,hi = np.broadcast_to(lo,(n,)).copy(),np.broadcast_to(hi,(n,)).copy()
    if s.remaining < 2*n:
        return lo,hi,np.zeros(n,dtype=bool)
    z = s.query(np.r_[x+lo[:,None]*v,x+hi[:,None]*v])>0
    left, right = z[:n].copy(), z[n:].copy()
    span = hi-lo
    for k in range(tries):
        il = np.flatnonzero(left)
        ir = np.flatnonzero(~right)
        if len(il)+len(ir)==0 or s.remaining<len(il)+len(ir):
            break
        lo[il] -= (1.45**k)*span[il]
        hi[ir] += (1.45**k)*span[ir]
        lo[il] = np.maximum(-13.,lo[il])
        hi[ir] = np.minimum(13.,hi[ir])
        z = s.query(np.r_[x[il]+lo[il,None]*v,x[ir]+hi[ir,None]*v])>0
        left[il], right[ir] = z[:len(il)], z[len(il):]
    return lo,hi,(~left)&right


def rootsteps(s, x, v, lo, hi, iters):
    for _ in range(iters):
        if s.remaining < len(x):
            break
        mid = (lo+hi)*.5
        z = s.query(x+mid[:,None]*v)>0
        lo,hi = np.where(z,lo,mid), np.where(z,mid,hi)
    return lo,hi


def roots(s,x,v,lo,hi,steps,tries=4):
    a,b,ok = brackets(s,x,v,lo,hi,tries=tries)
    a[ok],b[ok] = rootsteps(s,x[ok],v,a[ok],b[ok],steps) if np.any(ok) else (a[ok],b[ok])
    return (a+b)*.5, ok


def feature_matrix(z, full=True):
    l = min(7,z.shape[1])
    t = z[:,:l]
    H2 = (t**2-1.)/np.sqrt(2.)
    i,j = np.triu_indices(min(6,l),1)
    a = [np.ones((len(z),1)), z, H2, t[:,i]*t[:,j]]
    if full:
        k = min(4,l)
        H3 = (t[:,:k]**3-3*t[:,:k])/np.sqrt(6.)
        H4 = (t[:,:k]**4-6*t[:,:k]**2+3.)/np.sqrt(24.)
        ii,jj = np.where(~np.eye(k,dtype=bool))
        a += [H3, H4, H2[:,ii]*t[:,jj]]
    return np.concatenate(a,axis=1)


def learn_basis(s, v, V, t):
    d = s.d
    k = 3 if d<=32 else 3
    N = d-1
    A = s.rng.standard_normal((k,N))
    XA = A@V.T
    sample = np.r_[XA,-XA]*.87
    r0,ok0 = roots(s,sample,v,t-.78,t+.21,13)
    if np.count_nonzero(ok0)<2*k:
        return V
    h = .41
    ref = (sample[:,None,:]+h*V.T[None,:,:]).reshape(-1,d)
    mid = np.repeat(r0,N)
    step_size=.025 if d>=32 else .037
    rr,ok = roots(s,ref,v,mid-step_size,mid+step_size,8,tries=3)
    if not np.all(ok):
        return V
    g = (rr.reshape(2*k,N)-r0[:,None])/h
    B = ((g[:k]-g[k:])/.87).T
    if np.linalg.norm(B)<1e-4:
        return V
    # Two-sided slope sketches.  A low-rank ridge completion is used only
    # to order the net and form the control variate.
    Q = linalg.qr(B,mode='full')[0]
    W = np.linalg.pinv(A@Q[:,:k])
    small = .5*(Q[:,:k].T@B)@W
    U = linalg.eigh((small+small.T)*.5)[1]
    Q[:,:k] = Q[:,:k]@U
    log("digital slope singular values",np.linalg.svd(B,compute_uv=False))
    return V@Q


def digital_price(s,x,y):
    hi_val = float(np.max(y))
    yes = y>0
    if min(np.sum(yes),np.sum(~yes)) < 12 and s.remaining>9000:
        q = sobol(512,s.d,s.rng)
        x,y = np.r_[x,q],np.r_[y,s.query(q)]
        yes = y>0
    if np.sum(yes)<4 or np.sum(~yes)<4:
        return None
    v = logistic_axis(x,yes)
    tt, ok = roots(s,np.zeros((1,s.d)),v,-5.5,5.5,28,tries=1)
    if not ok[0]:
        return None
    t = tt[0]
    V = orth(v)[:,1:]
    h=.24
    rr, ok = roots(s,h*V.T,v,t-.19,t+.19,12)
    if np.all(ok):
        v = normalize(v - V @ ((rr-t)/h))
        V = orth(v)[:,1:]
        tt, ok = roots(s,np.zeros((1,s.d)),v,t-.35,t+.35,21)
        if ok[0]:
            t = tt[0]
    if s.remaining>11200 and s.d>=8:
        V=learn_basis(s,v,V,t)
    log("digital anchor",t,"used",s.used)
    n = 256 if s.remaining>9200 else 192
    z = sobol(n,s.d-1,s.rng)
    q = z@V.T
    rr, ok = roots(s,q,v,t-.66,t+.09,10,tries=3)
    if np.count_nonzero(ok) < .85*n:
        return None
    z, rr = z[ok], rr[ok]
    F = feature_matrix(z)
    G = F.T@F
    diag = np.arange(len(G))
    G[diag[1:],diag[1:]] += .32
    # A constant and centered Gaussian Hermite products have known means.
    bt,bg = linalg.solve(G,F.T@np.c_[rr,ndtr(-rr)],assume_a='pos').T
    val = float(bg[0])
    if not (0. < val < 1.):
        return None
    err = rr-F@bt
    width = max(.006, 3.0*np.sqrt(np.mean(err**2)))
    log("digital root train residual",np.sqrt(np.mean(err**2)),"surrogate",val,"used",s.used)
    if s.remaining<300 or s.elapsed>17.:
        return val*hi_val
    n = 1024 if s.remaining>=6300 else 512 if s.remaining>=3000 else 256
    reps=[]
    wr=[]
    for rep in range(3):
        if s.remaining<max(260,3*n) or s.elapsed>19.5:
            break
        z = sobol(n,s.d-1,s.rng)
        q = z@V.T
        F = feature_matrix(z)
        t_hat = np.clip(F@bt,t-5.,t+.29)
        # Use held-out thresholds, with enlargement on the actual queries.
        w = width*(.93+.13*np.maximum(0.,np.linalg.norm(z[:,:4],axis=1)-1))
        lo,hi,ok = brackets(s,q,v,t_hat-w,t_hat+w,tries=6)
        if not np.all(ok):
            sel = np.flatnonzero(~ok)
            lo[sel],hi[sel],ok[sel] = brackets(s,q[sel],v,t-6.,t+3.,tries=0)
        iters = min(6,max(0,s.remaining//n-1))
        if rep==0 and s.remaining>=9500 and n==1024:
            iters=min(iters,4)
        if s.remaining<n:
            break
        il = np.flatnonzero(ok)
        lo[il], hi[il] = rootsteps(s,q[il],v,lo[il],hi[il],iters)
        P0,P1 = ndtr(lo),ndtr(hi)
        u = np.clip(P0+(P1-P0)*s.rng.random(n),1e-300,1.-1e-16)
        isup = s.query(q+ndtri(u)[:,None]*v)>0
        Y = np.where(ok,ndtr(-hi)+(P1-P0)*isup,ndtr(-t_hat))
        r = Y-F@bg
        reps.append(r.mean())
        wr.append(n)
        log("digital correction",r.mean(),"sd",np.std(r),"slices",n,"used",s.used)
        if rep==0 and s.remaining<3*n:
            n = max(128, 1<<(max(1,s.remaining//5).bit_length()-1))
            n = min(n,512)
        elif rep>0:
            n=min(256, n)
    if wr:
        val += float(np.average(reps,weights=wr))
    return float(np.clip(val,0.,1.)*hi_val)
