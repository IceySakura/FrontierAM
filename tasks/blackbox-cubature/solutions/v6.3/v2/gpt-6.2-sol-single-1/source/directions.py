"""A second way to recover sparse spectra, using directional curvatures."""
import itertools

import numpy as np
from scipy.linalg import LinAlgError, solve, svd
from scipy.optimize import least_squares, minimize_scalar

from common import Model, legmat, multiindices, solve_ls
from periods import peak_moment, periodic_peak_profile
from ridges import gradients


def hessians(orc,points,scale,h=2.6e-4):
    d=orc.d
    ij=np.array(list(itertools.combinations(range(d),2)))
    directions=np.vstack([np.eye(d),np.eye(d)[ij[:,0]]+np.eye(d)[ij[:,1]]])
    q = np.concatenate([points[:,None,:],points[:,None,:]+h*directions,
                        points[:,None,:]-h*directions],axis=1)
    f=orc.sample(q.reshape(-1,d)).reshape(len(points),-1)/scale
    n=len(directions)
    p,m=f[:,1:n+1]-f[:,0,None],f[:,n+1:]-f[:,0,None]
    v=(p+m)/h**2
    H=np.zeros((len(points),d,d))
    H[:,np.arange(d),np.arange(d)]=v[:,:d]
    off=(v[:,d:]-v[:,ij[:,0]]-v[:,ij[:,1]])*.5
    H[:,ij[:,0],ij[:,1]]=off
    H[:,ij[:,1],ij[:,0]]=off
    g=(p[:,:d]-m[:,:d])/(2*h)
    return H,g


def common_shapes(hh,rng):
    d=hh.shape[1]
    basis,s,_=svd(np.concatenate(hh,axis=1),full_matrices=False,check_finite=False)
    if s[0] < 1e-13:
        return None
    rank=int(np.count_nonzero(s>s[0]*1.5e-5))
    if rank<2 or rank>d:
        return None
    q=basis[:,:rank]
    hh=q.T@hh@q
    opt=None
    n=len(hh)
    # Vary the reference Hessian to avoid an unlucky near-zero term.
    coeff=np.vstack([np.eye(n),rng.uniform(-1,1,(9,n))])
    for a in coeff:
        B=np.einsum("k,kij->ij",a,hh)
        cb=np.linalg.cond(B)
        if cb > 5e5:
            continue
        for b in rng.uniform(-1,1,(5,n)):
            A=np.einsum("k,kij->ij",b,hh)
            try:
                J=solve(B.T,A.T,check_finite=False).T
                e,v=np.linalg.eig(J)
                if np.any(e.imag!=0):
                    continue
                v=v.real
                if np.linalg.cond(v)>180:
                    continue
                c=solve(v,np.eye(rank),check_finite=False)
            except LinAlgError:
                continue
            C=c@hh@c.T
            diag=np.diagonal(C,axis1=1,axis2=2)
            ratio=np.linalg.norm(C-diag[:,:,None]*np.eye(rank))/np.linalg.norm(C)
            gap=np.min(np.diff(np.sort(e.real)))
            if gap<1e-5:
                continue
            score=max(ratio,1e-14)*(max(1.,cb/60)**.2)/(min(1.,gap)**.24)
            if opt is None or score<opt[0]:
                opt=score,ratio,(q@v).T
    if opt is None or opt[1]>9e-4:
        return None
    return opt[2]/np.linalg.norm(opt[2],axis=1)[:,None]


def frequency_from_derivative(t,y):
    tt=t-np.mean(t)
    span=np.ptp(tt)
    if span<.15 or np.ptp(y)<1e-10:
        return None
    # A floating phase and constant offset are included in the periodogram.
    y=y-np.mean(y)
    n=max(200,int(span*220))
    ff=np.linspace(.38,230.,n)
    uu=ff[:,None]*tt
    c,s=np.cos(uu),np.sin(uu)
    c -= c.mean(axis=1)[:,None]
    s -= s.mean(axis=1)[:,None]
    cc,ss,cs=np.sum(c*c,axis=1),np.sum(s*s,axis=1),np.sum(c*s,axis=1)
    a,b=c@y,s@y
    r=(ss*a*a+cc*b*b-2*cs*a*b)/np.maximum(cc*ss-cs*cs,1e-23)
    j=np.argmax(r)
    def res(w):
        u=w*tt
        A=np.column_stack([np.ones(len(tt)),np.cos(u),np.sin(u)])
        h=A@solve_ls(A,y)-y
        return float(h@h)
    out=minimize_scalar(res,method="bounded",bounds=(ff[max(j-1,0)],ff[min(j+1,len(ff)-1)]),
                        options={"xatol":1e-10})
    if out.fun > .034*(y@y):
        return None
    return float(out.x)


def _refine(orc,x,y,scale,w0):
    k,d=w0.shape
    n=int(len(x)*.83)
    ax,ay=orc.data()
    xx,yy=np.vstack([x[:n],ax[len(x):]])-.5, np.r_[y[:n],ay[len(x):]/scale]
    best=None
    for deg in (0,1,2):
        idx=multiindices(d,deg,deg)
        pol=legmat(xx+.5,idx)
        p=pol.shape[1]
        def design(z,w,base):
            f=z@w.T
            return np.hstack([base,np.cos(f),np.sin(f)])
        A=design(xx,w0,pol)
        c=solve_ls(A,yy)
        if np.std(A@c-yy)>.22*np.std(yy):
            continue
        def unpack(q):
            return q[:p+2*k],q[p+2*k:].reshape(k,d)
        def fun(q):
            c,w=unpack(q)
            return design(xx,w,pol)@c-yy
        def jac(q):
            c,w=unpack(q)
            a=design(xx,w,pol)
            z=a[:,p:p+k]*c[p+k:]-a[:,p+k:]*c[p:p+k]
            return np.hstack([a,(z[:,:,None]*xx[:,None,:]).reshape(len(xx),-1)])
        sol=least_squares(fun,np.r_[c,w0.ravel()],jac=jac,max_nfev=46,
                          x_scale="jac",ftol=4e-12,xtol=3e-12,gtol=1e-13)
        c,w=unpack(sol.x)
        def pred(u,c=c,w=w,idx=idx):
            return design(u-.5,w,legmat(u,idx))@c
        err=np.max(abs(pred(x[n:])-y[n:]))
        if err<.05*np.ptp(y) and (best is None or err<best.error):
            val=c[0]+np.prod(np.sinc(w/(2*np.pi)),axis=1)@c[p:p+k]
            best=Model(pred,float(val),"directional spectrum",float(err))
        if err < 1e-11:
            break
    return best


def directional_spectrum(orc,x,y,scale):
    d=orc.d
    minimum=3*(d*d+d)+28*d+60
    if d < 3 or d>9 or orc.left<minimum:
        return None
    good=np.flatnonzero(np.all((x>.011)&(x<.989),axis=1))
    if len(good)<35:
        return None
    ids=orc.rng.choice(good,3,replace=False)
    hh,g=hessians(orc,x[ids],scale)
    dirs=common_shapes(hh[1:]-hh[:1],orc.rng)
    if dirs is None:
        dirs=common_shapes(hh,orc.rng)
    if dirs is None:
        return None
    count=min(27,(orc.left-53)//(2*d))
    if count<13:
        return None
    jj=orc.rng.choice(np.setdiff1d(good,ids),count,replace=False)
    gg=gradients(orc,x[jj],scale,order=2,step=.000022)
    T=(x[np.r_[ids,jj]]-.5)@dirs.T
    C=solve_ls(dirs.T,np.vstack((g,gg)).T).T
    freq=[frequency_from_derivative(t,v) for t,v in zip(T.T,C.T)]
    if any(a is None for a in freq):
        return None
    w=dirs*np.asarray(freq)[:,None]
    return _refine(orc,x,y,scale,w)


def periodic_ridges(orc,x,y,scale):
    """Learn a few independent directions and their wrapped peak profiles."""
    d=orc.d
    if d<2 or d>9 or orc.left<6*(d*d+d)+350:
        return None
    valid=np.flatnonzero(np.all((x>.018)&(x<.982),axis=1))
    if len(valid)<30:
        return None
    valid=valid[np.argsort(y[valid])]
    ids=np.unique(np.r_[valid[int(.18*len(valid))],valid[int(.88*len(valid))],valid[-2]])
    H1,_=hessians(orc,x[ids],scale,.00016)
    H2,_=hessians(orc,x[ids],scale,.00008)
    H=(4*H2-H1)/3
    dirs=common_shapes(H,orc.rng)
    if dirs is None or len(dirs)>3:
        dirs=common_shapes(H[1:]-H[:1],orc.rng)
    if dirs is None or len(dirs)>3 or np.linalg.cond(dirs)>12:
        return None
    k=len(dirs)
    dual=np.linalg.pinv(dirs)
    w,phase,concentration = [],[],[]
    nline=min(151,(orc.left-72)//k)
    if nline<55:
        return None
    for v,di in zip(dual.T,dirs):
        radius=.485/np.max(abs(v))
        t=np.linspace(-radius,radius,nline)
        vals=orc.sample(.5+t[:,None]*v)/scale
        u=periodic_peak_profile(t,vals,rough=True)
        if u is None or u[0]>.018:
            return None
        _,_,am,cc,f,ph=u
        w.append(di*f)
        phase.append(ph)
        concentration.append(cc)
    w=np.asarray(w)
    phase=np.asarray(phase)
    logk=np.log(concentration)
    n=int(.82*len(x))
    xa,ya=orc.data()
    z,ys=np.vstack((x[:n],xa[len(x):]))-.5,np.r_[y[:n],ya[len(x):]/scale]
    best=None
    for degree in (0,1,2):
        ids=multiindices(d,degree,degree)
        poly=legmat(z+.5,ids)
        p=poly.shape[1]
        def design(z,base,lk,phi,w):
            a=z@w.T+phi
            g=np.cos(a)-1
            kap=np.exp(lk)
            e=np.exp(kap*g)
            return np.column_stack((base,e)),e,g,kap,np.sin(a)
        A=design(z,poly,logk,phase,w)[0]
        cc=solve_ls(A,ys)
        if np.std(A@cc-ys)>.19*np.std(ys):
            continue
        def unpk(q):
            return q[:p+k],q[p+k:p+2*k],q[p+2*k:p+3*k],q[p+3*k:].reshape(k,d)
        def fun(q):
            c,lk,phi,w0=unpk(q)
            return design(z,poly,lk,phi,w0)[0]@c-ys
        def jac(q):
            c,lk,phi,w0=unpk(q)
            A,e,a,kap,sa=design(z,poly,lk,phi,w0)
            db=c[p:]*e*kap
            de=-db*sa
            return np.column_stack((A,db*a,de,(de[:,:,None]*z[:,None,:]).reshape(len(z),-1)))
        q=np.r_[cc,logk,phase,w.ravel()]
        lo,hi=np.full(len(q),-np.inf),np.full(len(q),np.inf)
        lo[p+k:p+2*k],hi[p+k:p+2*k]=-3.1,6.4
        sol=least_squares(fun,q,jac=jac,bounds=(lo,hi),max_nfev=38,
                          x_scale="jac",ftol=4e-12,xtol=5e-12,gtol=2e-13)
        c,lk,phi,w0=unpk(sol.x)
        def predict(q,c=c,lk=lk,phi=phi,w0=w0,ids=ids):
            return design(q-.5,legmat(q,ids),lk,phi,w0)[0]@c
        err=np.max(np.abs(predict(x[n:])-y[n:]))
        if err<.07*np.ptp(y) and (best is None or err<best.error):
            val=c[0]+sum(v*peak_moment(a,ww,ph) for v,a,ww,ph in zip(c[p:],np.exp(lk),w0,phi))
            best=Model(predict,float(val),"localized periodic ridges",float(err))
        if err<1e-11:
            break
    return best
