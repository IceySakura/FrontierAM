"""Value-only detection of continuous internal kinks and quadratic controls."""

import itertools

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from core import basis_from, fd_gradients, fit_ridge, gaussian, net_integral
from ellipsoid import ray_directions
from quadratic import positive_part, positive_probability_diag, quad_values
from smooth import smooth_kernel


def bends(f, scale):
    d = f.d
    n = min(128, 46+2*d)
    U = ray_directions(n,d,f.rng)
    X = f.rng.standard_normal((n,d))
    a = np.arange(n)<(n//2)
    X[a] = U[a]*(np.sqrt(d)*f.rng.uniform(.56,1.30,a.sum()))[:,None]
    X[~a] *= f.rng.uniform(.64,1.04,(~a).sum())[:,None]
    h0 = min(1.30,.23*np.sqrt(d)+.10)
    grid=h0*np.arange(-4.,5.)
    Y=f.query(X[:,None,:]+U[:,None,:]*grid[None,:,None]).reshape(n,len(grid))
    dd=(Y[:,:-2]-2*Y[:,1:-1]+Y[:,2:])/h0
    l=dd[:,1:-2]-dd[:,:-3]
    r=dd[:,2:-1]-dd[:,3:]
    jd=l+r
    frac=r/np.where(abs(jd)>1e-15,jd,1e-15)
    vscore=np.abs(jd)*np.clip(1-1.35*np.maximum(-frac,frac-1),.0,1.)
    # Restrict multiple candidates from the same line, and retain both signs.
    row,col=np.indices(vscore.shape)
    I=np.argsort(-vscore.ravel())
    selected=[]
    count=np.zeros(n,dtype=int)
    for k in I:
        i,j=row.ravel()[k],col.ravel()[k]
        if count[i]>=1 or vscore[i,j] < max(1e-6,.016*scale):
            continue
        selected.append((i,j))
        count[i]+=1
        if len(selected)>=min(48,28+d//2):
            break
    if not selected:
        return None
    i,j=np.asarray(selected).T
    # dd[:,0] sits at grid[1], so the first admissible interval is [grid[2],grid[3]].
    t=grid[j+2]+h0*np.clip(frac[i,j],-.20,1.20)
    U,X=U[i],X[i]
    g=jd[i,j]
    H=np.full(len(t), .77*h0)
    # From three values on either side, extrapolate both quadratics to a root.
    for fac in (.31,.24,.16,.115,.08):
        xp = X[:,None,:]+U[:,None,:]*(t[:,None]+H[:,None]*np.array([-3,-2,-1,1,2,3]))[:,:,None]
        w = f.query(xp).reshape(len(t),6)
        left=w[:,:3][:,::-1]
        right=w[:,3:]
        diff=right-left
        q0 = diff @ np.array([3.,-3.,1.])
        q1 = (right+left) @ np.array([-2.5,4.,-1.5])
        q2 = diff @ np.array([.5,-1.,.5])
        root = -q0/np.where(abs(q1)>1e-16,q1,1e-16)
        disc = q1*q1-4*q0*q2
        use = disc>0.
        alt = -2*q0/(q1+np.copysign(np.sqrt(np.maximum(0,disc)),q1)+1e-30)
        root[use] = alt[use]
        jump=q1/H
        keep=(np.abs(root)<1.08)&(abs(jump)>.14*abs(g))&(abs(jump)>.014*scale)
        if len(t)<3:
            break
        t, H, U, X, g = (t+np.clip(root,-1.,1.)*H)[keep],H[keep]*fac,U[keep],X[keep],jump[keep]
        if len(t)==0:
            return None
    centers=X+t[:,None]*U
    # Verify that the derivative change persists at a scale where a smooth
    # bend would be indistinguishable from a straight line.
    tiny=np.maximum(4e-5, H*1.7)
    vals=f.query(centers[:,None,:]+U[:,None,:]*(tiny[:,None]*np.array([-2.,-1.,1.,2.]))[:,:,None]
                 ).reshape(len(t),4)
    jump=((vals[:,3]-vals[:,2])-(vals[:,1]-vals[:,0]))/tiny
    delta=(vals[:,1]+(vals[:,1]-vals[:,0])-vals[:,2]+(vals[:,3]-vals[:,2])) / np.where(abs(jump)>1e-15,jump,1e-15)
    ok=(abs(jump)>.33*abs(g))&(abs(delta)<1.05*tiny)&(abs(jump)>.015*scale)
    return centers[ok]+U[ok]*delta[ok,None], U[ok], jump[ok]


def kink_gradient(f, x, u, jump):
    d=f.d
    # The direction of the original section need not be normal to the surface.
    h=6e-6
    span=max(3e-5, 10*h*np.sqrt(d))
    P=np.vstack((np.zeros(d), h*np.eye(d)))
    D=f.query(x+P+span*u)-f.query(x+P-span*u)
    g=(D[1:]-D[0])/h
    if np.linalg.norm(g)<1e-11 or abs(g@u-jump)>max(.23*abs(jump),2e-4):
        return None
    return g


def fit_local_quadric(f,x,u,n):
    d=f.d
    # Once the normal is known, all stencil points remain strictly on their
    # respective sides, so subtraction removes the common smooth background.
    v=np.copysign(1.,u@n)*n/np.linalg.norm(n)
    h=1.8e-4
    delta=2.1*h

    def difference(P):
        return f.query(x+P+delta*v)-f.query(x+P-delta*v)

    E=np.eye(d)
    D=difference(np.vstack((np.zeros(d),h*E,-h*E)))
    g=(D[1:d+1]-D[d+1:])/(2*h)
    Dplus=D[1:d+1]
    denom=h*h
    baseD=D[0]
    H=None
    partial=None
    if d>=22:
        # Randomized low-rank reconstruction, with independent sketch columns
        # to verify that a full stencil is unnecessary.
        k=9 if d<=32 else 11
        Uall=np.linalg.qr(f.rng.standard_normal((d,k+3)),mode="complete")[0]
        W=Uall[:,:k+3].T.copy()
        dw=difference(h*W)
        dxw=difference(h*(E[:,None,:]+W[None,:,:])).reshape(d,k+3)
        M=(dxw-Dplus[:,None]-dw[None,:]+baseD)/denom
        Y=M[:,:k]
        S=W[:k]@Y
        val, rot=eigh((S+S.T)*.5)
        mag=np.max(np.abs(val))
        live=np.abs(val)>max(mag*.002, 6e-6)
        if np.any(live):
            B=Y@rot[:,live]
            Hr=(B/val[live])@B.T
            err=np.linalg.norm(Hr@W[k:].T-M[:,k:])/max(1e-12,np.linalg.norm(M[:,k:]))
            if err<.009:
                H=Hr
        partial=(W,M,Uall[:,k+3:])
    if H is None:
        if partial is None:
            i,j=np.triu_indices(d,1)
            if f.remaining < 2*len(i)+3400 or f.left()<7.:
                return None
            P=h*(E[i]+E[j])
            dd=(difference(P)-Dplus[i]-Dplus[j]+baseD)/denom
            H=np.diag((Dplus+D[d+1:]-2*baseD)/denom)
            H[i,j]=H[j,i]=dd
        else:
            W,M,P=partial
            r=P.shape[1]
            i,j=np.triu_indices(r)
            if f.remaining < 2*(r+len(i))+3300 or f.left()<7.:
                return None
            ps=P.T
            ds=difference(h*ps)
            zz=difference(h*(ps[i]+ps[j]))
            vv=(zz-ds[i]-ds[j]+baseD)/denom
            HH=np.zeros((r,r))
            HH[i,j]=HH[j,i]=vv
            ww=W@M
            cross=P.T@M
            cross=P@cross@W
            H=W.T@((ww+ww.T)*.5)@W+cross+cross.T+P@HH@P.T
    lam,V=eigh(H)
    keep=np.abs(lam)>max(2e-6,.0005*np.max(np.abs(lam)))
    H=(V[:,keep]*lam[keep])@V[:,keep].T
    b=g-H@x
    c=float(-.5*x@H@x-b@x)
    return .5*H,b,c


class KinkControl:
    """An affine/selection expression in a small number of quadratics."""
    def __init__(self, quadrics, name, indices, signs):
        self.quadrics=quadrics
        self.name=name
        self.indices=indices
        self.signs=np.array(signs,dtype=float)

    def eval_values(self, Q):
        return self.apply(Q[:,self.indices]*self.signs)

    def apply(self, q):
        if self.name=="sum":
            return np.sum(np.maximum(0.,q)*self.sgn,axis=1)
        x,y=q[:,0],q[:,1]
        if self.name=="top":
            return np.maximum(0.,np.maximum(x,y))
        if self.name=="bottom":
            return np.minimum(0.,np.minimum(x,y))
        if self.name=="cap":
            return np.maximum(0.,np.minimum(x,y))
        if self.name=="floor":
            return np.minimum(0.,np.maximum(x,y))
        return np.clip(x,np.minimum(0.,y),np.maximum(0.,y))

    def __call__(self, z):
        q = np.column_stack([quad_values(z,*p) for p in self.quadrics])
        return self.eval_values(q)

    def gradient(self,z):
        Q=np.column_stack([quad_values(z,*self.quadrics[i])*s for i,s in zip(self.indices,self.signs)])
        G=np.stack([(2*z@self.quadrics[i][0]+self.quadrics[i][1])*s for i,s in zip(self.indices,self.signs)],axis=1)
        if self.name=="sum":
            return np.sum(G*((Q>0.)*self.sgn)[...,None],axis=1)
        value=self.apply(Q)
        pick=(Q==value[:,None])&(value[:,None]!=0)
        return np.sum(G*pick[...,None],axis=1)

    def integral(self,f):
        if self.name=="sum":
            ans=0.
            for i,s,ss in zip(self.indices,self.signs,self.sgn):
                A,b,c=self.quadrics[i]
                ans+=ss*positive_part(s*A,s*b,s*c)
            return ans
        A1,b1,c1=self.quadrics[self.indices[0]]
        A2,b2,c2=self.quadrics[self.indices[1]]
        s,t=self.signs
        return integrated_pair(f,self,[(s*A1,s*b1,s*c1),(t*A2,t*b2,t*c2)])


def orthogonal_cap(model):
    if model.name not in ("cap", "top", "floor", "bottom"):
        return None
    reverse=model.name in ("floor","bottom")
    polarity=-1. if reverse else 1.
    i,j=model.indices
    k1,k2=model.signs*polarity
    A,b,c=tuple(k1*z for z in model.quadrics[i])
    C,v,w=tuple(k2*z for z in model.quadrics[j])
    S,U=eigh((A+A.T)*.5,check_finite=False)
    take=S<min(-2e-6,.0011*S[0])
    r1=np.sum(take)
    if r1 not in (1,2) or S[-1]>.002*abs(S[0]):
        return None
    W=U[:,take]
    P=W@W.T
    b0=W.T@b
    if np.linalg.norm(P@C)>.008*np.linalg.norm(C) or np.linalg.norm(P@v)>.012*max(.01,np.linalg.norm(v)):
        return None
    if np.linalg.norm(b-W@b0)>.015*max(.01,np.linalg.norm(b)):
        return None
    O=np.eye(len(b))-P
    C=O@C@O
    L,V=eigh((C+C.T)*.5,check_finite=False)
    take2=L<min(-2e-6,.0011*L[0])
    r2=np.sum(take2)
    if r2 not in (1,2) or L[-1]>.002*abs(L[0]):
        return None
    W2=V[:,take2].copy()
    v0=W2.T@v
    if np.linalg.norm(v-W2@v0)>.017*max(.01,np.linalg.norm(v)):
        return None
    Anew=(W*S[take])@W.T
    Cnew=(W2*L[take2])@W2.T
    U1=c-np.sum(b0*b0/(4*S[take]))
    U2=w-np.sum(v0*v0/(4*L[take2]))
    bound=min(U1,U2)
    if not 0.<bound<25.:
        return None

    def integrand(u):
        t=bound*(1-u*u)
        return 2*bound*u*positive_probability_diag(S[take],b0,c-t)*positive_probability_diag(L[take2],v0,w-t)

    cap=quad(integrand,0.,1.,epsabs=1e-9,limit=44)[0]
    if model.name in ("top","bottom"):
        cap=positive_part(Anew,W@b0,c)+positive_part(Cnew,W2@v0,w)-cap
    new=KinkControl([tuple(polarity*z for z in (Anew,W@b0,c)),
                     tuple(polarity*z for z in (Cnew,W2@v0,w))],model.name,[0,1],[1.,1.])
    return new,float(cap*polarity)


def integrated_pair(f,model,polys):
    """Analytically integrate the piece selection along one Gaussian axis."""
    d=f.d
    moment=np.zeros((d,d))
    for A,b,c in polys:
        moment+=4*A@A+np.outer(b,b)
    vals,V=eigh(moment)
    ix=np.sum(vals>max(1e-12,vals[-1]*1e-9))
    V=V[:,-ix:][:,::-1].copy()
    aa=[V.T@A@V for A,b,c in polys]
    bb=[V.T@b for A,b,c in polys]
    cc=[q[2] for q in polys]

    def condition(X):
        size=len(X)
        p=[]
        for A,b,c in zip(aa,bb,cc):
            p.append((A[0,0],2*X@A[0,1:]+b[0],quad_values(X,A[1:,1:],b[1:],c)))
        rp=[]
        a,b,c=p[0]
        ap,bp,cp=p[1]
        for v,w,z in (p[0],p[1],(a-ap,b-bp,c-cp)):
            D=w*w-4*v*z
            if abs(v)>1e-12:
                ss=-.5*(w+np.copysign(np.sqrt(D.clip(0)),w))
                x=ss/v
                y=np.divide(z,ss,out=np.zeros_like(z),where=abs(ss)>1e-30)
                x[D<0]=0
                y[D<0]=0
            else:
                x=np.divide(-z,w,out=np.zeros_like(z),where=abs(w)>1e-14)
                y=np.zeros(size)
            rp.extend((x,y))
        r=np.sort(np.stack(rp,axis=1).clip(-10,10),axis=1)
        r=np.column_stack((np.full(size,-10.),r,np.full(size,10.)))
        P=ndtr(r)
        den=np.exp(-.5*r*r)/np.sqrt(2*np.pi)
        w0=P[:,1:]-P[:,:-1]
        w1=den[:,:-1]-den[:,1:]
        w2=w0+(den*r)[:,:-1]-(den*r)[:,1:]
        tt=.5*(r[:,:-1]+r[:,1:])
        f1=(a*tt+b[:,None])*tt+c[:,None]
        f2=(ap*tt+bp[:,None])*tt+cp[:,None]
        f0=model.apply(np.column_stack((f1.ravel(),f2.ravel()))).reshape(tt.shape)
        r1=(f0==f1)&(f0!=0)
        r2=(f0==f2)&(f0!=0)&(~r1)
        return np.sum(r1*(a*w2+b[:,None]*w1+c[:,None]*w0)+r2*(ap*w2+bp[:,None]*w1+cp[:,None]*w0),axis=1)

    if ix<=1:
        return float(condition(np.zeros((1,0)))[0])
    acc=[]
    for rr in range(2):
        sob=qmc.Sobol(ix-1,scramble=True,bits=32,rng=f.rng)
        chunks=[]
        for k in range(32):
            if f.left() < 6.5 or (len(chunks)>=4 and f.left()<10.):
                break
            w=ndtri(np.clip(sob.random(8192),1e-13,1-1e-13))
            chunks.append(float(np.mean(condition(w))))
        if chunks:
            acc.append((np.mean(chunks),len(chunks)))
    if not acc:
        return None
    return np.average([x[0] for x in acc],weights=[x[1] for x in acc])


def collection(f, X, Y):
    scale=max(np.std(Y),1e-8)
    data=bends(f,scale)
    if data is None or len(data[0])<2:
        return None
    C,U,J=data
    quadrics=[]
    sign=[]
    d=f.d
    for i in range(len(J)):
        if len(quadrics)>=3 or f.remaining<4350 or f.left()<9.:
            break
        if quadrics:
            qvals=np.array([quad_values(C[i:i+1],*par)[0] for par in quadrics])
            qgrad=np.array([(2*par[0]@C[i]+par[1])@U[i] for par in quadrics])
            if np.any((abs(qvals)<max(.00025,.005*scale)) &
                      (abs(abs(qgrad)-abs(J[i]))<max(.001,.14*abs(J[i])))):
                continue
            if len(quadrics)>=2:
                qs=qvals[0]+np.array([-1.,1.])*qvals[1]
                js=qgrad[0]+np.array([-1.,1.])*qgrad[1]
                if np.any((abs(qs)<max(.0005,.010*scale)) &
                          (abs(abs(js)-abs(J[i]))<max(.0015,.19*abs(J[i])))):
                    continue
        g=kink_gradient(f,C[i],U[i],J[i])
        if g is None:
            continue
        cr=fit_local_quadric(f,C[i],U[i],g)
        if cr is None:
            continue
        A,b,c=cr
        # Check that the learned local gradient and curvature extrapolate to
        # other observed crossings, rather than accepting spurious smooth bends.
        qa=quad_values(C,*cr)
        js=(2*C@A+b)
        jj=np.einsum("ij,ij->i",js,U)
        linked=(abs(qa)<max(.002,.028*scale))&(abs(abs(jj)-abs(J))<.17*abs(J)+.001*scale)
        if np.sum(linked)<2 and len(quadrics)==0 and f.remaining<7300:
            continue
        if np.linalg.norm(A)+np.linalg.norm(b)<.1*abs(J[i]):
            continue
        quadrics.append(cr)
        sign.append(np.copysign(1.,J[i]))
    if not quadrics:
        return None
    models=[]
    for k in range(1,len(quadrics)+1):
        for ii in itertools.combinations(range(len(quadrics)),k):
            # q and -q differ only by an integrable quadratic in a hinge.
            m=KinkControl(quadrics,"sum",list(ii),[1.]*k)
            m.sgn=np.asarray(sign)[list(ii)]
            models.append(m)
    nfound=len(quadrics)
    pair_list=list(itertools.combinations(range(nfound),2))
    # If two measured surfaces are successive differences of three pieces,
    # their sum or difference supplies the remaining surface without queries.
    for i,j in itertools.combinations(range(nfound),2):
        for sg in (-1.,1.):
            common=tuple(x+sg*y for x,y in zip(quadrics[i],quadrics[j]))
            k=len(quadrics)
            quadrics.append(common)
            pair_list.extend(((i,k),(j,k)))
    for i,j in pair_list:
        for sg in itertools.product((-1.,1.),repeat=2):
            for kind in ("top","bottom","cap","floor","median"):
                models.append(KinkControl(quadrics,kind,[i,j],sg))
    Q=np.column_stack([quad_values(X,*p) for p in quadrics])
    N=len(C)
    D=.006
    Z=np.concatenate((C-2*D*U,C-D*U,C+D*U,C+2*D*U),axis=0)
    Qs=np.column_stack([quad_values(Z,*p) for p in quadrics])
    # Training fit removes a smooth polynomial, so selection primarily
    # measures accuracy of the kink network itself.
    p=min(2*d+1,len(X))
    FX=np.column_stack((np.ones(len(X)),X,(X*X-1)))[:,:p]
    inv=fit_ridge(FX,np.eye(len(X)),.35)
    Jscale=np.mean(J*J)+1e-20
    scores=[]
    for m in models:
        jk=m.eval_values(Qs).reshape(4,N)
        dk=((jk[3]-jk[2])-(jk[1]-jk[0]))/D
        rr=Y-m.eval_values(Q)
        rr-=FX@(inv@rr)
        scores.append(np.mean(rr*rr)/(scale*scale)+.46*np.mean((dk-J)**2)/Jscale)
    best=int(np.argmin(scores))
    if scores[best]>1.02:
        return None
    return models[best]


def finish_kinks(f,X,Y,model):
    simplified=orthogonal_cap(model)
    if simplified is None:
        mean=model.integral(f)
    else:
        model,mean=simplified
    if mean is None or not np.isfinite(mean) or f.remaining<128:
        return None
    d=f.d
    resid=Y-model(X)
    nadd = 768 if f.remaining > 6800 else (384 if f.remaining > 3400 else 0)
    if nadd:
        extra=gaussian(nadd,d,f.rng)
        rr=f.query(extra)-model(extra)
        X=np.vstack((X,extra))
        resid=np.r_[resid,rr]
    ig=f.rng.permutation(len(X))[:min(32,max(0,(f.remaining-4096)//d))]
    G=fd_gradients(f,X[ig],resid[ig]+model(X[ig]))-model.gradient(X[ig]) if len(ig) else np.zeros((0,d))
    nn=np.linalg.norm(G,axis=1)
    stable=np.minimum(1.,max(1e-12,np.median(nn)*2.4)/np.maximum(nn,1e-15)) if len(G) else np.empty(0)
    GG=G*stable[:,None]
    mom=GG.T@GG/max(1,len(G))
    V=basis_from(mom+1e-10*np.eye(d))
    B,g,poly=fit_background(X,resid,V,gradients=(X[ig],G,stable))
    kern=None
    if len(G)>=12 and f.left()>8.:
        kern=smooth_kernel(X,resid-poly(X),V,np.sum((GG@V)**2,axis=0))
    if kern is None:
        return net_integral(f,V,lambda z: model(z)+poly(z),float(mean), float(np.mean(Y)))
    return net_integral(f,V,lambda z: model(z)+poly(z)+kern(z),float(mean),float(np.mean(Y)))


def hermite_features(X,V,degree):
    w=X@V
    n,d=w.shape
    i,j=np.triu_indices(d,1)
    W=[w,w*w-1.0,np.sqrt(2.)*w[:,i]*w[:,j]]
    if degree >= 3:
        k=min(10,d)
        m=min(6,d)
        z=w[:,:k]
        i,j=np.where(~np.eye(k,dtype=bool))
        W += [(z*z-3)*z,(z*z-1)[:,i]*z[:,j]]
        tri=np.asarray(list(itertools.combinations(range(m),3)))
        if len(tri):
            W += [np.prod(z[:,tri],axis=2)]
    if degree >= 4:
        w=w[:,:min(8,d)]
        yy=w*w
        i,j=np.triu_indices(w.shape[1],1)
        W += [yy*(yy-6)+3, (yy-1)[:,i]*(yy-1)[:,j]]
    return np.column_stack(W)


def hermite_jacobian(w, degree, columns):
    n,d=w.shape
    out=np.zeros((n,d,columns))
    ind=np.arange(d)
    out[:,ind,ind]=1.
    out[:,ind,d+ind]=2*w
    i,j=np.triu_indices(d,1)
    col=2*d+np.arange(len(i))
    out[:,i,col]=np.sqrt(2)*w[:,j]
    out[:,j,col]=np.sqrt(2)*w[:,i]
    p=2*d+len(i)
    if degree>=3:
        k=min(10,d)
        m=min(6,d)
        ind=np.arange(k)
        out[:,ind,p+ind]=3*w[:,:k]**2-3.
        p+=k
        i,j=np.where(~np.eye(k,dtype=bool))
        col=p+np.arange(len(i))
        out[:,i,col]=2*w[:,i]*w[:,j]
        out[:,j,col]=w[:,i]**2-1.
        p+=len(i)
        tri=np.asarray(list(itertools.combinations(range(m),3)))
        if len(tri):
            col=p+np.arange(len(tri))
            i,j,k=tri.T
            out[:,i,col]=w[:,j]*w[:,k]
            out[:,j,col]=w[:,i]*w[:,k]
            out[:,k,col]=w[:,i]*w[:,j]
            p+=len(tri)
    if degree>=4:
        k=min(d,8)
        z=w[:,:k]
        ind=np.arange(k)
        out[:,ind,p+ind]=4*z**3-12*z
        p+=k
        i,j=np.triu_indices(k,1)
        col=p+np.arange(len(i))
        out[:,i,col]=2*z[:,i]*(z[:,j]**2-1)
        out[:,j,col]=2*z[:,j]*(z[:,i]**2-1)
    return out


def fit_background(X,Y,V,degree=4,gradients=None):
    d=X.shape[1]
    DX=hermite_features(X,V,degree)
    med=np.median(Y)
    dis=np.median(np.abs(Y-med))+.04*np.std(Y)+1e-16
    yc=np.clip(Y,med-5*dis,med+5*dis)-med
    reg=max(8.0,.075*DX.shape[1])
    whole=np.column_stack((np.ones(len(X)),DX))
    if gradients is not None and len(gradients[0])>=8:
        from scipy.linalg import solve
        XX,G,wt=gradients
        deriv=hermite_jacobian(XX@V,degree,DX.shape[1])
        deriv*=wt[:,None,None]
        df=deriv.reshape(-1,DX.shape[1])
        obs=((G@V)*wt[:,None]).ravel()
        wgt=.46*len(X)/len(XX)
        A=whole.T@whole
        A.flat[::len(A)+1]+=reg
        A[1:,1:]+=wgt*(df.T@df)
        rhs=whole.T@yc
        rhs[1:]+=wgt*(df.T@obs)
        beta=solve(A,rhs,assume_a="pos",check_finite=False)
    else:
        beta=fit_ridge(whole,yc,reg)
    return beta[0]+med,beta,lambda z: hermite_features(z,V,degree)@beta[1:]
