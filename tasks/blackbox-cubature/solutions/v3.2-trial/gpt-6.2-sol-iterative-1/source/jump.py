"""Closed-surface learning from comparisons on rays, with residual quadrature."""

import math

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gammaincinv, gammaln, roots_legendre

from common import gauss, gradient, ridge
from models import Quadratic, quadratic_law
from smooth import SmoothModel


class Separator:
    def __init__(self, w, x, y, scan, edges):
        self.w = w
        self.x = x
        _,_,p,u,_,delta,l,r,yl,yr = edges[0]
        v = u*np.sign(delta)
        h = max(2.e-4, .75*np.linalg.norm(l-r))
        h = min(.008,h)
        h = max(.0009,h)
        xx = p+np.array([-2.,-1.,1.,2.])[:,None]*h*v
        yj = w.ask(xx)
        base = 2*yj[1]-yj[0]
        jump = 2*yj[2]-yj[3]-base
        if jump<.5*abs(delta) or jump>1.5*abs(delta):
            jump,base = abs(delta),min(yl,yr)
        self.p = p
        self.amp = max(1.e-7,jump)
        self.base = base
        self.bmodel = None
        self.constant = self.amp>max(5.,18.*(np.quantile(y,.85)-np.quantile(y,.15)))
        if self.constant:
            self.beta = np.zeros(w.d)
            jumps = [abs(e[5]) for e in edges if e[0]=='jump' and .85*self.amp < abs(e[5]) < 1.15*self.amp]
            if jumps and np.ptp(jumps)<.00003*self.amp:
                self.amp = float(np.median(jumps+[self.amp]))
        else:
            _,g = gradient(w,p+np.array([-1.,1.])[:,None]*h*v,h=.000025)
            self.beta = (g[1]-g[0])/self.amp
            if np.linalg.norm(self.beta)>1.8:
                self.beta = np.zeros(w.d)
            nearby = [e for e in edges if e[0]=='jump']
            if len(nearby)>=max(4,w.d//3):
                A = np.array([e[2]-p for e in nearby])
                dy = np.log(np.maximum(1.e-9,[abs(e[5])/self.amp for e in nearby]))
                G = A.T@A
                G.flat[::w.d+1] += .003
                self.beta += np.linalg.solve(G,A.T@(dy-A@self.beta))
            sx, sy = scan[-2].reshape(-1,w.d), scan[-1].ravel()
            norm = np.linalg.norm(sx,axis=1)
            okay = norm<max(10.,1.5*np.sqrt(w.d))
            X = np.concatenate([x,sx[okay][::3]])
            Y = np.r_[y,sy[okay][::3]]
            J = self.weight(X)
            B = np.full(len(Y),base)
            scale = np.full(w.d,min(.052,1.35/w.d))
            for _ in range(2):
                b = Y-J*(Y>B+J*.50)
                keep = np.abs(b-base)<max(self.amp*.45,.22)
                if np.sum(keep)<100:
                    break
                self.bmodel = SmoothModel(X[keep],b[keep],scale,.014)
                B = self.bmodel(X)

    def weight(self, x):
        return self.amp*np.exp(np.clip((x-self.p)@self.beta,-14.,10.))

    def label(self, x, values):
        b = self.bmodel(x) if self.bmodel is not None else self.base
        return values>b+.5*self.weight(x)

    def labels(self,x):
        return self.label(x,self.w.ask(x))


def _anchor(w, S, x, y, scan, edges):
    origin = np.zeros((1,w.d))
    if S.labels(origin)[0]:
        return origin[0]
    X = np.concatenate([x,scan[-2].reshape(-1,w.d)])
    Y = np.r_[y,scan[-1].ravel()]
    is_in = S.label(X,Y)
    if np.any(is_in):
        p = X[is_in]
        # Averaging several nearby members moves towards the interior of a convex set.
        med = p.mean(axis=0)
        if S.labels(med[None,:])[0]:
            return med
        return p[np.argmin(np.linalg.norm(p,axis=1))]
    e = edges[0]
    out = np.asarray(e[7] if e[9]>e[8] else e[6])
    return out


def _symmetric_chords(w, S, a, V):
    """Track the two branches locally on signed axes before fitting off-diagonals."""
    d = w.d
    n = len(V)
    k = np.array([0.,.35,.58,.78,.97,1.19,1.47,1.86,2.40,3.30,4.6])
    t = k*np.sqrt(d)
    t = np.minimum(t,39.-np.max(np.abs(a)))
    info = np.flatnonzero(np.r_[True,np.diff(t)>1.e-10])
    t = t[info]
    pts = a+t[None,:,None]*V[:,None,:]
    flat = pts.reshape(-1,d)
    y = w.ask(flat).reshape(n,-1)
    J = S.weight(flat).reshape(n,-1)
    gp = (S.bmodel(flat).reshape(n,-1) if S.bmodel is not None
          else np.ones_like(y)*S.base)
    score = np.full((n,len(t)-1),np.inf)
    tmer = np.diff(t)
    for i in range(len(t)-1):
        f = y - J*(np.arange(len(t))[None,:]<=i)
        if i>0:
            trend = np.abs(np.diff(np.diff(f,axis=1)/tmer[None,:],axis=1))
            cost = np.mean(np.minimum(trend**2,.65),axis=1)
        else:
            cost = 0.
        close = (np.abs(f[:,i+1]-f[:,i]) / (.05+min(3.,tmer[i])))**2
        deviation = np.minimum(.45,(f-gp)**2)
        baseline = np.mean(deviation,axis=1)
        score[:,i] = cost+.20*close+.055*baseline
    j = np.argmin(score,axis=1)
    rows = np.arange(n)
    lo,hi = t[j].copy(),t[j+1].copy()
    a0=y[rows,j]-J[rows,j]
    b0=y[rows,j+1]
    for _ in range(14 if d>=40 else 17):
        mid=.5*(lo+hi)
        X = a+mid[:,None]*V
        jump = S.weight(X)
        value = w.ask(X)
        est=.5*(a0+b0)
        ins = value>est+.5*jump
        lo[ins],hi[~ins]=mid[ins],mid[~ins]
        a0[ins],b0[~ins]=value[ins]-jump[ins],value[~ins]
    return 2./np.maximum(1.e-10,lo+hi)


def comparative_geometry(w,S,a,tracked=False):
    """Diagonal and linear terms from signed chords, off-diagonal terms by bisection."""
    d = w.d
    eye = np.eye(d)
    V = np.concatenate([eye,-eye])
    if tracked:
        radius = _symmetric_chords(w,S,a,V)
    else:
        top = np.full(2*d,.8)
        bot = np.zeros(2*d)
        for _ in range(6):
            inside = S.labels(a+V/top[:,None])
            if np.all(inside):
                break
            top[~inside] *= 2.
        h = min(20, max(14, 16+int(d<=32)*3))
        for _ in range(h):
            mid = (top+bot)*.5
            inside = S.labels(a+V/np.maximum(mid[:,None],1.e-15))
            top[inside]=mid[inside]
            bot[~inside]=mid[~inside]
        radius = .5*(top+bot)
    pp,nn = radius.reshape(2,d)
    diag = pp*nn
    b = pp-nn
    diag = np.clip(diag,1.e-12,np.inf)
    aa = np.diag(diag)

    def compare(i,j,value):
        # Evaluate exactly at the trial quadratic threshold. Larger coefficients are outside.
        coupling = (np.where(S.beta[i]*S.beta[j] < 0., -1., 1.)
                    if tracked and d>=40 else np.ones(len(i)))
        bj = (b[i]+coupling*b[j])/math.sqrt(2.)
        k = np.maximum(1.e-13,.5*(diag[i]+diag[j])+coupling*value)
        sign = np.where(bj>=0,1.,-1.)
        if tracked:
            s = S.beta[i]+coupling*S.beta[j]
            sign = np.where(np.abs(s)<.01,sign,np.sign(s))
        mu = sign*bj
        sq = np.sqrt(bj*bj+4*k)
        recip = np.where(mu>=0,.5*(mu+sq),2*k/np.maximum(sq-mu,1.e-20))
        P = np.broadcast_to(a,(len(i),d)).copy()
        mul = sign/(math.sqrt(2.)*recip)
        rows = np.arange(len(i))
        P[rows,i] += mul
        P[rows,j] += coupling*mul
        return S.labels(P) == (coupling>0)

    def bisect(i,j,low,high,steps):
        low,high = low.copy(),high.copy()
        for k in range(steps):
            mid = (low+high)*.5
            smaller = compare(i,j,mid)
            high[smaller]=mid[smaller]
            low[~smaller]=mid[~smaller]
        return low,high

    # A few pivoted columns allow early completion when the ellipsoid occupies a low-rank subspace.
    P = np.empty((d,0))
    active = np.ones(d,dtype=bool)
    max_diag = diag.max()
    pivot_steps = min(4, d//4) if not tracked or d <= 32 else 0
    for k in range(pivot_steps):
        res = np.maximum(0.,diag-np.sum(P*P,axis=1))
        i0 = int(np.argmax(np.where(active,res,-1.)))
        if res[i0] < max_diag*.0006:
            break
        j = np.flatnonzero(active & (np.arange(d)!=i0))
        if len(j)==0 or w.left < len(j)*10+3100:
            break
        i = np.full(len(j),i0)
        cent = P[j]@P[i0]
        width = np.sqrt(res[i0]*res[j])+.00025*max_diag
        lo,hi = bisect(i,j,cent-width,cent+width,15 if d<=32 else 10)
        val = .5*(lo+hi)
        aa[i,j]=val
        aa[j,i]=val
        pp = np.zeros(d)
        pp[i0] = np.sqrt(res[i0])
        pp[j] = (val-cent)/pp[i0]
        P = np.column_stack([P,pp])
        active[i0]=False

    res = np.maximum(0.,diag-np.sum(P*P,axis=1))
    noise = None
    i,j=np.triu_indices(d,1)
    mask=active[i]&active[j]
    i,j=i[mask],j[mask]
    if len(i)==0:
        steps=16
        model_err=1.e-5
    elif (res[active].max() < .0007*max_diag) and P.shape[1]>0:
        z=P[i]@P[j].T if len(i)<50 else None
        value = np.sum(P[i]*P[j],axis=1) if z is None else np.diag(z)
        aa[i,j]=aa[j,i]=value
        steps=14
        model_err=1.1*max_diag*np.sqrt(d)*3.e-4
    else:
        cent = np.sum(P[i]*P[j],axis=1)
        scale = np.sqrt(res[i]*res[j])
        sd = np.std(np.divide(aa[np.triu_indices(d,1)],max_diag)) if P.size else .1
        if d>=40:
            if tracked:
                sk = w.rng.choice(len(i), min(80,len(i)), replace=False)
                sva = scale[sk]+.00065*max_diag
                scr = bisect(i[sk],j[sk],cent[sk]-sva,cent[sk]+sva,5)
                spec = np.quantile(np.abs((.5*(scr[0]+scr[1])-cent[sk])/sva),.82)
                frac = float(np.clip(1.70*spec,.16,.78))
            else:
                frac = float(np.clip(5.*sd,.21,.58))
        else:
            frac = 1.
        full = scale+.00065*max_diag
        wid = (frac*scale)+.00065*max_diag
        reserve = (3450 if tracked else 4300) if d>=40 else 3000
        steps = int(min(18,(w.left - reserve)//len(i)))
        steps = max(0,steps)
        if steps:
            lo,hi = bisect(i,j,cent-wid,cent+wid,steps)
            at_low = (hi<cent-wid+2.1*wid/(2**steps)) & (wid<full*.999)
            at_high = (lo>cent+wid-2.1*wid/(2**steps)) & (wid<full*.999)
            hot = at_low | at_high
            if np.any(hot):
                pp = min(steps+1,(w.left-(3000 if tracked else 3450))//hot.sum())
                if pp>0:
                    l0=np.where(at_low,cent-full,lo)
                    h0=np.where(at_high,cent+full,hi)
                    lo[hot],hi[hot]=bisect(i[hot],j[hot],l0[hot],h0[hot],pp)
            if tracked and d>=40:
                extra = min(len(i), max(0,w.left-3800))
                if extra>0:
                    p = np.argsort(hi-lo)[-extra:]
                    lo[p],hi[p] = bisect(i[p],j[p],lo[p],hi[p],1)
            val = .5*(lo+hi)
            aa[i,j]=aa[j,i]=val
            noise = np.sqrt(np.mean((hi-lo)**2)/12.)
            model_err = noise*max(1,d)*(2.8 if tracked else 2.5*math.sqrt(12.))
        else:
            aa[i,j]=aa[j,i]=cent
            model_err=float(np.sum(res)*.4)
    l,V = eigh(aa,check_finite=False)
    cutoff = max(1.e-8, 2.e-4*np.max(l))
    if steps<=6:
        cutoff = max(cutoff,.10*model_err/max(1.,math.sqrt(d)))
    if tracked and noise is not None and d>=40 and steps<=7:
        cutoff = max(cutoff, 2.3*noise*np.sqrt(d))
        if np.sum(l>cutoff)<.7*d:
            reliable = l>cutoff
            projected = V[:,reliable]@(V[:,reliable].T@b)
            if np.linalg.norm(projected-b)<max(.20*np.linalg.norm(b),3.*noise*np.sqrt(d)):
                b = projected
    l=np.where(l>cutoff,l,0.)
    A=(V*l)@V.T
    if tracked and noise is not None and d>=40 and steps<=7:
        J = np.sqrt(np.clip(diag/np.maximum(1.e-12,np.diag(A)), .60, 1.65))
        A=(A*J[:,None])*J[None,:]
    if not np.any(l):
        raise ArithmeticError('ellipsoid fit has zero rank')
    # Restore small truncation-induced normalization changes exactly at the anchor.
    return Quadratic(float(a@A@a-b@a-1.),b-2*A@a,A),model_err


def regular_geometry(w,S,anchor,pilot,edges):
    """Fit a quadric from short, locally tracked boundary chords."""
    d = w.d
    p = d*(d+3)//2
    n = min(max(190,int(1.50*p)), 2600)
    n = min(n, (w.left-3500)//4)
    P = w.normal(n,d)
    # Keep almost all of the sphere, but turn the very weakest jumps around.
    P /= np.linalg.norm(P,axis=1,keepdims=True)
    rr0 = float(np.median(np.linalg.norm(pilot,axis=1)))
    P[P@S.beta < -.65/rr0] *= -1
    e0 = np.asarray([np.linalg.norm(e[2]-anchor) for e in edges if e[0]=='jump'])
    nominal = np.clip(np.median(e0),.55*rr0,1.45*rr0)
    lo = np.full(n,.45*nominal)
    hi = np.full(n,1.65*nominal)
    ZA,ZB=anchor+lo[:,None]*P,anchor+hi[:,None]*P
    ya,yb=w.ask(np.concatenate([ZA,ZB])).reshape(2,n)
    sa,sb=S.label(ZA,ya),S.label(ZB,yb)
    edge_guard=min(1.75, 38./(1.65*nominal))
    faila,failb=~sa,sb
    if np.any(faila) or np.any(failb):
        lo[faila]*=.50
        hi[failb]*=edge_guard
        XA,XB=anchor+lo[faila,None]*P[faila],anchor+hi[failb,None]*P[failb]
        yy=w.ask(np.concatenate([XA,XB]))
        ya[faila], yb[failb] = yy[:faila.sum()],yy[faila.sum():]
        sa[faila],sb[failb]=S.label(XA,ya[faila]),S.label(XB,yb[failb])
    ok=sa&~sb
    if np.sum(ok)<min(p*.65, .74*n):
        raise ArithmeticError('too few tracked boundary chords')
    lo,hi,P,ya,yb=lo[ok],hi[ok],P[ok],ya[ok],yb[ok]
    J0=S.weight(anchor+lo[:,None]*P)
    base_a=ya-J0
    base_b=yb
    n = len(P)
    steps=min(19, (w.left-3250)//n)
    steps=max(0,steps)
    for it in range(steps):
        mid=.5*(lo+hi)
        pts=anchor+mid[:,None]*P
        val=w.ask(pts)
        J=S.weight(pts)
        local=.5*(base_a+base_b)
        if S.bmodel is not None:
            g=S.bmodel(pts)
            drift=np.abs(base_a-base_b)
            weight=np.clip(.20*(hi-lo),.10,.80)
            weight=np.where(drift>J*.90,.92,weight)
            local=(1-weight)*local+weight*g
        sel=val>local+.5*J
        lo[sel],hi[~sel]=mid[sel],mid[~sel]
        base_a[sel],base_b[~sel]=val[sel]-J[sel],val[~sel]

    T=.5*(lo+hi)
    width=.5*(hi-lo)
    ii,jj=np.triu_indices(d,1)
    Z=(T/rr0*np.sqrt(d))[:,None]*P
    F=np.column_stack([Z, Z*Z, math.sqrt(2.)*Z[:,ii]*Z[:,jj]])
    weights=np.minimum(3.,(rr0/T)**3)
    # Penalize distance to the spherical fit only when there are very few bits per chord.
    target=np.r_[np.zeros(d),np.full(d,(rr0/nominal)**2/d),np.zeros(len(ii))]
    resid=1.-F@target
    best=target+ridge(F*weights[:,None],resid*weights,max(1.e-7,n*(.012 if steps<=3 else .000025)))
    if steps>=5:
        for _ in range(2):
            e=F@best-1.
            er = max(.002,2.9*np.median(np.abs(e)))
            wt=weights*np.minimum(1.,er/np.maximum(1.e-15,np.abs(e)))
            best=target+ridge(F*wt[:,None],resid*wt,max(1.e-7,n*.000004))
    b=best[:d]*(math.sqrt(d)/rr0)
    A=np.diag(best[d:2*d])
    A[ii,jj]=A[jj,ii]=best[2*d:]/math.sqrt(2.)
    A*=d/rr0**2
    eig,V=eigh(A,check_finite=False)
    cutoff = eig.max()*(.00075 if steps>=9 else .009)
    A=(V*np.where(eig>cutoff,eig,0.))@V.T
    AA=Quadratic(float(anchor@A@anchor-b@anchor-1.),b-2*A@anchor,A)
    error=float(np.quantile(np.abs(AA(anchor+T[:,None]*P)),.88))
    error+=float(np.quantile(width/T,.80))*2.4
    return AA,max(1.e-5,error)


def _root_validation(w,S,q,anchor,err):
    d = w.d
    nr = 96 if d<=32 else 128
    if w.left<nr*3+900:
        return max(err,.012)
    dirs = w.rng.standard_normal((nr,d))
    dirs /= np.linalg.norm(dirs,axis=1,keepdims=True)
    a = np.sum((dirs@q.A)*dirs,axis=1)
    b = dirs@(q.b+2*q.A@anchor)
    D = np.sqrt(np.maximum(1.e-15,b*b+4*a))
    recip = .5*(b+D)
    r = 1./np.maximum(1.e-12,recip)
    # Exclude nearly unbounded rays: they provide little useful calibration in Gaussian space.
    ok = (r<26.) & (r>1.e-4)
    if np.sum(ok) < 8:
        return max(err,.005)
    dirs,a,b,r,D = dirs[ok],a[ok],b[ok],r[ok],D[ok]
    width = float(np.clip(max(.0018,err)*1.35,.002,.38))
    ok0=np.ones(len(r),dtype=bool)
    for _ in range(3):
        off = (width/np.maximum(D,1.e-9)).clip(.00001,.34*r)
        P = np.concatenate([anchor+(r-off)[:,None]*dirs,anchor+(r+off)[:,None]*dirs])
        label = S.labels(P).reshape(2,-1)
        ok0=label[0]&~label[1]
        if np.mean(ok0)>=.955 or w.left<3*len(r)+900:
            break
        width *= 2.7
    return width


def _radial_residual(w, q, S, gp, R, band, n, use_band=True):
    d = w.d
    lam, V = eigh(q.A,check_finite=False)
    act = lam>max(1.e-11,lam.max()*1.e-7)
    k = int(act.sum())
    if k<d and np.linalg.norm(q.b@V[:,~act])>1.e-4*max(1.,np.linalg.norm(q.b)):
        k=d
    E=V[:, -k:][:,::-1]
    if k<d:
        P=V[:,:d-k]
    U=w.uniforms(n,d+1)
    direction=gauss(U[:,:k])
    direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    X=direction@E.T
    off=np.zeros((n,d)) if k==d else gauss(U[:,k:d])@P.T
    a=np.sum((X@q.A)*X,axis=1)
    b=X@q.b+2*np.sum((X@q.A)*off,axis=1)
    c=q(off)
    edges=[np.zeros(n),np.ones(n)]
    for side in (-band,band):
        disc=b*b-4*a*(c-side)
        zero=np.maximum(a,1.e-30)
        s=np.sqrt(np.maximum(0.,disc))
        for r in ((-b-s)/(2*zero),(-b+s)/(2*zero)):
            r=np.maximum(0.,r)*(disc>0)
            edges.append(gammainc(k*.5,r*r*.5))
    edges=np.sort(np.asarray(edges),axis=0)
    W=(edges[1:]-edges[:-1])
    points=[]; weights=[]
    for i,cuts in enumerate(W):
        j=cuts>2.e-15
        if not np.any(j):
            continue
        uniform=(U[j,-1]+i*.245122333753)%1.
        t=np.clip(edges[i,j]+uniform*cuts[j],1.e-16,1-1.e-16)
        rad=np.sqrt(2*gammaincinv(k*.5,t))
        points.append(off[j]+rad[:,None]*X[j])
        weights.append(cuts[j])
    xx=np.concatenate(points)
    if len(xx)>w.left:
        xx=w.normal(w.left)@R.T
        weights=np.ones(len(xx))
        n=len(xx)
    else:
        weights=np.concatenate(weights)
    f=w.ask(xx)-S.weight(xx)*(q(xx)<=0.)
    if gp is not None:
        f-=gp(xx@R)
    return float(np.dot(weights,f)/n)


def _locally_sliced_residual(w,q,S,gp,R,band):
    """Recenter the radial control separately on each angle, then correct it."""
    d=w.d
    steps=2
    factor=3+2+steps
    n=2**int(np.floor(np.log2(max(1,w.left//factor))))
    if n<128:
        return None
    lam,V=eigh(q.A,check_finite=False)
    # Here the whole sphere is used: the reference anchor is the Gaussian mean.
    U=w.uniforms(n,d+3)
    ang=gauss(U[:,:d])
    ang/=np.linalg.norm(ang,axis=1,keepdims=True)
    D=ang@V.T
    A=ang**2 @ np.maximum(lam,1.e-20)
    B=D@q.b
    norm0=B*B

    def root(level):
        zz=np.sqrt(np.maximum(0.,norm0+4*A*(level-q.c)))
        return np.where(B>0, 2*(level-q.c)/np.maximum(1.e-15,zz+B), (zz-B)/(2*A))

    t=root(0.)
    bo=min(band,max(.006,-q.c*.48))
    lo=np.maximum(0.,root(-bo))
    hi=np.minimum(30.,root(bo))
    zz0,zz1=lo[:,None]*D,hi[:,None]*D
    a,b=w.ask(np.concatenate([zz0,zz1])).reshape(2,-1)
    left,right=S.label(zz0,a),S.label(zz1,b)
    ok=(left&~right)&(t<=hi)&(lo<t)
    p0,p1=lo.copy(),hi.copy()
    ya=a-S.weight(zz0)
    yb=b
    for _ in range(steps):
        mid=(lo+hi)*.5
        X=mid[:,None]*D
        f=w.ask(X)
        jump=S.weight(X)
        lab=f>.5*(ya+yb+jump)
        lab &= ok
        left, right = lab, (~lab)&ok
        lo[left],hi[right]=mid[left],mid[right]
        ya[left],yb[right]=f[left]-jump[left],f[right]
    new=.5*(lo+hi)
    new[~ok]=t[~ok]
    lo[~ok],hi[~ok]=p0[~ok],p1[~ok]
    low=np.minimum(lo,hi)
    high=np.maximum(lo,hi)
    nq,wq=roots_legendre(8)
    midpoint=.5*(new+t)
    widths=.5*(new-t)
    rr=midpoint[:,None]+widths[:,None]*nq
    normconst=(d*.5-1)*math.log(2.)+float(gammaln(d*.5))
    mass=np.exp((d-1)*np.log(np.maximum(rr,1.e-20))-.5*rr**2-normconst)
    beta=D@S.beta
    jump=S.amp*np.exp(np.clip(rr*beta[:,None]-S.p@S.beta,-14.,10.))
    corr=widths*np.sum(mass*jump*wq,axis=1)
    cuts=gammainc(.5*d,.5*np.stack([low,high])**2)
    edges=np.vstack([np.zeros(n),cuts,np.ones(n)])
    delta=np.diff(edges,axis=0)
    XX=[]
    rads=[]
    for j in range(3):
        u=np.clip(edges[j]+U[:,d+j]*delta[j],1.e-16,1-1.e-16)
        rr=np.sqrt(2*gammaincinv(.5*d,u))
        XX.append(rr[:,None]*D)
        rads.append(rr)
    X=np.concatenate(XX)
    raw=(w.ask(X)-S.weight(X)*(np.concatenate(rads)<=np.tile(new,3)))
    if gp is not None:
        raw-=gp(X@R)
    return float(np.mean(corr+np.sum(delta*raw.reshape(3,-1),axis=0)))


def discontinuous(w,x,y,scan,edges):
    S=Separator(w,x,y,scan,edges)
    a=_anchor(w,S,x,y,scan,edges)
    if S.constant:
        geo,err = comparative_geometry(w,S,a)
    elif w.d>=24:
        geo,err = comparative_geometry(w,S,a,tracked=True)
    else:
        geo,err = regular_geometry(w,S,a,x,edges)
    band=_root_validation(w,S,geo,a,err)
    amp=S.amp*np.exp(float(-S.p@S.beta+.5*S.beta@S.beta))
    mean=amp*quadratic_law(geo.translated(S.beta).scale(-1.),'cdf')
    pred = lambda z: S.weight(z)*(geo(z)<=0.)
    s1 = S.label(x,y)
    rem = y-S.weight(x)*s1
    from smooth import infer_frame
    # Smooth residual fitting operates on the two learned analytic branches.
    R,sc=infer_frame(w,x,rem,pred,nmax=17)
    scales=np.clip(.022+sc*.68,.019,.6)
    safe = np.abs(rem-np.median(rem))<max(.25,5*np.quantile(np.abs(rem-np.median(rem)),.75))
    gp = None
    if np.sum(safe)>100:
        gp=SmoothModel((x@R)[safe],rem[safe],scales*.60,4.e-5)
    base=0. if gp is None else gp.mean
    if w.left<256:
        return float(mean+base)
    if band<.005 and S.amp<4:
        xx=w.normal(w.left)@R.T
        f=w.ask(xx)-pred(xx)
        add=np.mean(f if gp is None else f-gp(xx@R))
    elif band<.0001:
        xx=w.normal(w.left)@R.T
        f=w.ask(xx)-pred(xx)
        add=np.mean(f if gp is None else f-gp(xx@R))
    else:
        add=None
        if (not S.constant and w.d>=32 and band>.013 and np.linalg.norm(a)<1.e-12):
            add=_locally_sliced_residual(w,geo,S,gp,R,band)
        if add is None:
            n = min(4096, 2**int(np.floor(np.log2(max(1,w.left//3)))))
            if geo.c>0.:
                n = min(n,2**int(np.floor(np.log2(max(1,w.left//5)))))
            add = _radial_residual(w,geo,S,gp,R,band,n)
    return float(mean+base+add)
