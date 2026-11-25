"""Projected ellipsoid chord geometry with constant or log-linear jumps."""
import numpy as np
from scipy import linalg
from scipy.special import chndtr

from common import log, normalize
from hard_geom import diffs


class Jump:
    def __init__(self, a, b, level):
        self.a, self.b, self.level = a, b, level
        self.coef = None
        self.aq = None
        self.reflect = None

    def heights(self,z):
        return self.a*np.exp(np.clip(z@self.b,-35.,35.))

    def center(self,z):
        if self.coef is None:
            return self.level
        loc = self.coef[0]+z@self.coef[1:]
        return np.clip(loc, self.level-self.band, self.level+self.band)

    def classify(self,z,f):
        if self.aq is not None:
            p,n,a,rank = self.aq
            u=z-p
            if rank==len(p):
                q=a*np.sum(u**2,axis=1) + u@n
            else:
                q=a*(u@normalize(n))**2+u@n
            diff=q-1.
            return np.where(np.abs(diff)>1e-5,diff<0,
                            f>self.center(z)+.5*self.heights(z))
        return f>self.center(z)+.5*self.heights(z)

    def residual(self,z,f):
        a = self.heights(z)
        hit = self.classify(z,f)
        if self.reflect is not None and self.aq is None:
            s,p,u,amag,ng,endlevel,side = self.reflect
            if self.coef is not None and s.remaining>8 and s.elapsed<22.:
                center = self.center(z)
                band = max(.037,self.band*.70)
                Y=z-p
                axial=Y@u
                T=1.-axial*(amag*axial+ng)
                hit[T<-2e-6]=False
                uncertainty=abs(f-center-.5*a)
                w=(T>2e-6)&(a<4.1*band)&(uncertainty<1.18*band)
                ix=np.flatnonzero(w)
                if len(ix):
                    ix=ix[np.argsort(uncertainty[ix]/np.maximum(a[ix],.05))]
                    ix=ix[:min(len(z)//36+3,s.remaining)]
                    V=Y[ix]-axial[ix,None]*u
                    shrink=np.minimum(.14,.54/(1e-20+np.linalg.norm(V,axis=1)))
                    C=1.-shrink*shrink*T[ix]
                    rad=np.sqrt(np.maximum(0.,ng*ng+4.*amag*C))
                    axial1=(-ng+side*rad)/(2.*amag)
                    zz=p+axial1[:,None]*u+shrink[:,None]*V
                    hh=self.heights(zz)
                    keep=(hh>max(.11,2.*band))&(np.max(abs(zz),axis=1)<39.9)
                    if np.any(keep):
                        g0=s.query(zz[keep])
                        hit[ix[keep]]=g0>endlevel+.5*hh[keep]
        return f-a*hit

    def fit_base(self,x,y):
        r = self.residual(x,y)
        baseline = float(np.median(r))
        rband = np.quantile(abs(r-baseline),.90)
        if not (0. < rband < .75):
            return
        for _ in range(3):
            m = abs(r-baseline)<3*rband+.03
            z = np.c_[np.ones(np.sum(m)),x[m]]
            G = z.T@z
            G.flat[::len(G)+1] += .38
            self.coef = linalg.solve(G,z.T@r[m],assume_a='pos')
            self.level,self.band=baseline, 2.0*rband+.015
            r=self.residual(x,y)


def get_jump(s, events):
    e=max(events,key=lambda a:abs(a.d[0])*.7+abs(a.d[1]))
    d=s.d
    dh=np.asarray([abs(q.d[0]) for q in events])
    if np.ptp(dh)<max(1e-7,8e-8*np.median(dh)) and abs(e.d[1])<1e-6:
        return Jump(float(np.median(dh)),np.zeros(d),float(np.median([q.base for q in events])))
    h=.0010
    X=np.r_[np.zeros((1,d)),h*np.eye(d),-h*np.eye(d)]
    D = diffs(s,e.z,e.v,X,.011,5)
    if np.any(D*D[0]<=0) or np.max(np.abs(D/D[0]-1.))>.02:
        h=.00032
        D = diffs(s,e.z,e.v,X*(h/.001),.012,5)
    if np.any(D*D[0]<=0) or np.max(np.abs(D/D[0]-1.))>.02:
        return None
    D=np.log(np.abs(D))
    b=(D[1:d+1]-D[d+1:])/(2*h)
    A=float(np.exp(D[0]-e.z@b))
    if np.linalg.norm(b)>2.5 or not (0.03 < A < 1e6):
        return None
    Ze=np.stack([q.z for q in events])
    expect=A*np.exp(Ze@b)
    if np.max(np.abs(expect-dh)/dh)>.02:
        return None
    log("jump coefficient",A,"tilt norm",np.linalg.norm(b))
    return Jump(A,b,float(np.median([q.base for q in events])))


def find_exits(s,model,p,V,repair=True):
    N=len(V)
    if N==0:
        return np.empty(0), np.empty(0,dtype=bool)
    cap=np.minimum((39.6-np.max(np.abs(p)))/np.max(np.abs(V),axis=1),64.)
    R=np.zeros(N)
    L=np.zeros(N)
    H=np.minimum(1.3,cap)
    ok=np.zeros(N,dtype=bool)
    ix=np.arange(N)
    for k in (1.0,1.75,2.7,3.9,5.55,7.8,11.,16.,23.,34.,48.,70.):
        if len(ix)==0 or s.remaining<len(ix)+64:
            break
        h=np.minimum(cap[ix],k)
        z=p+h[:,None]*V[ix]
        inside=model.classify(z,s.query(z))
        ok[ix[~inside]]=True
        H[ix[~inside]]=h[~inside]
        L[ix[inside]]=h[inside]
        ix=ix[inside & (h<cap[ix])]
    ix=np.flatnonzero(ok)
    num=len(ix)
    if num>0 and s.remaining>2*num+40:
        u = V[ix]
        a,b = L[ix],H[ix]
        iters=min(29,(s.remaining-2*num-40)//num)
        for _ in range(iters):
            h=(a+b)*.5
            z = p+h[:,None]*u
            hit=model.classify(z,s.query(z))
            a,b=np.where(hit,h,a),np.where(hit,b,h)
        r=(a+b)*.5
        q=p+(np.r_[r-.00032,r+.00032])[:,None]*np.tile(u,(2,1))
        fy=s.query(q)
        gap=fy[:num]-fy[num:]
        good=gap>.65*model.heights(p+r[:,None]*u)
        ok[ix[~good]]=False
        R[ix[good]]=r[good]
    else:
        ok[:]=False
    left=np.flatnonzero(~ok)
    if not repair or len(left)==0 or s.remaining<600 or s.elapsed>15.:
        return R,ok
    # If a smooth fluctuation hid the jump from the level test, find
    # the actual raw drop on a denser linear scan of that same ray.
    left=left[:min(len(left),s.remaining//120,25)]
    if not len(left):
        return R,ok
    en=np.minimum(cap[left],32.)
    frac=np.linspace(0,1,67)
    ts=en[:,None]*frac
    q=p+ts[:,:,None]*V[left,None,:]
    yf=s.query(q.reshape(-1,s.d)).reshape(len(left),len(frac))
    yf1=yf[:,:-1]-yf[:,1:]
    r=(ts[:,:-1]+ts[:,1:])*.5
    q=p+r[:,:,None]*V[left,None,:]
    coeff=model.heights(q.reshape(-1,s.d)).reshape(r.shape)
    score=yf1/np.maximum(.065,coeff)
    jj=np.argmax(score,axis=1)
    ii=np.arange(len(left))
    gr=(score[ii,jj]>.62)&(yf1[ii,jj]>.05)
    left,i,j=left[gr],ii[gr],jj[gr]
    if not len(left):
        return R,ok
    a,b=ts[i,j],ts[i,j+1]
    v=V[left]
    midf=.5*(yf[i,j]+yf[i,j+1])
    for _ in range(min(29,(s.remaining-2*len(left)-24)//len(left))):
        r=(a+b)*.5
        hit=s.query(p+r[:,None]*v)>midf
        a,b=np.where(hit,r,a),np.where(hit,b,r)
    r=(a+b)*.5
    if s.remaining>=2*len(left):
        fv=s.query(p+(np.r_[r-.00035,r+.00035])[:,None]*np.tile(v,(2,1)))
        good=(fv[:len(left)]-fv[len(left):])>.62*model.heights(p+r[:,None]*v)
        ok[left[good]]=True
        R[left[good]]=r[good]
    return R,ok


def chords(s,model,p,V,repair=True):
    R,ok=find_exits(s,model,p,np.r_[V,-V],repair)
    N=len(V)
    good=ok[:N]&ok[N:]
    R=np.maximum(R,1e-100)
    a=1./(R[:N]*R[N:])
    b=1./R[:N]-1./R[N:]
    return a,b,good


def jump_integral(s,x,y,events,lines):
    model=get_jump(s,events)
    if model is None:
        return None
    model.fit_base(x,y)
    X,V,t,fy=lines
    xp=(X[:,None]+t[None,:,None]*V[:,None]).reshape(-1,s.d)
    x1=np.r_[x,xp]
    y1=np.r_[y,fy.ravel()]
    tt=model.classify(x1,y1)
    if not np.any(tt):
        return None
    points=x1[tt]
    if len(points)>1:
        best=points[np.argmin(np.linalg.norm(points,axis=1))]
        center=.75*points.mean(axis=0)+.25*best
    else:
        center=points[0]
    e=max(events,key=lambda e:abs(e.d[0]))
    Xtrial=np.r_[[center, .7*center+.3*best if len(points)>1 else center],points[:6],
                 e.z+np.sign(e.d[0])*np.array([.13,.27,.39])[:,None]*e.v]
    obs=s.query(Xtrial)
    valid=model.classify(Xtrial,obs)
    if not np.any(valid):
        return None
    p=Xtrial[np.flatnonzero(valid)[0]]
    # Orthogonal chords give the complete gradient and the trace.
    d=s.d
    Q=linalg.qr(s.rng.standard_normal((d,d)),mode='economic')[0].T
    A,B,ok=chords(s,model,p,Q)
    if np.count_nonzero(ok)<max(1,d//3):
        return None
    if not np.all(ok) and s.remaining>(d-np.count_nonzero(ok))*128+1600:
        u=Q[np.argmax(np.where(ok,A,-np.inf))]
        a0=A[np.argmax(np.where(ok,A,-np.inf))]
        ix=np.flatnonzero(~ok)
        V=np.r_[Q[ix]+2.*u,Q[ix]-2.*u]
        vv=np.linalg.norm(V,axis=1)
        aa,bb,gg=chords(s,model,p,V/vv[:,None])
        n=len(ix)
        okay=gg[:n]&gg[n:]
        t=ix[okay]
        A[t]=(.5*(aa[:n]*vv[:n]**2+aa[n:]*vv[n:]**2)-4.*a0)[okay]
        B[t]=(.5*(bb[:n]*vv[:n]+bb[n:]*vv[n:]))[okay]
        ok[t]=True
    log("chords good",np.count_nonzero(ok),"of",d,"used",s.used)
    if not np.all(ok):
        return None
    n=Q.T@B
    if np.linalg.norm(n)<1e-8:
        return None
    shift=p-model.b
    vs=np.r_[normalize(n)[None,:],normalize(shift)[None,:],normalize(s.rng.standard_normal((3,d)))]
    aa,bb,good=chords(s,model,p,vs)
    if not np.all(good[:2]):
        return None
    a=aa[0]
    k=int(round(A.sum()/a))
    if not (1<=k<=d) or abs(A.sum()/a-k)>.016:
        return None
    if np.max(np.abs(vs[good]@n-bb[good]))>3e-6*max(1.,np.linalg.norm(n)):
        return None
    if np.max(A)>a*(1.+.0001) or np.min(A)<-a*.0001:
        return None
    aa0=aa[1]*(shift@shift)
    lam=(aa0-shift@n+(n@n)/(4.*a))/a
    r2=(1.+(n@n)/(4.*a))/a
    prob=chndtr(r2,k,max(0.,lam))
    I=model.a*np.exp(.5*(model.b@model.b))*prob
    if not np.isfinite(I) or I<0.:
        return None
    if k==d or k==1:
        model.aq=p,n,a,k
    elif np.linalg.norm(model.b)>1e-6:
        u=normalize(n)
        ng=np.linalg.norm(n)
        side=1. if model.b@u>=0 else -1.
        edge=(-ng+side*np.sqrt(ng*ng+4.*a))/(2.*a)
        cp=p+(edge+side*.000025)*u
        if np.max(abs(cp))<39.8 and s.remaining>50:
            endlevel=s.query(cp)[0]
            if abs(endlevel-model.level)<.6:
                model.reflect=s,p,u,a,ng,endlevel,side
    log("closed jump model rank",k,"radius",np.sqrt(r2),"noncentral",lam,"hit probability",prob,"mean jump",I)
    return I,model.residual
