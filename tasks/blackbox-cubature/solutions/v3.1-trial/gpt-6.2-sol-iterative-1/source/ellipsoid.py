"""Geometry learning and conditional correction for a closed jump surface."""

import numpy as np
from scipy.linalg import qr
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincinv, ndtr, ndtri

from cubature import Quadratic, norm_rows, quadratic_cdf, uniform
from kinks import debug
from smooth import smooth_control, smooth_correct


class Jump:
    def __init__(self, scale, tilt, anchor, base):
        self.scale, self.tilt, self.anchor, self.base = scale, tilt, anchor, base
        self.level = scale*np.exp(np.clip(-tilt@anchor,-200,200))
        self.linear = None

    def amplitude(self, z):
        return self.scale*np.exp(np.clip((z-self.anchor)@self.tilt, -200, 200))

    def indicator(self, z, y):
        pred = self.base if self.linear is None else self.base + np.tanh(z@self.linear/.3)*.3
        return y > pred + .5*self.amplitude(z)

    def smooth(self, z, y):
        return y-self.amplitude(z)*self.indicator(z,y)


def _jump_at(s, atlas, i, j):
    d = s.d
    v, base = atlas.v[i], atlas.base[i]
    h = atlas.h
    yy = atlas.y[i]
    dif = np.diff(yy)
    k = min(range(max(1,j-2),min(len(dif)-2,j+2)), key=lambda k: -abs(dif[k]))
    # Interpolate the two branches from outside the suspect cell.
    a, b = atlas.t[k], atlas.t[k+1]
    yl, yr = yy[k], yy[k+1]
    tl = np.arange(-2,1)
    tr = -tl[::-1]
    g1 = np.polynomial.polynomial.polyfit(tl, yy[k-2:k+1], 2)
    g2 = np.polynomial.polynomial.polyfit(tr, yy[k+1:k+4], 2)
    l0, r0 = a, b
    for _ in range(12):
        t = .5*(a+b)
        l = np.polynomial.polynomial.polyval((t-l0)/h,g1)
        r = np.polynomial.polynomial.polyval((t-r0)/h,g2)
        u = s.query([base+t*v])[0]
        if abs(u-l) < abs(u-r):
            a, yl = t, u
        else:
            b, yr = t, u
    amp = abs(yl-yr)
    if amp < .075*atlas.scale or amp < .25*abs(dif[k]):
        return None
    # At this scale the difference of the two analytic extensions is a local
    # exponential model. The large-constant case needs no gradient queries.
    root = base+.5*(a+b)*v
    high = base+(b if yr>yl else a)*v
    value = min(yl,yr)
    if amp > 15*atlas.scale:
        tilt = np.zeros(d)
        amp = _refine_gap(s, root, v, np.sign(yr-yl), h, amp)[0]
    else:
        sa = np.sign(yr-yl)
        off = max(3e-5, (b-a)*0.9)
        pts = root+sa*np.array([-off,off])[:,None]*v
        step = 6.e-6
        z = np.concatenate([pts[:,None,:],pts[:,None,:]+step*np.eye(d),
                            pts[:,None,:]-step*np.eye(d)],axis=1)
        Y = s.query(z).reshape(2,1+2*d)
        grad = (Y[:,1:d+1]-Y[:,d+1:])/(2*step)
        amp1, value1 = _refine_gap(s, root, v, sa, h, amp)
        if amp1 <= 0 or abs(amp1-amp) > 0.02*amp:
            return None
        amp, value = amp1, value1
        tilt = (grad[1]-grad[0])/amp
        if np.linalg.norm(tilt) > 1.8 or abs(Y[1,0]-Y[0,0]-amp) > .02*amp:
            return None
    if not np.isfinite(amp) or amp <= 0.0:
        return None
    return Jump(amp, tilt, root, value), high


def _refine_gap(s, root, v, sign, h, old):
    off = .04*h*np.arange(1,6)
    u = np.r_[-off,off]
    y = s.query(root+sign*u[:,None]*v)
    a = np.polynomial.polynomial.polyfit(-off/off[-1], y[:5], 4)[0]
    b = np.polynomial.polynomial.polyfit(off/off[-1], y[5:], 4)[0]
    return b-a, a


def get_jump(s, rng, atlas, zp, yp):
    for i,j in atlas.candidates()[:12]:
        a = atlas.y[i]
        delta = max(abs(np.diff(a)[max(0,j-2):j+2]), default=0.)
        if delta < .078*atlas.scale:
            break
        result = _jump_at(s, atlas, i,j)
        if result is not None:
            jump, inner = result
            # Fit a bounded first-order guess of the smooth branch. This is
            # only for identifying which side a point belongs to.
            train = np.vstack([zp, atlas.z.reshape(-1,s.d)])
            data = np.r_[yp, atlas.y.ravel()]
            for _ in range(3):
                gg = jump.smooth(train,data)
                center = np.median(gg)
                sc = max(np.quantile(abs(gg-center),.75), 1e-6)
                wt = np.minimum(1, 1.8*sc/np.maximum(1e-20,abs(gg-center)))
                if abs(center-jump.base) > .5*jump.scale and jump.scale < 2*atlas.scale:
                    break
                jump.base = center
                jump.linear = np.linalg.solve(train.T@(wt[:,None]*train) +
                                              .5*len(train)*np.eye(s.d), train.T@(wt*(gg-center)))
            return jump, inner
    return None


def search_large_jump(s,rng,atlas,zp,yp):
    """A second, sparse-event search, not requiring an atlas line to hit."""
    from cubature import normals
    d=s.d
    Z=np.r_[zp,atlas.z.reshape(-1,d)]
    Y=np.r_[yp,atlas.y.ravel()]
    scale=max(1e-10,np.median(Y))
    if Y.max()<14*scale and s.remaining>12000:
        zz=np.vstack([np.zeros((1,d)), .46*normals(rng,8,d),
                      normals(rng,9,d),1.55*normals(rng,9,d)])
        yy=s.query(zz)
        Z,Y=np.r_[Z,zz],np.r_[Y,yy]
    labels=Y>14*scale
    if not np.any(labels):
        return None
    small=Z[~labels]
    u=Z[np.flatnonzero(labels)[np.argmin(np.linalg.norm(Z[labels],axis=1))]]
    v=small[np.argmin(np.linalg.norm(small-u,axis=1))]-u
    r=np.linalg.norm(v)
    if r==0:
        return None
    v/=r
    a,b=0.,r
    for _ in range(16):
        t=(a+b)*.5
        x=s.query([u+t*v])[0]
        if x>14*scale:
            a=t
        else:
            b=t
    p=u+.5*(a+b)*v
    h=max(5e-4,min(.10,.035*r))
    amp,base=_refine_gap(s,p,v,-1.,h,scale)
    if amp>13*scale:
        return Jump(amp,np.zeros(d),p,base),u
    return None


def _anchor(s, rng, jump, inner, zp, yp, atlas):
    d = s.d
    x = np.r_[np.zeros((1,d)), .45*zp[:96], .73*zp[:64], 1.65*zp[:128]]
    y = s.query(x)
    z = np.vstack([x,zp,atlas.z.reshape(-1,d),inner[None]])
    yy = np.r_[y,yp,atlas.y.ravel(),s.query([inner])]
    inside = jump.indicator(z,yy)
    p = z[inside]
    if not len(p):
        return None
    # Averaging several interior points is safe for the convex model. Verify
    # the centre and prefer zero when it is inside with a sensible margin.
    order = np.argsort(np.linalg.norm(p,axis=1))
    u = p[order]
    cand = np.stack([u[0], np.mean(u[:min(10,len(u))],axis=0),np.mean(p,axis=0)])
    good = jump.indicator(cand,s.query(cand))
    if good[0] and np.linalg.norm(cand[0]) < 1e-15:
        return cand[0]
    if good[1]:
        return cand[1]
    if good[2]:
        return cand[2]
    return cand[0]


def _radial_axes(s, jump, anchor, directions, steps):
    n = len(directions)
    limit = np.min(np.where(abs(directions)>1e-12,
                            (39.999-np.copysign(1.,directions)*anchor)/
                            np.maximum(abs(directions),1e-12), 1e10),axis=1)
    lo, hi = np.zeros(n), limit.copy()
    u = np.minimum(2.5,limit)
    located = np.zeros(n,dtype=bool)
    for j in range(steps):
        inside = jump.indicator(anchor+u[:,None]*directions,s.query(anchor+u[:,None]*directions))
        lo = np.where(inside,u,lo)
        hi = np.where(inside,hi,u)
        located |= ~inside
        u = np.where(located, .5*(lo+hi), np.minimum(hi,u*2))
        # If the inside anchor lies close to the surface, keep relative rather
        # than absolute resolution as soon as a positive lower end is known.
        u = np.where((lo>0)&located, np.sqrt(lo*hi), u)
    t = np.sqrt(lo*hi)
    if np.linalg.norm(jump.tilt) > 1e-8:
        h = np.maximum(.0012,.0005*t)
        ctr = anchor+t[:,None]*directions
        Y = s.query(np.r_[ctr-h[:,None]*directions,ctr+h[:,None]*directions]).reshape(2,n)
        amp = jump.amplitude(ctr)
        bad = (~located)|(lo<=0)|(abs((Y[0]-Y[1])-amp)>.47*amp)
        if np.sum(bad) > min(15,n//3):
            debug("too many suspect axial rays",np.sum(bad))
            return None
        for j in np.flatnonzero(bad):
            t[j] = _rescan_ray(s, jump, anchor, directions[j], limit[j], max(t[j],np.median(t)))
            if t[j] <= 0:
                return None
        debug("repaired axial rays",np.sum(bad))
    elif not located.all() or np.any(lo<=0):
        return None
    return 1/t


def _rescan_ray(s,jump,p,v,limit,r):
    end = min(limit,max(10.,1.7*r))
    for _ in range(2):
        uu = np.linspace(0,end,min(193,int(9*end)+9))
        z = p+uu[:,None]*v
        val = s.query(z)
        drop = val[:-1]-val[1:]
        excess = drop[2:-2]-.25*(drop[1:-3]+drop[3:-1]+drop[:-4]+drop[4:])
        half = .5*(uu[2:-3]+uu[3:-2])
        amp = jump.amplitude(p+half[:, None]*v)
        score = excess/np.maximum(amp,1e-9)
        skip = (score < .43)|(score>1.82)
        if not np.all(skip):
            k = np.argmin(np.where(skip,100.,abs(score-1)))+2
            idx = np.arange(3)
            L = np.polynomial.polynomial.polyfit(idx-2,val[k-2:k+1],2)
            R = np.polynomial.polynomial.polyfit(idx,val[k+1:k+4],2)
            a,b=uu[k:k+2]
            dx=b-a
            lo,hi=a,b
            for _ in range(15):
                m=.5*(lo+hi)
                y=s.query([p+m*v])[0]
                l=np.polynomial.polynomial.polyval((m-a)/dx,L)
                r=np.polynomial.polynomial.polyval((m-b)/dx,R)
                if abs(y-l)<abs(y-r):
                    lo=m
                else:
                    hi=m
            return .5*(lo+hi)
        if end>=limit:
            break
        end=min(limit,end*2)
    return 0.


class EllipsoidFit:
    def __init__(self, s, jump, anchor, U, a, b):
        self.s, self.jump, self.anchor, self.U = s, jump, anchor, U
        self.a, self.b = a,b
        self.R = np.eye(len(a))
        self.W = np.zeros_like(self.R)
        self.std = 1.0

    def measure(self, i, j, count, scale=None, lower=None, upper=None):
        if len(i) == 0:
            return
        a,b = self.a,self.b
        csum = .5*(a[i]+a[j])
        mult = np.sqrt(a[i]*a[j])
        # Use the sign of the ray that shortens the excursion. The off-diagonal
        # sign can also flip to avoid nearly null directions.
        sign = np.where(abs(b[i]-b[j]) > abs(b[i]+b[j]),-1.,1.)
        bv = (b[i]+sign*b[j])/np.sqrt(2)
        V = ((self.U[:,i]+sign*self.U[:,j])/np.sqrt(2)).T
        tilt = V @ self.jump.tilt
        sgn = np.where(abs(tilt) > 1e-9, np.where(tilt < 0,-1.,1.),
                       np.where(bv < 0,-1.,1.))
        V *= sgn[:,None]
        bv *= sgn
        limit = np.min(np.where(abs(V)>1e-12,
                                (39.999-np.copysign(1.,V)*self.anchor)/np.maximum(abs(V),1e-12),
                                1.e10),axis=1)
        aa = -np.ones(len(i)) if lower is None else lower
        bb = np.ones(len(i)) if upper is None else upper
        if scale is not None:
            co = np.clip(scale, .06, .85)
            xa, xb = ndtr(aa/co), ndtr(bb/co)
        else:
            xa, xb = np.zeros(len(i)), np.ones(len(i))
        for _ in range(count):
            xm = .5*(xa+xb)
            if scale is None:
                corr = aa+(bb-aa)*xm
            else:
                corr = co*ndtri(xm)
            qa = np.maximum(1e-18, csum+sign*mult*corr)
            disc = np.sqrt(bv*bv+4*qa)
            root = np.where(bv < 0, (disc-bv)/(2*qa), 2/np.maximum(disc+bv,1e-50))
            pt = self.anchor+V*np.minimum(root,limit)[:,None]
            high = (~self.jump.indicator(pt,self.s.query(pt))) == (sign>0)
            xa = np.where(high,xm,xa)
            xb = np.where(high,xb,xm)
        xm = .5*(xa+xb)
        if scale is None:
            corr = aa + (bb-aa)*xm
            w = (bb-aa)*(xb-xa)
        else:
            corr = co*ndtri(xm)
            w = co*(ndtri(xb)-ndtri(xa))
        self.R[i,j] = self.R[j,i] = corr
        self.W[i,j] = self.W[j,i] = w

    def solve(self, available):
        d = len(self.a)
        m = min(d, 10)
        ii,jj = np.triu_indices(d,1)
        first = jj < m
        self.measure(ii[first],jj[first],14 if d<40 else 11)
        available -= np.count_nonzero(first)*(14 if d<40 else 11)
        subset = self.R[:m,:m]
        vals, vec = np.linalg.eigh(subset)
        spread = np.sqrt(np.mean(self.R[ii[first],jj[first]]**2))
        if d>=28 and vals[1] > .006 and spread > (.20 if d < 40 else .19):
            md = min(d, 18)
            second = (jj<md)&(~first)
            self.measure(ii[second],jj[second],12)
            available -= 12*np.count_nonzero(second)
            m = md
            first |= second
            vals,vec = np.linalg.eigh(self.R[:m,:m])
        # If there are several resolved null directions, fit all leading
        # columns and complete the PSD Gram matrix by a truncated Nyström step.
        nonzero = vals > max(.006, .0018*max(vals))
        r = np.count_nonzero(nonzero)
        ids = np.arange(m)
        if r < m-2 and r >= 1:
            mat=vec[:,nonzero].T
            _,_,piv=qr(mat,pivoting=True,mode="economic")
            ids=np.sort(piv[:min(m,r+2)])
        selected=np.zeros(d,dtype=bool)
        selected[ids]=True
        tr = (selected[ii]|selected[jj])&(~first)
        kcost = np.count_nonzero(tr)
        if r <= m-2 and r >= 1 and kcost*8 <= available and vals[0] > -.06:
            self.measure(ii[tr],jj[tr],min(18 if d<40 else 15,available//max(1,kcost)))
            lv,lvv=np.linalg.eigh(self.R[np.ix_(ids,ids)])
            C = self.R[:,ids] @ lvv[:,-r:]
            Z = C/np.sqrt(np.maximum(lv[-r:],1e-7))
            err = np.max(abs(np.sum(Z*Z,axis=1)-1))
            if err < .068:
                Z = norm_rows(Z)
                self.R = Z@Z.T
                debug("ellipse rank",r,"diagonal error",err)
                return r, max(.00010,.00011+err*.24)
            available-=min(18 if d<40 else 15,available//max(1,kcost))*kcost
            first |= tr
        off = self.R[ii[first],jj[first]]
        self.std = np.clip(np.sqrt(np.mean(off*off)), .022, .85)
        left=np.logical_not(first)
        n = np.count_nonzero(left)
        if n:
            actual = min(16,available//n)
            if actual < 3:
                return None, 1.
            self.measure(ii[left],jj[left],actual,scale=min(.89,1.68*self.std))
        A = self.R
        vals, V = np.linalg.eigh(A)
        # Projection to the PSD cone, then diagonal equilibration. A weak
        # spectral threshold is used only when a numerical nullspace is clear.
        keep = vals > max(.004, .0016*max(vals), -.75*vals.min())
        B = V[:,keep]*np.sqrt(vals[keep])
        B = norm_rows(B)
        self.R = B@B.T
        off_w = self.W[ii,jj]
        wid = np.sqrt(np.mean(np.minimum(off_w,5*np.median(off_w)+.0004)**2))/np.sqrt(12)
        wid = max(wid,.00025)
        debug("ellipse bits",actual if n else 14,"rank",np.count_nonzero(keep),"width",wid)
        return np.count_nonzero(keep),wid


def _ellipse_correct(s,rng,jump,model,eps,nmax):
    """Integral of the label correction restricted to |q| < eps."""
    d = s.d
    lam,b,c,U = model.shifted(jump.tilt).spectrum()
    k = int(np.count_nonzero(lam>lam.max()*1e-10))
    central = c < -eps
    level = 3 if central and eps>.007 else (1 if central and eps>.002 else 0)
    n = min(2048,nmax//(level+1))
    if n < 256:
        return 0.,False
    power = n.bit_length()-1
    N = 1 << power
    uu = uniform(rng,power,d+1)
    u = norm_rows(ndtri(uu[:,:k]))
    direction = u @ U[:,-k:].T
    center = np.broadcast_to(jump.tilt,(N,d)).copy()
    if k < d:
        center += ndtri(uu[:,k:d]) @ U[:,:-k].T
    a = (u*u)@lam[-k:]
    b = u@b[-k:]

    def interval(q):
        r = np.sqrt(np.maximum(0,b*b-4*a*(c-q)))
        lo = np.maximum(0,(-b-r)/(2*a))
        hi = np.maximum(lo,(-b+r)/(2*a))
        return gammainc(.5*k,.5*lo*lo),gammainc(.5*k,.5*hi*hi)

    ol,oh=interval(eps)
    il,ih=interval(-eps)
    ml,mh=interval(0.)
    E0 = np.maximum(0.,mh-ml)-np.maximum(0.,ih-il)

    def values(p):
        r = np.sqrt(2*gammaincinv(.5*k,np.clip(p,1e-15,1-1e-15)))
        z = center+r[:,None]*direction
        return jump.indicator(z,s.query(z))

    u0=uu[:,-1]
    if level:
        size = np.maximum(0.,oh-ih)
        frac=.38 if level>=3 else 1.
        l = mh-frac*(mh-ih)
        h = mh+frac*(oh-mh)
        for _ in range(level):
            mid=.5*(l+h)
            inside=values(mid)
            l=np.where(inside,mid,l)
            h=np.where(inside,h,mid)
        mid=.5*(l+h)
        p0=.18
        v = np.where(u0<p0,ih+u0/p0*size,l+(u0-p0)/(1-p0)*(h-l))
        inv = size/(p0+np.where((v>=l)&(v<=h),
                                (1-p0)*size/np.maximum(h-l,1e-100),0.))
        y = (mid-ih)+inv*(values(v).astype(float)-(v<mid))
    else:
        has=(ih>il)
        d1 = np.where(has,il-ol,oh-ol)
        d2 = np.where(has,oh-ih,0.)
        size = np.maximum(0,d1+d2)
        delta=u0*size
        p=np.where(delta<d1,ol+delta,ih+delta-d1)
        y=size*values(p)
    out = np.mean(y-E0)
    debug("band correction",out,"mean radial mass",size.mean(),"N",N,"cuts",level)
    return out,True


def _refine_eigen_rays(s,jump,model,anchor,width):
    lam,b,c,U = model.shifted(anchor).spectrum()
    good = lam > max(0.045*lam.max(),width*.04)
    k = int(good.sum())
    if k==0 or c>=-.15 or s.remaining < 3120+8*k:
        return model
    sgn=np.where(U[:,good].T@jump.tilt < -1e-10,-1.,1.)
    if np.linalg.norm(jump.tilt)<1e-8:
        sgn=np.where(b[good]<0,-1.,1.)
    v=U[:,good].T*sgn[:,None]
    lin=b[good]*sgn
    est=lam[good]
    margin=np.maximum(.16*est,8*width*np.trace(model.A)/s.d)
    a=np.maximum(1e-12,est-margin)
    b2=est+margin
    for _ in range(8):
        x=.5*(a+b2)
        disc=np.sqrt(np.maximum(0,lin*lin-4*c*x))
        t=np.where(lin>=0,-2*c/np.maximum(lin+disc,1e-30),(disc-lin)/(2*x))
        z=anchor+t[:,None]*v
        ans=jump.indicator(z,s.query(z))
        a=np.where(ans,a,x)
        b2=np.where(ans,x,b2)
    new=lam.copy()
    new[good]=.5*(a+b2)
    A=(U*new)@U.T
    # Preserve the axial linear terms about the original interior anchor.
    return Quadratic.centered(A,model.b+2*model.A@anchor,anchor,c)


def _rare_radial(s,rng,jump,model,width,nmax):
    """Radial probability correction when zero is inside a large jump."""
    if model.c>=-.07 or nmax<1400:
        return 0.,False
    if not jump.indicator(np.zeros((1,s.d)),s.query(np.zeros((1,s.d))))[0]:
        return 0.,False
    d=s.d
    n=min(1024,nmax//6)
    if n<256:
        return 0.,False
    m=n.bit_length()-1
    N=1<<m
    uu=uniform(rng,m,d+1)
    lam,b,c,U = model.spectrum()
    # An angular central Gaussian based on the quadratic Chernoff tilt.
    # Its explicit density reweights angles without rejection.
    def deriv(t):
        den=1-2*t*lam
        return c+np.sum(lam/den+t*b*b*(1-t*lam)/(den*den))
    t=0.
    if deriv(0.)>0 and deriv(-.5e3)<0:
        t=.76*brentq(deriv,-.5e3,0.)
    cv=1/np.maximum(1-2*t*lam,.5)
    du=norm_rows(ndtri(uu[:,:d])*np.sqrt(cv))
    V=du@U.T
    w=np.exp(.5*(np.log(cv).sum()+d*np.log((du*du)@(1/cv))))
    a=(du*du)@lam
    b=du@b
    r=-2*c/np.maximum(b+np.sqrt(np.maximum(0,b*b-4*a*c)),1e-40)
    est=gammainc(.5*d,.5*r*r)
    margin=np.minimum(.14,max(.0010,2.7*width))

    def query(radius,directions):
        return jump.indicator(radius[:,None]*directions,s.query(radius[:,None]*directions))
    left,right=r*(1-margin),r*(1+margin)
    yes,no=query(left,V),~query(right,V)
    extra=0
    for scale in (3.,9.):
        i,j=np.flatnonzero(~yes),np.flatnonzero(~no)
        if not len(i)+len(j) or 5*N+extra+len(i)+len(j)>nmax:
            break
        left[i]=r[i]*max(.1,1-scale*margin)
        right[j]=r[j]*(1+scale*margin)
        yes[i]=query(left[i],V[i])
        no[j]=~query(right[j],V[j])
        extra+=len(i)+len(j)
    l=np.where(yes,gammainc(.5*d,.5*left*left),0.)
    h=np.where(no,gammainc(.5*d,.5*right*right),1.)
    # These brackets use a globally single threshold, as supplied by an
    # ellipsoid with an interior origin; no assertion for arbitrary labels.
    for k in range(3):
        p=.5*(l+h) if k<2 else l+(h-l)*uu[:,-1]
        radius=np.sqrt(2*gammaincinv(.5*d,np.clip(p,1e-15,1-1e-15)))
        inside=query(radius,V)
        if k<2:
            l=np.where(inside,p,l)
            h=np.where(inside,h,p)
    out=np.mean(w*(l+(h-l)*inside-est))
    debug("rare radial correction",out,"fibres",N,"loose",np.mean(~yes|~no))
    return out,True


def ellipse_estimate(s,rng,jump,inner,zp,yp,atlas):
    d = s.d
    anchor = _anchor(s,rng,jump,inner,zp,yp,atlas)
    if anchor is None:
        return None
    # Random orientation gives a fairly balanced set of radius probes even
    # when the true ellipsoid is cylindrical in a very small subspace.
    U = np.linalg.qr(rng.standard_normal((d,d)))[0]
    if np.linalg.norm(jump.tilt) > 1e-8:
        u = U.T @ (jump.tilt/np.linalg.norm(jump.tilt))
        w = u - np.full(d,1/np.sqrt(d))
        if w@w > 1e-20:
            U -= 2*np.outer(U@w,w)/(w@w)
    ns = 20 if d < 40 else 17
    ab = _radial_axes(s,jump,anchor,np.r_[U.T,-U.T],ns)
    if ab is None:
        return None
    x,y = ab[:d],ab[d:]
    a,b = x*y,x-y
    l = EllipsoidFit(s,jump,anchor,U,a,b)
    reserve = 5300 if d <= 16 else 5000 if d < 40 else 3200
    planned = s.remaining-reserve
    if planned <= 0:
        return None
    rank,width = l.solve(planned)
    if rank is None:
        return None
    A = l.R*np.sqrt(a[:,None]*a[None])
    model = Quadratic.centered(U@A@U.T,U@b,anchor,-1)
    lam, bb, c, V = model.spectrum()
    bb[lam<1e-10*lam.max()]=0.
    model = Quadratic(model.A,V@bb,c)
    if d>=40 and d>rank>=d//2 and width>.004 and jump.level>12*atlas.scale:
        model=_refine_eigen_rays(s,jump,model,anchor,width)
    cp = quadratic_cdf(model.shifted(jump.tilt))
    # Sanity check the fitted model against probes not used in the tomography.
    label = jump.indicator(zp,yp)
    approx = model(zp)<0
    different = np.mean(label!=approx)
    rare = jump.level > 12*atlas.scale
    constant_jump = rare and not np.any(jump.tilt)
    if different > .15 or (rare and cp > .18):
        debug("ellipsoid rejected",different,cp)
        return None
    debug("jump",jump.level,"tilt",np.linalg.norm(jump.tilt),"prob",cp,"validation",different,
          "used",s.used)
    ep = jump.level*np.exp(.5*(jump.tilt@jump.tilt))
    eps = min(.16,max(.00055,4.5*width))
    if np.any(label!=approx):
        zz=abs(model(zp[label!=approx]))
        soft = zz[zz<max(.022,5*width)]
        if len(soft):
            eps = max(eps,1.35*np.quantile(soft,.72))
    powerbudget = min(4800,max(0,s.remaining-(3300 if d<40 else 2250)))
    radial=False
    response=None
    if constant_jump and d>=40 and rank==d and .003<width<.017 and different<.032:
        from response import radial_response
        response=radial_response(s,rng,jump,model,anchor,width,
                                 min(4850,s.remaining-1500))
        if response is not None:
            radial=True
            band=response
        powerbudget = min(powerbudget,max(0,s.remaining-2050))
    if not radial and constant_jump and (d>=40 and width>.002):
        band,radial=_rare_radial(s,rng,jump,model,width,powerbudget)
    if radial:
        active=False
    else:
        band,active = _ellipse_correct(s,rng,jump,model,eps,powerbudget)
    cp+=band

    class Background:
        def __init__(self, ss):
            self.ss = ss

        def __getattr__(self, x):
            return getattr(self.ss,x)

        def query(self,z):
            z = np.asarray(z).reshape(-1,d)
            y = self.ss.query(z)
            if radial:
                return jump.smooth(z,y)
            q = model(z)
            indicator = q < 0
            if active:
                ii = np.abs(q)<eps
                indicator[ii] = jump.indicator(z[ii],y[ii])
            return y-jump.amplitude(z)*indicator

    bsession = Background(s)
    rot, cv, exact = smooth_control(bsession,None,rng,reserve=2100 if d<40 else 1950,
                                    allow_curvature=(d<40 or s.remaining>4500))
    smooth = smooth_correct(bsession,rng,rot,cv,exact)
    debug("ellipse components",ep*cp,smooth,"band",eps)
    return ep*cp+smooth
