"""Cubature with inferred axis-aligned breakpoints and supports."""
import itertools
import math

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog

from common import gl_interval, legmat, multiindices, residual, solve_ls
from projection import Projection
from structure import (axes_separable, integrate_axes, polynomial, section_points,
                       transformed, _step_brackets)


class BoxOracle:
    def __init__(self, oracle, a, b, offset=0.):
        self.oracle, self.d, self.rng = oracle, oracle.d, oracle.rng
        self.a, self.w, self.offset = a, b-a, offset

    @property
    def left(self):
        return self.oracle.left

    def sample(self, x):
        return self.oracle.sample(self.a+x*self.w)-self.offset

    def new_sobol(self, size):
        return self.oracle.new_sobol(size)


def _trace_jumps(orc, points, a, b, yl, yr, scale, rounds):
    yl,yr = yl.copy(),yr.copy()
    a,b = a.copy(),b.copy()
    for j in range(rounds):
        h = (a+b)*.5
        y = orc.sample(points(h))/scale
        move = abs(y-yl) <= abs(y-yr)
        a[move],yl[move] = h[move],y[move]
        b[~move],yr[~move] = h[~move],y[~move]
    return a,b,yl,yr


def jump_trace(orc, line, scale):
    c,v,t,vals = line
    y = vals/scale
    trial = _step_brackets(t,y)
    if not len(trial) or orc.left < 12:
        return False
    k = trial[np.argmax(np.abs(y[trial+1]-y[trial]))]
    a,b = t[[k]],t[[k+1]]
    aa,bb,yl,yr = _trace_jumps(orc,lambda h:c+h[:,None]*v,
                               a,b,y[[k]],y[[k+1]],scale,8)
    return abs(yr[0]-yl[0]) > .2*abs(y[k+1]-y[k])


def support_box(orc, scale, x, y):
    """Locate an axis rectangle of nonconstant support if the data allow it."""
    d = orc.d
    if d>10 or orc.left < 150+18*d:
        return None
    # Near-zero values are not zeros; tiny smooth tails are informative
    # elsewhere and must not be interpreted as a discontinuity.
    uu,ct = np.unique(y,return_counts=True)
    idx = np.argmax(ct)
    if ct[idx] < .12*len(y) or (uu[idx]!=0. and ct[idx]<.35*len(y)):
        return None
    off = float(uu[idx])
    xx,raw = orc.data()
    yy = raw/scale-off
    pos = np.abs(yy)>0.
    if not np.any(pos):
        return None
    anchor = xx[np.argmax(np.abs(yy))]
    t = np.r_[0.,.032,.135,.29,.47,.65,.81,.945,1.]
    ts = np.sort(np.column_stack((np.tile(t,(d,1)),anchor)),axis=1)
    ps = np.broadcast_to(anchor,(d,len(t)+1,d)).copy()
    ps[np.arange(d),:,np.arange(d)] = ts
    us = orc.sample(ps.reshape(-1,d)).reshape(ts.shape)/scale-off
    left,right = np.zeros(d),np.ones(d)
    a,b,ls,rs,axes = [],[],[],[],[]
    lower = []
    for j in range(d):
        nz = np.nonzero(us[j] != 0.)[0]
        if not len(nz) or len(nz) != (nz[-1]-nz[0]+1):
            return None
        i,k = nz[0],nz[-1]
        if i:
            a.append(ts[j,i-1])
            b.append(ts[j,i])
            ls.append(us[j,i-1])
            rs.append(us[j,i])
            axes.append(j)
            lower.append(True)
        left[j] = ts[j,max(0,i-1)]
        if k < len(t):
            a.append(ts[j,k])
            b.append(ts[j,k+1])
            ls.append(us[j,k])
            rs.append(us[j,k+1])
            axes.append(j)
            lower.append(False)
        right[j] = ts[j,min(len(t),k+1)]
    if not len(axes) or np.any((xx[pos] < left) | (xx[pos] > right)):
        return None
    axes,lower = np.asarray(axes),np.asarray(lower)
    if orc.left < 23*len(axes)+60:
        return None
    # The membership predicate uses the exactly flat piece of the function,
    # rather than comparing magnitudes to a guessed jump height.
    a,b = np.array(a),np.array(b)
    pp = np.broadcast_to(anchor,(len(axes),d)).copy()
    niter = min(44,(orc.left-63)//len(axes))
    for k in range(niter):
        m = (a+b)*.5
        pp[np.arange(len(axes)),axes] = m
        values = orc.sample(pp)/scale-off
        go_left = (values!=0.) == lower
        a[~go_left],b[go_left] = m[~go_left],m[go_left]
    left[axes[lower]] = (a[lower]+b[lower])*.5
    right[axes[~lower]] = (a[~lower]+b[~lower])*.5
    width = right-left
    if np.any(width <= 0.):
        return None
    cover = np.all((xx > left+4e-13) & (xx < right-4e-13),axis=1)
    outside = np.any((xx < left-4e-13) | (xx > right+4e-13),axis=1)
    if np.any(cover & ~pos) or np.any(outside & pos):
        return None
    # Query on both sides to check for a domain face that depends on
    # another coordinate, or a separated component of the support.
    view = BoxOracle(orc,left,right,off*scale)
    can_use = min(196,max(44,orc.left//4))
    n = min(can_use,orc.left)
    xi = orc.new_sobol(n)
    vi = view.sample(xi)
    si = np.max(np.abs(vi))
    if si <= 0. or np.any(vi == 0.):
        return None
    vi = vi/si
    audit = []
    for j,low in zip(axes,lower):
        k = min(6,(orc.left-32)//max(1,2*len(axes)))
        if k<=0:
            break
        zz = left+width*orc.rng.uniform(.03,.97,(k,d))
        zz[:,j] = max(0.,left[j]-2e-8) if low else min(1.,right[j]+2e-8)
        audit.append(zz)
    if audit:
        u = orc.sample(np.vstack(audit))/scale
        if np.any(u != off):
            return None
    vol = math.prod(width)
    def result(s):
        return off + vol*(si/scale)*s
    for mod in (polynomial(xi,vi),transformed(xi,vi)):
        if mod is not None and residual(mod,xi,vi)[1] < 3e-10:
            return result(mod.integral)
    args = axes_separable(view,xi,vi,si)
    if args is not None:
        return result(integrate_axes(view,args,si))
    if d<=3 and orc.left >= 4**d:
        n = min(33,int(orc.left**(1/d)+1e-10))
        while n**d > orc.left:
            n-=1
        z,w = gl_interval(n)
        pts = np.array(list(itertools.product(z,repeat=d)))
        ws = np.prod(np.array(list(itertools.product(w,repeat=d))),axis=1)
        v = view.sample(pts)/si
        return result(float(v@ws))
    if orc.left:
        z = orc.new_sobol(orc.left)
        v = view.sample(z)/si
        return result(float(np.mean(v)))
    return result(float(np.mean(vi)))


def piecewise_grid(orc, x, y, scale):
    """Find common axis cuts, then integrate cell constants plus a smooth term."""
    d = orc.d
    if orc.left < 180+24*d or d>9:
        return None
    anchor = x[np.argmax(np.abs(y-np.median(y)))]
    ts = np.linspace(0.,1.,25)
    vals = orc.sample(section_points(anchor,ts).reshape(-1,d)).reshape(d,-1)/scale
    trial = [(j,k) for j,v in enumerate(vals) for k in _step_brackets(ts,v)]
    if not trial or len(trial) > 24:
        return None
    axes,idx = np.array(trial).T
    cloud = np.broadcast_to(anchor,(len(trial),d)).copy()
    rows = np.arange(len(trial))
    def points(h):
        cloud[rows,axes] = h
        return cloud
    yl,yr=vals[axes,idx],vals[axes,idx+1]
    a,b,vl,vr = _trace_jumps(orc,points,ts[idx],ts[idx+1],yl,yr,scale,9)
    good = np.abs(vl-vr) > .20*np.abs(yl-yr)
    if not np.any(good):
        return None
    a,b,vl,vr,axes = [z[good] for z in (a,b,vl,vr,axes)]
    rows = np.arange(len(axes))
    cloud = cloud[good]
    cuts = [np.r_[0.,(a[axes==j]+b[axes==j])*.5,1.] for j in range(d)]
    dims = np.array([len(a)-1 for a in cuts])
    cells = int(np.prod(dims))
    if cells > min(512,orc.left//2):
        return None
    nref = min(38,(orc.left-cells-90)//len(axes))
    if nref < 13:
        return None
    a,b,_,_ = _trace_jumps(orc,points,a,b,vl,vr,scale,nref)
    cuts = [np.r_[0.,(a[axes==j]+b[axes==j])*.5,1.] for j in range(d)]
    low = np.array(list(itertools.product(*(a[:-1] for a in cuts))))
    hi = np.array(list(itertools.product(*(a[1:] for a in cuts))))
    half = (hi-low)*.5
    mid = (low+hi)*.5
    weight = np.prod(hi-low,axis=1)
    cv = orc.sample(mid)/scale
    xx,yy = orc.data()
    tx,ty = np.vstack((x[:len(x)*4//5],xx[len(x):])),np.r_[y[:len(x)*4//5],yy[len(x):]/scale]
    vi = orc.new_sobol(48)
    vy = orc.sample(vi)/scale
    def classify(p):
        idxs = [np.searchsorted(a[1:-1],p[:,j]) for j,a in enumerate(cuts)]
        return np.ravel_multi_index(idxs,dims)
    ids = classify(tx)
    nonclose = np.ones(len(tx),dtype=bool)
    for j,a in enumerate(cuts):
        for c in a[1:-1]:
            nonclose &= np.abs(tx[:,j]-c)>2e-10
    tx,ty,ids = tx[nonclose],ty[nonclose],ids[nonclose]
    val_x,val_y = np.vstack((x[len(x)*4//5:],vi)),np.r_[y[len(x)*4//5:],vy]
    val_ids = classify(val_x)
    best = None
    for deg in (0,1,2,3,4):
        powers = multiindices(d,deg,min(deg,2))
        if powers.shape[1] > len(tx)*.55:
            break
        t1 = legmat(tx,powers)
        tm = legmat(mid,powers)
        w = solve_ls(t1[:,1:]-tm[ids,1:],ty-cv[ids]) if deg else np.empty(0)
        coeff = np.r_[0.,w]
        vv = cv[val_ids]+(legmat(val_x,powers)-tm[val_ids])@coeff
        err = np.max(np.abs(vv-val_y))
        if err < 2e-10:
            return float(weight@(cv-tm@coeff))
        if best is None or err < best[0]:
            best = err, float(weight@(cv-tm@coeff))
    # With few cells, integrate each smooth piece without assuming a
    # common polynomial trend across the cuts.
    if d <= 3 and best and best[0] < .2*max(1e-20,np.ptp(y)) and orc.left >= cells*4**d:
        n = min(22, int((orc.left//cells)**(1/d)+1e-10))
        while cells*n**d>orc.left:
            n-=1
        z,w = gl_interval(n,-1.,1.)
        zz = np.array(list(itertools.product(z,repeat=d)))
        ww = np.prod(np.array(list(itertools.product(w*.5,repeat=d))),axis=1)
        pts = (mid[:,None,:]+half[:,None,:]*zz).reshape(-1,d)
        return float(weight@(orc.sample(pts).reshape(cells,-1)/scale@ww))
    return None


def halfspace_cut(orc,x,y,scale):
    """Model a two-level jump, or a smooth function on one side of a plane."""
    d = orc.d
    if d<2 or d>9 or orc.left < 28*(d+2)+80:
        return None
    levels = np.unique(y)
    if len(levels)==2:
        low,high=levels
        pure = True
    elif len(levels)>2 and np.count_nonzero(y==0.)>.12*len(y):
        low,high=0.,1.
        pure = False
    else:
        return None
    def inside(ys):
        return (ys==high) if pure else (ys!=0.)
    xp, yp = orc.data()
    yp=yp/scale
    label=inside(yp)
    if min(np.sum(label),np.sum(~label))<1:
        return None
    train=inside(y)
    coords=x
    if min(np.sum(train),np.sum(~train))<2:
        select=np.linspace(0,len(xp)-1,min(len(xp),len(x)+100),dtype=int)
        coords,train=xp[select],label[select]
    z = np.column_stack((coords-.5,np.ones(len(coords))))
    lab = np.where(train,1.,-1.)
    a=np.column_stack((-lab[:,None]*z,np.ones(len(z))))
    score=np.r_[np.zeros(d+1),-1.]
    opt=linprog(score,A_ub=a,b_ub=np.zeros(len(z)),
                bounds=[(-1,1)]*d+[(-d,d),(0,d)],method="highs-ds")
    if not opt.success or opt.x[-1]<1e-9:
        return None
    w,b=opt.x[:d],-opt.x[d]
    j=np.argmax(abs(w))
    if abs(w[j])<1e-10:
        return None
    # Pick affinely independent lines whose endpoints straddle the fit.
    frac=np.clip(.5+b/np.sum(abs(w)),.001,.999)
    mid=np.where(w>=0,frac,1.-frac)
    grid=orc.rng.uniform(.0003,.9997,(1000,d))
    tail=mid + orc.rng.uniform(-1,1,(500,d))*min(frac,1.-frac)*2
    grid=np.vstack((grid,np.clip(tail,0.,1.)))
    grid[:,j]+=(b-(grid-.5)@w)/w[j]
    grid=grid[(grid[:,j]>.0001)&(grid[:,j]<.9999)]
    if len(grid)<d:
        return None
    ab=grid[np.linspace(0,len(grid)-1,min(len(grid),3*d+20),dtype=int)]
    ul,ur=ab.copy(),ab.copy()
    ul[:,j],ur[:,j]=0.,1.
    values=orc.sample(np.vstack((ul,ur)))/scale
    ll,rr=inside(values).reshape(2,-1)
    use=(ll!=rr)&(rr==(w[j]>0.))
    if np.sum(use)<d:
        return None
    sub=ab[use]
    oth=np.arange(d)!=j
    X=np.column_stack([np.ones(len(sub)),sub[:,oth]-.5])
    _,_,piv=qr(X.T,pivoting=True,mode="economic",check_finite=False)
    piv=piv[:min(d+2,len(sub))]
    X,rp=X[piv],sub[piv].copy()
    left,right=np.zeros(len(rp)),np.ones(len(rp))
    depth=min(44,(orc.left-76)//len(rp))
    if depth<21:
        return None
    for _ in range(depth):
        m=(left+right)*.5
        rp[:,j]=m
        g=inside(orc.sample(rp)/scale)
        lower=g != (w[j]>0.)
        left[lower],right[~lower]=m[lower],m[~lower]
    c=solve_ls(X,(left+right)*.5)
    if np.max(abs(X@c-(left+right)*.5)) > max(5e-11,2.**(-depth+1)):
        return None
    mvec=np.ones(d)
    mvec[oth]=-c[1:]
    intercept=c[0]-.5
    sgn=np.sign(w[j])
    w,b=mvec*sgn,intercept*sgn
    def loc(p):
        return (p-.5)@w-b
    vv=loc(xp)
    if np.any((vv>0.)!=label) and np.max(abs(vv[(vv>0.)!=label]),initial=0)>max(5e-9,20.*2.**(-depth)):
        return None
    audit=orc.new_sobol(44)
    sub2=sub[np.linspace(0,len(sub)-1,min(12,len(sub)),dtype=int)]
    val0=c[0]+(sub2[:,oth]-.5)@c[1:]
    a0,b0=sub2.copy(),sub2.copy()
    h=max(1e-7,64.*2.**(-depth))
    a0[:,j],b0[:,j]=val0-h,val0+h
    audit=np.vstack((audit,np.clip(a0,0,1),np.clip(b0,0,1)))
    va=orc.sample(audit)/scale
    if np.any(inside(va)!=(loc(audit)>0)):
        return None
    proj=Projection(w)
    mass=proj.moment(.5-b/proj.span,0)
    if pure:
        return float(low+(high-low)*mass)
    good=label
    if good.sum() < 2*d+4:
        return None
    p,targ = xp[good],yp[good]
    v=va[inside(va)]
    q=audit[inside(va)]
    if len(v)<5:
        return None
    m=polynomial(p,targ)
    gsign=np.sign(np.median(targ))
    if m is not None and residual(m,q,v)[1] < 1e-10:
        smooth=m.predict
    elif np.all(targ*gsign>0.):
        u=p-.5
        feat=np.column_stack((np.ones(len(u)),u,u*u))
        theta=solve_ls(feat,np.log(targ*gsign))
        def smooth(p):
            u=p-.5
            z=theta[0]+u@theta[1:d+1]+(u*u)@theta[d+1:]
            return gsign*np.exp(np.clip(z,-745,690))
        if (np.max(abs(smooth(p)-targ))>1e-9
                or np.max(abs(smooth(q)-v))>1e-9):
            return None
    else:
        return None
    node,wt=gl_interval(22)
    def conditional(z):
        root=np.clip(c[0]+(z-.5)@c[1:],0.,1.)
        a,l=(root,1.-root) if sgn>0 else (np.zeros(len(z)),root)
        pts=np.zeros((len(z),len(node),d))
        pts[:,:,oth]=z[:,None,:]
        pts[:,:,j]=a[:,None]+l[:,None]*node
        vals=smooth(pts.reshape(-1,d)).reshape(len(z),-1)
        return (vals@wt)*l
    if d==2:
        cuts=np.r_[0.,1.]
        if abs(c[1])>1e-14:
            br=.5+(np.array([0.,1.])-c[0])/c[1]
            cuts=np.sort(np.r_[cuts,br[(br>0)&(br<1)]])
        return math.fsum(float(wt@conditional(t[:,None])) for t,wt in
                         [gl_interval(28,a,b) for a,b in zip(cuts[:-1],cuts[1:])])
    z=orc.new_sobol(1<<18,d=d-1)
    vals=[]
    for i in range(0,len(z),4096):
        vals.append(conditional(z[i:i+4096]))
    return float(np.mean(np.concatenate(vals)))
