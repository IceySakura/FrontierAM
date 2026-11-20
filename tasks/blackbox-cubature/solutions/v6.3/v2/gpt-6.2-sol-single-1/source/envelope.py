"""Recover and integrate convex or concave piecewise affine functions."""
import itertools
import math

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from common import Model, gl_interval
from ridges import gradients


def _polytope_integral(w,b,anchors):
    d=w.shape[1]
    eye=np.eye(d)
    v,h=0.,0.
    for i,(wi,bi,p) in enumerate(zip(w,b,anchors)):
        keep=np.arange(len(w))!=i
        a=np.vstack((w[keep]-wi,eye,-eye))
        c=np.r_[b[keep]-bi,np.full(2*d,-.5)]
        nn=np.linalg.norm(a,axis=1)
        use=nn>1e-14
        if np.any(c[~use]>0):
            continue
        a,c=a[use]/nn[use,None],c[use]/nn[use]
        z=p-.5
        if np.max(a@z+c)>-2e-8:
            eq=np.column_stack((a,np.ones(len(a))))
            obj=np.r_[np.zeros(d),-1.]
            res=linprog(obj,A_ub=eq,b_ub=-c,bounds=[(-.5,.5)]*d+[(0,.5)],
                        method="highs-ds")
            if not res.success or res.x[-1]<1e-12:
                continue
            z=res.x[:-1]
        try:
            points=HalfspaceIntersection(np.column_stack([a,c]),z).intersections
            points=np.unique(points,axis=0)
            if len(points)<=d:
                return None
            faces=points[ConvexHull(points).simplices]
        except QhullError:
            return None
        det=abs(np.linalg.det(faces-z))/math.factorial(d)
        mid=(np.sum(faces,axis=1)+z)/(d+1)
        v+=det.sum()
        h+=det@(mid@wi+bi)
    if abs(v-1)>3e-10:
        return None
    return float(h/v)


def _conditional_integral(orc,w,b):
    d=w.shape[1]
    j=np.argmax(np.ptp(w,axis=0))
    other=np.arange(d)!=j
    a=w[:,j]
    l,r=np.triu_indices(len(a),1)
    use=np.abs(a[l]-a[r])>1e-14
    l,r=l[use],r[use]
    def cond(z):
        C=(z-.5)@w[:,other].T+b
        root=np.clip(.5+(C[:,r]-C[:,l])/(a[l]-a[r]),0,1)
        zz=np.column_stack((np.zeros(len(z)),np.sort(root,axis=1),np.ones(len(z))))
        h=np.diff(zz,axis=1)
        mid=(zz[:,:-1]+zz[:,1:])*.5-.5
        vals=np.max(C[:,None,:]+mid[:,:,None]*a,axis=2)
        return np.sum(vals*h,axis=1)
    if d==2:
        R,S=[],[]
        for l,r in itertools.combinations(range(len(w)),2):
            u,v=b[r]-b[l], (w[r,other]-w[l,other])[0]
            E=a[l]-a[r]
            if abs(E)<1e-14:
                if abs(v)>1e-14:
                    S.append(.5-u/v)
            else:
                R.append((.5+u/E,v/E))
        for u,v in R:
            if abs(v)>1e-14:
                S.extend([.5-u/v,.5+(1.-u)/v])
        for (u,v),(p,q) in itertools.combinations(R,2):
            if abs(v-q)>1e-14:
                S.append(.5+(p-u)/(v-q))
        cuts=np.array(S)
        cuts=np.r_[0.,np.unique(cuts[(cuts>0)&(cuts<1)]),1.]
        return math.fsum(float(wt@cond(t[:,None])) for t,wt in
                         [gl_interval(4,a,b) for a,b in zip(cuts[:-1],cuts[1:])])
    count=1<<18 if len(b)>7 else 1<<19
    u=orc.new_sobol(count,d=d-1)
    return math.fsum(float(cond(u[i:i+4096]).sum()) for i in range(0,len(u),4096))/count


def _edge_probes(orc,w,b):
    w,b=np.asarray(w),np.asarray(b)
    d=w.shape[1]
    k=len(w)
    per=min(9,(orc.left-24)//max(1,k*(k-1)//2))
    if per<2:
        return np.empty((0,d))
    pts=[]
    for i,j in itertools.combinations(range(k),2):
        dw=w[i]-w[j]
        a=np.argmax(np.abs(dw))
        if abs(dw[a])<1e-10:
            continue
        x=orc.rng.uniform(0,1,(180,d))
        x[:,a]+=(b[j]-b[i]-(x-.5)@dw)/dw[a]
        x=x[(x[:,a]>.0001)&(x[:,a]<.9999)]
        y=(x-.5)@w.T+b
        use=(y[:,i]>=np.max(y,axis=1,initial=-np.inf)-2e-10)
        if np.any(use):
            xx=x[use]
            pts.extend(xx[np.linspace(0,len(xx)-1,min(per,len(xx)),dtype=int)])
    return np.asarray(pts).reshape(-1,d)


def linear_envelope(orc,x,y,scale,sign,maxr=12):
    if orc.left < 100+2*orc.d or orc.d>8:
        return None
    d=orc.d
    xx,yy=orc.data()
    yy=sign*yy/scale
    w,b,anchors=[],[],[]
    done=True
    last=None
    for j in range(maxr):
        if orc.left<75+2*d or orc.elapsed>10:
            done=False
            break
        if j:
            u=np.max((xx-.5)@np.asarray(w).T+b,axis=1)
            error=yy-u
            if error.min() < -2e-8:
                return None
            if error.max()<6e-10:
                if last == len(w):
                    break
                n=min(52,max(14,orc.left//5))
                audit=orc.new_sobol(n)
                if len(w)>1:
                    edge = _edge_probes(orc,w,b)
                    room = max(0,orc.left-n-25)
                    audit=np.vstack((audit,edge[:room]))
                v=orc.sample(audit)/scale*sign
                xx=np.vstack((xx,audit))
                yy=np.r_[yy,v]
                u=np.max((xx-.5)@np.asarray(w).T+b,axis=1)
                error=yy-u
                last=len(w)
                if error.min()<-2e-8:
                    return None
                if error.max()<6e-10:
                    break
            idx=int(np.argmax(error))
        else:
            idx=int(np.argmax(yy))
        p=np.clip(xx[idx],.000036,.999964)
        g=gradients(orc,p[None,:],scale,order=2,step=.00012)[0]*sign
        vp=orc.sample(p[None,:])[0]*sign/scale
        c=vp-(p-.5)@g
        if any(np.linalg.norm(g-old)+abs(c-bo)<3e-8 for old,bo in zip(w,b)):
            return None
        w.append(g)
        b.append(c)
        anchors.append(p)
    else:
        done=False
    if len(w)<2 or not done:
        return None
    w,b=np.array(w),np.array(b)
    def pred(p):
        return sign*np.max((p-.5)@w.T+b,axis=1)
    err=np.max(abs(pred(xx)*sign-yy))
    if err>8e-10:
        return None
    integral=_polytope_integral(w,b,anchors) if d<=5 else None
    if integral is None:
        integral=_conditional_integral(orc,w,b)
    return Model(pred,sign*integral,"piecewise affine envelope",float(err))
