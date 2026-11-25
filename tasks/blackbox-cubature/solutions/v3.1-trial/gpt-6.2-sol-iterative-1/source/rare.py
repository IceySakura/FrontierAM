"""Defensive, payoff-weighted importance sampling for sparse payoffs."""

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

from cubature import net_sizes,normals,ordered_basis
from kinks import debug


class Gaussian:
    def __init__(self,mean,basis,sd):
        self.mean,self.basis,self.sd = mean,basis,sd
        self.norm = np.log(sd).sum()

    def points(self,x):
        return self.mean+(x*self.sd)@self.basis.T

    def ratio(self,z):
        x = ((z-self.mean)@self.basis)/self.sd
        return np.exp(np.clip(.5*(np.sum(z*z-x*x,axis=1))-self.norm,-650,650))


def _proposal(z,y,rat,v,initial):
    d = len(v)
    w = y/np.maximum(rat,1.e-8)
    sw=w.sum()
    if sw<=0:
        return initial
    w/=sw
    eff = 1/max(1.e-15,w@w)
    m = w@z
    xc = z-m
    cov = (xc.T*w)@xc
    P = np.eye(d)-np.outer(v,v)
    vt = v@cov@v
    lin = P@cov@v
    # Treat the terminal coordinate separately. Remaining mean and
    # covariance directions are shrunk according to effective sample size.
    qm = P@m
    shrink = max(.04,1-.85*(d-1)/max(1.,eff*(qm@qm)))
    mm = m@v*v+shrink*qm
    eps = np.sqrt(d/max(eff,10))
    R = ordered_basis(P@cov@P,v)
    var = np.diag(R.T@cov@R).copy()
    margin = .93*eps
    var[1:] = 1+np.sign(var[1:]-1)*np.maximum(0.,abs(var[1:]-1)-margin)*.87
    var[1:] = np.clip(var[1:], .40, 1.23)
    var[0] = np.clip(.94*vt+.03,.032,1.15)
    cv = (R*var)@R.T + .5*(np.outer(v,lin)+np.outer(lin,v))
    va, U = np.linalg.eigh(cv)
    # PCA of the weighted positive paths also supplies the QMC ordering:
    # narrow effective coordinates are stratified first.
    s = np.sqrt(np.clip(va,.024,1.5))
    debug("payoff proposal", "effective",round(eff,2),"mean",np.linalg.norm(mm),
          "terminal",float(m@v),vt,"variance",va[:min(5,d)])
    return Gaussian(mm,U,s)


def _pilot(s,rng,proposal,n1,n0,basis):
    blocks = [proposal.points(normals(rng,p,s.d)) for p in net_sizes(n1)]
    blocks += [normals(rng,p,s.d)@basis.T for p in net_sizes(n0)]
    z=np.vstack(blocks)
    q=(n0+n1*proposal.ratio(z))/(n0+n1)
    return z,s.query(z),q


class _MaskedModel:
    def __init__(self,proposal,z,y):
        self.P=proposal
        self.k=min(7,len(proposal.sd))
        self.i,self.j=np.triu_indices(self.k)
        h=self.features(z)
        labels=(y>0)
        ridge=.0012

        def obj(b):
            a=h@b
            val=np.mean(np.logaddexp(0.,a)-a*labels)+.5*ridge*(b[1:]@b[1:])
            g=h.T@(expit(a)-labels)/len(labels)
            g[1:]+=ridge*b[1:]
            return val,g

        b0=np.zeros(h.shape[1])
        proportion=np.clip(np.mean(labels),1e-4,1-1e-4)
        b0[0]=np.log(proportion/(1-proportion))
        self.classifier=minimize(obj,b0,jac=True,method='L-BFGS-B',
                                 options={'maxiter':55,'ftol':3e-9}).x
        r=h[labels]
        yy=y[labels]
        self.upper=max(1e-10,1.6*np.quantile(yy,.94))
        w=r.T@r
        w.flat[::len(w)+1]+=.12*len(r)
        w[0,0]-=.12*len(r)
        self.values=np.linalg.solve(w,r.T@yy)

    def features(self,z):
        x=np.clip(((z-self.P.mean)@self.P.basis)/self.P.sd,-4.3,4.3)
        a=np.tanh(x[:,:self.k]*.56)
        return np.c_[np.ones(len(z)),x,a[:,self.i]*a[:,self.j]*2.,
                     a[:, :3]**3]

    def predict(self,z):
        h=self.features(z)
        p=np.clip(expit(h@self.classifier),1e-6,1-1e-6)
        y=np.clip(h@self.values,.018*self.upper,self.upper)
        return p*y


def _masked_mean(s,rng,proposal,model,z,y,previous,alpha):
    R=alpha+(1-alpha)*proposal.ratio(z)
    cv=model.predict(z)
    var1=np.mean(y*y/(R*previous))
    var2=np.mean((y-cv)**2/(R*previous))
    debug("sparse control proxy",var2/max(var1,1e-20))
    if var2>.79*var1 or s.elapsed>s.limit-9.:
        return None
    return float(np.mean([model.predict(normals(rng,14,s.d)@proposal.basis.T).mean()
                          for _ in range(8)]))


def estimate_rare(s,rng,v,g,z,y):
    d=s.d
    B=ordered_basis(g.T@g,v)
    survivors=y>0
    t=z[survivors]@v
    if len(t)<3 or s.remaining<8000:
        return None
    w=y[survivors]**.7
    w=w/w.sum()
    mt=t@w
    st=np.sqrt(max(.16,((t-mt)**2)@w))
    st=np.clip(st*1.24,.44,1.14)
    prop=Gaussian(np.clip(mt,-3.,4.)*v,B,np.r_[st,np.ones(d-1)])
    z1,y1,q1 = _pilot(s,rng,prop,2048,256,B)
    Z=np.r_[z,z1]
    Y=np.r_[y,y1]
    R=np.r_[np.ones(len(y)),q1]
    P=_proposal(Z,Y,R,v,prop)
    z2,y2,q2 = _pilot(s,rng,P,1024,256,B)
    P=_proposal(np.r_[Z,z2],np.r_[Y,y2],np.r_[R,q2],v,P)
    # The dedicated ordinary block and the importance block have known
    # masses, so the balance heuristic exactly integrates their mixture.
    n0=512 if s.remaining>=4608 else 256
    n1=s.remaining-n0
    alpha=n0/(n0+n1)
    model,exact=None,0.
    if np.count_nonzero(Y>0)>=45 and np.count_nonzero(y2>0)>=24:
        candidate=_MaskedModel(P,Z,Y)
        fit=_masked_mean(s,rng,P,candidate,z2,y2,q2,alpha)
        if fit is not None and np.isfinite(fit):
            model,exact=candidate,fit
    total,count=0.,0
    for p in net_sizes(n0):
        x=normals(rng,p,d)@B.T
        f=s.query(x)
        if model is not None:
            f-=model.predict(x)
        total+=np.sum(f/(alpha+(1-alpha)*P.ratio(x)))
        count+=len(x)
    for p in net_sizes(n1):
        x=P.points(normals(rng,p,d))
        f=s.query(x)
        if model is not None:
            f-=model.predict(x)
        total+=np.sum(f/(alpha+(1-alpha)*P.ratio(x)))
        count+=len(x)
    return exact+total/count
