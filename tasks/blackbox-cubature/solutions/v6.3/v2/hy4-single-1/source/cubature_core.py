"""Prototype value-driven cubature with data-driven control-variate models."""
from __future__ import annotations
import math, itertools, json, sys, time
import numpy as np
from scipy.stats import qmc
from scipy.special import eval_legendre
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import least_squares, minimize_scalar
from numpy.polynomial.legendre import leggauss

GL_X16, GL_W16 = leggauss(24)
GL_X16 = 0.5*(GL_X16+1.0); GL_W16 = 0.5*GL_W16

def _sinc(a):
    a=np.asarray(a,float)
    return np.where(np.abs(a)<1e-10,1.0,np.sin(a/2)/(a/2))

def _sinc_prod(a):
    return float(np.prod(_sinc(a)))

def _unique_nodes(t, f):
    order=np.argsort(t); t=np.asarray(t)[order]; f=np.asarray(f)[order]
    tt=[t[0]]; ff=[f[0]]
    for i in range(1,len(t)):
        if t[i]-tt[-1] < 1e-12:
            ff[-1]=0.5*(ff[-1]+f[i])
        else:
            tt.append(t[i]); ff.append(f[i])
    return np.asarray(tt),np.asarray(ff)

class Model:
    def __init__(self, name, integral, predict, complexity=0.0, fold_predict=None):
        self.name=name; self.integral=integral; self.predict=predict
        self.complexity=complexity; self.fold_predict=fold_predict
    def values(self,X): return np.asarray(self.predict(X),float)
    def estimates(self,X,y,folds):
        # returns per-fold estimates and out-of-fold residual variance
        K=int(folds.max())+1
        es=np.empty(K); resid=np.empty(len(y))
        for r in range(K):
            m=folds==r
            g=self.predict(X[m]) if self.fold_predict is None else self.fold_predict(r,X[m])
            I=self.integral if not isinstance(self.integral,(list,tuple,np.ndarray)) else self.integral[r]
            es[r]=I+np.mean(y[m]-g)
            resid[m]=y[m]-g
        return es,resid

# ---------------- polynomial / spline control variates ----------------
def legendre_1d(x,K):
    z=2*np.asarray(x)-1
    return np.column_stack([math.sqrt(2*k+1)*eval_legendre(k,z) for k in range(K+1)])

def design_matrix(X,deg,fourier=0,hinges=0,steps=0):
    n,d=X.shape; cols=[]
    if deg>0:
        L=legendre_1d(X.ravel(),deg).reshape(n,d,deg+1)
        for exps in itertools.product(range(deg+1),repeat=d):
            s=sum(exps)
            if s==0 or s>deg: continue
            c=np.ones(n)
            for i,e in enumerate(exps):
                if e: c=c*L[:,i,e]
            cols.append(c)
    for k in range(1,fourier+1):
        for i in range(d):
            cols.append(math.sqrt(2)*np.sin(2*math.pi*k*X[:,i]))
            cols.append(math.sqrt(2)*np.cos(2*math.pi*k*X[:,i]))
    for i in range(d):
        for t in np.linspace(0.05,0.95,hinges):
            cols.append(np.abs(X[:,i]-t)-(t*t+(1-t)*(1-t))/2)
        for t in np.linspace(0.05,0.95,steps):
            cols.append((X[:,i]>t).astype(float)-(1-t))
    return np.column_stack(cols) if cols else np.zeros((n,0))

def polynomial_models(X,y,d,folds,seed):
    """Cross-fitted linear control variates. Returns list of Model objects."""
    N=len(y); K=int(folds.max())+1
    maxdeg={2:7,3:5,5:4,8:3}.get(d,3)
    if d>10: maxdeg=1
    elif d>8: maxdeg=2
    configs=[]
    for deg in range(1,maxdeg+1): configs.append((deg,0,0,0))
    for deg in (1,2,min(3,maxdeg)):
        configs.append((deg,0,9,9))
        configs.append((deg,6,0,0))
    seen=set(); configs=[c for c in configs if not (c in seen or seen.add(c))]
    out=[]
    for cfg in configs:
        Psi=design_matrix(X,*cfg); p=Psi.shape[1]
        if p==0 or p>N//3: continue
        # moments of full data and folds
        G=Psi.T@Psi; b=Psi.T@y; c=Psi.sum(0); ys=y.sum()
        Gf=np.stack([Psi[folds==r].T@Psi[folds==r] for r in range(K)])
        bf=np.stack([Psi[folds==r].T@y[folds==r] for r in range(K)])
        cf=np.stack([Psi[folds==r].sum(0) for r in range(K)])
        yf=np.array([y[folds==r].sum() for r in range(K)])
        nf=np.array([np.sum(folds==r) for r in range(K)])
        ntr=N-nf
        betas=np.empty((K,p)); intercepts=np.empty(K)
        for r in range(K):
            A=(G-Gf[r])-np.outer(c-cf[r],c-cf[r])/ntr[r]
            rhs=(b-bf[r])-(c-cf[r])*(ys-yf[r])/ntr[r]
            lam=1e-15*np.trace(A)/max(p,1)+1e-14
            A.flat[::p+1]+=lam
            try: beta=np.linalg.solve(A,rhs)
            except np.linalg.LinAlgError: beta=np.linalg.lstsq(A,rhs,rcond=None)[0]
            betas[r]=beta
            intercepts[r]=(ys-yf[r]-(c-cf[r])@beta)/ntr[r]
        # fold-specific prediction and integral: features have zero analytic mean
        avg_beta=betas.mean(0); avg_int=float(np.mean(intercepts))
        def predict_all(Xr,cfg=cfg,avg_beta=avg_beta,avg_int=avg_int):
            return avg_int+design_matrix(Xr,*cfg)@avg_beta
        def fp(r,Xr,cfg=cfg,betas=betas,intercepts=intercepts):
            return intercepts[r]+design_matrix(Xr,*cfg)@betas[r]
        name='poly'+str(cfg)
        out.append(Model(name,intercepts,predict_all,complexity=2.0*p,fold_predict=fp))
    return out

# ---------------- fixed models from coordinate slices ----------------
def gauss_integral(interp, nodes=None):
    if nodes is None or len(nodes)<2:
        vals=interp(GL_X16)
        return float(np.dot(GL_W16,vals))
    t=np.asarray(nodes,float)
    edges=np.concatenate(([0.0],t,[1.0])) if t[0]>0 or t[-1]<1 else t
    # include [0,1] boundaries and all interior nodes
    edges=np.unique(np.clip(np.concatenate(([0.0],t,[1.0])),0,1))
    total=0.0
    for a,b in zip(edges[:-1],edges[1:]):
        if b-a<1e-14: continue
        x=a+(b-a)*GL_X16
        total+=(b-a)*float(np.dot(GL_W16,interp(x)))
    return total

def build_interpolators(t_nodes,f_nodes,kind,c0):
    """Return list of (evaluator, integral) for z=f-c0 on each coordinate."""
    out=[]
    for t,f in zip(t_nodes,f_nodes):
        tt,ff=_unique_nodes(t,f)
        z=ff-c0
        if kind=='linear':
            # piecewise linear with flat extrapolation to endpoints
            def ev(x,t=tt,z=z): return np.interp(x,t,z)
            # composite trapezoid on uniform fine grid
            xg=np.linspace(0,1,2049); vals=np.interp(xg,tt,z)
            J=float(np.sum(0.5*(vals[:-1]+vals[1:])*np.diff(xg)))
            out.append((ev,J)); continue
        if kind=='logcubic':
            if np.any(z<=0) or np.any(z>=0)==False: pass
            if np.all(z>0) or np.all(z<0):
                sgn=1.0 if z[0]>0 else -1.0
                lz=np.log(np.abs(z))
                if len(tt)>=4:
                    cs=CubicSpline(tt,lz,extrapolate=True)
                    def ev(x,cs=cs,sgn=sgn): return sgn*np.exp(np.clip(cs(x),-700,700))
                    J=gauss_integral(ev,tt)
                    out.append((ev,J)); continue
            out.append((None,None)); continue
        if len(tt)<4:
            out.append((None,None)); continue
        try:
            if kind=='pchip': interp=PchipInterpolator(tt,z,extrapolate=True)
            else: interp=CubicSpline(tt,z,extrapolate=True)
            def ev(x,interp=interp): return interp(x)
            J=gauss_integral(ev,tt)
            out.append((ev,J))
        except Exception:
            out.append((None,None))
    return out

def product_models(t_nodes,f_nodes,f0,d,c0_candidates):
    models=[]
    for c0 in c0_candidates:
        z0=f0-c0
        scale=max(1e-12,np.max(np.abs(np.concatenate([f for f in f_nodes]))))
        if abs(z0)<1e-7*scale: continue
        for kind in ('pchip','cubic','logcubic','linear'):
            its=build_interpolators(t_nodes,f_nodes,kind,c0)
            if any(ev is None or not np.isfinite(J) for ev,J in its): continue
            ratios_J=np.array([J/z0 for ev,J in its])
            if not np.all(np.isfinite(ratios_J)): continue
            if any(J==0 for ev,J in its): continue
            logmag=(1.0-d)*math.log(abs(z0))+sum(math.log(abs(J)) for ev,J in its)
            if logmag>50: continue
            I_z=z0*float(np.prod(ratios_J))
            I_g=c0+I_z
            if not np.isfinite(I_g): continue
            def predict(X,its=its,z0=z0,c0=c0,d=d):
                logp=np.zeros(len(X)); signp=np.ones(len(X))
                for i,(ev,J) in enumerate(its):
                    v=ev(np.clip(X[:,i],0,1))/z0
                    logp+=np.log(np.abs(np.maximum(v,1e-300)))
                    signp*=np.sign(v)
                prod=signp*np.exp(np.clip(logp,-700,700))
                return c0+z0*prod
            models.append(Model(f'product_{kind}_c{c0:.3g}',I_g,predict,complexity=8*d))
    return models

def _fit_1d_sinusoid(t,f,omega_grid):
    # fit [1, cos(wt), sin(wt)] by least squares for every candidate frequency
    best=(np.inf,0.0,None)
    C=np.cos(np.outer(t,omega_grid)); S=np.sin(np.outer(t,omega_grid))
    one=np.ones((len(t),1))
    for j,w in enumerate(omega_grid):
        M=np.column_stack((one,C[:,j],S[:,j]))
        coef,_,_,_=np.linalg.lstsq(M,f,rcond=None)
        r=f-M@coef
        sse=float(r@r)
        if sse<best[0]: best=(sse,w,coef)
    return best

def _refine_1d_sinusoid(t,f,w0,step):
    one=np.ones(len(t))
    def fit(w):
        M=np.column_stack((one,np.cos(w*t),np.sin(w*t)))
        coef,_,_,_=np.linalg.lstsq(M,f,rcond=None)
        r=f-M@coef
        return float(r@r),coef
    lo=max(1e-6,w0-2*step); hi=w0+2*step
    try:
        res=minimize_scalar(lambda w: fit(w)[0],bounds=(lo,hi),method='bounded',
                            options={'xatol':max(1e-11,step*1e-6)})
        w=float(res.x); sse,coef=fit(w)
        return sse,w,coef
    except Exception:
        return fit(w0)[0],w0,fit(w0)[1]

def _circular_spread(angles):
    z=np.mean(np.exp(1j*np.asarray(angles)))
    return 1-abs(z),np.angle(z)

def _fit_joint_sinusoid(t_nodes,f_nodes,x0,d,omega_grid):
    """Fit one linear-phase sinusoid using coordinate slices only."""
    tt=np.concatenate(t_nodes); ff=np.concatenate(f_nodes)
    dims=np.concatenate([np.full(len(t),i) for i,t in enumerate(t_nodes)])
    base=np.tile(x0,(len(tt),1))
    for i in range(d):
        m=dims==i; base[m,i]=tt[m]
    # per-coordinate periodogram
    fits=[_fit_1d_sinusoid(t,f,omega_grid) for t,f in zip(t_nodes,f_nodes)]
    sse0=sum(float(np.var(f))*len(f) for f in f_nodes)
    sse1=sum(f[0] for f in fits)
    if sse1>0.80*sse0: return None
    step=omega_grid[1]-omega_grid[0]
    fits=[_refine_1d_sinusoid(t,f,ft[1],step) for (t,f),ft in zip(zip(t_nodes,f_nodes),fits)]
    omegas=np.array([f[1] for f in fits])
    active=[i for i,w in enumerate(omegas) if w>1e-6]
    if len(active)>10:
        # deterministic pseudo-random subset of sign choices for high d
        rr=np.random.default_rng(12345)
        combos=[rr.choice([1,-1],size=len(active)) for _ in range(512)]
    else:
        combos=itertools.product([1,-1],repeat=len(active))
    best=None
    # For each sign choice, fit global (c0,A,B) linearly and keep the best.
    for bits in combos:
        s=np.ones(d)
        for i,b in zip(active,bits): s[i]=b
        a=s*omegas
        z=base@a
        M=np.column_stack((np.ones(len(z)),np.cos(z),np.sin(z)))
        coef,_,_,_=np.linalg.lstsq(M,ff,rcond=None)
        r=ff-M@coef
        sse=float(r@r)
        if best is None or sse<best[0]: best=(sse,a.copy(),coef.copy())
    if best is None: return None
    a0,c0AB=best[1],best[2]
    p0=np.concatenate([a0,c0AB])  # a, c0, A, B
    def residual(p):
        z=base@p[:d]
        return p[d]+p[d+1]*np.cos(z)+p[d+2]*np.sin(z)-ff
    try:
        res=least_squares(residual,p0,max_nfev=200,ftol=1e-13,xtol=1e-13,gtol=1e-13)
        if float(res.cost)<=float(np.sum(residual(p0)**2))+1e-12:
            p=res.x
        else: p=p0
    except Exception:
        p=p0
    return p

def make_sinusoid_model(p,name,complexity):
    d=(len(p)-3)
    a=p[:d]; c0=p[d]; A=p[d+1]; B=p[d+2]
    sp=_sinc_prod(a); half=0.5*float(np.sum(a))
    I=c0+sp*(A*math.cos(half)+B*math.sin(half))
    def predict(X,a=a,c0=c0,A=A,B=B): return c0+A*np.cos(X@a)+B*np.sin(X@a)
    return Model(name,I,predict,complexity=complexity)

def sinusoid_models(t_nodes,f_nodes,x0,d,y_pilot,max_components=2):
    models=[]
    K=min(len(t) for t in t_nodes)
    om_max=min(260.0,math.pi*(K-1))
    omega_grid=np.linspace(0.35,om_max,min(2200,int(om_max*10)+1))
    p=_fit_joint_sinusoid(t_nodes,f_nodes,x0,d,omega_grid)
    if p is None: return []
    models.append(make_sinusoid_model(p,'sinusoid',4*d+10))
    if max_components>=2:
        # second sinusoid by matching pursuit on slice residuals, then joint refinement
        a=p[:d]; c0=p[d]; A=p[d+1]; B=p[d+2]
        resid_nodes=[]
        for i,(t,f) in enumerate(zip(t_nodes,f_nodes)):
            z=a[i]*t+sum(a[j]*x0[j] for j in range(d) if j!=i)
            resid_nodes.append(f-(c0+A*np.cos(z)+B*np.sin(z)))
        p2=_fit_joint_sinusoid(t_nodes,resid_nodes,x0,d,omega_grid)
        if p2 is not None:
            tt=np.concatenate(t_nodes); ff=np.concatenate(f_nodes)
            dims=np.concatenate([np.full(len(t),i) for i,t in enumerate(t_nodes)])
            base=np.tile(x0,(len(tt),1))
            for i in range(d):
                m=dims==i; base[m,i]=tt[m]
            p20=np.concatenate([p[:d],p2[:d],[p[d]+p2[d],p[d+1],p[d+2],p2[d+1],p2[d+2]]])
            def residual(q):
                z1=base@q[:d]; z2=base@q[d:2*d]
                return q[2*d]+q[2*d+1]*np.cos(z1)+q[2*d+2]*np.sin(z1)+q[2*d+3]*np.cos(z2)+q[2*d+4]*np.sin(z2)-ff
            try:
                res=least_squares(residual,p20,max_nfev=150,ftol=1e-12,xtol=1e-12)
                q=res.x if res.cost*2<=np.sum(residual(p20)**2)+1e-12 else p20
                a1=q[:d];a2=q[d:2*d];c0,A,B,A2,B2=q[2*d:]
                sp1=_sinc_prod(a1);h1=.5*float(np.sum(a1));sp2=_sinc_prod(a2);h2=.5*float(np.sum(a2))
                I=c0+sp1*(A*math.cos(h1)+B*math.sin(h1))+sp2*(A2*math.cos(h2)+B2*math.sin(h2))
                def predict(X,a1=a1,a2=a2,c0=c0,A=A,B=B,A2=A2,B2=B2):
                    return c0+A*np.cos(X@a1)+B*np.sin(X@a1)+A2*np.cos(X@a2)+B2*np.sin(X@a2)
                models.append(Model('sinusoid2',I,predict,complexity=8*d+20))
            except Exception: pass
    return models

def checkerboard_models(t_nodes,f_nodes,x0,d,f0):
    models=[]
    thresholds=[]; jumps=[]
    for i,(t,f) in enumerate(zip(t_nodes,f_nodes)):
        # skip the known base-point node in jump detection
        dif=np.abs(np.diff(f)); j=int(np.argmax(dif))
        thr=0.5*(t[j]+t[j+1]); jump=f[j+1]-f[j]
        thresholds.append(thr); jumps.append(jump)
    thresholds=np.asarray(thresholds)
    signs=np.where(x0>thresholds,1.0,-1.0)
    amps=[]
    for i,j in enumerate(jumps):
        other=np.prod(np.delete(signs,i))
        amps.append(j/(2*other))
    amp=float(np.median(amps)) if amps else 0.0
    lev=[]
    for f in f_nodes: lev.extend([f.min(),f.max()])
    c0=0.5*(min(lev)+max(lev))
    I=c0+amp*float(np.prod(1-2*thresholds))
    def predict(X,c0=c0,amp=amp,th=thresholds):
        s=np.where(X>th,1.0,-1.0)
        return c0+amp*np.prod(s,axis=1)
    models.append(Model('checkerboard',I,predict,complexity=4*d))
    return models

# ---------------- selection and main solve ----------------
def solve(oracle, d, budget, seed, verbose=False):
    budget=max(64,int(budget))
    sseed=int(seed)&0xffffffff
    rng=np.random.default_rng(sseed^0x5f3759df)
    probe_budget=min(512,max(128,budget//4))
    coarse_budget=probe_budget//2
    n_refine=probe_budget-coarse_budget
    K1=max(8,coarse_budget//d)
    n_coarse=d*K1
    n_probes=n_coarse+n_refine
    n_qmc=max(1,budget-n_probes)
    m=1<<int(math.ceil(math.log2(max(1,n_qmc))))
    q=qmc.Sobol(d=d,scramble=True,seed=sseed)
    Xall=q.random(m)[:n_qmc]
    npilot=min(512,max(64,n_qmc//3))
    y_pilot=oracle(Xall[:npilot])
    sc=np.abs(y_pilot-np.median(y_pilot))
    j0=int(np.argmax(sc)); x0=Xall[j0].copy(); f0=float(y_pilot[j0])
    t1=np.linspace(0.0,1.0,K1)
    coarse=np.vstack([np.tile(x0,(K1,1)) for _ in range(d)])
    for i in range(d): coarse[i*K1:(i+1)*K1,i]=t1
    y1=oracle(coarse)
    t_nodes=[np.concatenate([t1,[x0[i]]]) for i in range(d)]
    f_nodes=[np.concatenate([y1[i*K1:(i+1)*K1],[f0]]) for i in range(d)]
    for i in range(d):
        t_nodes[i],f_nodes[i]=_unique_nodes(t_nodes[i],f_nodes[i])
    # adaptive refinement of coordinate slices
    rounds=max(1,n_refine//16); per_round=max(1,n_refine//rounds)
    scale=max(1e-12,float(np.max(np.abs(np.concatenate(f_nodes)))))
    for rnd in range(rounds):
        cands=[]; allints=[]
        for i in range(d):
            t=t_nodes[i]; f=f_nodes[i]; n=len(t)
            for j in range(n-1):
                w=t[j+1]-t[j]
                if w<1e-10: continue
                jump=abs(f[j+1]-f[j])
                curv=0.0
                if j>0: curv+=abs(f[j+1]-2*f[j]+f[j-1])
                if j+2<n: curv+=abs(f[j+2]-2*f[j+1]+f[j])
                mag=0.5*(abs(f[j])+abs(f[j+1]))+0.05*scale
                score=(jump+0.5*curv)*mag*(w**0.25)
                item=(score,i,0.5*(t[j]+t[j+1]))
                cands.append(item); allints.append(item)
        if not cands: break
        cands.sort(key=lambda z:z[0],reverse=True)
        chosen=[]; seen=set()
        for it in cands:
            key=(it[1],round(it[2],12))
            if key in seen: continue
            seen.add(key); chosen.append(it)
            if len(chosen)>=max(1,per_round*3//4): break
        # exploration: random intervals weighted by width, to find missed narrow peaks
        if allints and len(chosen)<per_round:
            idx=list(range(len(allints)))
            probs=np.array([max(1e-9,it[0]) for it in allints],float)
            probs=0.15*probs/probs.sum()+0.85*np.array([
                (t_nodes[it[1]][np.searchsorted(t_nodes[it[1]],it[2])]-t_nodes[it[1]][np.searchsorted(t_nodes[it[1]],it[2])-1]) if 0<np.searchsorted(t_nodes[it[1]],it[2])<len(t_nodes[it[1]]) else 1e-6
                for it in allints])
            probs=np.maximum(probs,1e-12); probs/=probs.sum()
            take=per_round-len(chosen)
            for jj in rng.choice(len(allints),size=min(take,len(allints)),replace=False,p=probs):
                it=allints[jj]; key=(it[1],round(it[2],12))
                if key not in seen: seen.add(key); chosen.append(it)
                if len(chosen)>=per_round: break
        if not chosen: break
        pts=[]; meta=[]
        for score,i,tm in chosen[:per_round]:
            p=x0.copy(); p[i]=tm; pts.append(p); meta.append((i,tm))
        pts=np.asarray(pts)
        vals=oracle(pts)
        for (i,tm),v in zip(meta,vals):
            t_nodes[i]=np.append(t_nodes[i],tm); f_nodes[i]=np.append(f_nodes[i],float(v))
        for i in range(d):
            t_nodes[i],f_nodes[i]=_unique_nodes(t_nodes[i],f_nodes[i])
    # remaining QMC points
    y_rest=oracle(Xall[npilot:]) if n_qmc>npilot else np.empty(0)
    X=Xall; y=np.concatenate([y_pilot,y_rest])
    c0s=[0.0,float(np.median(y)),float(0.5*(np.min(y)+np.max(y))),float(np.mean(y))]
    c0s=list(dict.fromkeys([round(c,12) for c in c0s]))
    models=[Model('zero',0.0,lambda X: np.zeros(len(X)),complexity=0.0)]
    models+=product_models(t_nodes,f_nodes,f0,d,c0s)
    models+=_multi_sinusoid_models(t_nodes,f_nodes,x0,d,max_components=4,Xq=Xall[:npilot],yq=y_pilot)
    models+=checkerboard_models(t_nodes,f_nodes,x0,d,f0)
    K=4
    perm=rng.permutation(len(y)); folds=np.empty(len(y),dtype=int); folds[perm]=np.arange(len(y))%K
    models+=polynomial_models(X,y,d,folds,seed)
    best=None; scores=[]
    yvar=float(np.var(y))
    for model in models:
        try:
            es,resid=model.estimates(X,y,folds)
            if not np.all(np.isfinite(resid)) or not np.all(np.isfinite(es)): continue
            rv=float(np.var(resid))
            ev=float(np.var(es,ddof=1)) if K>1 else rv
            score=rv*(1.0+0.002*model.complexity)+0.05*ev
            scores.append((score,rv,ev,model,es))
        except Exception:
            continue
    if not scores: return float(np.mean(y))
    scores.sort(key=lambda z:z[0])
    score,rv,ev,model,es=scores[0]
    est=float(np.mean(es))
    plain=float(np.mean(y))
    if model.name!='zero' and yvar>0:
        w=max(0.0,min(1.0,1.0-2.0*score/max(yvar,1e-300)))
        est=w*est+(1-w)*plain
    if not np.isfinite(est): est=plain
    return est


# ---------------- multi-component linear-phase sinusoids ----------------
def _integral_sinusoid_params(a,A,B):
    a=np.asarray(a,float); sp=_sinc_prod(a); h=0.5*float(np.sum(a))
    return sp*(A*math.cos(h)+B*math.sin(h))

def _multi_sinusoid_models(t_nodes,f_nodes,x0,d,max_components=4,Xq=None,yq=None):
    tt=np.concatenate(t_nodes); ff=np.concatenate(f_nodes)
    dims=np.concatenate([np.full(len(t),i) for i,t in enumerate(t_nodes)])
    base=np.tile(x0,(len(tt),1))
    for i in range(d):
        m=dims==i; base[m,i]=tt[m]
    K=min(len(t) for t in t_nodes)
    om_max=min(260.0,math.pi*(K-1))
    omega_grid=np.linspace(0.35,om_max,min(2200,int(om_max*10)+1))
    sse0=float(ff@ff)/max(1.0,float(np.mean(ff**2))*len(ff))
    resid=[f.copy() for f in f_nodes]
    comps=[]; c0_0=0.0; models=[]
    for k in range(max_components):
        p=_fit_joint_sinusoid(t_nodes,resid,x0,d,omega_grid)
        if p is None: break
        a=p[:d]; c0=p[d]; A=p[d+1]; B=p[d+2]
        if k==0: c0_0=c0
        comps.append((a.copy(),A,B))
        # subtract this component (and its fitted offset) from slice residuals
        for i,(t,f) in enumerate(zip(t_nodes,resid)):
            z=a[i]*t+sum(a[j]*x0[j] for j in range(d) if j!=i)
            resid[i]=f-(c0+A*np.cos(z)+B*np.sin(z))
        # build a jointly refined model with all components found so far
        m=len(comps)
        q0=[]
        for a,A,B in comps: q0.extend(a.tolist())
        q0.append(c0_0)
        for a,A,B in comps: q0.extend([A,B])
        q0=np.asarray(q0,float)
        npar=len(q0)
        def residual(q,m=m,d=d,base=base,ff=ff):
            zz=base@q[:m*d].reshape(m,d).T
            out=q[m*d]+np.zeros(len(ff))
            for j in range(m): out=out+q[m*d+1+2*j]*np.cos(zz[:,j])+q[m*d+2+2*j]*np.sin(zz[:,j])
            return out-ff
        def jac(q,m=m,d=d,base=base):
            zz=base@q[:m*d].reshape(m,d).T
            J=np.empty((len(ff),npar))
            J[:,m*d]=1.0
            for j in range(m):
                A=q[m*d+1+2*j]; B=q[m*d+2+2*j]
                cz=np.cos(zz[:,j]); sz=np.sin(zz[:,j])
                J[:,m*d+1+2*j]=cz; J[:,m*d+2+2*j]=sz
                g=-A*sz+B*cz
                for i in range(d): J[:,j*d+i]=base[:,i]*g
            return J
        try:
            res=least_squares(residual,q0,jac=jac,max_nfev=120,ftol=1e-12,xtol=1e-12,gtol=1e-12)
            q=res.x if res.cost*2<=np.sum(residual(q0)**2)+1e-12 else q0
        except Exception:
            q=q0
        aall=q[:m*d].reshape(m,d); c0=q[m*d]
        A=q[m*d+1::2][:m]; B=q[m*d+2::2][:m]
        I=c0+sum(_integral_sinusoid_params(aall[j],A[j],B[j]) for j in range(m))
        def predict(X,aall=aall,c0=c0,A=A,B=B,m=m):
            zz=X@aall.T
            out=c0+np.zeros(len(X))
            for j in range(m): out=out+A[j]*np.cos(zz[:,j])+B[j]*np.sin(zz[:,j])
            return out
        models.append(Model('sinusoid%d'%m,I,predict,complexity=4*d*m+10*m))
        if Xq is not None and yq is not None and len(yq)>4*m+4:
            bq=np.asarray(Xq,float); fq=np.asarray(yq,float)
            def rq(qq,m=m,d=d,base=bq,ff=fq):
                zz=base@qq[:m*d].reshape(m,d).T
                out=qq[m*d]+np.zeros(len(ff))
                for j in range(m): out=out+qq[m*d+1+2*j]*np.cos(zz[:,j])+qq[m*d+2+2*j]*np.sin(zz[:,j])
                return out-ff
            def jq(qq,m=m,d=d,base=bq):
                zz=base@qq[:m*d].reshape(m,d).T
                J=np.empty((len(fq),npar)); J[:,m*d]=1.0
                for j in range(m):
                    A=qq[m*d+1+2*j]; B=qq[m*d+2+2*j]
                    cz=np.cos(zz[:,j]); sz=np.sin(zz[:,j]); g=-A*sz+B*cz
                    J[:,m*d+1+2*j]=cz; J[:,m*d+2+2*j]=sz
                    for i in range(d): J[:,j*d+i]=base[:,i]*g
                return J
            try:
                rr=least_squares(rq,q,jac=jq,max_nfev=80,ftol=1e-12,xtol=1e-12,gtol=1e-12)
                qr=rr.x if rr.cost*2<=np.sum(rq(q)**2)+1e-12 else q
                ar=qr[:m*d].reshape(m,d); c0r=qr[m*d]; Ar=qr[m*d+1::2][:m]; Br=qr[m*d+2::2][:m]
                Ir=c0r+sum(_integral_sinusoid_params(ar[j],Ar[j],Br[j]) for j in range(m))
                def predictr(X,ar=ar,c0=c0r,A=Ar,B=Br,m=m):
                    zz=X@ar.T; out=c0+np.zeros(len(X))
                    for j in range(m): out=out+A[j]*np.cos(zz[:,j])+B[j]*np.sin(zz[:,j])
                    return out
                models.append(Model('sinusoid%d_qmc'%m,Ir,predictr,complexity=4*d*m+10*m+5))
            except Exception: pass
        rms=float(np.mean(residual(q)**2))
        if rms<1e-20*max(1.0,float(np.mean(ff**2))): break
    return models
