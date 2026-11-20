"""Low-startup HiGHS interface for the polyhedral model."""
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
import sys
from time import perf_counter

import numpy as np


_site = os.path.dirname(os.path.dirname(np.__file__))


def extension(module, directory, basename):
    if module in sys.modules:
        return sys.modules[module]
    directory = os.path.join(_site, 'scipy', directory)
    path = next(os.path.join(directory, f) for f in os.listdir(directory)
                if f.startswith(basename+'.') and f.endswith('.so'))
    spec = spec_from_file_location(module, path)
    obj = module_from_spec(spec)
    sys.modules[module] = obj
    spec.loader.exec_module(obj)
    return obj


def norm(v):
    return float(np.max(np.abs(v),initial=0.))


class Csc:
    def __init__(self, m, n, ptr, idx, val):
        self.m, self.n = m, n
        self.ptr, self.idx, self.val = (np.ascontiguousarray(ptr,dtype=np.int32),
                                       np.ascontiguousarray(idx,dtype=np.int32),
                                       np.ascontiguousarray(val,dtype=np.float64))

    def canonical(self,sp):
        n, m, p, i, v = self.n,self.m,self.ptr,self.idx,self.val
        if not sp.csr_has_canonical_format(n,p,i):
            sp.csr_sort_indices(n,p,i,v)
            sp.csr_sum_duplicates(n,m,p,i,v)
        self.idx, self.val = i[:p[-1]], v[:p[-1]]
        if np.any(self.val == 0.):
            sp.csr_eliminate_zeros(n,p,self.idx,self.val)
            self.idx, self.val = i[:p[-1]], v[:p[-1]]

    def mv(self, x, sp, transpose=False):
        if transpose:
            y = np.zeros(self.n)
            sp.csr_matvec(self.n,self.m,self.ptr,self.idx,self.val,x,y)
        else:
            y = np.zeros(self.m)
            sp.csc_matvec(self.m,self.n,self.ptr,self.idx,self.val,x,y)
        return y

    def t(self,sp):
        p = np.empty(self.m+1,dtype=np.int32)
        i = np.empty(len(self.idx),dtype=np.int32)
        v = np.empty(len(self.val))
        sp.csr_tocsc(self.n,self.m,self.ptr,self.idx,self.val,p,i,v)
        return Csc(self.n,self.m,p,i,v)

    def filtered(self, mask, newrow=None):
        inner = mask[self.idx]
        ptr = np.empty(len(inner)+1,dtype=np.int32)
        ptr[0] = 0
        np.cumsum(inner,dtype=np.int32,out=ptr[1:])
        i = self.idx[inner]
        if newrow is not None:
            i = newrow[i]
        return Csc(int(mask.sum()),self.n,ptr[self.ptr],i,self.val[inner])

    def lower(self):
        cols = np.repeat(np.arange(self.n,dtype=np.int32),np.diff(self.ptr))
        sel = self.idx >= cols
        ptr = np.empty(len(sel)+1,dtype=np.int32)
        ptr[0] = 0
        np.cumsum(sel,dtype=np.int32,out=ptr[1:])
        return Csc(self.m,self.n,ptr[self.ptr],self.idx[sel],self.val[sel])


class Data:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as f:
            self.q, self.b = q,b = [np.ascontiguousarray(f[k],dtype=np.float64) for k in ('q','b')]
            self.n, self.m = n,m = len(q),len(b)
            self.A = Csc(m,n,f['A_indptr'],f['A_indices'],f['A_data'])
            self.P = Csc(n,n,f['P_indptr'],f['P_indices'],f['P_data'])
            meta = json.loads(str(f['metadata_json']))
            self.spec = [(c['type'],c['dim']) for c in meta['cones']]
            self.offset = float(meta.get('objective_offset',0.))
        self.linear = all(k in ('zero','nonnegative') or (k=='psd' and d==1)
                          for k,d in self.spec)
        self.sections = []
        self.eq = None
        if self.linear:
            p = 0
            self.eq = np.zeros(m,dtype=bool)
            for kind,d in self.spec:
                if kind == 'zero':
                    self.eq[p:p+d] = True
                else:
                    self.sections.append((p,p+d))
                p += d
        self.bn,self.qn = norm(b),norm(q)

    def inflate(self):
        from scipy.sparse import csc_matrix
        from model import Problem, Layout
        mats = []
        for M in (self.P,self.A):
            mat = csc_matrix((M.val,M.idx,M.ptr),shape=(M.m,M.n))
            mat.sum_duplicates()
            mat.eliminate_zeros()
            mats.append(mat)
        return Problem(mats[0],self.q,mats[1],self.b,Layout(self.spec),self.offset)

    def optimal(self,x,z,sp):
        if len(x)!=self.n or len(z)!=self.m or not np.isfinite(x).all() or not np.isfinite(z).all():
            return None
        Ax = self.A.mv(x,sp)
        Px = self.P.mv(x,sp)
        Az = self.A.mv(z,sp,transpose=True)
        r = norm(Px+self.q+Az)
        ds = 1.+max(norm(Px),norm(Az),self.qn)
        quad = .5*float(x@Px)
        p,d = quad+float(x@self.q), -quad-float(z@self.b)
        eps = 9.5e-9
        if (not np.isfinite(r+ds+p+d) or r > eps*ds or
                not np.isfinite([p+self.offset,d+self.offset]).all() or
                abs(p-d) > eps*max(1.,abs(p),abs(d))):
            return None
        s = self.b-Ax
        if not np.isfinite(s).all():
            return None
        # Represent genuinely active equalities with exact zero slack.
        corrected = s.copy()
        corrected[self.eq] = 0.
        nonnegative = ~self.eq
        corrected[nonnegative] = np.maximum(0.,s[nonnegative])
        delta = norm(Ax+corrected-self.b)
        if delta <= eps*(1.+max(norm(Ax),norm(corrected),self.bn)):
            s = corrected
        elif norm(s[self.eq]) > eps:
            return None
        for a,b in self.sections:
            w,v = s[a:b],z[a:b]
            lo = float(v.min())
            if lo < 0. and -lo > eps*max(1.,-lo,float(v.max())):
                return None
            if s is not corrected:
                lo = float(w.min())
                if lo < 0. and -lo > eps*max(1.,-lo,float(w.max())):
                    return None
        if np.isfinite(s@z):
            return {'status':'solved','x':x,'s':s,'z':z}
        return None

    def certify(self,y,dual,sp):
        a = norm(self.A.val) or 1.
        yn = norm(y)
        if yn==0. or not np.isfinite(yn):
            return None
        y = y/yn
        A = Csc(self.m,self.n,self.A.ptr,self.A.idx,self.A.val/a)
        tol = 9.4e-9
        if not dual:
            if not self.bn:
                return None
            beta = -float((self.b/self.bn)@y)
            if (not np.isfinite(beta) or beta <= 1.01e-8 or
                norm(A.mv(y,sp,transpose=True)) > tol*beta or
                np.min(y[~self.eq],initial=0.) < -tol*beta):
                return None
            return {'status':'primal_infeasible','z':y}
        if not self.qn:
            return None
        alpha = -float((self.q/self.qn)@y)
        if not np.isfinite(alpha) or alpha <= 1.01e-8:
            return None
        if len(self.P.val):
            P = Csc(self.n,self.n,self.P.ptr,self.P.idx,self.P.val/(norm(self.P.val) or 1.))
            if norm(P.mv(y,sp)) > tol*alpha:
                return None
        s = -A.mv(y,sp)
        if norm(s[self.eq]) > tol*alpha:
            return None
        for i,j in self.sections:
            w = s[i:j]
            if -float(w.min()) > tol*alpha*max(1.,norm(w)):
                return None
        return {'status':'dual_infeasible','x':y,'s':s*a} if np.isfinite(s*a).all() else None


class Bounds:
    """Fold simple explicit rows into HiGHS column bounds."""
    def __init__(self, p, A, choose):
        self.keep = keep = ~choose
        rm = np.cumsum(keep,dtype=np.int32)-1
        self.M = A.filtered(keep,rm)
        n = A.n
        entries = np.flatnonzero(choose[A.idx])
        cols = np.searchsorted(A.ptr[1:],entries,side='right').astype(np.int32)
        rs = A.idx[entries]
        aa = A.val[entries]
        lim = p.b[rs]/aa
        self.lb = lb = np.full(n,-np.inf)
        self.ub = ub = np.full(n,np.inf)
        low, high = (aa<0.) | p.eq[rs], (aa>0.) | p.eq[rs]
        np.maximum.at(lb,cols[low],lim[low])
        np.minimum.at(ub,cols[high],lim[high])
        self.rows = rs
        self.val = aa
        self.li = self.best(cols,aa,low&(lim==lb[cols]),n)
        self.ui = self.best(cols,aa,high&(lim==ub[cols]),n)
        self.ok = np.isfinite(lim).all()

    @staticmethod
    def best(cols,a,mask,n):
        score = np.zeros(n)
        c, z = cols[mask],np.abs(a[mask])
        np.maximum.at(score,c,z)
        idx = np.flatnonzero(mask)
        idx = idx[z == score[c]]
        who = np.full(n,-1,dtype=np.int32)
        who[cols[idx]] = idx
        return who

    def restore(self,rd,cd,m):
        z = np.zeros(m)
        z[self.keep] = rd
        low = (cd>0.) & (self.li>=0)
        high = (cd<0.) & (self.ui>=0)
        for mask,idx in [(low,self.li),(high,self.ui)]:
            at = idx[mask]
            z[self.rows[at]] = -cd[mask]/self.val[at]
        return z


class Ranges:
    def __init__(self,A,eq,b,sp):
        self.available = False
        if A.m<30 or A.m>480000 or len(A.val)>2_200_000 or np.count_nonzero(~eq)<24:
            return
        import native
        if native.lib is None:
            return
        gp = native.group_rows(A.t(sp))
        if gp is None:
            return
        which,chosen,mul = gp
        if A.m-len(chosen)<max(5,A.m*.035):
            return
        self.mul,self.m = mul,A.m
        self.l, self.u = lo,hi = np.full(len(chosen),-np.inf),np.full(len(chosen),np.inf)
        bound = b/mul
        if not np.isfinite(bound).all():
            return
        lm, hm = (mul<0.) | eq, (mul>0.) | eq
        np.maximum.at(lo,which[lm],bound[lm])
        np.minimum.at(hi,which[hm],bound[hm])
        self.li = Bounds.best(which,mul,lm & (bound==lo[which]),len(chosen))
        self.ui = Bounds.best(which,mul,hm & (bound==hi[which]),len(chosen))
        self.contradiction = None
        invalid = lo>hi
        if np.any(invalid):
            j = int(np.argmax(np.where(invalid,lo-hi,-np.inf)))
            l,u = self.li[j],self.ui[j]
            f = min(abs(mul[l]),abs(mul[u]))
            z = np.zeros(A.m)
            z[l],z[u] = -f/mul[l],f/mul[u]
            self.contradiction = z
            return
        mask = np.zeros(A.m,dtype=bool)
        mask[chosen] = True
        self.M = A.filtered(mask,which)
        self.available = True

    def recover(self,y):
        z = np.zeros(self.m)
        for select,inds in [((y<0.)&(self.li>=0),self.li),((y>0.)&(self.ui>=0),self.ui)]:
            j=inds[select]
            z[j] = y[select]/self.mul[j]
        return z


def budget_quad(p,sp,diag):
    eig=np.flatnonzero(p.eq)
    if len(eig)!=1:
        return None
    counts=np.bincount(p.A.idx,minlength=p.m)
    row=eig[0]
    if counts[row]<3 or np.any(counts[~p.eq]>1):
        return None
    active=(counts==1)&~p.eq
    B=Bounds(p,p.A,active)
    if not B.ok or np.any(B.lb>B.ub):
        return None
    ids=~B.keep
    if np.any(p.b[~p.eq & ~ids]<0.):
        return None
    # The remaining nonzero row is the equality. Empty feasible rows keep
    # zero multipliers.
    rid=np.flatnonzero(B.keep)
    at=int(np.searchsorted(rid,row))
    T=B.M.t(sp)
    pos=slice(T.ptr[at],T.ptr[at+1])
    a=np.zeros(p.n)
    a[T.idx[pos]]=T.val[pos]
    c=p.q
    lo,hi=B.lb,B.ub
    rhs=p.b[row]
    qt=-c/diag
    d=a/diag
    step0=max(1.e-8,max(1.,norm(qt))/max(norm(d),1.e-30))
    tol=7.e-12*(1.+abs(rhs))

    def val(t):
        y=qt-t*d
        x=np.clip(y,lo,hi)
        f=float(a@x)-rhs
        return x,y,f

    x,y,f=val(0.)
    l,h=-np.inf,np.inf
    t=0.
    for _ in range(38):
        if abs(f)<=tol:
            rm=np.zeros(len(rid))
            rm[at]=t
            g=c+diag*x+t*a
            z=B.restore(rm,g,p.m)
            return p.optimal(x,z,sp)
        if f>0.:
            l=t
        else:
            h=t
        valid=(y>lo)&(y<hi)
        deriv=float(a[valid]@d[valid])
        trial=t+f/deriv if deriv>0. else np.nan
        if np.isfinite(l) and np.isfinite(h):
            if not l+.03*(h-l)<trial<h-.03*(h-l):
                trial=.5*(l+h)
        elif not np.isfinite(trial) or trial<=l or trial>=h or abs(trial-t)>5.*step0:
            trial=t+np.sign(f)*step0
            step0*=2.
        t=trial
        x,y,f=val(t)
        if not np.isfinite(f):
            break
    return None


def few_rows_quad(p,sp,diag,until):
    A=p.A
    counts=np.bincount(A.idx,minlength=p.m)
    multi=(counts>1)
    k=int(multi.sum())
    if not 1<=k<=16 or k*p.n>4_500_000:
        return None
    B=Bounds(p,A,counts==1)
    if not B.ok or np.any(B.lb>B.ub):
        return None
    rid=np.flatnonzero(B.keep)
    nz=multi[rid]
    rhs=p.b[rid[nz]]
    eq=p.eq[rid[nz]]
    part=B.M.filtered(nz,np.cumsum(nz,dtype=np.int32)-1)
    G=np.zeros((k,p.n))
    G[part.idx,np.repeat(np.arange(p.n),np.diff(part.ptr))]=part.val
    iv=1./diag
    E=1./np.sqrt(np.maximum(1.e-100,(G*G)@iv))
    G*=E[:,None]
    rhs=rhs*E
    lo,hi=B.lb,B.ub
    u=np.zeros(k)
    q=p.q
    rtol=1.e-10*(1.+norm(rhs))
    for it in range(32):
        if perf_counter()>until:
            break
        r=q+G.T@u
        v=-r*iv
        x=np.clip(v,lo,hi)
        g=rhs-G@x
        act=eq|(u>0.)|(g<0.)
        measure=norm(g[act])
        if measure<rtol:
            z=np.zeros(len(rid))
            z[nz]=E*u
            ans=p.optimal(x,B.restore(z,r+diag*x,p.m),sp)
            if ans is not None:
                return ans
        s=(v>lo)&(v<hi)
        H=G[act][:,s]
        gg=g[act]
        if not gg.size:
            break
        V=(H*iv[s])@H.T
        V[np.diag_indices(len(gg))]+=1.e-10
        z=u.copy()
        try:
            z[act]-=np.linalg.solve(V,gg)
        except np.linalg.LinAlgError:
            z[act]-=gg
        z[~eq]=np.maximum(0.,z[~eq])
        d=z-u
        grad=float(g@d)
        if not grad < -1.e-20:
            z=u-8.*g
            z[~eq]=np.maximum(0.,z[~eq])
            d=z-u
            grad=float(g@d)
            if not grad < 0.:
                break
        t=G.T@d
        f=float(rhs@d)
        l,h=0.,1.
        fl=grad

        def df(a):
            return f-float(t@np.clip(-(r+a*t)*iv,lo,hi))

        fh=df(h)
        a=h
        if fh>0.:
            for j in range(19):
                a=np.clip((l*fh-h*fl)/(fh-fl),l+.05*(h-l),h-.05*(h-l))
                fm=df(a)
                if fm>0.:
                    h,fh=a,fm
                else:
                    l,fl=a,fm
                if abs(fm)<1.e-7*abs(grad) or h-l<1.e-6:
                    break
        u+=a*d
    return None


def create(high,A,q,l,u,lb,ub,P,seconds,mode):
    hh = high._Highs()
    hh.setOptionValue('output_flag',False)
    h = high.HighsOptions()
    h.output_flag = h.log_to_console = False
    h.threads, h.parallel = 1, 'off'
    h.time_limit = max(.005,seconds)
    h.solver = mode
    h.small_matrix_value, h.large_matrix_value = 1.e-12, 1.e19
    h.infinite_bound = h.infinite_cost = 1.e30
    h.primal_feasibility_tolerance = h.dual_feasibility_tolerance = 1.e-9
    h.ipm_optimality_tolerance = 1.e-11
    if mode=='ipm' and A.n+A.m>2500:
        h.run_crossover = 'off'
    hh.passOptions(h)
    if lb is None:
        lb,ub = np.full(A.n,-np.inf),np.full(A.n,np.inf)
    ret = hh.passModel(A.n,A.m,len(A.val),1,1,0.,q,lb,ub,l,u,A.ptr,A.idx,A.val,
                       np.zeros(A.n,dtype=np.int32))
    if ret == high.HighsStatus.kError:
        return None
    if P is not None:
        ret = hh.passHessian(A.n,len(P.val),1,P.ptr,P.idx,P.val)
        hh.setOptionValue('qp_regularization_value',1.e-12)
        if ret == high.HighsStatus.kError:
            return None
    return hh


def equilibrate(A,P,q,l,u,lb,ub):
    vmax = norm(A.val)
    if vmax==0.:
        return A,P,q,l,u,lb,ub,None
    amin = float(np.min(np.abs(A.val),initial=np.inf))
    qmax = norm(q)
    if 2.e-11 < amin and vmax < 1.e10 and 1.e-7 <= (qmax or 1.) <= 1.e7:
        return A,P,q,l,u,lb,ub,None
    na = np.repeat(np.arange(A.n,dtype=np.int32),np.diff(A.ptr))
    if P is not None:
        pc = np.repeat(np.arange(A.n,dtype=np.int32),np.diff(P.ptr))
        pv = P.val.copy()
    a = A.val.copy()
    D,E,c = np.ones(A.n),np.ones(A.m),1.
    for _ in range(5):
        r,z = np.zeros(A.m),np.zeros(A.n)
        np.maximum.at(r,A.idx,np.abs(a))
        np.maximum.at(z,na,np.abs(a))
        if P is not None:
            v = np.abs(pv)
            np.maximum.at(z,pc,v)
            np.maximum.at(z,P.idx,v)
        sd=1./np.sqrt(np.where(z>1.e-25,z,1.))
        se=1./np.sqrt(np.where(r>1.e-25,r,1.))
        sd=np.clip(sd,1.e-12/D,1.e12/D)
        se=np.clip(se,1.e-12/E,1.e12/E)
        D*=sd
        E*=se
        a*=se[A.idx]*sd[na]
        if P is not None:
            pv*=sd[P.idx]*sd[pc]
    ob=max(norm(q*D),(norm(pv)*.1 if P is not None else 0.))
    if ob and np.isfinite(ob):
        c=float(np.clip(1./ob,1.e-16,1.e16))
    return (Csc(A.m,A.n,A.ptr,A.idx,a),
            None if P is None else Csc(P.m,P.n,P.ptr,P.idx,pv*c),
            q*D*c,l*E,u*E,None if lb is None else lb/D,None if ub is None else ub/D,(D,E,c))


def dense_polish(p, x, z, sp):
    if p.n>300:
        return None
    n,m = p.n,p.m
    s = p.b-p.A.mv(x,sp)
    tight = (s <= 5.e-7*np.maximum(1.,np.abs(p.b))) | p.eq | (z>1.e-6*max(1.e-6,norm(z)))
    k = int(tight.sum())
    if n+k>390:
        return None
    I = np.cumsum(tight,dtype=np.int32)-1+n
    N = n+k
    K = np.zeros((N,N))
    P,A = p.P,p.A
    K[P.idx,np.repeat(np.arange(n),np.diff(P.ptr))] = P.val
    j = np.repeat(np.arange(n),np.diff(A.ptr))
    flag = tight[A.idx]
    K[I[A.idx[flag]],j[flag]] = K[j[flag],I[A.idx[flag]]] = A.val[flag]
    r = np.r_[-p.q,p.b[tight]]
    c = 1./np.sqrt(np.maximum(1.e-6,np.abs(K).max(axis=0)))
    K *= c[:,None]*c
    r *= c
    y = np.r_[x,z[tight]]/c
    R = K.copy()
    R[np.diag_indices(N)] += np.r_[np.full(n,2.e-12),np.full(k,-2.e-12)]
    try:
        for _ in range(2):
            y += np.linalg.solve(R,r-K@y)
    except np.linalg.LinAlgError:
        return None
    y *= c
    zz = np.zeros(m)
    zz[tight] = y[n:]
    return p.optimal(y[:n],zz,sp)


def run(p,start):
    """Return (verified answer, nearby candidate, attempted)."""
    if not p.linear or not p.n or not p.m:
        return None, None, False
    A,P = p.A,p.P
    quadratic = bool(len(P.val))
    small_qp = p.n<=480 and p.n+p.m<=6500
    diag_qp = quadratic and p.n>=80 and p.n==len(P.val) and p.m>=2
    band_qp = (quadratic and 1200<=p.n<=430000 and p.m<=1800000
               and len(P.val)<=8*p.n and len(A.val)<=7*p.m
               and not os.environ.get('CONIC_NO_BANDQP'))
    core_qp = (quadratic and 108<=p.n<=22000 and 60<=p.m<=95000
               and len(A.val)+len(P.val)<=1_450_000
               and (not small_qp or p.m>=.88*p.n or len(P.val)>12*p.n)
               and not os.environ.get('CONIC_NO_COREQP'))
    if quadratic and not small_qp and not (diag_qp or band_qp or core_qp):
        return None,None,False
    sp = extension('scipy.sparse._sparsetools','sparse','_sparsetools')
    A.canonical(sp)
    P.canonical(sp)
    diag_candidate = None
    AT = None
    if diag_qp and not os.environ.get('CONIC_NO_DIAGQP') and (len(P.val)==p.n and np.all(P.val>0) and
                    np.all(P.idx==np.arange(p.n,dtype=np.int32)) and
                    np.all(np.diff(P.ptr)==1)):
        if p.m>32 and len(A.val)<=p.m+p.n:
            ans=budget_quad(p,sp,P.val)
            if ans is not None:
                return ans,None,True
        if p.n>=100 and p.m>=12 and perf_counter()-start<.4:
            ans=few_rows_quad(p,sp,P.val,min(start+.49,perf_counter()+.15))
            if ans is not None:
                return ans,None,True
        import native
        if native.lib is not None:
            qp_limit = max(.035,min(.25,.54-(perf_counter()-start)))
            AT = A.t(sp)
            diag_candidate = native.diagonal_qp(AT,P.val,p.q,p.b,p.eq,qp_limit)
            if diag_candidate is not None:
                ans = p.optimal(*diag_candidate,sp)
                if ans is not None:
                    return ans,diag_candidate,True
    if (quadratic and p.qn>0. and p.n>90 and p.n+p.m<=42000 and
            norm(P.val)*(1.+min(p.bn,100.))<.14*p.qn and
            perf_counter()-start<.38 and not os.environ.get('CONIC_NO_QPVERTEX')):
        high = extension('scipy.optimize._highspy._core','optimize/_highspy','_core')
        from qpvertex import solve
        ans,pt=solve(p,sp,high,min(start+.65,perf_counter()+.21),diag_candidate)
        if ans is not None:
            return ans,pt,True
        if pt is not None:
            diag_candidate=pt
    if band_qp:
        import native
        left = min(.54,.77-(perf_counter()-start))
        if native.lib is not None and left>.08:
            if AT is None:
                AT = A.t(sp)
            status,x,z = native.band_qp(AT,P,p.q,p.b,p.eq,left)
            if status==1:
                ans = p.optimal(x,z,sp)
                if ans is not None:
                    return ans,(x,z),True
                if diag_candidate is None:
                    diag_candidate = x,z
            elif status==2:
                ans = p.certify(z,False,sp) or p.certify(x,True,sp)
                if ans is not None:
                    return ans,None,True
    if core_qp:
        import native
        left=min(.61 if not small_qp else .155,.805-(perf_counter()-start))
        if native.lib is not None and left>.09:
            if AT is None:
                AT=A.t(sp)
            status,x,z=native.core_qp(AT,P,p.q,p.b,p.eq,left,
                                   plain=(p.n>900 and np.count_nonzero(p.eq)<.12*p.n),
                                   warm=diag_candidate)
            if status==1:
                ans=p.optimal(x,z,sp)
                if ans is None and p.n<=300:
                    ans=dense_polish(p,x,z,sp)
                if ans is not None:
                    return ans,(x,z),True
                diag_candidate=x,z
            elif status==2:
                ans=p.certify(z,False,sp) or p.certify(x,True,sp)
                if ans is not None:
                    return ans,None,True
    if quadratic and not small_qp:
        return None,diag_candidate,False
    high = extension('scipy.optimize._highspy._core','optimize/_highspy','_core')
    H = P.lower() if quadratic else None
    row_count = np.bincount(A.idx,minlength=p.m)
    bad = np.flatnonzero((row_count==0)&((p.b<0.) | (p.eq&(p.b!=0.))))
    if len(bad):
        y = np.zeros(p.m)
        j = bad[np.argmax(np.abs(p.b[bad]))]
        y[j] = -np.sign(p.b[j])
        ans = p.certify(y,False,sp)
        if ans is not None:
            return ans,None,True
    B = None
    rows = row_count==1
    if p.m>=20 and np.count_nonzero(rows) >= max(12,p.m*.12):
        bound = Bounds(p,A,rows)
        if bound.ok and not np.any(bound.lb>bound.ub):
            B = bound
        elif bound.ok:
            j = int(np.argmax(np.where(bound.lb>bound.ub,bound.lb-bound.ub,-np.inf)))
            lo,hi = bound.li[j],bound.ui[j]
            a,b = bound.val[lo],bound.val[hi]
            mn = min(abs(a),abs(b))
            y = np.zeros(p.m)
            y[bound.rows[lo]],y[bound.rows[hi]] = -mn/a,mn/b
            ans = p.certify(y,False,sp)
            if ans is not None:
                return ans,None,True
    if B is not None:
        W = B.M
        eq,u = p.eq[B.keep],p.b[B.keep]
        lb,ub = B.lb,B.ub
    else:
        W = A
        eq,u,lb,ub = p.eq,p.b,None,None
    l = np.where(eq,u,-np.inf)
    size = p.n+W.m
    if (not quadratic and not os.environ.get('CONIC_NO_NETWORK') and B is not None and 1500<p.n<1_100_000 and W.m>2
            and np.all(eq) and len(W.val)<=2*p.n and np.max(np.diff(W.ptr),initial=0)<=2):
        import native
        if native.lib is not None:
            limit_network = min(.29,.57-(perf_counter()-start))
            if limit_network>.03:
                net = native.network(W,p.q,u,lb,ub,limit_network)
                if net is not None:
                    x,d = net
                    g = p.q+W.mv(d,sp,transpose=True)
                    g = np.where(((x==lb)&(g>=0.)) | ((x==ub)&(g<=0.)),g,0.)
                    ans = p.optimal(x,B.restore(d,g,p.m),sp)
                    if ans is not None:
                        return ans,None,True
    row_ranges = None
    if quadratic and perf_counter()-start < .47 and not os.environ.get('CONIC_NO_RANGES'):
        r = Ranges(W,eq,u,sp)
        if r.available:
            W,l,u = r.M,r.l,r.u
            size = p.n+W.m
            row_ranges = r
        elif getattr(r,'contradiction',None) is not None:
            z = r.contradiction
            if B is not None:
                z = B.restore(z,np.zeros(p.n),p.m)
            ans = p.certify(z,False,sp)
            if ans is not None:
                return ans,None,True

    scaling = None

    def reshape_solution(sol):
        x,z = np.asarray(sol.col_value),-np.asarray(sol.row_dual)
        cd = None if B is None else np.asarray(sol.col_dual)
        if scaling is not None:
            D,E,c = scaling
            x=x*D
            z=z*E/c
            if cd is not None:
                cd=cd/(c*D)
        if row_ranges is not None:
            z = row_ranges.recover(z)
        if B is not None:
            z = B.restore(z,cd,p.m)
        return x,z
    mode = ('ipm' if not quadratic and size>1250 and len(W.val)>4500
            and W.n>50 and W.m>70 else 'choose')
    if os.environ.get('CONIC_LPMETHOD'):
        mode = os.environ['CONIC_LPMETHOD']
    q1 = p.q
    if quadratic and W.m and len(W.val)<2_250_000 and perf_counter()-start<.46:
        W,H,q1,l,u,lb,ub,scaling = equilibrate(W,H,q1,l,u,lb,ub)
    write_reserve = .028 + (8*p.n+16*p.m)/1.e9
    limit = max(.035,.951-write_reserve-(perf_counter()-start))
    if quadratic:
        limit = min(limit,.12 if p.n>115 else .24)
    if size>260000:
        limit = max(limit,1.5)
    hh = create(high,W,q1,l,u,lb,ub,H,limit,mode)
    if hh is None:
        return None,None,False
    hh.run()
    st = hh.getModelStatus()
    S = high.HighsModelStatus
    point = None
    if st in (S.kOptimal,S.kTimeLimit,S.kIterationLimit,S.kUnknown):
        sol = hh.getSolution()
        if sol.value_valid:
            x,z = reshape_solution(sol)
            point = (x,z)
            ans = p.optimal(x,z,sp)
            if ans is None and st==S.kOptimal and p.n<=300 and p.n+p.m<1500:
                ans = dense_polish(p,x,z,sp)
            if ans is None and not quadratic and mode=='ipm' and st==S.kOptimal:
                left = .895-(perf_counter()-start)
                if left > .06:
                    hh.setOptionValue('time_limit',left)
                    status = hh.crossover(sol)
                    if status!=high.HighsStatus.kError:
                        sol = hh.getSolution()
                        if sol.value_valid:
                            x,z = reshape_solution(sol)
                            ans = p.optimal(x,z,sp)
                            if ans is not None:
                                return ans,(x,z),True
            return ans,point,True
    left = .93-(perf_counter()-start)
    if left <= .03:
        return None,point,True
    if st==S.kInfeasible and p.bn>0.:
        W = A.t(sp)
        b = np.zeros(p.n)
        hh = create(high,W,p.b/p.bn,b,b,np.where(p.eq,-1.,0.),np.ones(p.m),
                    None,left,'choose')
        if hh is not None:
            hh.run(); sol=hh.getSolution()
            if sol.value_valid:
                ans=p.certify(np.asarray(sol.col_value),False,sp)
                return ans,None,True
    if st in (S.kUnbounded,S.kUnboundedOrInfeasible) and p.qn>0. and not quadratic:
        b = np.zeros(p.m)
        h = np.where(p.eq,b,-np.inf)
        hh = create(high,A,p.q/p.qn,h,b,-np.ones(p.n),np.ones(p.n),None,left,'choose')
        if hh is not None:
            hh.run(); sol=hh.getSolution()
            if sol.value_valid:
                ans=p.certify(np.asarray(sol.col_value),True,sp)
                return ans,None,True
    return None,point,True


def output(path,r):
    v = (r or {'status':'unresolved'}).copy()
    meta = np.array(json.dumps({'schema_version':1,'status':v.pop('status')}))
    with open(path,'wb') as f:
        np.savez(f,metadata_json=meta,**v)
