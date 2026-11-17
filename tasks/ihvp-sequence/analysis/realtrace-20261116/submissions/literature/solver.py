"""Research implementations of three literature-informed matrix-free solvers.

RR augmented CG, energy-POD augmented CG, scaled-spectral PCG. These are bounded
window HVP adaptations, not verbatim reproductions of author software or the
POD paper's complete three-stage algorithm. No generator/teacher access.
"""
from collections import deque
import time
import ctypes
import sys
import numpy as np
from scipy.linalg import qr, solve_triangular, eigh


# Numerical workspaces are freed by Python, but macOS malloc can retain hundreds
# of MiB of empty pages between solves. Release ONLY allocator-owned free pages
# at a step boundary; actual live/peak RSS remains monitored by the evaluator.
_libc = ctypes.CDLL(None)
_release = None
if sys.platform == 'darwin':
    _release = _libc.malloc_zone_pressure_relief
    _release.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _release.restype = ctypes.c_size_t
elif sys.platform.startswith('linux') and hasattr(_libc, 'malloc_trim'):
    _release = _libc.malloc_trim
    _release.argtypes = [ctypes.c_size_t]
    _release.restype = ctypes.c_int


def release_free_pages():
    if _release is not None:
        if sys.platform == 'darwin':_release(None,0)
        else:_release(0)


class StopSolve(Exception):
    pass


def mm(A,B,out=None):
    # GEMV columns avoid large persistent Accelerate GEMM workspaces on macOS.
    # All operations remain float64; this is also a bounded-memory CPU path.
    if out is None:out=np.empty((A.shape[0],B.shape[1]),order='F')
    target=out[:,:B.shape[1]]
    for j in range(B.shape[1]):np.matmul(A,B[:,j],out=target[:,j])
    return target


def rayleigh_ritz(S, HS, rank, which, workspace=None, output=None):
    """Two-pass MGS with cached H images, followed by Rayleigh-Ritz."""
    n,m=S.shape
    if workspace is None:Q=np.empty((n,m),order='F');Hq=np.empty_like(Q)
    else:Q,Hq=workspace
    count=0
    for j in range(m):
        v=S[:,j].copy();hv=HS[:,j].copy()
        initial=float(np.linalg.norm(v))
        if initial<=1e-200 or not np.isfinite(initial):continue
        v/=initial;hv/=initial
        if count:
            for _ in range(2):
                c=Q[:,:count].T@v
                v-=Q[:,:count]@c;hv-=Hq[:,:count]@c
        norm=float(np.linalg.norm(v))
        if norm<1e-9:continue
        Q[:,count]=v/norm;Hq[:,count]=hv/norm;count+=1
    if not count:return None,None,None
    Q,Hq=Q[:,:count],Hq[:,:count]
    T=mm(Q.T,Hq)
    vals,vecs=eigh((T+T.T)*.5,check_finite=False)
    positive=vals>max(1e-200,vals[-1]*1e-13)
    vals,vecs=vals[positive],vecs[:,positive]
    if not len(vals):return None,None,None
    take=np.arange(min(rank,len(vals))) if which=='small' else np.arange(max(0,len(vals)-rank),len(vals))
    U=mm(Q,vecs[:,take],output);lam=vals[take]
    theta=float(vals[-1] if which=='small' else vals[max(0,len(vals)-len(take)-1)])
    return U,lam,theta


def energy_pod(S, HS, rank, output=None):
    """POD minimizing total snapshot projection error in the current H metric."""
    G = mm(S.T, HS)
    vals, V = eigh((G+G.T)*.5, check_finite=False)
    keep = np.flatnonzero(vals > max(1e-200, vals[-1]*1e-12))[-rank:]
    if not len(keep):
        return None, None
    sigma = np.sqrt(vals[keep])
    U = mm(S, V[:, keep] / sigma,output)
    return np.asfortranarray(U), sigma


class Solver:
    def __init__(self, dimension, seed, options):
        self.n = dimension
        self.method = options.get('method', 'ritz')
        self.rank = int(options.get('rank', 4))
        self.window = int(options.get('window', 32))
        self.which = options.get('which', 'small')
        self.forgetting = float(options.get('forgetting', .8))
        self.cold_weight = float(options.get('cold_weight', 3.))
        self.basis = None
        self.values = None
        self.theta = None
        self.sigma = None
        self.previous = None
        self.hvp_seconds = .002
        self.extract_seconds = .04
        pool=self.window+self.rank+1
        self._snap=np.empty((dimension,self.window),order='F')
        self._hsnap=np.empty_like(self._snap)
        self._S=np.empty((dimension,pool),order='F');self._HS=np.empty_like(self._S)
        self._Q=np.empty_like(self._S);self._HQ=np.empty_like(self._S)
        self._Ub=np.empty((dimension,self.rank),order='F')
        self._HUold=np.empty_like(self._Ub)
        self._Uc=np.empty_like(self._Ub);self._HUc=np.empty_like(self._Ub)

    def solve(self, hvp, b, context):
        release_free_pages()
        begin = time.monotonic()
        changed = bool(context.get('phase_changed', context['step']==0))
        if context['step']==0:self.previous=None
        if changed:
            self.basis = self.values = self.theta = self.sigma = None
        remaining = context['steps']-context['step']
        finish = float(context['deadline'])-.15
        weight = self.cold_weight if changed else 1.
        end = begin+max(0.,finish-begin)*weight/(remaining-1+weight)
        reserve = min(.18, max(.015, 1.3*self.extract_seconds)) if remaining>1 else .003
        x = np.zeros_like(b)
        r = b.copy()
        calls = vectors = iterations = 0
        oldU = self.basis
        oldHU = None
        U = HU = None
        snapshot_count=0
        setup_start = time.monotonic()

        def product(v, setup=False):
            nonlocal calls, vectors
            now=time.monotonic()
            k=1 if v.ndim==1 else v.shape[1]
            if now + max(.001,1.3*k*self.hvp_seconds) >= end-reserve or now>=finish-.02:
                raise StopSolve()
            y = np.asarray(hvp(v),dtype=np.float64)
            k=1 if v.ndim==1 else v.shape[1]
            self.hvp_seconds=.8*self.hvp_seconds+.2*(time.monotonic()-now)/k
            calls+=1;vectors+=k
            return y

        try:
            # Exact one-dimensional quadratic minimization is safe in H-energy;
            # a Euclidean residual growth test would discard useful warm starts.
            if self.previous is not None:
                hp=product(self.previous,True)
                den=float(self.previous@hp)
                if den>0 and np.isfinite(den):
                    scale=float(self.previous@b)/den
                    x=scale*self.previous
                    r=b-scale*hp
            if self.method in ('ritz','pod') and oldU is not None:
                # Block refresh is charged by actual elapsed wall time and columns.
                oldHU=self._HUold[:,:oldU.shape[1]]
                for j in range(oldU.shape[1]):oldHU[:,j]=product(oldU[:,j],True)
                E=mm(oldU.T,oldHU)
                vals,V=eigh((E+E.T)*.5,check_finite=False)
                keep=vals>max(1e-200,vals[-1]*1e-12)
                if np.any(keep):
                    Z=V[:,keep]/np.sqrt(vals[keep])
                    U=mm(oldU,Z,self._Uc);HU=mm(oldHU,Z,self._HUc)
                    c=U.T@r
                    x+=U@c;r-=HU@c
            setup_seconds=time.monotonic()-setup_start
            setup_vectors=vectors
            def precondition(v):
                if self.method=='scaled' and oldU is not None:
                    coefficients=self.theta/self.values-1.
                    return v+oldU@(coefficients*(oldU.T@v))
                if U is not None:
                    return v-U@(HU.T@v)
                return v.copy()
            z=precondition(r);rz=float(r@z);p=z.copy()
            bnorm=float(b@b)
            while rz>0 and np.isfinite(rz) and float(r@r)>1e-24*bnorm and iterations<100000:
                if time.monotonic()+max(.001,1.3*self.hvp_seconds)>=end-reserve:
                    break
                q=product(p)
                den=float(p@q)
                if not np.isfinite(den) or den<=0:break
                alpha=rz/den
                slot=snapshot_count%self.window
                if self.method=='pod':scale=alpha
                else:
                    norm=float(np.linalg.norm(p));scale=1./norm if norm>1e-200 else 0.
                np.multiply(p,scale,out=self._snap[:,slot]);np.multiply(q,scale,out=self._hsnap[:,slot])
                snapshot_count+=1
                x+=alpha*p;r-=alpha*q
                iterations+=1
                z=precondition(r);newrz=float(r@z)
                p*=newrz/rz;p+=z;rz=newrz
            if U is not None:
                c=U.T@r;x+=U@c;r-=HU@c
            self.previous=x.copy()
            extract_start=time.monotonic()
            extracted=False
            if remaining>1 and snapshot_count and extract_start+self.extract_seconds<end:
                used=0
                if self.method!='scaled' and oldU is not None and oldHU is not None:
                    k=oldU.shape[1]
                    if self.method=='pod' and self.sigma is not None:scale=self.forgetting*self.sigma
                    else:scale=np.ones(k)
                    np.multiply(oldU,scale,out=self._S[:,:k]);np.multiply(oldHU,scale,out=self._HS[:,:k]);used+=k
                if self.method=='pod':
                    self._S[:,used]=x;np.subtract(b,r,out=self._HS[:,used]);used+=1
                k=min(snapshot_count,self.window)
                self._S[:,used:used+k]=self._snap[:,:k];self._HS[:,used:used+k]=self._hsnap[:,:k];used+=k
                S,HS=self._S[:,:used],self._HS[:,:used]
                if self.method=='pod':
                    candidate,sigma=energy_pod(S,HS,self.rank,self._Ub)
                    if candidate is not None:self.basis,self.sigma=candidate,sigma;extracted=True
                else:
                    candidate,vals,theta=rayleigh_ritz(S,HS,self.rank,self.which,(self._Q,self._HQ),self._Ub)
                    if candidate is not None:
                        self.basis=candidate;self.values=np.clip(vals,context['mu'],context['L'])
                        self.theta=float(np.clip(theta,context['mu'],context['L']));extracted=True
                self.extract_seconds=.5*self.extract_seconds+.5*(time.monotonic()-extract_start)
            return x,{'method':self.method,'iterations':iterations,'hvp_vectors':vectors,
                      'setup_vectors':setup_vectors,'setup_seconds':setup_seconds,
                      'basis_seconds':time.monotonic()-extract_start,'basis_updated':extracted,
                      'basis_rank':0 if self.basis is None else self.basis.shape[1]}
        except Exception as error:
            if isinstance(error,StopSolve) or type(error).__name__=='BudgetExceeded':
                if np.isfinite(x).all():self.previous=x.copy();return x,{'early_budget':True,'iterations':iterations,'hvp_vectors':vectors}
            raise


def make_solver(dimension,seed,options):
    return Solver(dimension,seed,options or {})
