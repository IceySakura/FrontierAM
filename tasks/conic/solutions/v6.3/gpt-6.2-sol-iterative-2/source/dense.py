"""Reduced dense Newton solves for medium-size systems with high fill."""
import numpy as np
from scipy import sparse
from scipy.linalg import lapack

from model import maxabs
import native


def suitable(A,K):
    m,n = A.shape
    return (170 <= n <= 1250 and 700 < n+m < 12800 and not K.psd and
            A.nnz < 150000 and A.nnz > 5.7*m and len(K.eq)<.55*n and
            len(K.nonlinear) < 2200 and all(d<=18 for (k,d) in K.groups))


class DenseKkt:
    def __init__(self,A,P,K,reg):
        self.n,self.m = A.shape[1],A.shape[0]
        self.eq = eq = K.eq
        self.F = A[eq,:].tocsr()
        self.P = P
        self.N = N = self.n+len(eq)
        mask = np.ones(self.m,dtype=bool)
        mask[eq] = False
        self.ret = mask
        self.mi = mi = self.m-len(eq)
        rr = np.cumsum(mask,dtype=np.int32)-1
        self.lpos = rr[K.lpos]
        self.NN = int(self.n)
        a = A.tocsr()
        aind, aval = a.indices, a.data
        bits = np.repeat(np.arange(self.m,dtype=np.int32),np.diff(a.indptr))
        include = np.zeros(self.m,dtype=bool)
        include[K.lpos] = True
        keep = include[bits]
        chunks = [rr[bits[keep]].astype(np.int64)*self.n+aind[keep]]
        vs = [aval[keep]]
        self.rotations = {}
        keys = {}
        limit = float(np.dot(np.diff(a.indptr)[K.lpos].astype(float),
                             np.diff(a.indptr)[K.lpos].astype(float)))
        for (k,d), starts in K.groups.items():
            r = starts[:,None]+np.arange(d,dtype=np.int32)
            idx = rr[r]
            own = np.full(self.m,-1,dtype=np.int32)
            own[r] = np.arange(len(starts),dtype=np.int32)[:,None]
            groups = own[bits]
            kq = groups>=0
            g = groups[kq]
            flat,inv = np.unique(g.astype(np.int64)*self.n+aind[kq],return_inverse=True)
            locs = (flat//self.n).astype(np.int32)
            cols = (flat%self.n).astype(np.int32)
            ct = np.bincount(locs).astype(float)
            limit += d*float(ct@ct)
            if limit > 23_000_000:
                raise ValueError('dense reduction would have too many contributions')
            xy = np.zeros((len(cols),d))
            xy[inv,bits[kq]-starts[g]] = aval[kq]
            ix = idx[locs].astype(np.int64)*self.n+cols[:,None]
            chunks.append(ix.reshape(-1))
            vs.append(np.zeros(ix.size))
            keys[k,d] = ix.reshape(-1)
            self.rotations[k,d] = (r,idx,locs,xy)
        field = np.concatenate(chunks)
        v = np.concatenate(vs)
        order = np.argsort(field,kind='stable')
        field = field[order]
        ptr = np.r_[0,np.cumsum(np.bincount(field//self.n,minlength=mi))].astype(np.int32)
        cols = (field%self.n).astype(np.int32)
        self.base_values = v[order]
        self.B = sparse.csr_matrix((self.base_values.copy(),cols,ptr),shape=(mi,self.n))
        self.pos = {k:np.searchsorted(field,v) for k,v in keys.items()}
        self.base = np.zeros((N,N),order='F')
        self.base[:self.n,:self.n] = P.toarray()
        self.base[self.n:,:self.n] = self.F.toarray()
        self.base[:self.n,self.n:] = self.base[self.n:,:self.n].T
        self.reg = reg
        self.ix = np.diag_indices(N)
        self.sign = np.r_[np.ones(self.n),-np.ones(len(eq))]
        self.lwork = int(lapack.dsytrf_lwork(N,lower=1)[0])
        self.eq_norm = np.max(np.abs(self.base[self.n:,:self.n]),axis=0,initial=0.)

    def factor(self,con,initial=False):
        n = self.n
        M = self.base.copy(order='F')
        self.R = {}
        self.eq_h = 1. if initial else 0.
        if initial:
            j = np.arange(n,self.N)
            M[j,j] = -1.
        H = np.ones(self.mi)
        self.B.data[:] = self.base_values
        if not initial:
            H[self.lpos] = con.linear_h
        for key,(r,idx,g,coeff) in self.rotations.items():
            if initial:
                v = coeff
            else:
                st = con.groups[key]
                R = self.R[key] = st.rotation
                H[idx] = st.diagonal
                v = np.einsum('kij,kj->ki',R[g],coeff)
            self.B.data[self.pos[key]] = v.reshape(-1)
        self.H=H=np.maximum(H,3.e-13)
        self.hi=hi=1./H
        if native.lib is not None:
            native.gram_into(self.B,hi,M)
        else:
            T = self.B.multiply(hi[:,None])
            M[:n,:n] += (self.B.T@T).toarray()
        D = 1./np.sqrt(np.maximum(1.e-10,np.maximum(M.diagonal()[:n],self.eq_norm)))
        if len(self.eq):
            norms = np.max(np.abs(self.base[n:,:n]*D),axis=1,initial=0.)
            E = 1./np.maximum(1.e-4,norms)
            if initial:
                E = np.minimum(E,1.)
            sc = np.r_[D,E]
        else:
            sc = D
        self.scale = sc
        M *= sc[:,None]
        M *= sc
        M[self.ix] += self.reg*self.sign
        R,S,info = lapack.dsytrf(M,lower=1,lwork=self.lwork,overwrite_a=1)
        if info:
            raise ValueError('dense condensed factorization failed')
        self.L,self.piv = R,S

    def solve_core(self,bx,be,bi):
        hi = self.hi
        r = np.r_[bx + self.B.T@(hi*bi),be]*self.scale
        x = lapack.dsytrs(self.L,self.piv,r,lower=1,overwrite_b=1)[0]*self.scale
        return x[:self.n],x[self.n:],(self.B@x[:self.n]-bi)*hi

    def solve(self,bx,bz,refine=2):
        be = bz[self.eq]
        bi = bz[self.ret]
        for key,R in self.R.items():
            _,idx,_,_ = self.rotations[key]
            bi[idx] = np.einsum('kij,kj->ki',R,bi[idx])
        x,ze,zi = self.solve_core(bx,be,bi)
        B,F,P,H = self.B,self.F,self.P,self.H
        size = max(maxabs(bx),maxabs(be),maxabs(bi))
        last = np.inf
        saved = x,ze,zi
        for i in range(refine+1):
            rx = bx-(P@x+F.T@ze+B.T@zi)
            re = be-F@x+self.eq_h*ze
            ri = bi-B@x+H*zi
            val = max(maxabs(rx),maxabs(re),maxabs(ri))
            if not val<last:
                x,ze,zi = saved
                break
            saved = x,ze,zi
            last = val
            if i==refine or val<=6.e-13*(1.+size):
                break
            c,e,v = self.solve_core(rx,re,ri)
            x,ze,zi = x+c,ze+e,zi+v
        for key,R in self.R.items():
            _,idx,_,_ = self.rotations[key]
            zi[idx] = np.einsum('kji,kj->ki',R,zi[idx])
        z = np.empty(self.m)
        z[self.eq], z[self.ret] = ze,zi
        return x,z
