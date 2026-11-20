"""Sparse augmented Newton systems with optional PSD Schur elimination."""
import os
import numpy as np
from scipy import sparse
from scipy.linalg import lapack
from scipy.sparse.linalg import splu

from model import smat, svec, upper, maxabs
import native


class SdpCoeff:
    def __init__(self, A, d, start, stop):
        rows = A[start:stop, :].tocsc()
        self.d, self.start, self.stop = d, start, stop
        self.cols = np.flatnonzero(np.diff(rows.indptr))
        self.part = part = rows[:, self.cols].tocsc()
        self.ind = ind = np.repeat(np.arange(len(self.cols)), np.diff(part.indptr))
        i, j, w = upper(d)
        self.i = i[part.indices].astype(np.int32)
        self.j = j[part.indices].astype(np.int32)
        self.w = w[part.indices]
        self.val = part.data
        self.wval = self.val*self.w
        self.sval = self.val/self.w
        self.simple = len(ind) == len(self.cols)
        self.source = None
        if not self.simple and len(ind)**2 <= 1_500_000:
            self.source = sparse.csr_matrix((np.ones(len(ind)), (ind, np.arange(len(ind)))),
                                            shape=(len(self.cols), len(ind)))

    def hessian(self, p):
        self.G = G = p.G
        n = len(self.cols)
        if native.lib is not None and 600 < len(self.val) and len(self.val)**2 < 13_000_000:
            return native.sdp_schur(self.d,self.part.indptr,self.i,self.j,self.wval,G)
        if self.simple or self.source is not None:
            i, j = self.i, self.j
            vals = self.val * self.w
            H = G[i[:, None], i]*G[j[:, None], j]
            H += G[i[:, None], j]*G[j[:, None], i]
            H *= .5 * vals[:, None]*vals
            if self.simple:
                return H
            H = self.source @ H
            return (self.source @ H.T).T
        self.Ri = Ri = p.Ri
        self.B = B = None
        if self.d**2 * n <= 12_000_000:
            U = np.zeros((n, self.d, self.d))
            v = self.val / self.w
            U[self.ind, self.i, self.j] = U[self.ind, self.j, self.i] = v
            i, j, w = upper(self.d)
            B = self.B = (Ri @ U @ Ri.T)[:, i, j]*w
            return B @ B.T
        H = np.empty((n, n))
        for j in range(n):
            r = slice(self.part.indptr[j], self.part.indptr[j+1])
            v = self.val[r]/self.w[r]
            u = np.zeros((self.d, self.d))
            u[self.i[r], self.j[r]] = u[self.j[r], self.i[r]] = v
            H[j, :] = self.part.T @ self.transform(svec(u))
        return (H+H.T)*.5

    def transform(self, z):
        G = self.G
        return svec(G @ smat(z, self.d) @ G)

    def augment_rhs(self, rhs, bz):
        rhs[self.cols] += self.part.T @ self.transform(bz[self.start:self.stop])

    def recover(self, x, bz):
        V = self.part @ x[self.cols] - bz[self.start:self.stop]
        return self.transform(V)

    def scaled_coeff(self, Ri):
        n, d = len(self.cols), self.d
        if native.lib is not None and n > 140 and len(self.val) < n*(1.+.85*d):
            return native.sdp_columns(d,self.part.indptr,self.i,self.j,self.sval,Ri)
        U = np.zeros((n, d, d))
        v = self.val/self.w
        U[self.ind, self.i, self.j] = U[self.ind, self.j, self.i] = v
        i, j, w = upper(d)
        return ((Ri @ U @ Ri.T)[:, i, j] * w).T


def can_qr(A, K):
    m, n = A.shape
    return (0 < n <= 480 and K.psd and not K.groups and (m+n)*n <= 2_300_000)


class QrKkt:
    """Square-root Schur solve; form the orthogonal residual without subtraction."""
    def __init__(self, A, P, K, reg):
        if not can_qr(A, K):
            raise ValueError('QR backend is not suited to this structure')
        m, n = A.shape
        self.A, self.P = A, P
        self.m, self.n = m, n
        self.rotations = {}
        self.ret = np.zeros(m, dtype=bool)
        self.eq, self.lin = K.eq, K.lpos
        self.sdp = [SdpCoeff(A, d, a, b) for k, d, a, b in K.nonlinear if k == 'psd']
        self.sqrtp = None
        if P.nnz:
            from scipy.linalg import eigh
            mat = P.toarray()
            e, V = eigh((mat+mat.T)*.5, check_finite=False)
            keep = e > 0.
            self.sqrtp = np.sqrt(e[keep])[:, None]*V[:, keep].T
        self.N = m+n+(0 if self.sqrtp is None else self.sqrtp.shape[0])
        self.reg = min(1.e-15, reg)
        self.lwork = int(lapack.dgeqrf_lwork(self.N,n)[0])
        self.qwork = None

    def factor(self, con, initial=False):
        m, n = self.m, self.n
        B = np.zeros((self.N, n), order='F')
        self.Ri = {}
        self.half = half = np.ones(m)
        if initial:
            B[:m] = self.A.toarray()
        else:
            idx = self.lin
            if len(idx):
                half[idx] = 1./np.sqrt(con.linear_h)
                B[idx] = half[idx, None]*self.A[idx, :].toarray()
            for p in self.sdp:
                Ri = self.Ri[p.start] = con.psd[p.start].Ri
                B[np.ix_(np.arange(p.start,p.stop),p.cols)] = p.scaled_coeff(Ri)
            if len(self.eq):
                half[self.eq] = max(2.e4, min(1.e10, 20.*maxabs(B)))
                B[self.eq] = half[self.eq, None]*self.A[self.eq, :].toarray()
        if self.sqrtp is not None:
            B[m:self.N-n] = self.sqrtp
        self.sc = sc = 1./np.maximum(1.e-10, np.sqrt(np.einsum('ij,ij->j', B, B)))
        B *= sc
        B[self.N-n+np.arange(n),np.arange(n)] = np.sqrt(self.reg)
        self.Q, self.t, _, info = lapack.dgeqrf(B, lwork=self.lwork, overwrite_a=1)
        if info:
            raise ValueError('QR factorization failed')
        self.R = np.asfortranarray(self.Q[:n, :])
        if self.qwork is None:
            work = lapack.dormqr(b'L', b'T', self.Q, self.t, np.zeros((self.N,1)), -1)[1]
            self.qwork = int(work[0])

    def solve_one(self, bx, bz):
        m, n = self.m, self.n
        rhs = np.zeros((self.N,1))
        rhs[:m,0] = self.half*bz
        for p in self.sdp:
            if p.start in self.Ri:
                Ri = self.Ri[p.start]
                a, b = p.start, p.stop
                rhs[a:b,0] = svec(Ri @ smat(bz[a:b],p.d) @ Ri.T)
        v, _, _ = lapack.dormqr(b'L', b'T', self.Q, self.t, rhs, self.qwork, overwrite_c=1)
        w, _ = lapack.dtrtrs(self.R, (self.sc*bx).reshape(-1,1), trans=1)
        u, _ = lapack.dtrtrs(self.R, v[:n]+w)
        v[:n] = w
        v[n:] *= -1.
        vz, _, _ = lapack.dormqr(b'L', b'N', self.Q, self.t, v, self.qwork, overwrite_c=1)
        z = self.half*vz[:m, 0]
        for p in self.sdp:
            if p.start in self.Ri:
                Ri = self.Ri[p.start]
                a, b = p.start, p.stop
                z[a:b] = svec(Ri.T @ smat(vz[a:b,0],p.d) @ Ri)
        return self.sc*u[:,0], z

    def solve(self, bx, bz, refine=2):
        x, z = self.solve_one(bx, bz)
        if refine:
            r = bx-self.P@x-self.A.T@z
            last = maxabs(r)
            target = 2.e-12*(1.+maxabs(bx))
            for i in range(refine):
                if last <= target:
                    break
                dx, dz = self.solve_one(r, np.zeros(self.m))
                xx, zz = x+dx, z+dz
                r = bx-self.P@xx-self.A.T@zz
                residue = maxabs(r)
                if residue >= last*.96:
                    break
                x, z, last = xx, zz, residue
        return x, z


class RowSchur:
    """Constant sparse Gram pattern for orthant rows with short supports."""
    def __init__(self,A,K,rows,rowcount):
        self.A = A[rows,:].tocsr()
        self.rows = rows
        self.lpos = np.searchsorted(K.lpos,rows)
        nn = rowcount[rows]
        ri,ci,vv,which = [],[],[],[]
        for d in np.unique(nn):
            ix = np.flatnonzero(nn==d).astype(np.int32)
            ind = self.A.indptr[ix,None]+np.arange(d,dtype=np.int32)
            cols = self.A.indices[ind]
            a = self.A.data[ind]
            ri.append(np.broadcast_to(cols[:,:,None],(len(ix),d,d)).reshape(-1))
            ci.append(np.broadcast_to(cols[:,None,:],(len(ix),d,d)).reshape(-1))
            vv.append((a[:,:,None]*a[:,None,:]).reshape(-1))
            which.append(np.repeat(ix,d*d))
        self.r,self.c,self.v,self.which = [np.concatenate(t) for t in (ri,ci,vv,which)]

    def add(self,X,slot,f):
        self.inv = f
        if native.lib is not None:
            native.scatter(slot,self.which,self.v,f,X)
        else:
            np.add.at(X,slot,self.v*f[self.which])


class Kkt:
    def __init__(self, A, P, K, condense=True, reg=1.e-11, rotate=True, bounds=True,
                 row_schur=True):
        m, n = A.shape
        self.m, self.n, self.reg = m, n, reg
        self.sdp, self.dense_records, self.arrow_records = [], {}, {}
        self.ret = np.ones(m, dtype=bool)
        self.reg_expand = None
        self.scipy_order = 'MMD_AT_PLUS_A'
        self.block_modes = {}
        self.bound = None
        self.general = None
        sdp_nnz = 0
        for k, d, a, b in K.nonlinear:
            if k != 'psd':
                continue
            if d >= 5 and condense:
                coeff = SdpCoeff(A, d, a, b)
                active = len(coeff.cols)
                if active**2 <= (b-a)**2 or d >= 17:
                    sdp_nnz += active**2
                    if sdp_nnz > 6_000_000:
                        raise ValueError('PSD Schur matrix is too large')
                    self.ret[a:b] = False
                    self.sdp.append(coeff)
                    self.block_modes[a] = 'condense'
                    continue
            sdp_nnz += (b-a)**2
            if sdp_nnz > 6_000_000:
                raise ValueError('Uncondensed PSD scaling is too large')
            self.block_modes[a] = 'dense'
        count = None
        if bounds and m+n >= 120 and K.lpos.size and A.nnz:
            count = np.bincount(A.indices,minlength=m)
            single = count[K.lpos] == 1
            if np.count_nonzero(single) >= max(35,.07*m):
                removed = np.zeros(m,dtype=bool)
                removed[K.lpos[single]] = True
                ins = removed[A.indices]
                t = np.flatnonzero(ins)
                br = A.indices[t]
                vc = np.searchsorted(A.indptr[1:],t,side='right').astype(np.int32)
                self.bound = (br,vc,A.data[t])
                self.bound_pos = np.searchsorted(K.lpos,br)
                self.ret[br] = False
        if (row_schur and not K.nonlinear and 420 < n+m and 0 < n <= 185000
                and K.lpos.size >= 80 and A.nnz):
            if count is None:
                count = np.bincount(A.indices,minlength=m)
            ln = count[K.lpos]
            for width in (8,4,2):
                flag = (ln>=2) & (ln<=width)
                have = ln[flag]
                if len(have)<max(65,.16*m):
                    break
                if int(have@have)<=2_300_000:
                    go = K.lpos[flag]
                    self.general = RowSchur(A,K,go,count)
                    self.ret[go] = False
                    break
        self.mr = mr = int(np.count_nonzero(self.ret))
        self.iscond = bool(self.sdp) or self.bound is not None or self.general is not None
        if self.iscond:
            Ar = A[self.ret, :].tocsc()
            rr = np.cumsum(self.ret, dtype=np.int32)-1
        else:
            Ar = A
            rr = np.arange(m, dtype=np.int32)
        self.rr = rr
        self.nplus = np.zeros(sum(len(v)*2 for (k,d),v in K.groups.items()
                                  if k == 'soc' and d > 32), dtype=bool)
        E = len(self.nplus)
        N = self.N = n + mr + E
        self.layout = K
        self.eq_diag = rr[K.eq]+n
        self.keep_lh = self.ret[K.lpos]
        self.pos_diag = rr[K.lpos[self.keep_lh]]+n
        chunk, values, loc = [], [], {}
        places = 0

        def append_vector(i, j, val=None, label=None):
            nonlocal places
            v = np.asarray(j, dtype=np.int64)*N + i
            pos = v.size
            if label is not None:
                loc[label] = slice(places, places+pos)
            places += pos
            chunk.append(v.reshape(-1))
            values.append(val)

        pcol = np.repeat(np.arange(n, dtype=np.int32), np.diff(P.indptr))
        acol = np.repeat(np.arange(n, dtype=np.int32), np.diff(Ar.indptr))
        append_vector(P.indices, pcol, P.data)
        self.rotations = {}
        if rotate and n > 0:
            rotated_rows = np.zeros(mr, dtype=bool)
            for (kind, d), starts in K.groups.items():
                if kind not in ('soc','exponential') or d > 18:
                    continue
                k = (kind, d)
                i = rr[starts, None] + np.arange(d, dtype=np.int32)
                owner = np.full(mr, -1, dtype=np.int32)
                owner[i] = np.arange(len(starts), dtype=np.int32)[:, None]
                ids = owner[Ar.indices]
                mask = ids >= 0
                bk = ids[mask]
                cols, ix = np.unique(bk.astype(np.int64)*n+acol[mask], return_inverse=True)
                if cols.size*d > max(8*int(mask.sum()), 5000) or cols.size*d*d > 8_000_000:
                    continue
                group, col = (cols//n).astype(np.int32), (cols%n).astype(np.int32)
                offset = Ar.indices[mask] - rr[starts[bk]]
                block = np.zeros((cols.size, d))
                block[ix, offset] = Ar.data[mask]
                j = i[group]+n
                append_vector(j, col[:, None], label=(k,'AR'))
                append_vector(col[:, None], j, label=(k,'ART'))
                self.rotations[k] = (group, block, i, starts)
                rotated_rows[i] = True
            keep_row = ~rotated_rows[Ar.indices]
            row = Ar.indices[keep_row]+n
            col = acol[keep_row]
            data = Ar.data[keep_row]
        else:
            row, col, data = Ar.indices+n, acol, Ar.data
        append_vector(row, col, data)
        append_vector(col, row, data)
        allrows = np.arange(N, dtype=np.int32)
        append_vector(allrows, allrows, label='diag')
        if self.general is not None:
            append_vector(self.general.r,self.general.c,label='orthant_gram')

        self.extra_info = {}
        naux = n+mr
        for (kind, d), starts in K.groups.items():
            c = rr[starts]+n
            i = c[:, None] + np.arange(d, dtype=np.int32)
            key = (kind, d)
            if key in self.rotations:
                continue
            if kind == 'soc' and d > 32:
                length = c.size
                u = naux + np.arange(length, dtype=np.int32)
                v = u + length
                self.nplus[naux-(n+mr):naux+length-(n+mr)] = True
                append_vector(i, u[:, None], label=(key, 'ur'))
                append_vector(u[:, None], i, label=(key, 'uc'))
                append_vector(i, v[:, None], label=(key, 'vr'))
                append_vector(v[:, None], i, label=(key, 'vc'))
                self.extra_info[key] = i, u, v
                naux += 2*length
            else:
                append_vector(i[:, :, None], i[:, None, :], label=key)
        for k, d, a, b in K.nonlinear:
            if k == 'psd' and self.ret[a]:
                r = rr[a:b]+n
                append_vector(r[:, None], r, label=('psd', a))
        for p in self.sdp:
            append_vector(p.cols[:, None], p.cols, label=('schur', p.start))

        flat = np.concatenate(chunk)
        keep, back = np.unique(flat, return_inverse=True)
        self.col = (keep // N).astype(np.int32)
        self.row = (keep % N).astype(np.int32)
        constant = np.zeros(len(keep))
        pos = 0
        for v, val in zip(chunk, values):
            if val is not None and val.size:
                np.add.at(constant, back[pos:pos+v.size], val)
            pos += v.size
        self.map = {k: back[v].astype(np.int32) for k,v in loc.items()}
        self.dg = dg = self.map['diag']
        self.sign = np.ones(N)
        self.sign[n:n+mr] = -1.
        self.sign[n+mr:] = np.where(self.nplus, 1., -1.)
        constant[dg[n+mr:]] = self.sign[n+mr:]
        self.constant = constant
        self.ptr = np.r_[0, np.cumsum(np.bincount(self.col, minlength=N), dtype=np.int32)].astype(np.int32)
        self.mode = ('posdef' if mr+E == 0 else 'dense') if (
            N <= 90 or (N < 1500 and len(keep) > .22*N*N)) else 'sparse'
        if self.mode == 'dense':
            self.lwork = int(lapack.dsytrf_lwork(N, lower=1)[0])
        self.perm = None
        self.lu_perm = None
        self.lu_count = 0
        self.band = None
        if self.mode=='sparse' and mr+E==0 and 180 < N <= 600000:
            dif = self.row-self.col
            bw = int(np.max(np.abs(dif),initial=0))
            if bw<=40 and N*(bw+1)<7_000_000:
                low = dif>=0
                self.band = bw,dif[low],self.col[low],np.flatnonzero(low)
                self.mode='band'
        self.try_native = (native.lib is not None and 90 < N < 300_000
                           and os.environ.get('CONIC_LDL','1') != '0')
        self.ldl = None
        self.native_active = False
        self.A = A
        self.P = P

    def factor(self, con, initial=False):
        X = self.constant.copy()
        dd = self.dg
        self.R = {}
        if self.general is not None:
            g = self.general
            f = (np.ones(len(g.rows)) if initial else
                 1./np.maximum(con.linear_h[g.lpos],1.e-13))
            g.add(X,self.map['orthant_gram'],f)
        if self.bound is not None:
            _,col,aa = self.bound
            if initial:
                self.bound_inv = np.ones(len(aa))
            else:
                h = con.linear_h[self.bound_pos]
                self.bound_inv = 1./np.maximum(h,1.e-14*aa*aa)
            self.bound_a = aa*self.bound_inv
            self.bound_d = np.bincount(col,weights=aa*self.bound_a,minlength=self.n)
            X[dd[:self.n]] += self.bound_d
        for key, (group, block, ii, starts) in self.rotations.items():
            if initial:
                v = block.reshape(-1)
            else:
                st = con.groups[key]
                R = self.R[key] = st.rotation
                X[dd[ii.reshape(-1)+self.n]] -= st.diagonal.reshape(-1)
                v = np.einsum('kij,kj->ki', R[group], block).reshape(-1)
            X[self.map[key,'AR']] = X[self.map[key,'ART']] = v
        if initial:
            for p in self.sdp:
                p.G = np.eye(p.d)
                X[self.map['schur', p.start]] += (p.part.T @ p.part).toarray().reshape(-1)
            X[dd[self.n:self.n+self.mr]] -= 1.
        else:
            X[dd[self.pos_diag]] -= con.linear_h[self.keep_lh]
            for key, st in con.groups.items():
                if key in self.rotations:
                    continue
                if key in self.extra_info:
                    i,u,v = self.extra_info[key]
                    X[dd[i.reshape(-1)]] -= (st.eta2[:, None]*np.r_[.5,np.ones(i.shape[1]-1)]).reshape(-1)
                    for d, a in [('u', st.u), ('v', st.v)]:
                        X[self.map[key, d+'r']] = a.reshape(-1)
                        X[self.map[key, d+'c']] = a.reshape(-1)
                else:
                    X[self.map[key]] -= st.h.reshape(-1)
            for start, st in con.psd.items():
                if not st.condensed:
                    X[self.map['psd', start]] -= st.h.reshape(-1)
            for p in self.sdp:
                X[self.map['schur', p.start]] += p.hessian(con.psd[p.start]).reshape(-1)
        if not np.isfinite(X).all():
            raise ValueError('Nonfinite KKT matrix')
        # Symmetric equilibration is refreshed as the cone scaling changes.
        # PSD Schur diagonal equilibration is particularly important near the
        # boundary of the cone.
        row = self.row
        col = self.col
        d = self.scale = 1./np.sqrt(np.maximum(np.maximum.reduceat(np.abs(X), self.ptr[:-1]), 1.e-10))
        V = X*d[row]*d[col]
        self.unreg = sparse.csc_matrix((V, row, self.ptr), shape=(self.N, self.N))
        V = V.copy()
        V[dd] += self.reg*self.sign
        if self.mode == 'band':
            bw,r,c,k = self.band
            M = np.zeros((bw+1,self.N),order='F')
            M[r,c] = V[k]
            C,info = lapack.dpbtrf(M,lower=1,overwrite_ab=1)
            if info:
                self.mode='sparse'
                self.factor_lu(V)
            else:
                self.direct=C
        elif self.mode == 'sparse':
            if self.ldl is not None and self.ldl.factor(V):
                self.native_active = True
                self.values = V
            else:
                self.native_active = False
                self.factor_lu(V)
                if self.try_native:
                    self.try_native = False
                    self.ldl = native.Ldl(self.N,self.ptr,self.row,self.direct.perm_c,self.sign,
                                          2_500_000,35_000_000.)
                    if not self.ldl.ptr:
                        self.ldl = None
        else:
            M = np.zeros((self.N, self.N), order='F')
            M[row, col] = V
            if self.mode == 'posdef':
                R, info = lapack.dpotrf(M, lower=1, overwrite_a=1, clean=0)
                if info == 0:
                    self.direct = (R, None)
                    return
                M[row, col] = V
            R, ix, info = lapack.dsytrf(M, lower=1, overwrite_a=1, lwork=self.lwork if
                                        hasattr(self, 'lwork') else 64*self.N)
            if info:
                raise ValueError('Singular dense Newton system')
            self.direct = (R, ix)

    def factor_lu(self,V):
        self.lu_count += 1
        if self.lu_perm is not None:
            perm,inv,ix,row,ptr=self.lu_perm
            S=sparse.csc_matrix((V[ix],row,ptr),shape=(self.N,self.N))
            self.direct=splu(S,permc_spec='NATURAL',diag_pivot_thresh=.02,
                             options={'SymmetricMode':True})
            self.ordered_lu=True
            return
        S = sparse.csc_matrix((V,self.row,self.ptr),shape=(self.N,self.N))
        self.direct = splu(S,permc_spec=self.scipy_order,diag_pivot_thresh=.02,
                           options={'SymmetricMode':True})
        self.ordered_lu=False
        if (self.lu_count==1 and self.N>350 and len(V)<5_600_000
                and os.environ.get('CONIC_NO_REORDER') is None):
            perm=self.direct.perm_c.astype(np.int32,copy=True)
            inv=np.argsort(perm).astype(np.int32)
            i,j=perm[self.row],perm[self.col]
            ix=np.argsort(j.astype(np.int64)*self.N+i).astype(np.int32)
            ptr=np.r_[0,np.cumsum(np.bincount(j,minlength=self.N))].astype(np.int32)
            self.lu_perm=perm,inv,ix,i[ix],ptr

    def _solve(self, rhs):
        if self.mode == 'band':
            return lapack.dpbtrs(self.direct,rhs,lower=1)[0]
        if self.mode == 'sparse':
            if self.native_active:
                return self.ldl.solve(rhs)
            if self.ordered_lu:
                perm,inv,*rest=self.lu_perm
                return self.direct.solve(rhs[inv])[perm]
            return self.direct.solve(rhs)
        R, ix = self.direct
        if ix is None:
            return lapack.dpotrs(R, rhs, lower=1)[0]
        return lapack.dsytrs(R, ix, rhs, lower=1)[0]

    def solve(self, bx, bz, refine=2):
        n, mr = self.n, self.mr
        v = np.zeros(self.N)
        v[:n] = bx
        v[n:n+mr] = bz[self.ret] if self.iscond else bz
        for key, R in self.R.items():
            _, _, i, starts = self.rotations[key]
            v[n+i] = np.einsum('kij,kj->ki', R, v[n+i])
        for p in self.sdp:
            p.augment_rhs(v[:n], bz)
        if self.bound is not None:
            i,j,aa = self.bound
            v[:n] += np.bincount(j,weights=self.bound_a*bz[i],minlength=n)
        if self.general is not None:
            g = self.general
            v[:n] += g.A.T@(g.inv*bz[g.rows])
        b = v*self.scale
        u = self._solve(b)
        if refine:
            tol = 1.e-13 + 5.e-13 * maxabs(b)
            r = b-self.unreg @ u
            old = maxabs(r)
            for _ in range(refine):
                if old <= tol or not np.isfinite(old):
                    break
                du = self._solve(r)
                w = u + du
                r = b-self.unreg @ w
                rn = maxabs(r)
                if not rn < .96*old:
                    break
                u, old = w, rn
            if self.native_active and old > 5.e-8*(1.+maxabs(b)):
                self.native_active = False
                self.factor_lu(self.values)
                u = self._solve(b)
                u += self._solve(b-self.unreg@u)
        v = u*self.scale
        x = v[:n]
        for key, R in self.R.items():
            _, _, i, starts = self.rotations[key]
            v[n+i] = np.einsum('kji,kj->ki', R, v[n+i])
        if not self.iscond:
            return x, v[n:n+mr]
        z = np.empty(self.m)
        z[self.ret] = v[n:n+mr]
        for p in self.sdp:
            z[p.start:p.stop] = p.recover(x, bz)
        if self.general is not None:
            g = self.general
            z[g.rows] = (g.A@x-bz[g.rows])*g.inv
        if self.bound is not None:
            i,j,aa = self.bound
            z[i] = (aa*x[j]-bz[i])*self.bound_inv
            dual_error = bx-self.P@x-self.A.T@z
            # Reconstruct bound directions without amplifying cancellation in
            # A*dx-bz when both the bound and its KKT pivot are tiny.
            rewt = self.bound_a/np.maximum(self.bound_d[j],.2)
            z[i] += rewt*dual_error[j]
        if (self.general is not None or self.sdp) and refine:
            r = bx-self.P@x-self.A.T@z
            before = maxabs(r)
            thresh = 4.e-11*(1.+maxabs(bx))
            for _ in range(2 if self.sdp else 1):
                if before<=thresh or not np.isfinite(before):
                    break
                ze = np.zeros(self.m)
                u,v = self.solve(r,ze,refine=0)
                new = r-self.P@u-self.A.T@v
                score = maxabs(new)
                if score>=.84*before:
                    break
                x,z = x+u,z+v
                r,before = new,score
        return x, z
