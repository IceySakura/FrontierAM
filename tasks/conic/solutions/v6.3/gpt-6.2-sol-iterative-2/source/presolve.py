"""Reversible elimination of singleton equalities and exposed cone faces."""
import math

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from model import Layout, Problem, maxabs, smat, svec, upper
import native


def constant_certificate(kind, d, b):
    if kind == 'soc':
        n = np.linalg.norm(b[1:])
        if b[0] < n:
            return np.r_[1., -b[1:]/n] if n > 0. else np.r_[1., np.zeros(d-1)]
    elif kind == 'psd':
        e, V = eigh(smat(b,d),subset_by_index=(0,0),check_finite=False)
        if e[0] < 0.:
            return svec(np.outer(V[:,0],V[:,0]))
    else:
        x,y,z = b
        if y < 0.:
            return np.array([0.,1.,0.])
        if z < 0.:
            return np.array([0.,0.,1.])
        if y > 0. and z > 0.:
            t = math.log(z)-math.log(y)
            if x <= y*t:
                return None
        elif y > 0.:
            t = min(70.,x/y-1.)
        elif x > 0.:
            t = (math.log(z)-math.log(x)+1.) if z > 0. else 0.
        else:
            return None
        if t > 0.:
            return np.array([-1.,t-1.,np.exp(-t)])
        else:
            e = np.exp(t)
            return np.array([-e,e*(t-1.),1.])
    return None


class Undo:
    def __init__(self, p, keep, fixed, cols, pivots, a, transform, faces, lift=None):
        self.p = p
        self.keep, self.fixed = keep, fixed
        self.cols, self.pivots, self.a = cols, pivots, a
        self.transform = transform
        self.faces = faces
        self.lift = lift

    def finish_z(self, z, x=None):
        p = self.p
        for kind, idx, info in self.faces:
            if kind == 'psd':
                mag = max(1.e-5,maxabs(z[idx]))
                diag, d = info
                z[diag] = min(1.e100, 3.e4*np.sqrt(d)*mag)
            elif kind == 'soc0':
                z[idx] = np.linalg.norm(z[info])
            elif kind == 'socray':
                rg, ray = info
                mag = 3.e4*max(1.e-5,maxabs(z[rg]))
                z[rg] += mag*ray
            elif kind == 'exp-y':
                a,b,c = idx
                u = -z[a]
                if u > 0.:
                    if z[c] <= 0.:
                        z[c] = u*1.e-15
                    z[b] = max(0., -u*(1.+math.log(z[c])-math.log(u)))
            elif kind == 'exp-w':
                a,b,c = idx
                u = -z[a]
                if u > 0.:
                    target = math.log(u)-1.-z[b]/u
                    z[c] = np.exp(target+1.e-9) if target < 120. else 1.e10*max(1.,u,abs(z[b]))
                elif z[b] < 0.:
                    z[c] = 1.e10*max(1.,abs(z[b]))
        if self.cols.size:
            res = (p.A.T@z)[self.cols]
            if x is not None:
                res += (p.P@x+p.q)[self.cols]
            z[self.pivots] = -res/self.a
        return z

    def restore(self, ans):
        status = ans['status']
        if status == 'solved':
            if self.lift is None:
                x = self.fixed.copy()
                x[self.keep] = ans['x']
            else:
                x = self.fixed + self.lift@ans['x']
            z = self.finish_z(self.transform.T@ans['z'], x)
            r = self.p.optimal(x,z)
            if r is not None:
                return r
            return None
        if status == 'primal_infeasible':
            z = self.finish_z(self.transform.T@ans['z'])
            return self.p.primal_cert(z)
        if status == 'dual_infeasible':
            if self.lift is None:
                x = np.zeros(self.p.n)
                x[self.keep] = ans['x']
            else:
                x = self.lift@ans['x']
            return self.p.dual_cert(x)
        return None


def step(p):
    A, b, P, q = p.A, p.b, p.P, p.q
    m,n = A.shape
    keep, fixed = np.ones(n,dtype=bool), np.zeros(n)
    csr = A.tocsr()
    eq = p.K.eq
    el = eq[np.diff(csr.indptr)[eq]==1]
    epos = csr.indptr[el]
    cand_col = csr.indices[epos]
    # Choose the best-scaled pivot when several equations fix a variable.
    if len(cand_col):
        mx = np.zeros(n)
        np.maximum.at(mx,cand_col,np.abs(csr.data[epos]))
        best = np.abs(csr.data[epos]) == mx[cand_col]
        el, epos, cand_col = el[best],epos[best],cand_col[best]
    cols, which = np.unique(cand_col,return_index=True)
    piv = el[which]
    coeff = csr.data[epos[which]]
    lift = None
    if not cols.size and eq.size and native.lib is not None:
        chosen = native.eq_pivots(n,eq,csr,A,P)
        ix = chosen>=0
        if np.any(ix):
            piv = eq[ix]
            cj = chosen[ix]
            cols = csr.indices[cj]
            coeff = csr.data[cj]
    if not cols.size and p.K.lpos.size:
        col = np.flatnonzero((np.diff(A.indptr)==1) & (np.diff(P.indptr)==0))
        at = A.indptr[col]
        rows = A.indices[at]
        nn = np.zeros(m,dtype=bool)
        nn[p.K.lpos] = True
        use = nn[rows]
        col,at,rows=col[use],at[use],rows[use]
        cf=A.data[at]
        gain=q[col]*np.sign(cf)
        if np.any(gain>0.):
            ray=np.zeros(n)
            k=gain>0.
            ray[col[k]]=-np.sign(cf[k])
            cert=p.dual_cert(ray)
            if cert is not None:
                return p,None,cert
        take = gain<=0.
        rows,col,cf=rows[take],col[take],cf[take]
        pivot = np.zeros(m)
        np.maximum.at(pivot,rows,np.abs(cf))
        take=np.abs(cf)==pivot[rows]
        piv, at=np.unique(rows[take],return_index=True)
        cols,coeff=col[take][at],cf[take][at]
    if cols.size and np.any(np.diff(csr.indptr)[piv]>1):
        keep[cols] = False
        new_idx = np.cumsum(keep,dtype=np.int32)-1
        part = csr[piv,:].tocsr()
        v = np.repeat(np.arange(len(piv)),np.diff(part.indptr))
        ext = keep[part.indices]
        loc = np.flatnonzero(keep)
        lift = sparse.csc_matrix((np.r_[np.ones(len(loc)),-part.data[ext]/coeff[v[ext]]],
               (np.r_[loc,cols[v[ext]]],np.r_[new_idx[loc],new_idx[part.indices[ext]]])),
               shape=(n,len(loc)))
    if cols.size:
        keep[cols] = False
        fixed[cols] = b[piv]/coeff
        if not np.isfinite(fixed).all() or maxabs(fixed) > 1.e100:
            return p, None, None
        b = b-A@fixed
        b[piv] = 0.
        if lift is not None:
            A = (A@lift).tocsc()
            A.eliminate_zeros()
            q = lift.T@(P@fixed+q)
            P = (lift.T@P@lift).tocsc()
            P.eliminate_zeros()
        else:
            A = A[:,keep].tocsc()
            q = (P@fixed+q)[keep] if P.nnz else q[keep]
            P = P[keep,:][:,keep].tocsc()
    colct = np.bincount(A.indices,minlength=m)
    empty = colct == 0
    empty[piv] = True
    spec, rind, cind, rdata, faces = [], [], [], [], []
    pos = 0
    changed = bool(cols.size)

    def append(kind, d, ids, sign=1., aux=None):
        nonlocal pos
        ids = np.asarray(ids,dtype=np.int32).reshape(-1)
        end = pos+ids.size
        if end == pos:
            return
        idx = np.arange(pos,end,dtype=np.int32)
        rind.append(idx)
        cind.append(ids)
        rdata.append(np.broadcast_to(sign,(ids.size,)))
        if aux is not None:
            jj, vv = aux
            rind.append(idx[:len(jj)])
            cind.append(np.asarray(jj,dtype=np.int32))
            rdata.append(np.asarray(vv,dtype=float))
        if spec and spec[-1][0] == kind and kind in ('zero','nonnegative'):
            spec[-1] = (kind,spec[-1][1]+d)
        else:
            spec.append((kind,d))
        pos = end

    def cert(rows, values):
        z = np.zeros(m)
        z[rows] = values
        z = Undo(p,keep,fixed,cols,piv,coeff,None,faces,lift).finish_z(z)
        return p.primal_cert(z)

    def linear(kind, ids, sign=1.):
        nonlocal changed
        ids = np.asarray(ids,dtype=np.int32).reshape(-1)
        if not ids.size:
            return None
        ct = empty[ids]
        if np.any(ct):
            row = ids[ct]
            v = sign*b[row]
            invalid = (v != 0.) if kind=='zero' else (v < 0.)
            if np.any(invalid):
                bad = row[invalid][np.argmax(np.abs(v[invalid]))]
                ans = cert(bad,-np.sign(b[bad]))
                if ans is not None:
                    return ans
            ct[ct] = ~invalid
            if np.any(ct):
                changed = True
                ids = ids[~ct]
        append(kind,len(ids),ids,sign)
        return None

    for kind,d,start,stop in p.K.blocks:
        r = np.arange(start,stop,dtype=np.int32)
        if kind == 'zero' or kind == 'nonnegative' or (kind == 'psd' and d == 1):
            k = 'nonnegative' if kind == 'psd' else kind
            ans = linear(k,r)
            if ans is not None:
                return p,None,ans
            continue
        z = b[start:stop]
        e = empty[start:stop]
        if np.all(e):
            cz = constant_certificate(kind,d,z)
            if cz is None:
                changed = True
                continue
            ans = cert(r,cz)
            if ans is not None:
                return p,None,ans
        if kind == 'psd':
            i,j,_ = upper(d)
            diag = (i==j)
            bad = np.flatnonzero(diag & e & (z<0.))
            if len(bad):
                ans = cert(start+bad[np.argmin(z[bad])],1.)
                if ans is not None:
                    return p,None,ans
            remove = (e & (z==0.))[diag]
            if np.any(remove):
                changed = True
                rest = ~remove
                new = rest[i] & rest[j]
                cz = ~(new | diag)
                ans = linear('zero',r[cz])
                if ans is not None:
                    return p,None,ans
                sub = int(rest.sum())
                if sub:
                    append('psd',sub,r[new])
                faces.append(('psd',r,(r[diag][remove],d)))
                continue
            if d == 2:
                v = 2.**-.5
                append('soc',3,[start,start,start+1],[v,v,1.],
                       ([start+2,start+2],[v,-v]))
                changed = True
                continue
            support = (~e | (z!=0.)) & ~diag
            if d >= 5 and np.count_nonzero(support)<.30*d*d:
                from scipy.sparse.csgraph import connected_components
                adj = sparse.csr_matrix((np.ones(np.count_nonzero(support)),
                        (i[support],j[support])),shape=(d,d))
                num, lab = connected_components(adj,directed=False)
                if num>1:
                    same = lab[i]==lab[j]
                    for c in range(num):
                        section = r[same & (lab[i]==c)]
                        dim = int(np.count_nonzero(lab==c))
                        if dim==1:
                            ans=linear('nonnegative',section)
                            if ans is not None:
                                return p,None,ans
                        else:
                            append('psd',dim,section)
                    changed = True
                    continue
        elif kind == 'soc':
            if e[0]:
                v = np.where(e[1:],z[1:],0.)
                rnorm = np.linalg.norm(v)
                if z[0] < rnorm:
                    bdual = np.r_[1.,-v/rnorm] if rnorm else np.r_[1.,v]
                    ans = cert(r,bdual)
                    if ans is not None:
                        return p,None,ans
                elif z[0] == rnorm:
                    changed = True
                    ans = linear('zero',r[1:][~e[1:]])
                    if ans is not None:
                        return p,None,ans
                    if rnorm == 0.:
                        faces.append(('soc0',start,r[1:]))
                    else:
                        faces.append(('socray',start,(r,np.r_[1.,-v/rnorm])))
                    continue
            if d == 2:
                v = 2.**-.5
                append('nonnegative',2,[start,start],[v,v],
                       ([start+1,start+1],[v,-v]))
                changed = True
                continue
        elif kind == 'exponential':
            if e[1] and z[1] == 0.:
                changed = True
                ans = linear('nonnegative',[start],-1.) or linear('nonnegative',[start+2])
                if ans is not None:
                    return p,None,ans
                faces.append(('exp-y',(start,start+1,start+2),None))
                continue
            if e[2] and z[2] == 0.:
                changed = True
                ans = linear('nonnegative',[start],-1.) or linear('zero',[start+1])
                if ans is not None:
                    return p,None,ans
                faces.append(('exp-w',(start,start+1,start+2),None))
                continue
        append(kind,d,r)
    if not changed:
        return p,None,None
    if rind:
        T = sparse.csr_matrix((np.concatenate(rdata),(np.concatenate(rind),np.concatenate(cind))),
                              shape=(pos,m))
    else:
        T = sparse.csr_matrix((0,m))
    red = Problem(P,q,(T@A).tocsc(),T@b,Layout(spec))
    return red,Undo(p,keep,fixed,cols,piv,coeff,T,faces,lift),None


def reduce(p):
    cur, chain = p, []
    for _ in range(6):
        new, undo, ans = step(cur)
        if ans is not None:
            for undo in reversed(chain):
                ans = undo.restore(ans)
                if ans is None:
                    break
            if ans is not None:
                return p, [], ans
        if undo is None or new is cur:
            break
        cur = new
        chain.append(undo)
    return cur, chain, None
