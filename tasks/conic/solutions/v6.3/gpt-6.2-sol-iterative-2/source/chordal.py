"""Sparse PSD decomposition with original-coordinate dual completion."""
import heapq
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from model import Layout, Problem, smat, svec, upper


def cliques(d,i,j):
    g = [set() for _ in range(d)]
    for v,w in zip(i.tolist(),j.tolist()):
        g[v].add(w); g[w].add(v)
    q = [(len(k),i) for i,k in enumerate(g)]
    heapq.heapify(q)
    dead = np.zeros(d,dtype=bool)
    order,neighbor,bag = [],[],[]
    total = 0
    while q:
        k,v = heapq.heappop(q)
        if dead[v] or k != len(g[v]):
            continue
        if k>min(110,.56*d):
            return None
        dead[v] = True
        a = g[v].copy()
        order.append(v)
        neighbor.append(np.asarray(sorted(a),dtype=np.int32))
        bag.append(a|{v})
        total += (k+1)*(k+2)//2
        if total > 1_600_000:
            return None
        for w in a:
            old = len(g[w])
            g[w].discard(v)
            g[w].update(a)
            g[w].discard(w)
            if len(g[w]) != old:
                heapq.heappush(q,(len(g[w]),w))
    if len(order)!=d:
        return None
    maximal = []
    contain = [[] for i in range(d)]
    for s in sorted(bag,key=len,reverse=True):
        if any(s<=u for u in contain[min(s,key=lambda j:len(contain[j]))]):
            continue
        maximal.append(np.asarray(sorted(s),dtype=np.int32))
        for v in s:
            contain[v].append(s)
    entries = sum(len(x)*(len(x)+1)//2 for x in maximal)
    if len(maximal)<2 or entries>1.15*d*(d+1)/2:
        return None
    return maximal, np.array(order,dtype=np.int32), neighbor


class Undo:
    def __init__(self,p,which,ops):
        self.p,self.which,self.ops = p,which,ops
        self.n = p.n
        self.count = np.bincount(which,minlength=p.m)

    def complete(self,z,deadline,shift=0.):
        p = self.p
        u = np.bincount(self.which,weights=z,minlength=p.m)/np.maximum(1,self.count)
        for d,a,b,order,nn in self.ops:
            if perf_counter()>deadline:
                return None
            H = smat(u[a:b],d)
            norm = np.max(np.abs(H),initial=0.)
            if not norm:
                continue
            H /= norm
            H[np.diag_indices(d)] += shift
            known = np.zeros(d,dtype=bool)
            for k in range(d-1,-1,-1):
                v = order[k]
                N = nn[k]
                known[N] = True
                ic = order[k+1:]
                miss = ic[~known[ic]]
                known[N] = False
                if not miss.size or not N.size:
                    continue
                x = H[N,v]
                M = H[np.ix_(N,N)]
                if len(N)==1:
                    w = x/np.maximum(M[0,0],1.e-12)
                else:
                    lam,Q = eigh(M,check_finite=False)
                    inc = lam > max(2.e-12,2.e-11*lam[-1])
                    w = Q[:,inc]@((Q[:,inc].T@x)/lam[inc])
                h = w@H[np.ix_(N,miss)]
                if not np.isfinite(h).all() or np.max(np.abs(h),initial=0.)>10.:
                    return None
                H[miss,v] = H[v,miss] = h
                if k%64==0 and perf_counter()>deadline:
                    return None
            u[a:b] = svec(H)*norm
        return u

    def restore(self,r,deadline):
        p = self.p
        status = r['status']
        if status=='dual_infeasible':
            return p.dual_cert(r['x'][:self.n])
        for shift in (0.,1.5e-11,2.e-10):
            v = self.complete(r['z'],deadline,shift)
            if v is None:
                continue
            if status=='primal_infeasible':
                rr = p.primal_cert(v)
            elif status=='solved':
                rr = p.optimal(r['x'][:self.n],v,polish_slack=False)
            else:
                rr = None
            if rr is not None:
                return rr
        return None


def decompose(p,deadline):
    if not any(k=='psd' and d>=16 for k,d,_,_ in p.K.nonlinear):
        return p,None
    support = (np.bincount(p.A.indices,minlength=p.m)!=0) | (p.b!=0.)
    layout, orig, rows, ops = [],[],[],[]
    maps_r, maps_c, maps_x = [],[],[]
    offset, count = 0,0
    for k,d,a,b in p.K.blocks:
        info = None
        if k=='psd' and 16<=d<=1800 and 7*np.count_nonzero(support[a:b]) < d*d:
            i,j,_ = upper(d)
            active = support[a:b] & (i!=j)
            info = cliques(d,i[active],j[active])
        if perf_counter()>deadline:
            return p,None
        if info is None:
            layout.append((k,d))
            orig.append(np.arange(a,b,dtype=np.int32))
            rows.append(np.arange(offset,offset+b-a,dtype=np.int32))
            offset += b-a
            continue
        clique, order, neighbor = info
        owner = np.full(b-a,-1,dtype=np.int32)
        ops.append((d,a,b,order,neighbor))
        for c in clique:
            d0 = len(c)
            i,j,_ = upper(d0)
            u,v = c[i],c[j]
            ind = u+v*(v+1)//2
            ns = np.arange(offset,offset+len(ind),dtype=np.int32)
            inew = owner[ind]<0
            # An overlap auxiliary adds opposite svec entries in two cliques.
            old=owner[ind[~inew]]
            if len(old):
                cols = np.arange(count,count+len(old),dtype=np.int32)
                maps_r.extend([old,ns[~inew]])
                maps_c.extend([cols,cols])
                maps_x.extend([np.ones(len(old)),-np.ones(len(old))])
                count += len(old)
            owner[ind[inew]]=ns[inew]
            orig.append(ind+a)
            rows.append(ns[inew])
            layout.append(('psd',d0))
            offset += len(ind)
    if not ops or count>140000 or offset>850000:
        return p,None
    which = np.concatenate(orig)
    own = np.concatenate(rows)
    T = sparse.csr_matrix((np.ones(len(own)),(own,which[own])),shape=(offset,p.m))
    if count:
        ext = sparse.csc_matrix((np.concatenate(maps_x),(np.concatenate(maps_r),
                         np.concatenate(maps_c))),shape=(offset,count))
        A = sparse.hstack((T@p.A,ext),format='csc')
        P = sparse.block_diag((p.P,sparse.csc_matrix((count,count))),format='csc')
    else:
        A,P = (T@p.A).tocsc(),p.P
    return Problem(P,np.r_[p.q,np.zeros(count)],A,T@p.b,Layout(layout)),Undo(p,which,ops)
