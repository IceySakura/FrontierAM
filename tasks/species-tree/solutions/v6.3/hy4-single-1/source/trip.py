"""Triplet/quartet-based species tree search using votes from gene trees rooted at outgroup."""
import sys, time, re, random
import numpy as np
from collections import defaultdict
sys.path.insert(0,'/workspace/work')
from lib2 import (parse, leaves, unrooted_adj, gene_info, nj, compatible, orient,
                  complete_tree, tree_splits_from_newick, rf)

W = 4  # 64-bit words for 250 taxa

class GeneForest:
    """Global arrays for h(S1,S2) queries: number of gene-tree triplet votes
    (a,b)|c with a in S1, b in S2, c outside S1|S2, weighted by edge support."""
    def __init__(self, genelines, names, n, sup_thr=None, len_thr=None, sup_pow=0.0,
                 sup_default=None, weight_mode='edge'):
        self.n=n; self.ning=n-1
        allmasks=[]; parents=[]; children=[]; internal=[]; sizes=[]; weights=[]
        offset=0
        ntrees=0
        for line in genelines:
            adj=unrooted_adj(parse(line),names)
            cadj,C,sp,Dp,Dt=gene_info(adj,n,names,sup_thr,len_thr,sup_default)
            # root at leaf 0
            par=[-1]*C; pedge=[None]*C; order=[]; st=[0]; seen=[False]*C; seen[0]=True
            while st:
                v=st.pop(); order.append(v)
                for e in cadj[v]:
                    u=e[0]
                    if not seen[u]: seen[u]=True; par[u]=v; pedge[u]=e[1:]; st.append(u)
            sub=[0]*C
            for i in range(n): sub[i]=1<<i
            for v in reversed(order):
                if v>=n:
                    m=0
                    for e in cadj[v]:
                        if par[e[0]]==v: m|=sub[e[0]]
                    sub[v]=m
            # local node ids: 0..C-1 -> global offset+v
            ch=[[] for _ in range(C)]
            for v in range(C):
                if par[v]>=0: ch[par[v]].append(v)
            for v in range(C):
                m=sub[v]
                wds=[(m>>(64*k))&((1<<64)-1) for k in range(W)]
                allmasks.append(wds)
                parents.append(offset+par[v] if par[v]>=0 else -1)
                children.append(ch[v])
                isint = v>=n
                internal.append(isint)
                sizes.append(m.bit_count() if isint else 1)
                sup=pedge[v][1] if pedge[v] is not None else None
                if sup is None: sup=1.0
                if weight_mode=='edge':
                    wt=sup**sup_pow
                elif weight_mode=='none':
                    wt=1.0
                else:
                    wt=sup**sup_pow
                weights.append(wt)
            offset+=C; ntrees+=1
        self.masks=np.array(allmasks,dtype=np.uint64)
        self.parent=np.array(parents,dtype=np.int32)
        self.internal=np.array(internal,dtype=bool)
        self.intids=np.where(self.internal)[0]
        self.size=np.array(sizes,dtype=np.float64)[self.intids]
        self.wt=np.array(weights,dtype=np.float64)[self.intids]
        self.childids=np.where(self.parent>=0)[0].astype(np.int32)
        self.childpar=self.parent[self.childids]
        self.N=len(allmasks)
        self.ntrees=ntrees

    def words(self, mask):
        return np.array([(mask>>(64*k))&((1<<64)-1) for k in range(W)],dtype=np.uint64)

    def h(self, S1, S2):
        """Score for a species node with children S1,S2 (masks, disjoint, no bit0)."""
        w1=self.words(S1); w2=self.words(S2)
        x1=np.zeros(self.N,dtype=np.int32); x2=np.zeros(self.N,dtype=np.int32)
        for k in range(W):
            x1+=np.bitwise_count(self.masks[:,k]&w1[k]).astype(np.int32)
            x2+=np.bitwise_count(self.masks[:,k]&w2[k]).astype(np.int32)
        prod=x1.astype(np.int64)*x2
        cs=np.bincount(self.childpar,weights=prod[self.childids].astype(float),minlength=self.N)
        p=prod[self.intids]-cs[self.intids]
        q=self.ning-(S1.bit_count()+S2.bit_count())-self.size+x1[self.intids]+x2[self.intids]
        q=np.maximum(q,0)
        return float(np.sum(self.wt*p*q))

class SpeciesTree:
    """Rooted binary tree on ingroup taxa (bits 1..n-1), root mask = full^1."""
    __slots__=('n','forest','ch','par','mask','nodes','score','full','root')
    def __init__(self, forest, clades, extra=None):
        self.full=(1<<forest.n)-1
        self.n=forest.n; self.forest=forest
        # clades: list of masks oriented away from taxon0, compatible
        self._build(clades, extra)
    def _build(self, clades, extra):
        n=self.n; full=self.full
        cs=set(orient(c,n) for c in clades)
        if extra:
            for m in extra:
                if len(cs)>=n-3: break
                m2=orient(m,n)
                if m2 in (0,full) or m2 in cs: continue
                if all(compatible(m2,c,full) for c in cs): cs.add(m2)
        self.ch=[]; self.par=[]; self.mask=[]
        def newnode(L):
            v=len(self.ch); self.ch.append([]); self.par.append(-1); self.mask.append(L); return v
        def build(L):
            v=newnode(L)
            if L.bit_count()==1: return v
            ins=[c for c in cs if c!=L and (c&L)==c and c.bit_count()>=2 and (L&~c).bit_count()>=1]
            if ins:
                c=max(ins,key=lambda c:c.bit_count())
                a=build(c); b=build(L&~c)
            else:
                i=(L&-L).bit_length()-1
                a=newnode(1<<i); b=build(L^(1<<i))
            self.ch[v]=[a,b]; self.par[a]=v; self.par[b]=v
            return v
        self.root=build(full^1)
        self.nodes=[v for v in range(len(self.ch)) if self.ch[v]]
        self.score=self.total()
    def h(self,v):
        a,b=self.ch[v]
        return self.forest.h(self.mask[a],self.mask[b])
    def total(self):
        return sum(self.h(v) for v in self.nodes)
    def nni_pass(self, maxmoves=None, verbose=False):
        """Hill-climb by NNI swaps. Returns (#moves, score)."""
        F=self.forest; total=self.score; moves=0
        improved=True; rounds=0
        # cache h per node
        hv={v:self.h(v) for v in self.nodes}
        while improved:
            improved=False; rounds+=1
            order=list(self.nodes); random.shuffle(order)
            for v in order:
                p=self.par[v]
                if p<0 or not self.ch[v] or len(self.ch[v])!=2: continue
                sib=self.ch[p][0] if self.ch[p][1]==v else self.ch[p][1]
                for k in (0,1):
                    c1,c2=self.ch[v]; other=c2 if k==0 else c1; moved=c1 if k==0 else c2
                    newmask=(self.mask[other]|self.mask[sib])
                    # new h for v: children other, sib; new h for p: children v(new), moved
                    hnew_v=F.h(self.mask[other],self.mask[sib])
                    hnew_p=F.h(newmask,self.mask[moved])
                    delta=(hnew_v+hnew_p)-(hv[v]+hv[p])
                    if delta>1e-6:
                        # apply swap
                        self.ch[v]=[other,sib]; self.par[sib]=v
                        self.ch[p]=[v,moved]; self.par[moved]=p
                        self.mask[v]=newmask
                        hv[v]=hnew_v; hv[p]=hnew_p
                        total+=delta; moves+=1; improved=True
                        break
                if improved and maxmoves and moves>=maxmoves: break
            if maxmoves and moves>=maxmoves: break
            if verbose: print('   round',rounds,'score',total,'moves',moves,flush=True)
        self.score=total
        return moves,total
    def clades(self):
        return [self.mask[v] for v in self.nodes if self.mask[v]!=self.full^1]
    def newick(self,names):
        def rec(v):
            if not self.ch[v]: return str(names[self.mask[v].bit_length()-1])
            return '('+','.join(rec(c) for c in self.ch[v])+')'
        return '('+str(names[0])+','+rec(self.root)+');'

def load(case):
    lines=[l.strip() for l in open(f'/workspace/samples/cases/{case}/genes.tre') if l.strip()]
    names=leaves(parse(lines[0]))
    names=['0']+[x for x in names if x!='0']
    truth=parse(open(f'/workspace/samples/cases/{case}/truth.tre').read())
    tsp=gene_info(unrooted_adj(truth,names),len(names),names)[2]
    return names,lines,[m for m,_,_ in tsp]
