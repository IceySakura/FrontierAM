"""Prototype v2: gene tree parsing with reroot at outgroup, contractions, distances, NJ/consensus."""
import re, numpy as np
from collections import defaultdict

TOKEN = re.compile(r"""\s+|\[[^\[\]]*\]|'(?:[^']|'')*'|[^\s\[\]'(),:;]+|.""", re.DOTALL)
NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

class Node:
    __slots__=('ch','lab','L','sup','name')
    def __init__(self, ch, lab, L, name=None):
        self.ch=ch; self.lab=lab; self.L=L
        self.sup=float(lab) if lab is not None and NUM.match(lab) else None
        self.name=name

def parse(text):
    toks=[m.group(0) for m in TOKEN.finditer(text)
          if not m.group(0).isspace() and not m.group(0).startswith('[')]
    pos=0
    def sub():
        nonlocal pos
        if toks[pos]=='(':
            pos+=1
            ch=[sub()]
            while toks[pos]==',':
                pos+=1; ch.append(sub())
            assert toks[pos]==')'; pos+=1
            lab=None
            if pos<len(toks) and toks[pos] not in (',',')',':',';'):
                lab=toks[pos]; pos+=1
            L=0.0
            if pos<len(toks) and toks[pos]==':':
                pos+=1; L=float(toks[pos]); pos+=1
            return Node(ch,lab,L)
        name=toks[pos]; pos+=1
        name=name[1:-1].replace("''","'") if name.startswith("'") else name
        L=0.0
        if pos<len(toks) and toks[pos]==':':
            pos+=1; L=float(toks[pos]); pos+=1
        return Node([],None,L,name)
    r=sub()
    return r

def leaves(root):
    out=[]
    def rec(n):
        if not n.ch: out.append(n.name)
        else:
            for c in n.ch: rec(c)
    rec(root); return out

def unrooted_adj(root, names):
    """adjacency of all parsed vertices; leaves are vertices 0..n-1, other nodes after.
    Edges carry (length, support) as stored on the child node (edge toward parent in Newick)."""
    n=len(names); idx={x:i for i,x in enumerate(names)}
    adj=[[] for _ in range(n)]; cnt=[n]
    def rec(node):
        if not node.ch: return idx[node.name]
        v=cnt[0]; cnt[0]+=1; adj.append([])
        for c in node.ch:
            u=rec(c)
            adj[v].append((u,c.L,c.sup)); adj[u].append((v,c.L,c.sup))
        return v
    rec(root)
    return adj

def gene_info(adj, n, names, sup_thr=None, len_thr=None, sup_default=None):
    """Build contracted component graph, return (cadj, C, splits, D_pat, D_topo).
    splits: list of (mask, L, sup) for internal edges, oriented away from leaf 0 (masks never contain bit0)."""
    N=len(adj)
    par=list(range(N))
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: par[b]=a
    # merge degree-2 non-leaf vertices (serialization root)
    for v in range(n,N):
        if len(adj[v])==2: union(v,adj[v][0][0])
    if sup_thr is not None or len_thr is not None:
        for v in range(n,N):
            for (u,L,s) in adj[v]:
                if u<=v or u<n: continue
                ss = s if s is not None else sup_default
                bad=False
                if ss is not None and sup_thr is not None and ss<sup_thr: bad=True
                if len_thr is not None and L<len_thr: bad=True
                if bad: union(u,v)
    # component ids: leaves keep 0..n-1
    cidx={find(i):i for i in range(n)}
    nxt=[n]
    def cid(r):
        if r not in cidx: cidx[r]=nxt[0]; nxt[0]+=1
        return cidx[r]
    ed={}
    for v in range(N):
        cv=cid(find(v))
        for (u,L,s) in adj[v]:
            if u<v: continue
            cu=cid(find(u))
            if cu==cv: continue
            key=(min(cv,cu),max(cv,cu))
            if key in ed:
                L0,s0=ed[key]
                ed[key]=(L0+L, max(s0 or 0,s or 0) or None)
            else: ed[key]=(L,s)
    C=nxt[0]
    cadj=[[] for _ in range(C)]
    for (a,b),(L,s) in ed.items():
        cadj[a].append((b,L,s)); cadj[b].append((a,L,s))
    # root at leaf 0
    parent=[-1]*C; pedge=[None]*C; order=[]; stack=[0]; seen=[False]*C; seen[0]=True
    while stack:
        v=stack.pop(); order.append(v)
        for e in cadj[v]:
            u=e[0]
            if not seen[u]: seen[u]=True; parent[u]=v; pedge[u]=e[1:]; stack.append(u)
    sub=[0]*C
    for i in range(n): sub[i]=1<<i
    for v in reversed(order):
        if v>=n:
            m=0
            for (u,L,s) in cadj[v]:
                if parent[u]==v: m|=sub[u]
            sub[v]=m
    full=(1<<n)-1
    splits=[]
    for v in range(n,C):
        m=sub[v]
        if m&1: m=full^m
        if 2<=m.bit_count()<=n-2: splits.append((m,pedge[v][0],pedge[v][1]))
    # distances: patristic sum of branch lengths; topo: count internal vertices on path
    Dp=np.zeros((n,n)); Dt=np.zeros((n,n))
    for src in range(n):
        d1=[0.0]*C; d2=[0.0]*C; vis=[False]*C; vis[src]=True; st=[src]
        while st:
            v=st.pop()
            for (u,L,s) in cadj[v]:
                if vis[u]: continue
                vis[u]=True
                d1[u]=d1[v]+L
                d2[u]=d2[v]+(1.0 if (u>=n and v>=n) else (0.5 if (u>=n or v>=n) else 0.0))
                st.append(u)
        for j in range(n): Dp[src,j]=d1[j]; Dt[src,j]=d2[j]
    return cadj, C, splits, (Dp+Dp.T)/2, (Dt+Dt.T)/2

def nj(D):
    n=D.shape[0]; A=D.astype(float).copy(); A=np.pad(A,((0,n),(0,n)))
    active=list(range(n)); cl={i:1<<i for i in range(n)}; res=[]; nxt=n
    full=(1<<n)-1
    while len(active)>3:
        m=len(active)
        Ai=A[np.ix_(active,active)]
        r=Ai.sum(axis=1)
        Q=(m-2)*Ai-r[:,None]-r[None,:]
        np.fill_diagonal(Q,np.inf)
        ii,jj=np.unravel_index(np.argmin(Q),Q.shape)
        i,j=active[ii],active[jj]
        mk=cl[i]|cl[j]
        res.append((mk, float(A[i,j])))
        cl[nxt]=mk
        newrow=0.5*(A[i,active]+A[j,active]-A[i,j])
        for k,v in zip(active,newrow): A[nxt,k]=v; A[k,nxt]=v
        A[nxt,nxt]=0
        del cl[i]; del cl[j]
        active=[k for k in active if k!=i and k!=j]+[nxt]
        nxt+=1
    return res

def compatible(a,b,full):
    A=full^a; B=full^b
    return not ((a&b) and (a&B) and (A&b) and (A&B))

def greedy(cand, n, limit=None):
    """cand: list of (mask,score). Returns selected masks + list of (mask,score,accepted bool)."""
    full=(1<<n)-1
    chosen=[]
    for m,s in sorted(cand,key=lambda x:-x[1]):
        if limit and len(chosen)>=limit: break
        if all(compatible(m,a,full) for a in chosen): chosen.append(m)
    return chosen

def orient(m,n):
    full=(1<<n)-1
    return full^m if (m&1) else m

def complete_tree(chosen, dmat, names, n, extra_splits=None):
    """Add NJ splits restricted to unresolved groups until binary."""
    full=(1<<n)-1
    chosen=list(chosen)
    if extra_splits is None:
        extra_splits=[m for m,_ in nj(dmat)]
    # order extra splits by NJ edge length descending
    for m in extra_splits:
        if len(chosen)>=n-3: break
        m2=orient(m,n)
        if m2 in (0,full): continue
        if any(m2==orient(c,n) for c in chosen): continue
        if all(compatible(m2,c,full) for c in chosen): chosen.append(m2)
    # fill arbitrary
    clades=[orient(c,n) for c in chosen]
    def build(L, path=()):
        if L.bit_count()==1:
            return str(names[L.bit_length()-1])
        ins=[c for c in clades if (c&L)==c and c!=L and c.bit_count()>=2 and (L&~c).bit_count()>=1]
        if ins:
            c=max(ins,key=lambda c:c.bit_count())
            return '('+build(c)+','+build(L&~c)+')'
        bits=[i for i in range(n) if (L>>i)&1]
        s=str(names[bits[-1]])
        for i in bits[-2::-1]: s='('+str(names[i])+','+s+')'
        return s
    out=[]
    def collect(L):
        if L.bit_count()==1: return
        ins=[c for c in clades if (c&L)==c and c!=L and c.bit_count()>=2 and (L&~c).bit_count()>=1]
        if ins:
            c=max(ins,key=lambda c:c.bit_count()); collect(c); collect(L&~c)
        elif L.bit_count()>=2:
            bits=[i for i in range(n) if (L>>i)&1]
            for i in bits[:-2]:
                out.append(sum(1<<k for k in bits if k!=i))
                bits2=[k for k in bits if k!=i]
            # simpler: add ladder clades
    # use build directly (arbitrary resolution ok)
    ing=full^1
    return '('+str(names[0])+','+build(ing)+');', chosen

def tree_splits_from_newick(txt,names,n):
    r=parse(txt); adj=unrooted_adj(r,names)
    _,_,sp,_,_=gene_info(adj,n,names)
    return [m for m,_,_ in sp]

def rf(ts,ps,n):
    ts={orient(m,n) for m in ts}; ps={orient(m,n) for m in ps}
    return len(ts-ps)/max(1,len(ts))
