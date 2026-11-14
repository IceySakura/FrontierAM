#!/usr/bin/env python3
"""Species-tree inference from 50 short-gene estimated gene trees.

Method (all parameters are instance independent, learned on the supplied dev cases):
  * Each gene tree is read as an unrooted binary tree.  Internal branches are
    assigned a confidence from their branch length and approximate support.
  * Several "internode" distance matrices (ASTRID-like, confidence weighted) are
    averaged over loci, and neighbour-joining trees are built from each of them,
    from bootstrap resamples of loci and from constrained NJ runs that keep
    well-supported consensus clades.
  * Candidate bipartitions are the union of all these trees plus all gene-tree
    bipartitions.  Each candidate is scored by an ensemble vote, its gene-tree
    frequency, its distance-derived branch length and its (weighted) quartet
    support.
  * A maximum-weight compatible split system is found by dynamic programming;
    the resulting fully resolved tree on the 249 ingroup taxa is written with
    the outgroup (label 0) attached at the root.

Only Python 3 and numpy (present in the environment) are required.  If numpy is
unavailable a pure-python majority-rule fallback still writes a valid tree.
"""
import os
import sys
import math
import time
import random
from collections import Counter, defaultdict

START = time.time()
BL_SCALE = float(os.environ.get('SP_BLSCALE', '0.008'))
SUP_FLOOR = float(os.environ.get('SP_SUPFLOOR', '0.3'))
N = 250
ALL = (1 << N) - 1
LABEL_OUT = {i: str(i) for i in range(N)}
OUT = 1
ING = ALL ^ OUT

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# Newick parsing
# ---------------------------------------------------------------------------

class Tree:
    __slots__ = ('root', 'children', 'bl', 'sup', 'isleaf', 'leafbit', 'labels',
                 'v2label')

    def __init__(self):
        self.children = []
        self.bl = []
        self.sup = []
        self.isleaf = []
        self.leafbit = []
        self.labels = {}
        self.v2label = {}
        self.root = -1


def parse_newick(s):
    """Parse one Newick tree.  Returns a Tree."""
    T = Tree()
    ch, bl, sup, isl, lb, labels = (T.children, T.bl, T.sup, T.isleaf,
                                    T.leafbit, T.labels)
    i = 0
    L = len(s)

    def skip():
        nonlocal i
        while i < L and s[i].isspace():
            i += 1

    def token():
        nonlocal i
        skip()
        j = i
        while i < L and s[i] not in '(),:;':
            i += 1
        return s[j:i]

    def number():
        nonlocal i
        skip()
        j = i
        while i < L and (s[i].isdigit() or s[i] in '.eE+-'):
            i += 1
        try:
            return float(s[j:i])
        except ValueError:
            return 0.0

    def node():
        nonlocal i
        skip()
        v = len(ch)
        ch.append(None); bl.append(0.0); sup.append(-1.0)
        isl.append(False); lb.append(0)
        if s[i] == '(':
            i += 1
            kids = [node()]
            while True:
                skip()
                if i < L and s[i] == ',':
                    i += 1
                    kids.append(node())
                else:
                    break
            skip()
            assert i < L and s[i] == ')', 'bad Newick'
            i += 1
            skip()
            if i < L and s[i] not in ',):;':
                t = token()
                try:
                    sup[v] = float(t)
                except ValueError:
                    pass
            if i < L and s[i] == ':':
                i += 1
                bl[v] = number()
            ch[v] = kids
            m = 0
            for c in kids:
                m |= lb[c]
            lb[v] = m
        else:
            t = token()
            if t.startswith("'") and t.endswith("'"):
                t = t[1:-1].replace("''", "'")
            idx = int(t)
            LABEL_OUT[idx] = t
            labels[idx] = v
            T.v2label[v] = idx
            isl[v] = True
            lb[v] = 1 << idx
            if i < L and s[i] == ':':
                i += 1
                bl[v] = number()
        return v

    T.root = node()
    return T


def gene_edges(T):
    """Internal edges, oriented away from the outgroup (label 0)."""
    ch, bl, sup, isl, lb = T.children, T.bl, T.sup, T.isleaf, T.leafbit
    out = []

    def rec(v, isroot):
        if isl[v]:
            return
        for c in ch[v]:
            rec(c, False)
        if not isroot:
            m = lb[v]
            if m & OUT:
                m = ALL ^ m
            out.append((m, bl[v], sup[v]))

    rec(T.root, True)
    return out


# ---------------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------------

if HAVE_NUMPY:
    def popcnt(a):
        """Population count of a uint64 array, portable across numpy versions."""
        if hasattr(np, 'bitwise_count'):
            return np.bitwise_count(a).astype(np.int16)
        a = np.ascontiguousarray(a, dtype=np.uint64)
        v = a.view(np.uint8).reshape(a.shape + (8,))
        return np.unpackbits(v, axis=-1).sum(axis=-1).astype(np.int16)

    def words(ints):
        k = len(ints)
        W = np.zeros((k, 4), dtype=np.uint64)
        for z, m in enumerate(ints):
            W[z, 0] = m & 0xFFFFFFFFFFFFFFFF
            W[z, 1] = (m >> 64) & 0xFFFFFFFFFFFFFFFF
            W[z, 2] = (m >> 128) & 0xFFFFFFFFFFFFFFFF
            W[z, 3] = (m >> 192) & 0xFFFFFFFFFFFFFFFF
        return W

    def intersect_counts(cw, ew):
        """popcount(c_i & e_j) for all pairs -> (|c|,|e|) int16 matrix."""
        H = np.zeros((len(cw), len(ew)), dtype=np.int16)
        for k in range(4):
            H += popcnt(cw[:, k:k + 1] & ew[None, :, k])
        return H


def conf_weight(bl, sup, form='both', scale=None, p=1.0, q=0.5):
    if scale is None:
        scale = BL_SCALE
    """Confidence of a gene-tree internal branch. bl/sup are numpy arrays."""
    b = np.clip(bl / scale, 0.0, 1.0)
    s = np.where(sup < 0.0, 1.0, np.clip((sup - SUP_FLOOR) / (1.0 - SUP_FLOOR), 0.0, 1.0))
    if form == 'both':
        return (b ** p) * (s ** q)
    if form == 'sqrt':
        return np.sqrt(b * s)
    if form == 'sup':
        return s ** q
    if form == 'bl':
        return b ** p
    if form == 'one':
        return np.ones_like(bl)
    return (b ** p) * (s ** q)


# ---------------------------------------------------------------------------
# Distance matrices and neighbour joining
# ---------------------------------------------------------------------------

def prep_gene(T):
    """Per gene tree: edge cluster bitsets, branch lengths, supports, A matrix."""
    E = [e for e in gene_edges(T) if 2 <= e[0].bit_count() <= N - 2]
    U = [e[0] for e in E]
    bl = np.array([e[1] for e in E], dtype=np.float32)
    sup = np.array([e[2] for e in E], dtype=np.float32)
    A = np.zeros((len(U), N), dtype=np.float32)
    for k, m in enumerate(U):
        x = m
        while x:
            b = x & -x
            A[k, b.bit_length() - 1] = 1.0
            x ^= b
    return dict(U=U, bl=bl, sup=sup, A=A)


def gene_dist(g, w):
    """Internode distance matrix of one gene tree for edge weights w."""
    A = g['A']
    sw = np.sqrt(np.maximum(w, 0)).astype(np.float32)
    M = A * sw[:, None]
    G = M.T @ M
    s = w @ A
    D = s[:, None] + s[None, :] - 2.0 * G
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


def nj(D):
    """Neighbour joining. Returns the list of join clusters (python ints)."""
    n = D.shape[0]
    D = np.array(D, dtype=np.float64, copy=True)
    alive = list(range(n))
    clusters = [1 << i for i in range(n)]
    joins = []
    while len(alive) > 3:
        m = len(alive)
        r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), (m, m))
        dij = D[i, j]
        keep = [k for k in range(m) if k != i and k != j]
        D2 = np.zeros((m - 1, m - 1), dtype=np.float64)
        D2[:-1, :-1] = D[np.ix_(keep, keep)]
        for a, k in enumerate(keep):
            v = 0.5 * (D[i, k] + D[j, k] - dij)
            D2[a, -1] = v
            D2[-1, a] = v
        newc = clusters[alive[i]] | clusters[alive[j]]
        joins.append(newc)
        clusters.append(newc)
        alive = [alive[k] for k in keep] + [len(clusters) - 1]
        D = D2
    return joins


def nj_clusters(D):
    js = nj(D)
    out = set()
    for c in js:
        if c & OUT:
            c = ALL ^ c
        if 2 <= c.bit_count() <= N - 2:
            out.add(c)
    return out


def nj_small(D):
    """NJ on a small distance matrix; returns joins as bitmasks over indices."""
    n = D.shape[0]
    if n <= 3:
        return []
    return nj(D)


# ---------------------------------------------------------------------------
# Consensus and constrained NJ
# ---------------------------------------------------------------------------

def is_compatible(c, fam):
    for d in fam:
        if c == d:
            return False
        x = c & d
        if x != c and x != d and x != 0:
            return False
    return True


def greedy(weights, limit=N - 3):
    cand = sorted(((v, m) for m, v in weights.items()
                   if 2 <= m.bit_count() <= N - 2), reverse=True)
    chosen = []
    for v, m in cand:
        if len(chosen) >= limit:
            break
        if is_compatible(m, chosen):
            chosen.append(m)
    return chosen


def skeleton(chosen):
    fam = set(m for m in chosen if 2 <= m.bit_count() <= N - 2)
    fam |= {1 << i for i in range(1, N)}
    fam.add(ING)
    bysize = sorted(fam, key=lambda m: m.bit_count())
    subsets = {m: [] for m in fam}
    for j, c in enumerate(bysize):
        for d in bysize[:j]:
            if (d & c) == d:
                subsets[c].append(d)
    children = {}
    for c in fam:
        subs = subsets[c]
        maxi = [d for d in subs if not any((d & e) == d and e != d for e in subs)]
        cov = 0
        for d in maxi:
            cov |= d
        left = c ^ cov
        kids = list(maxi)
        while left:
            b = left & -left
            left ^= b
            kids.append(b)
        children[c] = kids
    return children


def constrained_nj(chosen, D):
    """Resolve polytomies of a consensus skeleton with NJ on supertaxa."""
    children = skeleton(chosen)
    newclusters = set(chosen)

    def resolve(c):
        kids = children[c]
        if c.bit_count() == 1:
            return
        for k in kids:
            resolve(k)
        if len(kids) <= 2:
            return
        outside = ALL ^ c
        groups = [np.fromiter((i for i in range(N) if (ch >> i) & 1),
                              dtype=np.int32, count=ch.bit_count())
                  for ch in kids]
        groups.append(np.fromiter((i for i in range(N) if (outside >> i) & 1),
                                  dtype=np.int32, count=outside.bit_count()))
        K = len(groups)
        if K <= 3:
            return
        Dm = np.zeros((K, K), dtype=np.float64)
        for a in range(K):
            ia = groups[a]
            for b in range(a + 1, K):
                ib = groups[b]
                v = float(D[np.ix_(ia, ib)].sum()) / (len(ia) * len(ib))
                Dm[a, b] = v
                Dm[b, a] = v
        joins = nj_small(Dm)
        oi = K - 1
        full = (1 << K) - 1
        for jc in joins:
            if (jc >> oi) & 1:
                jc = full ^ jc
            union = 0
            for a in range(K - 1):
                if (jc >> a) & 1:
                    union |= kids[a]
            if 2 <= union.bit_count() <= N - 2:
                newclusters.add(union)

    resolve(ING)
    return newclusters


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_newick(chosen):
    fam = set(m for m in chosen if 2 <= m.bit_count() <= N - 2)
    fam |= {1 << i for i in range(1, N)}
    fam.add(ING)
    children = skeleton(fam)
    seen = set()

    def emit(c):
        if c.bit_count() == 1:
            return LABEL_OUT[c.bit_length() - 1]
        assert c not in seen, 'cycle in tree'
        seen.add(c)
        parts = [emit(k) for k in children[c]]
        while len(parts) > 2:
            a = parts.pop()
            b = parts.pop()
            parts.append('(' + a + ',' + b + ')')
        return '(' + ','.join(parts) + ')'

    return '(0,' + emit(ING) + ');'


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_delta(D, cand):
    """OLS-like branch length estimate of each candidate split."""
    k = len(cand)
    A = np.zeros((k, N), dtype=np.float64)
    for z, m in enumerate(cand):
        x = m
        while x:
            b = x & -x
            A[z, b.bit_length() - 1] = 1.0
            x ^= b
    r = D.sum(axis=1)
    total = float(D.sum())
    X = A @ D
    withinP = np.einsum('ij,ij->i', X, A)
    sumP = A @ r
    between = sumP - withinP
    withinQ = total - withinP - 2.0 * between
    nP = A.sum(axis=1)
    nQ = N - nP
    mb = between / (nP * nQ)
    mp = withinP / np.maximum(nP * (nP - 1), 1)
    mq = withinQ / np.maximum(nQ * (nQ - 1), 1)
    return mb - 0.5 * (mp + mq)


def compute_qs(U, conf, cand, keep, ngenes=50):
    """Weighted gene-tree quartet support fraction of each candidate split."""
    U2 = [U[i] for i in keep]
    w = conf[keep].astype(np.float64)
    ew = words(U2)
    sU = np.array([m.bit_count() for m in U2], dtype=np.float64)
    sV = N - sU
    out = np.zeros(len(cand), dtype=np.float64)
    outnum = np.zeros(len(cand), dtype=np.float64)
    B = 250
    for st in range(0, len(cand), B):
        en = min(st + B, len(cand))
        cw = words(cand[st:en])
        H = intersect_counts(cw, ew)
        u = H.astype(np.float64)
        v = np.array([m.bit_count() for m in cand[st:en]],
                     dtype=np.float64)[:, None] - u
        uQ = sU[None, :] - u
        vQ = sV[None, :] - v
        c2u = u * (u - 1) * 0.5
        c2v = v * (v - 1) * 0.5
        c2uQ = uQ * (uQ - 1) * 0.5
        c2vQ = vQ * (vQ - 1) * 0.5
        num = c2u * c2vQ + c2uQ * c2v
        cross = u * uQ * v * vQ
        nm = (num * w[None, :]).sum(axis=1)
        dn = ((num + cross) * w[None, :]).sum(axis=1)
        out[st:en] = nm / np.maximum(dn, 1e-9)
        sz = np.array([m.bit_count() for m in cand[st:en]], dtype=np.float64)
        outnum[st:en] = nm / (ngenes * (sz * (sz - 1) * 0.5)
                              * ((250 - sz) * (249 - sz) * 0.5) + 1e-9)
    return out, outnum



def rooted_gene_nodes(T):
    """(L, R, confidence) for each internal node of a gene tree rooted at leaf 0."""
    ch, bl, sup, isl, lb = T.children, T.bl, T.sup, T.isleaf, T.leafbit
    n = len(ch)
    adj = [[] for _ in range(n)]
    for v in range(n):
        if ch[v]:
            for c in ch[v]:
                adj[v].append(c)
                adj[c].append(v)
    edgeinfo = {}
    for v in range(n):
        if ch[v]:
            for c in ch[v]:
                edgeinfo[(v, c)] = (bl[c], sup[c])

    def ew(a, b):
        return edgeinfo.get((a, b), edgeinfo.get((b, a), (0.0, -1.0)))

    leaf0 = T.labels[0]
    root = adj[leaf0][0]
    out = []

    def dfs(v, parent):
        kids = [u for u in adj[v] if u != parent and u != leaf0]
        if not kids:
            return lb[v]
        subs = [dfs(u, v) for u in kids]
        L = subs[0]
        R = 0
        for x in subs[1:]:
            R |= x
        b, su = ew(v, parent)
        bpart = min(1.0, max(0.0, b / BL_SCALE))
        spart = 1.0 if su < 0 else min(1.0, max(0.0, (su - SUP_FLOOR) / (1.0 - SUP_FLOOR)))
        conf = 1.0 if parent == leaf0 else bpart * spart
        out.append((L, R, conf))
        return L | R

    dfs(root, leaf0)
    return out


def compute_qexact(cand, gnodes, weighted=False, batch=150, ngenes=50):
    """Exact fraction of gene-tree quartets supporting each split."""
    G = len(gnodes)
    Ls = [g[0] for g in gnodes]
    Rs = [g[1] for g in gnodes]
    conf = np.array([g[2] if weighted else 1.0 for g in gnodes],
                    dtype=np.float64)
    sizeL = np.array([m.bit_count() for m in Ls], dtype=np.float64)
    sizeR = np.array([m.bit_count() for m in Rs], dtype=np.float64)
    sizeD = 250.0 - sizeL - sizeR
    wl, wr = words(Ls), words(Rs)
    out = np.zeros(len(cand), dtype=np.float64)
    for st in range(0, len(cand), batch):
        en = min(st + batch, len(cand))
        cl = cand[st:en]
        cw = words(cl)
        aL = intersect_counts(cw, wl).astype(np.float64)
        aR = intersect_counts(cw, wr).astype(np.float64)
        sP = np.array([m.bit_count() for m in cl],
                      dtype=np.float64)[:, None]
        sQ = 250.0 - sP
        aD = sP - aL - aR
        bL = sizeL[None, :] - aL
        bR = sizeR[None, :] - aR
        bD = sizeD[None, :] - aD
        c2 = lambda x: x * (x - 1) * 0.5
        term = (c2(aL) * (c2(bR) + bR * bD)
                + c2(bL) * (c2(aR) + aR * aD)
                + c2(aR) * (bL * bD) + c2(bR) * (aL * aD))
        out[st:en] = (term * conf[None, :]).sum(axis=1)
    sP = np.array([m.bit_count() for m in cand], dtype=np.float64)
    sQ = 250.0 - sP
    return out / (ngenes * (sP * (sP - 1) * 0.5) * (sQ * (sQ - 1) * 0.5) + 1e-9)


# ---------------------------------------------------------------------------
# Maximum weight compatible split system (DP)
# ---------------------------------------------------------------------------

def dp_select(cand, scores, topK=3000):
    k = len(cand)
    order = np.argsort(scores)[::-1][:min(topK, k)]
    sel = [cand[i] for i in order]
    Xset = set(sel)
    Xset |= {1 << i for i in range(1, N)}
    Xset.add(ING)
    X = sorted(Xset, key=lambda m: (m.bit_count(), m))
    Xw = words(X)
    sz = np.array([m.bit_count() for m in X], dtype=np.int32)
    idx = {m: i for i, m in enumerate(X)}
    wt = {m: float(scores[i]) for m, i in zip(cand, range(k))}
    for m in X:
        if m.bit_count() == 1 or m == ING:
            wt[m] = 0.0
    nX = len(X)
    S = np.full(nX, -1e30)
    best = [None] * nX
    for i in range(nX):
        if sz[i] == 1:
            S[i] = 0.0
    ordr = np.argsort(sz, kind='stable')
    for pos, ia in enumerate(ordr):
        if sz[ia] <= 1:
            continue
        Aw = Xw[ia]
        Bs = ordr[:pos]
        ok = np.all((Xw[Bs] & Aw) == Xw[Bs], axis=1)
        for ib in Bs[ok]:
            ib = int(ib)
            C = idx.get(X[ia] ^ X[ib])
            if C is None or C < ib:
                continue
            val = S[ib] + S[C] + wt.get(X[ib], -5.0) + wt.get(X[C], -5.0)
            if val > S[ia]:
                S[ia] = val
                best[ia] = (ib, C)
    iroot = idx[ING]
    if best[iroot] is None:
        return None
    chosen = []
    st = [iroot]
    while st:
        ia = st.pop()
        if sz[ia] == 1:
            continue
        b, c = best[ia]
        chosen.append(X[ia])
        st.append(b)
        st.append(c)
    return chosen


# ---------------------------------------------------------------------------
# Score-guided NNI refinement
# ---------------------------------------------------------------------------

def tree_structure(cs):
    """children / parent / sibling maps of a rooted (at outgroup) tree."""
    fam = set(m for m in cs if 2 <= m.bit_count() <= N - 2)
    fam |= {1 << i for i in range(1, N)}
    fam.add(ING)
    bysize = sorted(fam, key=lambda m: m.bit_count())
    subsets = {m: [] for m in fam}
    for j, c in enumerate(bysize):
        for d in bysize[:j]:
            if (d & c) == d:
                subsets[c].append(d)
    children = {}
    for c in fam:
        subs = subsets[c]
        maxi = [d for d in subs
                if not any((d & e) == d and e != d for e in subs)]
        cov = 0
        for d in maxi:
            cov |= d
        left = c ^ cov
        kids = list(maxi)
        while left:
            b = left & -left
            left ^= b
            kids.append(b)
        children[c] = kids
    parent, sibling = {}, {}
    for c, kids in children.items():
        for k in kids:
            parent[k] = c
    for c, kids in children.items():
        if len(kids) == 2:
            sibling[kids[0]] = kids[1]
            sibling[kids[1]] = kids[0]
    return children, parent, sibling


def refine_nni(chosen, scorer, max_iter=30, min_gain=1e-7):
    """Steepest-ascent NNI on the split-score objective."""
    cur = set(m for m in chosen if 2 <= m.bit_count() <= N - 2)
    for _ in range(max_iter):
        children, parent, sibling = tree_structure(cur)
        alts = {}
        for v, kids in children.items():
            if v == ING or v.bit_count() == 1 or len(kids) != 2:
                continue
            if v not in parent or v not in sibling:
                continue
            B, C = kids
            S = sibling[v]
            for nc in (B | S, C | S):
                sz = nc.bit_count()
                if 2 <= sz <= N - 2 and nc != v and nc not in cur:
                    alts[nc] = v
        if not alts:
            break
        allm = list(cur) + [m for m in alts if m not in cur]
        vals = scorer(allm)
        sv = {m: float(v) for m, v in zip(allm, vals)}
        best_gain, best_new, best_old = min_gain, None, None
        for nc, v in alts.items():
            g = sv.get(nc, -1e30) - sv.get(v, 0.0)
            if g > best_gain:
                best_gain, best_new, best_old = g, nc, v
        if best_new is None:
            break
        cur.discard(best_old)
        cur.add(best_new)
        if len(cur) != N - 3:
            return list(cur)
    return list(cur)


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

NJ_CONFIGS = [
    ('both', 0.50 * BL_SCALE, 1.0, 1.0),
    ('both', 1.00 * BL_SCALE, 1.0, 1.0),
    ('both', 1.00 * BL_SCALE, 1.0, 0.5),
    ('both', 1.00 * BL_SCALE, 0.5, 1.0),
    ('sup', 1.00 * BL_SCALE, 1.0, 1.0),
    ('bl', 1.00 * BL_SCALE, 1.0, 1.0),
    ('sqrt', 1.00 * BL_SCALE, 1.0, 1.0),
    ('both', 1.875 * BL_SCALE, 1.0, 2.0),
    ('both', 0.50 * BL_SCALE, 1.0, 2.0),
    ('sqrt', 1.875 * BL_SCALE, 1.0, 1.0),
]
BEST_CONFIG = ('both', 1.00 * BL_SCALE, 1.0, 0.5)


def infer(lines, deadline):
    genes = [prep_gene(parse_newick(x)) for x in lines]
    # per-gene edge confidence arrays for each NJ configuration
    confs = []
    for form, scale, p, q in NJ_CONFIGS:
        confs.append([conf_weight(g['bl'], g['sup'], form, scale, p, q)
                      for g in genes])
    best_i = NJ_CONFIGS.index(BEST_CONFIG)
    Dgenes = np.stack([gene_dist(g, confs[best_i][z])
                       for z, g in enumerate(genes)])

    # ---- ensemble of trees -------------------------------------------------
    trees = []   # (type, set of clusters)
    for ci, cfg in enumerate(NJ_CONFIGS):
        D = np.zeros((N, N), dtype=np.float64)
        for z, g in enumerate(genes):
            D += gene_dist(g, confs[ci][z])
        D /= len(genes)
        trees.append(('nj', nj_clusters(D)))
        if ci == best_i:
            Dbest = D

    if os.environ.get('SP_ALTD', '0') == '1':
        # A few structurally different locus-averaged distances add diversity.
        altDs = []
        for mode in ('bl', 'shared', 'topo', 'sqrtbl'):
            Da = np.zeros((N, N), dtype=np.float64)
            for g in genes:
                A = g['A'].astype(np.float64)
                if mode == 'bl':
                    w = np.clip(g['bl'].astype(np.float64) / BL_SCALE, 0, 1)
                elif mode == 'sqrtbl':
                    w = np.sqrt(np.clip(g['bl'].astype(np.float64) / BL_SCALE, 0, 1))
                elif mode == 'topo':
                    w = np.ones(len(g['U']), dtype=np.float64)
                else:
                    w = conf_weight(g['bl'], g['sup'], 'both', BL_SCALE, 1.0, 1.0).astype(np.float64)
                Mx = A * np.sqrt(np.maximum(w, 0))[:, None]
                G = Mx.T @ Mx
                sv = w @ A
                if mode == 'shared':
                    Da += -G
                else:
                    Da += sv[:, None] + sv[None, :] - 2.0 * G
            np.fill_diagonal(Da, 0.0)
            altDs.append(Da)
        for Da in altDs:
            trees.append(('nj', nj_clusters(Da)))

    rng = np.random.default_rng(999)
    nboot = int(os.environ.get('SP_NBOOT', '24'))
    for b in range(nboot):
        take = rng.integers(0, len(genes), len(genes))
        D = Dgenes[take].mean(axis=0)
        trees.append(('boot', nj_clusters(D)))

    # Extra diversity: half-sample trees and mildly perturbed distances.
    nextra = int(os.environ.get('SP_EXTRA', '0'))
    if nextra:
        scale = float(np.mean(Dbest[np.triu_indices(N, 1)]))
        for r in range(nextra):
            if r % 2 == 0:
                take = rng.choice(len(genes), len(genes) // 2, replace=False)
                D = Dgenes[take].mean(axis=0)
                D = D + rng.normal(0, 0.01 * scale, (N, N))
            else:
                D = Dbest + rng.normal(0, 0.03 * scale, (N, N))
            D = 0.5 * (D + D.T)
            np.fill_diagonal(D, 0.0)
            trees.append(('boot', nj_clusters(D)))

    # ---- split statistics from gene trees ---------------------------------
    cnt = Counter()
    sconf = defaultdict(float)
    sbl = defaultdict(float)
    ssup = defaultdict(float)
    chi = Counter()
    for z, g in enumerate(genes):
        w = conf_weight(g['bl'], g['sup'], 'both', BL_SCALE, 1.0, 1.0)
        for j, m in enumerate(g['U']):
            if not (2 <= m.bit_count() <= N - 2):
                continue
            cnt[m] += 1
            sconf[m] += float(w[j])
            sbl[m] += float(g['bl'][j])
            ssup[m] += (1.0 if g['sup'][j] < 0 else max(float(g['sup'][j]), 0.0))
            chi[m] += float(w[j]) > 0.5

    cons_weights = {
        'conf': {m: sconf[m] for m in cnt},
        'count': {m: float(cnt[m]) for m in cnt},
        'bl': {m: sbl[m] for m in cnt},
        'sup': {m: ssup[m] for m in cnt},
    }
    for mode, ws in cons_weights.items():
        trees.append(('cons', set(greedy(ws))))
        for th in (3.0, 8.0):
            trees.append(('cons', set(greedy({m: v for m, v in ws.items()
                                              if v >= th}))))
    for mode in ('conf', 'bl', 'sup'):
        ws = cons_weights[mode]
        for th in (1.0, 3.0, 8.0):
            ch = greedy({m: v for m, v in ws.items() if v >= th})
            trees.append(('cnj', constrained_nj(ch, Dbest)))

    # ---- candidate pool and votes -----------------------------------------
    pool = set(m for m in cnt if 2 <= m.bit_count() <= N - 2)
    vote = Counter()
    vote_type = defaultdict(Counter)
    for typ, cs in trees:
        for m in cs:
            if 2 <= m.bit_count() <= N - 2:
                pool.add(m)
                vote[m] += 1
                vote_type[typ][m] += 1
    cand = sorted(pool, key=lambda m: (m.bit_count(), m))
    k = len(cand)

    # ---- features ----------------------------------------------------------
    Uall = []
    for g in genes:
        Uall.extend(g['U'])
    confall = np.concatenate([conf_weight(g['bl'], g['sup'], 'both',
                                          BL_SCALE, 1.0, 1.0)
                              for g in genes]).astype(np.float64)
    keep = np.flatnonzero(confall > 1e-4)
    gnodes = []
    for line in lines:
        gnodes.extend(rooted_gene_nodes(parse_newick(line)))

    FEATURES = ['vote', 'voteb', 'votec', 'votecn', 'votenj', 'cnt', 'sconf',
                'chi', 'has', 'size', 'qs', 'qnum', 'q1', 'delta']
    # Fixed coefficients (tuned on the supplied development cases).  They may
    # be overridden with SP_WEIGHTS="vote=0.7,votecn=1,..." for development.
    COEF = dict(vote=0.6, voteb=-0.1, votec=-0.1, votecn=1.0, cnt=0.7,
                chi=0.7, qs=4.0, qnum=1.5, delta=1.0, has=-1.5, size=-0.4)
    try:
        ov = os.environ.get('SP_WEIGHTS')
        if ov:
            COEF = dict((k, float(v)) for k, v in
                        (kv.split('=') for kv in ov.split(',')))
        cv = os.environ.get('SP_COEFS')
        if cv:
            v = [float(x) for x in cv.split(',')]
            COEF = {'vote': v[0], 'cnt': v[1], 'sconf': v[2],
                    'qs': v[3], 'delta': v[4]}
    except Exception:
        pass
    wvec = np.array([COEF.get(f, 0.0) for f in FEATURES], dtype=np.float64)

    def features_of(clusters):
        m = len(clusters)
        F = np.zeros((m, len(FEATURES)), dtype=np.float64)
        for z, x in enumerate(clusters):
            F[z, 0] = np.log1p(vote[x])
            F[z, 1] = np.log1p(vote_type['boot'][x])
            F[z, 2] = np.log1p(vote_type['cons'][x])
            F[z, 3] = np.log1p(vote_type['cnj'][x])
            F[z, 4] = np.log1p(vote_type['nj'][x])
            F[z, 5] = np.log1p(cnt[x])
            F[z, 6] = np.log1p(sconf[x])
            F[z, 7] = np.log1p(chi[x])
            F[z, 8] = 1.0 if cnt[x] > 0 else 0.0
            F[z, 9] = np.log(x.bit_count())
        qs_v, qnum_v = compute_qs(Uall, confall, clusters, keep, len(lines))
        F[:, 10] = qs_v
        F[:, 11] = qnum_v
        if COEF.get('q1', 0.0) != 0.0:
            F[:, 12] = compute_qexact(clusters, gnodes, ngenes=len(lines))
        F[:, 13] = compute_delta(Dbest, clusters)
        return F

    cache = {}

    def scorer(clusters):
        missing = [x for x in clusters if x not in cache]
        if missing:
            F = features_of(missing)
            for x, row in zip(missing, F):
                cache[x] = row
        return np.array([cache[x] @ wvec for x in clusters], dtype=np.float64)

    score = scorer(cand)

    # ---- select a fully resolved tree -------------------------------------
    chosen = None
    if time.time() < deadline - 8:
        try:
            chosen = dp_select(cand, score, topK=int(os.environ.get('SP_TOPK', '3000')))
        except Exception as exc:
            sys.stderr.write('DP failed: %s\n' % exc)
            chosen = None
    if chosen is None or len(chosen) != N - 3:
        weights = {m: float(score[i]) for i, m in enumerate(cand)}
        chosen = greedy(weights)
    if os.environ.get('SP_NNI', '0') == '1' and time.time() < deadline - 5:
        try:
            chosen = refine_nni(chosen, scorer)
        except Exception as exc:
            sys.stderr.write('NNI refinement failed: %s\n' % exc)
    hyb = int(os.environ.get('SP_HYBRID', '0'))
    if hyb and chosen:
        try:
            sc = np.array([cache[m] @ wvec for m in chosen])
            thr = np.partition(sc, len(sc) - hyb)[len(sc) - hyb]
            cons = [m for m, v in zip(chosen, sc) if v >= thr]
            chosen = list(constrained_nj(cons, Dbest))
        except Exception as exc:
            sys.stderr.write('hybrid failed: %s\n' % exc)
    try:
        return build_newick(chosen)
    except Exception as exc:
        sys.stderr.write('tree build failed: %s\n' % exc)
        weights = {m: float(score[i]) for i, m in enumerate(cand)}
        return build_newick(greedy(weights))


# ---------------------------------------------------------------------------
# Pure python fallback (only used if numpy is unavailable)
# ---------------------------------------------------------------------------

def fallback(lines):
    weights = defaultdict(float)
    for line in lines:
        for m, bl, sup in gene_edges(parse_newick(line)):
            if 2 <= m.bit_count() <= N - 2:
                c = min(1.0, bl / 0.008) * max(0.0, (sup - 0.3) / 0.7)
                weights[m] += 0.1 + c
    return build_newick(greedy(dict(weights)))


def main():
    if len(sys.argv) < 3:
        sys.stderr.write('usage: main.py INPUT OUTPUT\n')
        return 2
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp, 'r', encoding='utf-8') as fh:
        raw = fh.read()
    # Strip Newick comments (not expected in the input, but harmless).
    if '[' in raw:
        import re as _re
        raw = _re.sub(r'\[[^\]]*\]', '', raw)
    # One tree per statement; tolerate arbitrary line breaks and blank lines.
    lines = [t.strip() + ';' for t in raw.split(';') if t.strip()]
    budget = float(os.environ.get('SP_TIME_BUDGET', '100'))
    deadline = START + budget
    try:
        if not HAVE_NUMPY:
            raise RuntimeError('numpy missing')
        text = infer(lines, deadline)
    except Exception as exc:
        sys.stderr.write('falling back: %r\n' % (exc,))
        import traceback
        traceback.print_exc(file=sys.stderr)
        text = fallback(lines)
    with open(outp, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
