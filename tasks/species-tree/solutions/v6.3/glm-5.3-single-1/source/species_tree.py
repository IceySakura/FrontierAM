#!/usr/bin/env python3
"""Species-tree inference from 50 estimated gene trees (250 taxa).

Method: maximise the exact quartet-agreement (ASTRAL-style) objective
    F(T) = sum over gene trees g of #{quartets q : T|q = g|q}
using neighbour-joining starts followed by NNI hill climbing with exact
incremental score deltas, optionally re-weighted by gene-tree support values.
"""
import sys
import os
import re
import time

import numpy as np

START = time.time()
TIME_LIMIT = float(os.environ.get('TIME_LIMIT', '110.0'))

_TOKEN = re.compile(r"\s+|\[[^\[\]]*\]|'(?:[^']|'')*'|[^\s\[\]'(),:;]+|.", re.DOTALL)


def tokenize(text):
    return [m[0] for m in _TOKEN.finditer(text)
            if not m[0][0].isspace() and not (m[0].startswith("[") and m[0].endswith("]"))]


def parse_newick(text):
    """-> (kids, leaves, top).  kids[i] = list of (child, length, support);
    child < 0 encodes leaf ~index into leaves."""
    toks = tokenize(text)
    n = len(toks)
    i = 0
    kids = []
    leaves = []
    leafidx = {}
    stack = [[]]
    while i < n:
        t = toks[i]
        if t == '(':
            stack.append([])
            i += 1
        elif t == ',':
            i += 1
        elif t == ')':
            i += 1
            lab = None
            ln = 0.0
            sup = None
            if i < n and toks[i] not in (')', '(', ',', ';', ':'):
                lab = toks[i]
                i += 1
            if i < n and toks[i] == ':':
                i += 1
                ln = float(toks[i])
                i += 1
            if lab is not None:
                try:
                    sup = float(lab)
                except ValueError:
                    sup = None
            nid = len(kids)
            kids.append(stack.pop())
            stack[-1].append((nid, ln, sup))
        elif t == ';':
            i += 1
        else:
            lab = t
            i += 1
            ln = 0.0
            if i < n and toks[i] == ':':
                i += 1
                ln = float(toks[i])
                i += 1
            if lab not in leafidx:
                leafidx[lab] = len(leaves)
                leaves.append(lab)
            stack[-1].append((~leafidx[lab], ln, None))
    return kids, leaves, stack[0]


class GTree:
    """Unrooted gene tree: leaves 0..n-1, internal n..N-1, edge metadata."""

    def __init__(self, text, taxon_index=None):
        kids, leaves, top = parse_newick(text)
        self.n = n = len(leaves)
        if taxon_index is None:
            self.taxa = list(leaves)
            idx = {t: i for i, t in enumerate(leaves)}
        else:
            self.taxa = list(taxon_index)
            idx = taxon_index
        N = n + len(kids)
        self.N = N
        self.eto = []
        self.elen = []
        self.esup = []
        self.adj = [[] for _ in range(N)]
        nid = {}
        for i in range(len(kids)):
            nid[i] = n + i

        def add(u):
            for (c, ln, sup) in kids[u - n]:
                v = idx[leaves[~c]] if c < 0 else nid[c]
                self._edge(u, v, ln, sup)
            for (c, ln, sup) in kids[u - n]:
                if c >= 0:
                    add(nid[c])

        if len(top) == 1:
            add(nid[top[0][0]])
        else:
            self.adj.append([])
            self.N = N + 1
            for (c, ln, sup) in top:
                v = idx[leaves[~c]] if c < 0 else nid[c]
                self._edge(self.N - 1, v, ln, sup)
        self._suppress()
        self._cache = {}

    def _edge(self, u, v, ln, sup):
        e = len(self.eto)
        self.eto.append((u, v))
        self.elen.append(ln if ln is not None else 0.0)
        self.esup.append(sup)
        self.adj[u].append((v, e))
        self.adj[v].append((u, e))

    def _suppress(self):
        changed = True
        while changed:
            changed = False
            for u in range(self.n, self.N):
                nb = self.adj[u]
                if len(nb) == 2:
                    (a, ea), (b, eb) = nb
                    if a == b:
                        continue
                    ln = self.elen[ea] + self.elen[eb]
                    sup = self.esup[ea] if self.esup[ea] is not None else self.esup[eb]
                    self.adj[u] = []
                    self.adj[a] = [x for x in self.adj[a] if x[0] != u]
                    self.adj[b] = [x for x in self.adj[b] if x[0] != u]
                    self._edge(a, b, ln, sup)
                    changed = True

    def dfs_from(self, root):
        par = [-1] * self.N
        epar = [-1] * self.N
        order = []
        seen = [False] * self.N
        seen[root] = True
        stack = [root]
        while stack:
            u = stack.pop()
            order.append(u)
            for v, e in self.adj[u]:
                if not seen[v]:
                    seen[v] = True
                    par[v] = u
                    epar[v] = e
                    stack.append(v)
        return par, order, epar


def masks_to_array(masks):
    K = len(masks)
    cl = np.zeros((K, 4), dtype=np.uint64)
    for i, m in enumerate(masks):
        cl[i, 0] = m & 0xFFFFFFFFFFFFFFFF
        cl[i, 1] = (m >> 64) & 0xFFFFFFFFFFFFFFFF
        cl[i, 2] = (m >> 128) & 0xFFFFFFFFFFFFFFFF
        cl[i, 3] = (m >> 192) & 0xFFFFFFFFFFFFFFFF
    return cl


class Scorer:
    """Exact quartet-agreement machinery.

    qs(A, B) = sum_g sum_{a,b in A, c,d in B} w * [g displays ab|cd]
             = sum over gene-tree internal nodes v of
               w_v * cA(x)*cA(y) * C(|B \\ clade(v)|, 2)
    """

    def __init__(self, trees, n, weight='none', alpha=1.0, root_leaf=0):
        self.n = n
        masks, popc, wts, c1, c2 = [], [], [], [], []
        for t in trees:
            par, order, epar = t.dfs_from(root_leaf)
            clade = [0] * t.N
            for u in reversed(order):
                if u < n:
                    clade[u] = 1 << u
                else:
                    m = 0
                    for v, e in t.adj[u]:
                        if v != par[u]:
                            m |= clade[v]
                    clade[u] = m
            nodeidx = {}
            for u in range(n, t.N):
                if not t.adj[u]:
                    continue
                kids = [v for v, e in t.adj[u] if v != par[u]]
                if len(kids) != 2:
                    continue
                nodeidx[u] = len(masks)
                masks.append(clade[u])
                popc.append(bin(clade[u]).count('1'))
                s = t.esup[epar[u]]
                b = t.elen[epar[u]]
                if weight == 'support':
                    w = 0.5 if s is None else s
                elif weight == 'sqrt':
                    w = np.sqrt(0.5 if s is None else s)
                elif weight == 'square':
                    w = (0.5 if s is None else s) ** 2
                elif weight == 'supx':
                    w = (0.5 if s is None else s) * (1.0 - np.exp(-alpha * (b or 0.0)))
                else:
                    w = 1.0
                wts.append(w)
                c1.append(kids[0])
                c2.append(kids[1])
            for u in range(n, t.N):
                if u not in nodeidx:
                    continue
                i = nodeidx[u]
                kids = [v for v, e in t.adj[u] if v != par[u]]
                c1[i] = nodeidx.get(kids[0], -1 - kids[0])
                c2[i] = nodeidx.get(kids[1], -1 - kids[1])
        M = len(masks)
        self.M = M
        self.cl = masks_to_array(masks)
        self.vpop = np.array(popc, dtype=np.int64)
        self.vw = np.array(wts, dtype=np.float64)
        r1 = np.array(c1, dtype=np.int64)
        r2 = np.array(c2, dtype=np.int64)
        self.leaf1 = r1 < 0
        self.leaf2 = r2 < 0
        self.t1 = np.where(r1 < 0, -1 - r1, 0)
        self.t2 = np.where(r2 < 0, -1 - r2, 0)
        self.r1 = np.where(r1 >= 0, r1, 0)
        self.r2 = np.where(r2 >= 0, r2, 0)
        self.comb2 = np.zeros(n + 2, dtype=np.float64)
        for k in range(n + 2):
            self.comb2[k] = k * (k - 1) / 2.0
        self.root_leaf = root_leaf
        self._lbuf = np.zeros(n, dtype=np.int64)

    def counts(self, A):
        """popcount(clade & A) for every gene-tree internal node."""
        A4 = np.array([A & 0xFFFFFFFFFFFFFFFF, (A >> 64) & 0xFFFFFFFFFFFFFFFF,
                       (A >> 128) & 0xFFFFFFFFFFFFFFFF,
                       (A >> 192) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        return np.bitwise_count(self.cl & A4).sum(axis=1).astype(np.int64)

    def leafmask(self, A):
        A4 = np.array([A & 0xFFFFFFFFFFFFFFFF, (A >> 64) & 0xFFFFFFFFFFFFFFFF,
                       (A >> 128) & 0xFFFFFFFFFFFFFFFF,
                       (A >> 192) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        return np.unpackbits(A4.view(np.uint8), bitorder='little')[:self.n].astype(np.int64)

    def _term(self, cntA, cntB, popB):
        """sum_v w * cA(x) cA(y) C(popB - cB(v), 2) using cached child arrays."""
        lv1 = np.where(self.leaf1, cntA[self.M + self.t1], cntA[self.r1])
        lv2 = np.where(self.leaf2, cntA[self.M + self.t2], cntA[self.r2])
        cb = cntB
        k = popB - cb
        np.clip(k, 0, None, out=k)
        return float((self.vw * (lv1 * lv2) * self.comb2[k]).sum())

    def qs(self, A, popA=None):
        """B = complement(A)."""
        n = self.n
        if popA is None:
            popA = bin(A).count('1')
        cntA = self.counts(A)
        cntB = self.vpop - cntA
        cntA_ext = np.concatenate([cntA, self.leafmask(A)])
        return self._term_ext(cntA_ext, cntB, n - popA)

    def _term_ext(self, cntA_ext, cntB, popB):
        lv1 = np.where(self.leaf1, cntA_ext[self.M + self.t1], cntA_ext[self.r1])
        lv2 = np.where(self.leaf2, cntA_ext[self.M + self.t2], cntA_ext[self.r2])
        k = popB - cntB
        np.clip(k, 0, None, out=k)
        return float((self.vw * (lv1 * lv2) * self.comb2[k]).sum())

    def qs2(self, A, B, popA=None, popB=None):
        """General A, B (not necessarily complementary)."""
        if popA is None:
            popA = bin(A).count('1')
        if popB is None:
            popB = bin(B).count('1')
        cntA = self.counts(A)
        cntB = self.counts(B)
        cntA_ext = np.concatenate([cntA, self.leafmask(A)])
        return self._term_ext(cntA_ext, cntB, popB)

    def objective(self, tree):
        """Exact F(T) = sum_e qs(clade(e)) - sum_v qs2(child clades)."""
        cl, par, order, kids = tree.clades()
        tot = 0.0
        n = self.n
        for u in tree.internal:
            if par[u] == self.root_leaf:
                # top node: only the node term (its edge is pendant)
                if len(kids[u]) == 2:
                    tot -= self.qs2(cl[kids[u][0]], cl[kids[u][1]])
                continue
            tot += self.qs(cl[u])
            if len(kids[u]) == 2:
                tot -= self.qs2(cl[kids[u][0]], cl[kids[u][1]])
        return tot

    def nni_delta(self, A1, A2, B1):
        """Score change of the NNI swapping subtrees A2 and B1.

        Before: u has children (A1, A2), v has children (u, B1).
        After:  u has children (A1, B1), v has children (u, A2).
        """
        n = self.n
        cA1 = self.counts(A1)
        cA2 = self.counts(A2)
        cB1 = self.counts(B1)
        lA1 = self.leafmask(A1)
        lA2 = self.leafmask(A2)
        lB1 = self.leafmask(B1)
        cA1e = np.concatenate([cA1, lA1])
        cA2e = np.concatenate([cA2, lA2])
        cB1e = np.concatenate([cB1, lB1])
        popA1 = int(lA1.sum()); popA2 = int(lA2.sum()); popB1 = int(lB1.sum())
        # --- edge term at u: qs(A1|B1) - qs(A1|A2)  (B = complement)
        cNew = cA1 + cB1
        cNewe = np.concatenate([cNew, lA1 + lB1])
        cOld = cA1 + cA2
        cOlde = np.concatenate([cOld, lA1 + lA2])
        d_edge = (self._term_ext(cNewe, self.vpop - cNew, n - popA1 - popB1)
                  - self._term_ext(cOlde, self.vpop - cOld, n - popA1 - popA2))
        # --- node term at u: qs2(A1,B1) - qs2(A1,A2)
        d_u = (self._term_ext(cA1e, cB1, popB1)
               - self._term_ext(cA1e, cA2, popA2))
        # --- node term at v: qs2(A1|B1, A2) - qs2(A1|A2, B1)
        d_v = (self._term_ext(cNewe, cA2, popA2)
               - self._term_ext(cOlde, cB1, popB1))
        return d_edge + d_u + d_v


class STree:
    """Candidate species tree: adjacency lists, leaves 0..n-1."""

    def __init__(self, adj, n):
        self.adj = adj
        self.n = n
        self.N = len(adj)
        self.internal = [u for u in range(n, self.N) if adj[u]]

    def root_at(self, leaf=0):
        par = [-1] * self.N
        order = []
        seen = [False] * self.N
        seen[leaf] = True
        stack = [leaf]
        kids = [[] for _ in range(self.N)]
        while stack:
            u = stack.pop()
            order.append(u)
            for v in self.adj[u]:
                if not seen[v]:
                    seen[v] = True
                    par[v] = u
                    kids[u].append(v)
                    stack.append(v)
        return par, order, kids

    def clades(self, leaf=0):
        par, order, kids = self.root_at(leaf)
        cl = [0] * self.N
        for u in reversed(order):
            if u < self.n:
                cl[u] = 1 << u
            else:
                m = 0
                for v in kids[u]:
                    m |= cl[v]
                cl[u] = m
        return cl, par, order, kids

    def to_newick(self, taxa):
        root = None
        for u in self.internal:
            if len(self.adj[u]) >= 3:
                root = u
                break
        if root is None:
            root = 0
        out = []

        def rec(u, pe):
            if u < self.n:
                out.append(taxa[u])
                return
            out.append("(")
            first = True
            for v in self.adj[u]:
                if v == pe:
                    continue
                if not first:
                    out.append(",")
                first = False
                rec(v, u)
            out.append(")")
        out.append("(")
        first = True
        for v in self.adj[root]:
            if not first:
                out.append(",")
            first = False
            rec(v, root)
        out.append(")")
        out.append(";")
        return "".join(out)

    def nni(self, u, v, a, b):
        """Swap neighbour a of u with neighbour b of v (u,v adjacent)."""
        au = self.adj[u]
        au[au.index(a)] = b
        bv = self.adj[v]
        bv[bv.index(b)] = a
        aa = self.adj[a]
        aa[aa.index(u)] = v
        bb = self.adj[b]
        bb[bb.index(v)] = u


def nj_tree(D, n):
    """Neighbour joining -> STree."""
    D = np.asarray(D, dtype=float).copy()
    np.fill_diagonal(D, 0.0)
    adj = [[] for _ in range(n)]
    nodes = list(range(n))
    nextid = n
    while len(nodes) > 2:
        m = len(nodes)
        r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = divmod(int(np.argmin(Q)), m)
        dij = D[i, j]
        dik = 0.5 * (dij + (r[i] - r[j]) / (m - 2))
        djk = dij - dik
        u = nextid
        nextid += 1
        adj.append([])
        ni, nj = nodes[i], nodes[j]
        adj[u].append(ni); adj[ni].append(u)
        adj[u].append(nj); adj[nj].append(u)
        newrow = 0.5 * (D[i, :] + D[j, :] - dij)
        keep = [k for k in range(m) if k != i and k != j]
        Dn = np.empty((m - 1, m - 1))
        Dn[:m - 2, :m - 2] = D[np.ix_(keep, keep)]
        Dn[:m - 2, m - 2] = newrow[keep]
        Dn[m - 2, :m - 2] = newrow[keep]
        Dn[m - 2, m - 2] = 0.0
        D = Dn
        nodes = [nodes[k] for k in keep] + [u]
    a, b = nodes
    adj[a].append(b); adj[b].append(a)
    return STree(adj, n)


def distance_matrix(trees, n, mode):
    """Average pairwise distances across gene trees."""
    D = np.zeros((n, n))
    for t in trees:
        if mode == 'topo':
            for s in range(n):
                seen = [False] * t.N
                seen[s] = True
                stack = [(s, 0)]
                while stack:
                    u, d = stack.pop()
                    if u < n:
                        D[s, u] += d
                    for v, e in t.adj[u]:
                        if not seen[v]:
                            seen[v] = True
                            stack.append((v, d + 1))
        else:
            elen = t.elen
            for s in range(n):
                seen = [False] * t.N
                seen[s] = True
                stack = [(s, 0.0)]
                while stack:
                    u, d = stack.pop()
                    if u < n:
                        D[s, u] += d
                    for v, e in t.adj[u]:
                        if not seen[v]:
                            seen[v] = True
                            stack.append((v, d + elen[e]))
    D /= len(trees)
    return D


def hill_climb(tree, sc, deadline, max_passes=400):
    n = tree.n
    root_leaf = sc.root_leaf
    passes = 0
    while time.time() < deadline and passes < max_passes:
        passes += 1
        cl, par, order, kids = tree.clades(root_leaf)
        improved = False
        edges = []
        for a in tree.internal:
            for b in tree.adj[a]:
                if b >= n and a < b:
                    edges.append((a, b))
        for (x, y) in edges:
            if y not in tree.adj[x] or x not in tree.adj[y]:
                continue
            if par[x] == y:
                u, v = x, y
            elif par[y] == x:
                u, v = y, x
            else:
                continue
            ku = [w for w in tree.adj[u] if w != v]
            kv = [w for w in tree.adj[v] if w != u]
            if len(ku) != 2 or len(kv) != 2:
                continue
            best = None
            for i in (0, 1):
                for j in (0, 1):
                    ka, kb = ku[i], kv[j]
                    a1, a2, b1 = cl[ka], cl[ku[1 - i]], cl[kb]
                    d = sc.nni_delta(a1, a2, b1)
                    if d > 1e-9 and (best is None or d > best[0]):
                        best = (d, ku[1 - i], kb)
            if best is not None:
                tree.nni(u, v, best[1], best[2])
                improved = True
                cl, par, order, kids = tree.clades(root_leaf)
        if not improved:
            break
    return tree


def read_genes(path):
    txt = open(path, encoding='utf-8').read()
    trees = []
    for chunk in txt.replace('\r', '\n').split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        trees.append(chunk + ';')
        if len(trees) == 1000:
            break
    return trees


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    seed = os.environ.get('SEED', '1')
    np.random.seed(int(seed) if seed.isdigit() else 1)
    raw = read_genes(inp)
    t0 = GTree(raw[0])
    n = t0.n
    taxa = list(t0.taxa)
    if '0' in taxa:
        taxa.remove('0')
        taxa = ['0'] + taxa
    order = {lab: i for i, lab in enumerate(taxa)}
    trees = [GTree(r, order) for r in raw]

    weight = os.environ.get('WEIGHT', 'none')
    sc = Scorer(trees, n, weight=weight)

    # starting trees from distance matrices
    starts = []
    try:
        Dp = distance_matrix(trees, n, 'pat')
        starts.append(nj_tree(Dp, n))
    except Exception:
        pass
    try:
        Dt = distance_matrix(trees, n, 'topo')
        starts.append(nj_tree(Dt, n))
    except Exception:
        pass
    if not starts:
        # fall back to a copy of the first gene tree
        t = trees[0]
        adj = [[v for v, e in t.adj[u]] for u in range(t.N)]
        starts.append(STree(adj, n))

    best_tree = None
    best_score = -np.inf
    use_hc = os.environ.get('HILLCLIMB', '0') == '1'
    select = os.environ.get('SELECT', '0') == '1'
    for st in starts:
        try:
            s = sc.objective(st) if select or use_hc else 0.0
        except Exception:
            s = -np.inf
        if use_hc:
            deadline = min(time.time() + max(5.0, (TIME_LIMIT - (time.time() - START)) / max(1, len(starts))),
                           START + TIME_LIMIT - 5.0)
            st = hill_climb(st, sc, deadline)
            try:
                s = sc.objective(st)
            except Exception:
                s = -np.inf
        if s > best_score:
            best_score = s
            best_tree = st
        if time.time() - START > TIME_LIMIT - 15:
            break

    out = best_tree.to_newick(taxa)
    with open(outp, 'w') as f:
        f.write(out)
        f.write("\n")


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    main()
