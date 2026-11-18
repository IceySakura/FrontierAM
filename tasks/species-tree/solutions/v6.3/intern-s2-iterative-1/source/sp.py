"""Species-tree inference library.

Rooted representation: every tree is rerooted so that taxon ``anchor``
(== "0", the outgroup) is a direct child of the root:

        root
       /    \
     anchor  rest...

In that representation every internal node other than the root corresponds
to exactly one unrooted split, and its mask is the set of leaves below it.
All masks are oriented *away* from the anchor (they never contain the anchor
bit), which makes the standard clade-compatibility test valid.
"""
import sys
import re
from collections import Counter, defaultdict, deque

sys.setrecursionlimit(100000)

TOKEN = re.compile(
    r"""\s+|\[[^\[\]]*\]|'(?:[^']|'')*'|[^\s\[\]'(),:;]+|.""", re.DOTALL
)
NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


# ---------------------------------------------------------------- Node ----
class Node:
    __slots__ = ("children", "label", "length", "support", "i")

    def __init__(self, children):
        self.children = children
        self.label = None
        self.length = None
        self.support = None
        self.i = None


class ON:
    """Unlabelled/internal node used while building."""
    __slots__ = ("children", "label")

    def __init__(self, children, label=None):
        self.children = children
        self.label = label


# -------------------------------------------------------------- parsing ----
def parse(text):
    """Parse a Newick string into a Node tree."""
    toks = [m[0] for m in TOKEN.finditer(text) if not m[0].isspace()]
    if not toks:
        raise ValueError("empty newick")
    # bracketed decorations are part of the preceding label, e.g. A[1] or
    # 'a,b'[1], so merge them back before the parser sees them.
    merged = []
    for t in toks:
        if t.startswith("[") and merged:
            merged[-1] += t
        else:
            merged.append(t)
    toks = merged
    # a leading bracketed annotation (e.g. "[&R](a,b);") is metadata, not a
    # label, so drop it rather than mis-parsing it as a taxon.
    while toks and toks[0].startswith("["):
        toks = toks[1:]
    if not toks:
        raise ValueError("empty newick")
    p = [0]

    def node():
        t = toks[p[0]]
        if t == "(":
            p[0] += 1
            ch = [node()]
            while p[0] < len(toks) and toks[p[0]] == ",":
                p[0] += 1
                ch.append(node())
            if p[0] >= len(toks) or toks[p[0]] != ")":
                raise ValueError("expected ')'")
            p[0] += 1
            n = Node(ch)
        else:
            n = Node([])
            n.label = t
            p[0] += 1
        if p[0] < len(toks) and NUM.match(toks[p[0]]) \
                and toks[p[0]] not in (",", ")", ";"):
            n.support = float(toks[p[0]])
            p[0] += 1
        if p[0] < len(toks) and toks[p[0]] == ":":
            p[0] += 1
            if p[0] >= len(toks) or not NUM.match(toks[p[0]]):
                raise ValueError("expected branch length")
            n.length = float(toks[p[0]])
            p[0] += 1
        return n

    r = node()
    if p[0] >= len(toks) or toks[p[0]] != ";":
        raise ValueError("expected ';'")
    return r


def leaves(u, out=None):
    if out is None:
        out = set()
    if not u.children:
        out.add(u.label)
    else:
        for c in u.children:
            leaves(c, out)
    return out


def allnodes(u, out=None):
    if out is None:
        out = []
    out.append(u)
    for c in u.children:
        allnodes(c, out)
    return out


def labelset(u):
    return {n.label for n in allnodes(u) if n.label is not None}


def adj_of(u):
    adj = defaultdict(set)

    def rec(v, par):
        if par is not None:
            adj[v].add(par)
            adj[par].add(v)
        for c in v.children:
            rec(c, v)

    rec(u, None)
    return adj


# --------------------------------------------------------------- reroot ----
def reroot_anchor(u, anchor):
    """Reroot so the root is binary with anchor as its first child."""
    A = adj_of(u)
    an = next(x for x in allnodes(u) if x.label == anchor)
    par = {an: None}
    q = deque([an])
    while q:
        v = q.popleft()
        for w in A[v]:
            if w not in par:
                par[w] = v
                q.append(w)

    def build(v, p):
        n = Node([])
        n.label = v.label
        n.support = v.support
        n.length = v.length
        for w in A[v]:
            if w is p:
                continue
            n.children.append(build(w, v))
        return n

    r = build(an, None)
    # In the unrooted tree the anchor is a leaf, so re-rooting on it yields a
    # degree-one root.  Wrap it into the required "(anchor, rest)" shape: the
    # root's children are the anchor leaf and the rest of the tree.
    if len(r.children) == 1:
        leaf = Node([])
        leaf.label = an.label
        leaf.length = an.length
        leaf.support = an.support
        r = Node([leaf, r.children[0]])
    return r

# --------------------------------------------------------------- splits ----
def precompute_masks(r):
    """Map every node to the leaf-mask of the clade below it."""
    nm = {}

    def walk(v):
        if not v.children:
            return 1 << v.i
        m = 0
        for c in v.children:
            m |= walk(c)
        nm[v] = m
        return m

    walk(r)
    return nm


def parent_of(r):
    par = {}

    def walk(v):
        for c in v.children:
            par[c] = v
            walk(c)

    walk(r)
    return par


def attach_index(r, idx):
    for n in allnodes(r):
        if n.label is not None:
            n.i = idx[n.label]


def clade_masks(r, idx, n):
    attach_index(r, idx)
    nm = precompute_masks(r)
    return {m for m in nm.values() if 2 <= m.bit_count() <= n - 2}, nm, parent_of(r)


# --------------------------------------------------------------- greedy ----
def greedy_clades(cnt, n, idx, anchor, weighting="support", minweight=1, sup=None, n_trees=None):
    """Greedy compatible-clade selection, descending weight.

    Masks that contain the anchor are flipped to the anchor-free side first,
    because the compatibility test is only meaningful on the canonical
    (anchor-free) orientation.
    """
    full = (1 << n) - 1
    anchor_bit = 1 << idx[anchor]
    accepted = {0, full} | {1 << idx[lab] for lab in idx}
    # deterministic order: descending weight, then ascending mask so the
    # result does not depend on dict construction order
    order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    for m, w in order:
        # minweight only applies to integer vote counts; for support-weighted
        # schemes every clade with positive weight must be considered, so the
        # cutoff is disabled there.
        if isinstance(minweight, int) and weighting == "vote" and w < minweight:
            break
        if m & anchor_bit:
            m = full ^ m
        k = m.bit_count()
        if k < 2 or k > n - 2:
            continue
        if any((m & a) and (a & (full ^ m)) and (m & (full ^ a)) and (a & (full ^ m))
               for a in accepted):
            continue
        accepted.add(m)
    return accepted


# ---------------------------------------------------------------- build ----
def build_tree(accepted, idx, n):
    """Build a rooted binary tree (anchor, rest) from accepted masks."""
    full = (1 << n) - 1

    def build(mask):
        subs = [c for c in accepted if c not in (0, mask) and (c & mask) == c
                and not any(c != d and d != mask and (c & d) == c
                            and (d & mask) == d for d in accepted)]
        if not subs:
            labs_m = [lab for lab, i in idx.items() if (mask >> i) & 1]
            if not labs_m:
                raise ValueError(f"empty mask {mask}")
            if len(labs_m) == 1:
                return ON([], labs_m[0])
            cur = ON([], labs_m[-1])
            for lab in reversed(labs_m[:-1]):
                cur = ON([ON([], lab), cur])
            return cur
        cov = 0
        kids = [build(s) for s in subs]
        for s in subs:
            cov |= s
        if cov != mask:
            gap = mask ^ cov
            labs_g = [lab for lab, i in idx.items() if (gap >> i) & 1]
            if labs_g:
                cur = ON([], labs_g[-1])
                for lab in reversed(labs_g[:-1]):
                    cur = ON([ON([], lab), cur])
                kids.append(cur)
        if len(kids) == 1:
            return kids[0]
        cur = kids[-1]
        for k in reversed(kids[:-1]):
            cur = ON([k, cur])
        return cur

    return build(full)


def to_newick(u):
    if u.label is not None and not u.children:
        return u.label
    return "(" + ",".join(to_newick(c) for c in u.children) + ")"


# ------------------------------------------------------------- utilities ----
def verify(u, anchor, expected):
    """Return (ok, message).  A single degree-two root is permitted (it
    re-represents the top unrooted edge and is collapsed by the scorer)."""
    lab = labelset(u)
    if lab != expected:
        return False, f"leaf set mismatch ({len(lab)} vs {len(expected)})"
    if len(lab) != len(expected):
        return False, "duplicate leaf labels"
    # unrooted binary: every internal node degree 3, except a serialization
    # root which may be degree 2 -- only one such node is allowed.
    A = adj_of(u)
    deg2 = 0
    for v, nb in A.items():
        d = len(nb)
        if d == 2:
            deg2 += 1
            if deg2 > 1:
                return False, f"node {v.label if v.label else '?'} has degree 2"
        elif d not in (1, 3):
            return False, f"node {v.label if v.label else '?'} has degree {d}"
    if len(A) != 2 * len(expected) - 2 + deg2:
        return False, f"node count {len(A)} != 2*{len(expected)}-2+{deg2}"
    return True, ""


def rf(pred, truth, idx, n, anchor):
    ps = clade_masks(pred, idx, n)[0]
    ts = clade_masks(truth, idx, n)[0]
    return len(ts - ps) / (n - 3), len(ts), len(ps), len(ps & ts)


def rand_tree(n, rng):
    """Random unrooted binary tree on 0..n-1 (Paupoid growth)."""
    leaves = [Node([], str(i)) for i in range(n)]
    nodes = leaves[:]
    while len(nodes) > 2:
        i, j = rng.sample(range(len(nodes)), 2)
        a, b = nodes[i], nodes[j]
        nodes = [Node([a, b])] + [nodes[k] for k in range(len(nodes)) if k not in (i, j)]
    return Node(nodes)
