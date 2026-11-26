#!/usr/bin/env python3
"""Infer a species tree from a set of estimated gene trees.

Input : a text file with one Newick gene tree per line (unrooted, binary).
Output: one Newick species tree (unrooted, binary) covering exactly the same
         leaf labels.

Method
------
The loci are short (200 bp), so single gene trees are noisy and disagree with
each other.  The program therefore combines the evidence of all loci twice:

1. *Consensus distance* (global, quartet-like evidence).  Every internal branch
   of every gene tree defines a cut of the leaf set and carries an approximate
   Bayesian support.  For two leaves i, j we accumulate

       d(i, j) = sum over loci and over internal branches that separate i and j
                 of the support of that branch,

   i.e. the support-weighted length of the path between the two leaves,
   summed over all loci.  A branch with a high support contributes a lot, a
   poorly supported branch contributes little, so the resulting matrix measures
   how strongly the loci agree that two lineages are close.  Neighbour joining
   on that matrix maximises a four-point (minimum evolution) criterion, which
   aggregates the conflicting per-locus evidence into a globally consistent
   tree and is able to place deep splits that no single locus resolves.

2. *Repair with the well supported splits* (local, direct evidence).  Splits
   that are present in many loci (a fifth of the loci) are essentially never
   wrong, but neighbour joining occasionally places such a split wrongly.  For
   every such missing split we look for a single nearest-neighbour-interchange
   (NNI) move that inserts it without destroying another well supported split
   (or only destroying a much less supported one; splits are compared by their
   summed support over the loci).  This fixes the local mistakes of the
   distance based tree without disturbing its deep structure.

Everything is deterministic; no random numbers are used.
"""
import re
import sys

TOKEN = re.compile(r"""\s+|\[[^\[\]]*\]|'(?:[^']|'')*'|[^\s\[\]'(),:;]+|.""", re.DOTALL)

CORE_MIN = 10        # a split seen in at least this many loci is "well supported"
CORE_RATIO = 1.2     # a repair may drop a well supported split only if the
                     # inserted one is at least CORE_RATIO times better supported


def core_min(n_loci):
    """Number of loci a split has to appear in to count as well supported.

    With the usual 50 loci this is 10 (a fifth of the loci), a threshold at
    which such splits are essentially always correct.  The rule scales with the
    number of loci so that instances with fewer (or more) loci are treated
    consistently.
    """
    return max(8, int(round(0.2 * n_loci)))


class ParseError(Exception):
    pass


class Node(object):
    __slots__ = ("name", "kids", "sup", "bl", "leaves")

    def __init__(self, name=None):
        self.name = name
        self.kids = []
        self.sup = None
        self.bl = None
        self.leaves = None


def parse_newick(text):
    """Parse one Newick string into a Node tree."""
    toks = [m.group(0) for m in TOKEN.finditer(text)
            if not m.group(0).isspace()
            and not (m.group(0).startswith("[") and m.group(0).endswith("]"))]
    pos = 0
    n = len(toks)

    def peek():
        return toks[pos] if pos < n else None

    def adv():
        nonlocal pos
        t = toks[pos]
        pos += 1
        return t

    def subtree():
        if peek() == "(":
            adv()
            node = Node()
            node.kids.append(subtree())
            while peek() == ",":
                adv()
                node.kids.append(subtree())
            if adv() != ")":
                raise ParseError("unbalanced parentheses")
            if peek() is not None and peek() not in {",", ":", ";", ")", "("}:
                # an internal label: normally an approximate Bayesian support.
                # Anything that is not a number is ignored (the branch is then
                # treated as unsupported).
                label = adv()
                try:
                    node.sup = float(label)
                except ValueError:
                    node.sup = None
            if peek() == ":":
                adv()
                if peek() is None:
                    raise ParseError("missing branch length")
                adv()
            return node
        name = adv()
        if name is None or name in {",", ":", ";", "(", ")"}:
            raise ParseError("bad leaf label")
        node = Node(name)
        if peek() == ":":
            adv()
            if peek() is None:
                raise ParseError("missing branch length")
            adv()
        return node

    root = subtree()
    if peek() != ";":
        raise ParseError("missing ';'")
    return root


def collect_leaves(node, out):
    if node.kids:
        s = set()
        for k in node.kids:
            s |= collect_leaves(k, out)
        node.leaves = s
        return s
    out.append(node.name)
    node.leaves = {node.name}
    return node.leaves


def internal_cuts(root, index):
    """Return [(leaf index set, support)] for every internal branch."""
    cuts = []

    def rec(node, parent):
        if node.kids:
            s = set()
            for k in node.kids:
                s |= rec(k, node)
            if parent is not None:
                cuts.append((s, node.sup))
            return s
        return {index[node.name]}

    rec(root, None)
    # A degree-two root sits in the middle of an edge; in that case its two
    # children describe one single edge and must not be counted twice.
    if len(root.kids) == 2 and all(k.kids for k in root.kids):
        a, b = root.kids
        if a.leaves and b.leaves and len(a.leaves) > 1 and len(b.leaves) > 1:
            cuts = [c for c in cuts if c[0] != b.leaves]
    return cuts


def read_gene_trees(path):
    """Read the input file and parse every tree.  Returns (roots, taxa, index)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    parsed = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            root = parse_newick(line)
        except ParseError:
            continue
        names = []
        collect_leaves(root, names)
        if len(names) < 2 or len(set(names)) != len(names):
            # not a usable tree (truncated or duplicated labels)
            continue
        parsed.append((root, names))
    roots = []
    taxa = []
    index = None
    if parsed:
        # the taxon set is taken from the tree with the most leaves
        best = max(range(len(parsed)), key=lambda i: len(parsed[i][1]))
        taxa = list(parsed[best][1])
        index = {t: i for i, t in enumerate(taxa)}
        want = set(taxa)
        for root, names in parsed:
            if set(names) == want:
                roots.append(root)
    if not roots:
        # Nothing usable: fall back to the labels that appear in the file so that
        # a (star) tree can still be written.
        labels = []
        for line in raw.split("\n"):
            for tok in TOKEN.finditer(line):
                t = tok.group(0)
                if t and t not in {"(", ")", ",", ":", ";", "[", "]", "'"} and not t.startswith("'"):
                    labels.append(t)
        seen = set()
        taxa = [t for t in labels if not (t in seen or seen.add(t))]
        if len(taxa) < 4:
            raise ParseError("no parsable trees")
        index = {t: i for i, t in enumerate(taxa)}
    return roots, taxa, index


def build_distance(roots, index, n):
    """Support weighted consensus distance matrix (numpy, with a pure python fallback)."""
    rows = []
    vals = []
    for root in roots:
        for s, sup in internal_cuts(root, index):
            if sup is None:
                continue
            if len(s) < 2 or len(s) > n - 2:
                continue
            w = float(sup)
            if w <= 0.0:
                continue
            rows.append(s)
            vals.append(w)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is None:
        D = [[0.0] * n for _ in range(n)]
        for s, w in zip(rows, vals):
            for i in s:
                for j in range(n):
                    if j not in s:
                        D[i][j] += w
                        D[j][i] += w
        return D
    C = np.zeros((len(rows), n), dtype=np.float64)
    for i, (s, w) in enumerate(zip(rows, vals)):
        C[i, list(s)] = w
    r = C.sum(axis=0)
    D = r[:, None] + r[None, :] - 2.0 * (C.T @ C)
    np.fill_diagonal(D, 0.0)
    return D


def neighbor_joining(D):
    """Classic neighbour joining.  Returns (kids, root) where kids maps a node id
    to the list of (leaf set, child id) of its children."""
    import numpy as np
    n = D.shape[0]
    d = D.astype(np.float64, copy=True)
    np.fill_diagonal(d, 0.0)
    active = list(range(n))
    clust = [frozenset([i]) for i in range(n)]
    kids = {}
    while len(active) > 2:
        m = len(active)
        r = d.sum(axis=1)
        Q = (m - 2) * d - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        idx = int(np.argmin(Q))
        i = idx // m
        j = idx % m
        u, v = active[i], active[j]
        dij = d[i, j]
        w = len(clust)
        kids[w] = [(clust[u], u), (clust[v], v)]
        clust.append(clust[u] | clust[v])
        newd = (d[i, :] + d[j, :] - dij) / 2.0
        d[i, :] = newd
        d[:, i] = newd
        d[i, i] = 0.0
        d = np.delete(np.delete(d, j, 0), j, 1)
        active[i] = w
        del active[j]
    u, v = active[0], active[1]
    w = len(clust)
    kids[w] = [(clust[u], u), (clust[v], v)]
    clust.append(clust[u] | clust[v])
    return kids, w


# --------------------------------------------------------------------------
# tree utilities on an unrooted adjacency structure
# --------------------------------------------------------------------------

def to_adjacency(kids, root, n):
    """Convert the rooted NJ structure into an unrooted adjacency dict."""
    adj = {i: [] for i in range(n)}
    counter = [n]

    def build(node):
        ch = kids.get(node)
        if not ch:
            return node
        counter[0] += 1
        me = counter[0]
        adj[me] = []
        for _, cid in ch:
            c = build(cid)
            adj[me].append(c)
            adj[c].append(me)
        return me

    build(root)
    return adj


def rooted_info(adj, n):
    """Root the tree at an internal node; return (parent, order, mask)."""
    root = next(x for x in adj if x >= n)
    parent = {}
    order = []
    stack = [(root, None)]
    while stack:
        node, p = stack.pop()
        parent[node] = p
        order.append(node)
        for nb in adj[node]:
            if nb != p:
                stack.append((nb, node))
    mask = {}
    for node in reversed(order):
        m = (1 << node) if node < n else 0
        for nb in adj[node]:
            if parent.get(nb) == node:
                m |= mask[nb]
        mask[node] = m
    return parent, order, mask


def canonical(mask, n):
    """Canonical side of a bipartition: the one that does not contain taxon 0."""
    full = (1 << n) - 1
    return mask if not (mask & 1) else full ^ mask


def current_splits(adj, n):
    """Canonical masks of the internal splits of the tree."""
    parent, order, mask = rooted_info(adj, n)
    out = set()
    for node in mask:
        if node < n or parent[node] is None:
            continue
        m = mask[node]
        if m.bit_count() >= 2 and n - m.bit_count() >= 2:
            out.add(canonical(m, n))
    return out


def repair_core_splits(adj, n, core_freq, core_weight=None):
    """Insert well supported splits that the tree misses, using NNI moves that do
    not destroy a well supported split (or only a much weaker one).

    core_freq   : canonical mask -> number of loci in which the split occurs
    core_weight : canonical mask -> summed support of the split (optional; used
                  instead of the frequency to rank and to compare splits)
    """
    full = (1 << n) - 1
    weight = core_weight if core_weight else core_freq
    present = current_splits(adj, n)
    applied = 0
    for _ in range(300):
        missing = {m: f for m, f in core_freq.items() if m not in present}
        if not missing:
            break
        parent, order, mask = rooted_info(adj, n)
        best = None
        for u in list(adj):
            if u < n:
                continue
            for v in adj[u]:
                if v < n or v < u:
                    continue
                a_branches = []
                b_branches = []
                for w in adj[u]:
                    if w == v:
                        continue
                    a_branches.append((w, mask[w] if parent.get(w) == u else full ^ mask[u]))
                for w in adj[v]:
                    if w == u:
                        continue
                    b_branches.append((w, mask[w] if parent.get(w) == v else full ^ mask[v]))
                if len(a_branches) != 2 or len(b_branches) != 2:
                    continue
                (na1, a1), (na2, a2) = a_branches
                (nb1, b1), (nb2, b2) = b_branches
                old = canonical(a1 | a2, n)
                if old in missing:
                    continue
                old_weight = weight.get(old, 0.0)
                for (x, sw_u, sw_v) in (((a1 | b1), na2, nb1), ((a1 | b2), na2, nb2)):
                    cm = canonical(x, n)
                    if cm not in missing:
                        continue
                    cw = weight.get(cm, 0.0)
                    if old_weight > 0 and cw < CORE_RATIO * old_weight:
                        continue
                    key = (cw, missing[cm])
                    if best is None or key > best[0]:
                        best = (key, u, v, sw_u, sw_v)
        if best is None:
            break
        _, u, v, sw_u, sw_v = best
        adj[u].remove(sw_u)
        adj[sw_u].remove(u)
        adj[v].remove(sw_v)
        adj[sw_v].remove(v)
        adj[u].append(sw_v)
        adj[sw_v].append(u)
        adj[v].append(sw_u)
        adj[sw_u].append(v)
        present = current_splits(adj, n)
        applied += 1
    return applied


def adjacency_to_newick(adj, n, taxa):
    """Write the unrooted tree as a Newick string with a trifurcating root."""
    parent, order, mask = rooted_info(adj, n)
    root = next(x for x in adj if x >= n)
    kids = {}
    for node in order:
        ch = [nb for nb in adj[node] if parent.get(nb) == node]
        kids[node] = ch

    def rec(node):
        ch = kids[node]
        if not ch:
            return str(taxa[node])
        return "(" + ",".join(rec(c) for c in ch) + ")"

    return rec(root) + ";"


def split_frequencies(roots, index, n):
    """Number of loci in which each internal split occurs (canonical masks)."""
    from collections import Counter
    cnt = Counter()
    full = (1 << n) - 1
    for root in roots:
        seen = set()
        for s, _sup in internal_cuts(root, index):
            if len(s) < 2 or len(s) > n - 2:
                continue
            mask = 0
            for x in s:
                mask |= 1 << x
            seen.add(canonical(mask, n))
        for m in seen:
            cnt[m] += 1
    return cnt


def split_support_sums(roots, index, n):
    """Summed support of each internal split over all loci (canonical masks)."""
    from collections import Counter
    cnt = Counter()
    for root in roots:
        for s, sup in internal_cuts(root, index):
            if sup is None:
                continue
            if len(s) < 2 or len(s) > n - 2:
                continue
            mask = 0
            for x in s:
                mask |= 1 << x
            cnt[canonical(mask, n)] += float(sup)
    return cnt


def fallback_tree(taxa):
    """A balanced binary tree over the given labels (used when nothing usable
    could be parsed, so that the output is at least a valid tree)."""
    labels = [str(t) for t in taxa]
    if len(labels) < 4:
        return "(" + ",".join(labels) + ");"
    while len(labels) > 2:
        nxt = []
        for i in range(0, len(labels) - 1, 2):
            nxt.append("(" + labels[i] + "," + labels[i + 1] + ")")
        if len(labels) % 2:
            nxt.append(labels[-1])
        labels = nxt
    return "(" + labels[0] + "," + labels[1] + ");"


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: species_tree.py INPUT OUTPUT\n")
        return 2
    inp, outp = argv[1], argv[2]
    roots, taxa, index = read_gene_trees(inp)
    n = len(taxa)
    if not roots or n < 4:
        # Nothing usable: write a valid (if uninformative) binary tree so that
        # the output is at least a legal tree over the input labels.
        out = fallback_tree(taxa)
        with open(outp, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        return 0
    D = build_distance(roots, index, n)
    kids, root = neighbor_joining(D)
    adj = to_adjacency(kids, root, n)
    freq = split_frequencies(roots, index, n)
    core = {m: c for m, c in freq.items() if c >= core_min(len(roots))}
    if core:
        weight = split_support_sums(roots, index, n)
        repair_core_splits(adj, n, core, weight)
    out = adjacency_to_newick(adj, n, taxa)
    with open(outp, "w", encoding="utf-8") as fh:
        fh.write(out + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as exc:  # pragma: no cover
        sys.stderr.write("error: %s\n" % exc)
        sys.exit(1)
