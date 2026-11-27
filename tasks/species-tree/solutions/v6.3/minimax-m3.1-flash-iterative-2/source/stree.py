#!/usr/bin/env python3
"""Species-tree inference from a collection of estimated gene trees.

The program reads a Newick file with one estimated gene tree per line and
writes a single unrooted binary species tree in Newick format.

Method
------
1.  Every gene tree is converted into its set of internal-edge splits.  For
    each split we accumulate how many gene trees contain it together with the
    reported node supports.
2.  A weighted greedy consensus selects a maximum-weight set of pairwise
    compatible splits (splits that can simultaneously occur in one binary
    tree).  The weight of a split combines the sum of the reported node
    supports with the number of gene trees in which the split is reported
    with (almost) unit support.
3.  The consensus leaves a few unresolved (polytomous) nodes.  They are
    resolved with a distance criterion: the average gene-tree distance
    (patristic distance between two groups of taxa) is used to agglomerate
    the children of a polytomy into a binary arrangement.

Only the Python standard library is used; numpy is used opportunistically when
it happens to be installed.
"""

import sys
import re
import collections

try:
    import numpy as np
except Exception:                                    # pragma: no cover
    np = None


# ---------------------------------------------------------------------------
# Newick parsing
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""\s+|\[[^\[\]]*\]|'(?:[^']|'')*'|[^\s\[\]'(),:;]+|.""", re.DOTALL)


class Node(object):
    __slots__ = ("mask", "children", "support", "blen", "parent", "depth")

    def __init__(self, mask, children, support, blen):
        self.mask = mask
        self.children = children
        self.support = support
        self.blen = blen
        self.parent = None
        self.depth = 0


class GeneTree(object):
    __slots__ = ("root", "nodes", "splits", "supports", "internal_masks",
                 "edge_len", "edge_mask", "order")


def parse_gene_tree(text, label_index, labels):
    """Parse one Newick string into (root, nodes)."""
    toks = [m[0] for m in _TOKEN_RE.finditer(text)]
    i = 0
    n = len(toks)
    nodes = []

    def parse_sub():
        nonlocal i
        if toks[i] == '(':
            i += 1
            kids = []
            while True:
                kids.append(parse_sub())
                if i < n and toks[i] == ',':
                    i += 1
                    continue
                break
            if toks[i] != ')':
                raise ValueError("malformed newick")
            i += 1
            lab = None
            if i < n and toks[i] not in '(),:;':
                lab = toks[i]
                i += 1
            sup = None
            if lab is not None:
                try:
                    sup = float(lab)
                except ValueError:
                    sup = None
            blen = 0.0
            if i < n and toks[i] == ':':
                i += 1
                blen = float(toks[i])
                i += 1
            mask = 0
            for k in kids:
                mask |= k.mask
            nd = Node(mask, kids, sup, blen)
            nodes.append(nd)
            for k in kids:
                k.parent = nd
            return nd
        name = toks[i]
        i += 1
        if name not in label_index:
            label_index[name] = len(labels)
            labels.append(name)
        idx = label_index[name]
        blen = 0.0
        if i < n and toks[i] == ':':
            i += 1
            blen = float(toks[i])
            i += 1
        nd = Node(1 << idx, [], None, blen)
        nodes.append(nd)
        return nd

    root = parse_sub()
    if i >= n or toks[i] != ';':
        raise ValueError("malformed newick")
    if len(root.children) not in (2, 3):
        raise ValueError("gene tree root must be binary or trifurcating")
    return root, nodes


def build_gene_tree(root, nodes):
    gt = GeneTree()
    gt.root = root
    gt.nodes = nodes
    stack = [(root, 0)]
    order = []
    while stack:
        nd, d = stack.pop()
        nd.depth = d
        order.append(nd)
        for c in nd.children:
            stack.append((c, d + 1))
    internal_masks = []
    edge_len = []
    edge_mask = []
    for nd in order:
        if nd.children:
            internal_masks.append(nd.mask)
            edge_len.append(nd.blen)
            edge_mask.append(nd.mask)
    gt.internal_masks = internal_masks[1:]      # the root contributes nothing
    gt.edge_len = edge_len[1:]
    gt.edge_mask = edge_mask[1:]
    gt.order = order
    gt.splits = []
    gt.supports = []
    return gt


def finalize_splits(gt, anchor_bit):
    """Canonical splits (the side not containing the anchor) of a gene tree.

    A tree whose root has only two children carries a degree-two
    serialisation root; the two children then describe the same edge, so the
    split is recorded only once.
    """
    full = gt.root.mask
    n = full.bit_count()
    gt.splits = []
    gt.supports = []
    seen = set()
    for nd in gt.order:
        if not nd.children or nd is gt.root:
            continue
        side = nd.mask
        if side & anchor_bit:
            side = full & ~side
        k = side.bit_count()
        if 2 <= k <= n - 2 and side not in seen:
            seen.add(side)
            gt.splits.append(side)
            gt.supports.append(nd.support)


def parse_gene_trees(path):
    label_index = {}
    labels = []
    trees = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            root, nodes = parse_gene_tree(line, label_index, labels)
            trees.append(build_gene_tree(root, nodes))
    anchor_bit = 1 << label_index.get("0", 0)
    for gt in trees:
        finalize_splits(gt, anchor_bit)
    return trees, labels, label_index


# ---------------------------------------------------------------------------
# split utilities
# ---------------------------------------------------------------------------

def compatible(x, y):
    """Can the two canonical splits occur together in one binary tree?"""
    if x & y == 0:
        return True
    if x & ~y == 0:
        return True
    if y & ~x == 0:
        return True
    return False


def greedy_consensus(candidates, weights):
    """Greedy maximum-weight set of pairwise compatible splits.

    Candidates are considered in decreasing order of weight and accepted when
    they are compatible with every split accepted so far.
    """
    order = sorted(range(len(candidates)), key=lambda k: (-weights[k], candidates[k]))
    chosen = []
    for k in order:
        s = candidates[k]
        ok = True
        for t in chosen:
            if not compatible(s, t):
                ok = False
                break
        if ok:
            chosen.append(s)
    return chosen


def build_hierarchy(splits, ingroup):
    """Containment hierarchy of a set of pairwise compatible splits."""
    ordered = sorted(splits, key=lambda s: -s.bit_count())
    children = {ingroup: []}
    processed = []
    for s in ordered:
        best = ingroup
        bestsize = ingroup.bit_count()
        for c in processed:
            if (c & s) == s and c.bit_count() < bestsize:
                best = c
                bestsize = c.bit_count()
        children.setdefault(best, []).append(s)
        children.setdefault(s, [])
        processed.append(s)
    return children


def node_parts(children, mask):
    """Children clusters plus the individual leaves hanging below ``mask``."""
    kids = list(children.get(mask, []))
    cov = 0
    for k in kids:
        cov |= k
    leaves = []
    bits = mask & ~cov
    while bits:
        b = bits & -bits
        leaves.append(b)
        bits ^= b
    return kids + leaves


# ---------------------------------------------------------------------------
# gene tree distance between two groups of taxa
# ---------------------------------------------------------------------------

def part_distance_matrix(trees, parts, metric="dist"):
    """Average gene-tree distance between every pair of ``parts``.

    ``dist``   : mean patristic distance (sum of branch lengths) between the
                 two groups of taxa, averaged over the gene trees.
    ``level``  : mean number of internal nodes between the two groups, i.e. the
                 mean depth of the gene tree LCA.
    """
    k = len(parts)
    sizes = [p.bit_count() for p in parts]
    nt = len(trees)
    usenp = np is not None and k >= 8
    if metric == "level":
        if usenp:
            acc = np.zeros((k, k), dtype=np.float64)
            for t in trees:
                for m in t.internal_masks:
                    av = np.fromiter(((p & m).bit_count() for p in parts),
                                     dtype=np.float64, count=k)
                    acc += np.outer(av, av)
        else:
            acc = [0.0] * (k * k)
            for t in trees:
                for m in t.internal_masks:
                    av = [(p & m).bit_count() for p in parts]
                    for i in range(k):
                        ai = av[i]
                        if not ai:
                            continue
                        base = i * k
                        for j in range(i + 1, k):
                            acc[base + j] += ai * av[j]
    else:
        if usenp:
            acc = np.zeros((k, k), dtype=np.float64)
            sarr = np.asarray(sizes, dtype=np.float64)
            for t in trees:
                masks = t.edge_mask
                lens = t.edge_len
                for idx in range(len(masks)):
                    m = masks[idx]
                    l = lens[idx]
                    if l <= 0.0:
                        continue
                    av = np.fromiter(((p & m).bit_count() for p in parts),
                                     dtype=np.float64, count=k)
                    cv = sarr - av
                    acc += l * (np.outer(av, cv) + np.outer(cv, av))
        else:
            acc = [0.0] * (k * k)
            for t in trees:
                masks = t.edge_mask
                lens = t.edge_len
                for idx in range(len(masks)):
                    m = masks[idx]
                    l = lens[idx]
                    if l <= 0.0:
                        continue
                    av = [(p & m).bit_count() for p in parts]
                    cv = [sizes[i] - av[i] for i in range(k)]
                    for i in range(k):
                        ai = av[i]
                        ci = cv[i]
                        if not ai and not ci:
                            continue
                        base = i * k
                        for j in range(i + 1, k):
                            acc[base + j] += l * (ai * cv[j] + av[j] * ci)
    out = np.zeros((k, k), dtype=np.float64) if np is not None else [0.0] * (k * k)
    for i in range(k):
        for j in range(i + 1, k):
            denom = float(sizes[i] * sizes[j]) * nt
            raw = acc[i, j] if usenp else acc[i * k + j]
            d = raw / denom if denom else 0.0
            if np is not None:
                out[i, j] = d
                out[j, i] = d
            else:
                out[i * k + j] = d
                out[j * k + i] = d
    return out


def agglomerate_binary(dist, k):
    """Average-linkage agglomeration returning a nested binary structure.

    A leaf of the result is an integer (index into ``parts``); an internal
    node is a two-element tuple of children.
    """
    if k == 1:
        return 0
    if k == 2:
        return (0, 1)
    isnp = np is not None and isinstance(dist, np.ndarray)
    nodes = {i: (i, 1) for i in range(k)}
    d = {}
    for i in range(k):
        for j in range(i + 1, k):
            d[(i, j)] = d[(j, i)] = float(dist[i][j] if isnp else dist[i * k + j])
    while len(nodes) > 1:
        keys = sorted(nodes)
        best = None
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                i, j = keys[a], keys[b]
                v = d[(i, j) if i < j else (j, i)]
                if best is None or v < best[0]:
                    best = (v, i, j)
        _, i, j = best
        ni, nj = (i, j) if i < j else (j, i)
        wi, wj = nodes[ni][1], nodes[nj][1]
        for kk in nodes:
            if kk in (ni, nj):
                continue
            ki = (ni, kk) if ni < kk else (kk, ni)
            kj = (nj, kk) if nj < kk else (kk, nj)
            v = (d[ki] * wi + d[kj] * wj) / (wi + wj)
            d[(ni, kk)] = d[(kk, ni)] = v
        nodes[ni] = ((nodes[ni][0], nodes[nj][0]), wi + wj)
        del nodes[nj]
    return nodes[list(nodes)[0]][0]


# ---------------------------------------------------------------------------
# polytomy resolution
# ---------------------------------------------------------------------------

def resolve_polytomies(chosen, ingroup, trees, metric="dist"):
    """Turn the consensus hierarchy into a fully resolved binary tree.

    Returns (children, splits) where ``children`` maps a cluster mask to the
    list of its children (singletons are allowed) and ``splits`` holds every
    internal split of the finished tree.
    """
    splits = list(chosen)
    children = build_hierarchy(splits, ingroup)
    stack = [ingroup]
    while stack:
        m = stack.pop()
        parts = node_parts(children, m)
        children[m] = list(parts)
        if len(parts) > 2:
            k = len(parts)
            dist = part_distance_matrix(trees, parts, metric)
            grp = agglomerate_binary(dist, k)

            def build(g):
                """Cluster of group ``g``; creates the children entries."""
                if isinstance(g, int):
                    return parts[g]
                a, b = g
                ma = build(a)
                mb = build(b)
                mm = ma | mb
                children[mm] = [ma, mb]
                splits.append(mm)
                return mm

            top_a, top_b = grp
            children[m] = [build(top_a), build(top_b)]
        for p in children.get(m, []):
            if p.bit_count() > 1:
                stack.append(p)
    return children, splits


# ---------------------------------------------------------------------------
# newick output
# ---------------------------------------------------------------------------

def render_newick(children, mask, labels):
    if mask.bit_count() == 1:
        return labels[mask.bit_length() - 1]
    kids = children.get(mask)
    if not kids:
        bits = mask
        out = []
        while bits:
            b = bits & -bits
            out.append(labels[b.bit_length() - 1])
            bits ^= b
        return "(" + ",".join(out) + ")"
    return "(" + ",".join(render_newick(children, k, labels) for k in kids) + ")"


def tree_to_newick(children, ingroup, labels):
    kids = children.get(ingroup, [])
    if len(kids) != 2:
        raise ValueError("ingroup is not binary: %d children" % len(kids))
    return "(0," + render_newick(children, ingroup, labels) + ");"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def split_weights(trees, mode="supgood", lam=0.5):
    """Weight of every split.

    ``sup``      sum of the reported node supports over the gene trees that
                 contain the split.
    ``supgood``  the same, plus half a point for every gene tree in which the
                 split is reported with (almost) unit support.  A split that
                 is confidently present in a gene tree is stronger evidence
                 than one that merely occurs often while being weakly
                 supported.
    ``count``    plain number of gene trees containing the split.
    """
    cnt = collections.Counter()
    supsum = collections.defaultdict(float)
    nsup = collections.Counter()
    nbest = collections.Counter()
    for t in trees:
        for s, sup in zip(t.splits, t.supports):
            cnt[s] += 1
            if sup is not None:
                supsum[s] += sup
                nsup[s] += 1
                if sup >= 0.999:
                    nbest[s] += 1
    if mode == "count":
        return {s: float(c) for s, c in cnt.items()}
    if mode == "sup":
        return {s: (supsum[s] if nsup[s] else float(cnt[s])) for s in cnt}
    if mode == "supgood":
        return {s: (supsum[s] if supsum[s] > 0.0 else float(cnt[s]))
                + lam * nbest[s] for s in cnt}
    raise ValueError(mode)


_LEAF_RE = re.compile(r"[(,]([^()[\s,;:'\"]+)")


def fallback_tree(inpath, outpath):
    """Write a trivial (caterpillar) tree over the leaf labels of the input.

    Used only if the normal inference fails: a syntactically valid tree with
    the right taxon set is preferable to no output at all.
    """
    labels = []
    seen = set()
    with open(inpath, "r") as fh:
        for line in fh:
            for m in _LEAF_RE.finditer(line):
                name = m.group(1)
                if name not in seen:
                    seen.add(name)
                    labels.append(name)
    if not labels:
        return False
    root = None
    if "0" in seen:
        root = "0"
        labels = [x for x in labels if x != "0"]
    if not labels:
        return False
    if len(labels) == 1:
        nwk = "(%s,%s);" % (root, labels[0]) if root else "(%s);" % labels[0]
    else:
        # caterpillar: (((a,b),c),d) ...
        inner = "(%s,%s)" % (labels[0], labels[1])
        for lab in labels[2:]:
            inner = "(%s,%s)" % (inner, lab)
        nwk = "(%s,%s);" % (root, inner) if root else "%s;" % inner
    with open(outpath, "w") as fh:
        fh.write(nwk + "\n")
    return True


def infer(inpath, outpath, metric="dist", mode="supgood", verbose=False):
    try:
        return _infer(inpath, outpath, metric, mode, verbose)
    except Exception as exc:                          # pragma: no cover
        sys.stderr.write("inference failed (%s); writing fallback tree\n" % exc)
        fallback_tree(inpath, outpath)
        raise SystemExit(0)


def _infer(inpath, outpath, metric="dist", mode="supgood", verbose=False):
    trees, labels, label_index = parse_gene_trees(inpath)
    n = len(labels)
    full = (1 << n) - 1
    anchor = 1 << label_index.get("0", 0)
    ingroup = full & ~anchor

    weights = split_weights(trees, mode)
    cands = list(weights.keys())
    chosen = greedy_consensus(cands, [weights[s] for s in cands])
    children, splits = resolve_polytomies(chosen, ingroup, trees, metric)
    nw = tree_to_newick(children, ingroup, labels)
    with open(outpath, "w") as fh:
        fh.write(nw + "\n")
    if verbose:
        sys.stderr.write("taxa=%d chosen=%d final=%d\n"
                         % (n, len(chosen), len(splits)))
    return nw, splits


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("usage: %s INPUT OUTPUT\n" % sys.argv[0])
        sys.exit(2)
    infer(sys.argv[1], sys.argv[2], verbose=True)
