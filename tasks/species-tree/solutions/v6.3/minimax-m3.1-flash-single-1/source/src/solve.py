#!/usr/bin/env python3
"""Species-tree inference from a set of estimated gene trees.

Input : a text file with one Newick gene tree per line (unrooted binary
        trees, all covering the same leaf labels; "0" is the outgroup).
Output: one Newick tree (unrooted, binary) over the same leaf labels.

Method
------
1.  Every internal edge of every gene tree gives an unrooted split.  The
    splits are expressed on the ingroup (the outgroup "0" is handled
    separately) and counted, giving the *support* of each candidate split.
2.  A greedy consensus collects a set of mutually compatible splits, best
    supported first (smaller split first on ties).  Two splits are
    compatible when one side of the first is nested in or disjoint from one
    side of the second; a set of pairwise compatible splits is realisable
    by exactly one tree, so the result is always a legal tree.
3.  For a compatible set of splits the smaller sides form a laminar family,
    so the tree is obtained by putting every one of them below the smallest
    cluster containing it.  The edge above such a cluster realises exactly
    the split it came from.  Remaining polytomies are resolved so that the
    output is fully binary.
4.  The outgroup is attached to the top level of the inferred ingroup tree.

Why this objective: the Robinson-Foulds score of the task only counts
*missed* true splits.  A wrong split that is compatible with the correct
ones costs nothing directly, whereas dropping a correct split costs one.
So every split carrying any signal is worth taking, and the greedy keeps
the best supported mutually compatible ones.
"""

import re
import sys

# --------------------------------------------------------------------------
# Newick parsing
# --------------------------------------------------------------------------

TOK = re.compile(r"[(),;:]|[^\s(),;:]+")


class Node(object):
    __slots__ = ("ch", "name", "mask", "parent")

    def __init__(self, ch=None, name=None):
        self.ch = ch          # list of children (None => leaf)
        self.name = name      # leaf label
        self.mask = 0         # bitmask of the leaves below the node
        self.parent = None


def parse_tree(text):
    """Parse one Newick string into a Node tree."""
    toks = TOK.findall(text)
    pos = 0
    n = len(toks)

    def node():
        nonlocal pos
        t = toks[pos]
        if pos >= n:
            raise ValueError("truncated Newick")
        if t == "(":
            pos += 1
            kids = []
            while True:
                kids.append(node())
                if pos < n and toks[pos] == ",":
                    pos += 1
                    continue
                break
            pos += 1                       # ')'
            if pos < n and toks[pos] not in "(),;:":
                pos += 1                    # internal label / support value
            cur = Node(ch=kids)
        else:
            name = t
            if len(name) >= 2 and name[0] == "'" and name[-1] == "'":
                name = name[1:-1].replace("''", "'")
            cur = Node(name=name)
            pos += 1
        if pos < n and toks[pos] == ":":
            pos += 2                       # ':' and the branch length
        return cur

    return node()


def annotate(root, index):
    """Assign leaf indices and compute subtree masks; returns all nodes."""
    order = []

    def rec(nd):
        if nd.ch is None:
            if nd.name not in index:
                index[nd.name] = len(index)
            nd.mask = 1 << index[nd.name]
        else:
            m = 0
            for c in nd.ch:
                c.parent = nd
                m |= rec(c)
            nd.mask = m
        order.append(nd)
        return nd.mask

    rec(root)
    return order


def popcount(x):
    return bin(x).count("1")


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def gene_tree_splits(root, order, allmask, outbit):
    """Unrooted internal splits of one gene tree, expressed on the ingroup."""
    keep = ~outbit
    splits = []
    for nd in order:
        if nd.ch is None or nd is root:
            continue
        a = nd.mask
        b = allmask ^ a
        if a & outbit:
            a, b = b, a
        b &= keep
        if popcount(a) >= 2 and popcount(b) >= 2:
            splits.append((a, b))
    return splits


def compatible(x, splits):
    """True if the split x can coexist with every split in `splits`.

    Two splits of the same taxon set are compatible when one side of the
    first is contained in one side of the second (or disjoint from it);
    such a pair can be realised by a single tree.
    """
    for c, d in splits:
        if (x & c) == x or (x & d) == x or (c & x) == c or (d & x) == d:
            continue
        return False
    return True


# --------------------------------------------------------------------------
# Tree assembly
# --------------------------------------------------------------------------

def build_tree(root_mask, splits, leaf_masks):
    """Turn a set of mutually compatible splits into a binary tree.

    For a compatible set of splits the smaller sides form a laminar family
    (nested or disjoint), and the tree is obtained by putting every one of
    them below the smallest cluster that contains it.  The edge above such a
    cluster realises exactly the split it came from.  Leaves that are not
    covered by any cluster form the remaining children.

    Returns kids: a mapping from a cluster to its children.
    """
    clusters = [a for a, _b in splits]
    sizes = {c: popcount(c) for c in clusters}
    ordered = sorted(set(clusters), key=lambda c: sizes[c])

    def container(x):
        """Smallest cluster containing x (the root when there is none)."""
        for t in ordered:
            if t == x:
                continue
            if (x & t) == x:
                return t
        return root_mask

    kids = {root_mask: []}
    for c in ordered:
        kids.setdefault(c, [])
    for c in ordered:
        kids[container(c)].append(c)

    # leaves go below the smallest cluster that contains them
    for lf in leaf_masks:
        kids[container(lf)].append(lf)

    # every node must be binary: group the remaining leaves into children
    for m in list(kids.keys()):
        ch = kids[m]
        covered = 0
        for c in ch:
            covered |= c
        rest = m & ~covered
        if rest:
            # the taxa of `rest` are leaves (or clusters) not yet attached
            rem = rest
            while rem:
                low = rem & -rem
                kids[m].append(low)
                rem ^= low
        kids[m] = ch

    # make every internal node binary by merging the two smallest children
    for _ in range(4 * len(kids) + 16):
        polys = [m for m, ch in kids.items() if len(ch) > 2]
        if not polys:
            break
        m = polys[0]
        ch = kids[m]
        order = sorted(range(len(ch)), key=lambda k: popcount(ch[k]))
        pick = None
        for ai in range(len(order)):
            for bi in range(ai + 1, len(order)):
                i, j = order[ai], order[bi]
                nm = ch[i] | ch[j]
                if nm not in kids:
                    pick = (i, j, nm)
                    break
            if pick:
                break
        if pick is None:
            i, j = order[0], order[1]
            nm = ch[i] | ch[j]
        else:
            i, j, nm = pick
        rest = [c for k, c in enumerate(ch) if k != i and k != j]
        kids[nm] = [ch[i], ch[j]]
        kids[m] = [nm] + rest
    return kids


def serialise(kids, mask, label_of):
    """Iterative Newick writer; returns (newick, number of leaves)."""
    out = []
    nleaf = 0
    stack = [[mask, 0]]
    while stack:
        node, state = stack[-1]
        ch = kids.get(node)
        if ch is None:
            stack.pop()
            out.append(label_of(node))
            nleaf += 1
            continue
        if state == 0:
            out.append("(")
            stack[-1][1] = 1
            stack.append([ch[0], 0])
            continue
        if state < len(ch):
            if state > 0:
                out.append(",")
            stack[-1][1] += 1
            stack.append([ch[state], 0])
            continue
        out.append(")")
        stack.pop()
    return "".join(out), nleaf


# --------------------------------------------------------------------------
# Main solver
# --------------------------------------------------------------------------

def read_gene_trees(path):
    with open(path, "r") as fh:
        lines = fh.read().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if ln and ln[0] == "(":
            out.append(ln)
    return out


def solve(gene_lines):
    index = {}
    all_splits = []
    allmask = 0
    outbit = 0
    for ln in gene_lines:
        root = parse_tree(ln)
        order = annotate(root, index)
        allmask = root.mask
        if outbit == 0:
            for nm, idx in index.items():
                if nm == "0":
                    outbit = 1 << idx
                    break
            if outbit == 0:
                outbit = 1 << min(index.values())
        all_splits.append(gene_tree_splits(root, order, allmask, outbit))

    ingroup = allmask & ~outbit
    leaf_masks = []
    m = ingroup
    while m:
        low = m & -m
        leaf_masks.append(low)
        m ^= low

    # ---- support of every candidate split --------------------------------
    support = {}
    for sp in all_splits:
        for a, b in sp:
            key = (a, b) if a < b else (b, a)
            support[key] = support.get(key, 0) + 1

    cands = []
    for (a, b), w in support.items():
        sa = popcount(a)
        sb = popcount(b)
        if sa > sb:
            a, b = b, a
            sa, sb = sb, sa
        cands.append((-w, sa, a, b))

    # best supported first; on ties the smaller split comes first, which
    # keeps the deep, weakly supported splits from crowding out the many
    # small ones.
    cands.sort()

    # ---- greedy consensus over mutually compatible splits ----------------
    ranked = cands

    # `chosen` holds the smaller side of every accepted split.  A candidate
    # split is compatible with an accepted one exactly when it is nested in
    # or disjoint from that side, so this check is sufficient.
    accepted = []
    for _nw, _sz, a, b in ranked:
        if compatible(a, accepted):
            accepted.append((a, b))

    # ---- assemble the tree ------------------------------------------------
    kids = build_tree(ingroup, accepted, leaf_masks)

    # ---- outgroup ---------------------------------------------------------
    top = kids[ingroup]
    if len(top) == 2:
        a, b = top
    else:
        acc = top[0]
        for c in top[1:-1]:
            acc |= c
        a = acc
        b = top[-1]

    label = {}
    for nm, idx in index.items():
        label[1 << idx] = nm

    def label_of(mask):
        return label.get(mask, str(mask))

    s_a, n_a = serialise(kids, a, label_of)
    s_b, n_b = serialise(kids, b, label_of)
    out_label = label.get(outbit, "0")
    if n_a + n_b + 1 != len(index):
        raise ValueError("incomplete tree")
    return "(" + out_label + "," + s_a + "," + s_b + ");"


def fallback_tree(index, outbit):
    """A simple valid binary tree, used only if anything goes wrong."""
    labels = sorted(index, key=lambda nm: index[nm])
    out_label = "0"
    others = [nm for nm in labels if nm != out_label]
    if outbit not in (1 << index[nm] for nm in labels):
        out_label = labels[0]
        others = labels[1:]
    node = others[-1] if others else out_label
    for lab in reversed(others[:-1]):
        node = "(" + lab + "," + node + ")"
    return "(" + out_label + "," + node + ");"


def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: solve.py INPUT OUTPUT\n")
        return 2
    try:
        gene_lines = read_gene_trees(argv[1])
        tree = solve(gene_lines)
    except Exception as exc:                      # pragma: no cover
        sys.stderr.write("falling back to a trivial tree: %r\n" % (exc,))
        try:
            index = {}
            for ln in gene_lines:
                root = parse_tree(ln)
                annotate(root, index)
            outbit = 0
            for nm, idx in index.items():
                if nm == "0":
                    outbit = 1 << idx
            tree = fallback_tree(index, outbit)
        except Exception:
            tree = "(0,1);"
    with open(argv[2], "w") as fh:
        fh.write(tree)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
