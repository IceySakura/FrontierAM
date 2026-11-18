"""Species-tree inference: greedy compatible-clade consensus of gene trees.

Reads a file of Newick gene trees (one per line, ';' terminated; internal
nodes may carry ``support:length``) and writes a single Newick species tree.

Every tree is re-rooted on taxon "0" (the outgroup) so that each internal
node corresponds to exactly one unrooted split.  Clades are weighted by the
sum of their bootstrap supports across the gene trees; clades are then
selected greedily in descending weight order, keeping only pairwise
compatible ones, and the unresolved remainder is completed as a caterpillar.

Tuning knobs (read once at start, safe to override per invocation):

    SPT_WEIGHT      support | vote | hybrid      how clades are weighted
    SPT_MINWEIGHT   int (1)  minimum weight to consider a clade
    SPT_MISSING     one | zero | count          value when a node reports no support
    SPT_SUPPORT_MIN float (0.0)   ignore a tree's contribution when its support < this

Defaults reproduce the configuration that scored best on the two published
development cases (0.162 / 0.239 normalized RF).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp  # noqa: E402

ANCHOR = "0"


def _env(name, default, cast=str):
    try:
        return cast(os.environ[name])
    except (KeyError, ValueError):
        return default


def weight_of(node, missing):
    """Weight contributed by one node: its support, or the missing-policy value."""
    if node.support is not None:
        return node.support
    if missing == "zero":
        return 0.0
    if missing == "count":
        return 1.0
    return 1.0


# --- weighting -----------------------------------------------------------

# SPT_MISSING controls the value given to an internal node whose support was
# not reported ("缺省表示没有报告，不能假设每条边都有数值").
#   one  -> 1.0   (neutral: the clade counts as present, unscaled)
#   zero -> 0.0   (absent support contributes nothing)
#   count-> 1.0   (identical to "one"; kept for backwards compatibility)
def weight_of(node, missing):
    """Weight contributed by one node: its support, or the missing-policy value."""
    if node.support is not None:
        return node.support
    if missing == "zero":
        return 0.0
    return 1.0


def count_clades(trees, idx, n, anchor, weighting, missing, support_min):
    """Return {clade_mask: weight} over all gene trees.

    Masks are returned in the *raw* orientation (as they appear below a node);
    the caller/selection routine flips them to the anchor-free side.
    """
    cnt = {}
    for t in trees:
        rr = sp.reroot_anchor(t, anchor)
        sp.attach_index(rr, idx)
        for node, mask in sp.precompute_masks(rr).items():
            k = mask.bit_count()
            if k < 2 or k > n - 2:
                continue
            w = weight_of(node, missing)
            if w < support_min:
                continue
            # canonical orientation: the side that does NOT contain the anchor
            if mask & (1 << idx[anchor]):
                mask = ((1 << n) - 1) ^ mask
            if weighting == "vote":
                cnt[mask] = cnt.get(mask, 0) + 1
            elif weighting == "support":
                cnt[mask] = cnt.get(mask, 0.0) + w
            else:  # hybrid: vote count, scaled by mean support
                v, s = cnt.get(mask, (0, 0.0))
                cnt[mask] = (v + 1, s + w)
    if weighting == "hybrid":
        cnt = {m: s / max(v, 1) for m, (v, s) in cnt.items()}
    return cnt


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: %s GENES_TRE OUT_TRE\n" % argv[0])
        return 2
    inp, out = argv[1], argv[2]

    weighting = _env("SPT_WEIGHT", "support")
    minweight = _env("SPT_MINWEIGHT", 1, int)
    missing = _env("SPT_MISSING", "zero")
    support_min = _env("SPT_SUPPORT_MIN", 0.0, float)
    if weighting not in ("vote", "support", "hybrid"):
        sys.stderr.write("[sptree] bad SPT_WEIGHT=%r, using support\n" % weighting)
        weighting = "support"
    if missing not in ("one", "zero", "count"):
        sys.stderr.write("[sptree] bad SPT_MISSING=%r, using zero\n" % missing)
        missing = "zero"

    t0 = time.perf_counter()
    with open(inp) as fh:
        raw = fh.read()
    # The file is a list of Newick trees, each terminated by ';'.  Parsing by
    # line keeps each tree's terminator intact.
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    trees = []
    bad = 0
    for ln in lines:
        try:
            trees.append(sp.parse(ln))
        except (ValueError, IndexError) as exc:
            bad += 1
            sys.stderr.write("[sptree] skipped unparseable tree: %s\n" % exc)
    if not trees:
        sys.stderr.write("[sptree] no parseable trees in %s\n" % inp)
        return 1

    first = sp.leaves(trees[0])
    L = len(first)
    labs = sorted(first, key=lambda x: (len(x), x))
    idx = {lab: i for i, lab in enumerate(labs)}
    if idx.get(ANCHOR) is None:
        sys.stderr.write("[sptree] anchor taxon %r not found\n" % ANCHOR)
        return 1

    t1 = time.perf_counter()
    cnt = count_clades(trees, idx, L, ANCHOR, weighting, missing, support_min)
    sys.stderr.write(
        "[sptree] %d trees, %d taxa, %d distinct clades (weight=%s minweight=%d)\n"
        % (len(trees), L, len(cnt), weighting, minweight)
    )

    accepted = sp.greedy_clades(
        cnt, L, idx, ANCHOR, weighting=weighting, minweight=minweight,
        n_trees=len(trees),
    )
    sys.stderr.write(
        "[sptree] accepted %d compatible clades (%.3f s)\n"
        % (len(accepted), time.perf_counter() - t1)
    )

    root = sp.build_tree(accepted, idx, L)
    text = sp.to_newick(root) + ";\n"

    parsed = sp.parse(text)
    ok, msg = sp.verify(parsed, ANCHOR, set(labs))
    if not ok:
        sys.stderr.write("[sptree] INVALID OUTPUT: %s\n" % msg)
        return 1
    with open(out, "w") as fh:
        fh.write(text)
    sys.stderr.write(
        "[sptree] wrote %s in %.3f s\n" % (out, time.perf_counter() - t0)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
