# Species tree inference from short-locus gene trees

`run.sh INPUT OUTPUT` reads a file of estimated gene trees (one Newick tree per
line) and writes a single unrooted binary species tree in Newick format to
`OUTPUT`.

```sh
./run.sh /absolute/path/genes.tre /absolute/path/pred.tre
```

## Problem

Each locus is short (200 bp), so a single estimated gene tree is noisy and the
loci disagree with each other and with the underlying species tree.  The task is
to recover as many of the 247 internal edges of the true species tree as
possible from 50 such gene trees.

## Method

The program combines the evidence of all loci twice: once globally (a
support-weighted consensus distance plus neighbour joining) and once locally
(repairing the splits that many loci agree on).

### 1. Support weighted consensus distance

Every internal branch of a gene tree defines a *cut* of the leaf set and
carries an approximate Bayesian support (the number after the closing
parenthesis; branches without a number are simply unweighted).  For two leaves
`i` and `j` the program accumulates

```
d(i, j) = sum over all loci and over all internal branches that separate i and j
           of the support of that branch
```

A branch separates `i` and `j` when exactly one of them lies below it.  `d` is
therefore the support-weighted length of the path between the two leaves,
summed over the loci: it is small when the loci consistently place two
lineages close (sisters, or members of the same small clade) and large when the
loci keep pulling them apart.  Because the supports are used as weights, a
branch that a locus does not support contributes little, exactly as in a
support-weighted summary method.

### 2. Neighbour joining

Neighbour joining is run on that matrix.  NJ selects at every step the pair of
clusters that minimises the four-point criterion, i.e. it chooses splits that
are simultaneously supported by the largest number of quartets across all
loci.  This aggregates the conflicting per-locus evidence into one globally
consistent tree and, unlike a plain consensus of splits, it also places deep
splits that no single locus resolves.

### 3. Repair with the well supported splits

Splits that occur in many loci (a fifth of the loci, at least eight) are
essentially never wrong, but neighbour joining occasionally misplaces one.
For every such split that the tree does not contain, the program looks for a
single nearest-neighbour-interchange (NNI) move that inserts it.  A move is
only used when it does not destroy another well supported split, or only a much
weaker one: the inserted split has to carry at least 1.2 times the summed
support of the split it replaces (splits are compared by their summed support,
which is a finer measure than the number of loci in which they occur).
The repair is applied repeatedly until no further split can be inserted.

### 4. Output

The tree is written in Newick notation; branch lengths and internal labels are
omitted because the evaluation ignores them.  Leaf labels are copied verbatim
from the input, each exactly once.  The root is a node of degree three (or a
node of degree two placed on an edge), and every other internal node has
degree three, as required.

## Files

| file | role |
|---|---|
| `run.sh` | entry point; calls `python3 species_tree.py INPUT OUTPUT` |
| `species_tree.py` | the whole implementation (parsing, distances, NJ, repair, output) |
| `build.sh` | optional; nothing to compile, only checks the interpreter |

## Dependencies

* Python 3 (developed with 3.12).
* NumPy, used for the distance matrix and neighbour joining.  If NumPy cannot
  be imported the program falls back to an equivalent pure-Python
  implementation of the same algorithm, so it always runs.

No network access, no data files and no external programs are needed; the
program only reads its input file and writes its output file (diagnostics, if
any, go to standard error).

## Runtime

About 0.6 s for a 250 taxon / 50 locus instance on one CPU (single threaded,
far below the 135 s limit).  Memory use is a few tens of MB.

## Reproducibility

The algorithm is fully deterministic - no random numbers are used - so the
output depends only on the input file.
