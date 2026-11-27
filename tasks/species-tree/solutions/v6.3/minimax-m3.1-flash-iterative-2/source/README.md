# Species-tree inference from 50 gene trees (250 species)

Infers one unrooted binary species tree from a file of estimated gene trees.

## Contents

| file | purpose |
|---|---|
| `stree.py` | the whole solution (Python 3, standard library only) |
| `run.sh` | entry point: `./run.sh INPUT OUTPUT` |
| `build.sh` | optional, no-argument build step (byte-compiles the sources) |

## Dependencies

* Linux, POSIX shell
* Python 3 (>= 3.8).  Only the standard library is used — `numpy` is used
  opportunistically when it happens to be installed, but the program works
  without it.

No compilation, no downloads and no network access are required.  `build.sh` is
a no-op apart from byte-compiling `stree.py`; the program also runs correctly
when `build.sh` is never executed.

## Usage

```sh
./run.sh /absolute/path/to/genes.tre /absolute/path/to/pred.tre
```

The first argument is a UTF-8 text file with one Newick gene tree per line
(50 lines in the task data, each an unrooted binary tree over the same 250
taxa, terminated by `;`).  Leaf labels are arbitrary strings; `0` is the
outgroup.  The second argument receives a single Newick tree, terminated by
`;`, containing every input leaf label exactly once.  The tree is written
rooted on the outgroup with a degree-two serialisation root, so after the root
is suppressed every internal node has degree three.

Progress information is written to standard error; standard output is unused.

## Method

1. **Split extraction.**  Each gene tree is parsed into a rooted tree and
   every internal edge is turned into a *split* — the bipartition of the taxon
   set induced by that edge.  A split is stored as a bitmask over the taxa,
   canonicalised as the side that does *not* contain the outgroup, so that two
   splits can be compared directly.  The number of gene trees containing a
   split and the sum of the reported node supports are accumulated for every
   split.

2. **Weighted greedy consensus.**  Splits that are pairwise compatible (i.e.
   that can all occur in a single binary tree) are selected greedily in
   decreasing order of weight.  The weight of a split is the sum of the node
   supports reported for it across the gene trees that contain it, plus half a
   point for every gene tree in which the split is reported with (almost) unit
   support.  A split that is confidently present in a gene tree is much
   stronger evidence than one that merely occurs often while being weakly
   supported, and this weighting measured better than the plain gene-tree
   count (normalised RF 0.224 vs. 0.247 on subsampled development
   instances).  The result is a hierarchy of clusters with a few unresolved
   (polytomous) nodes.

3. **Polytomy resolution.**  Unresolved nodes are made binary with a distance
   criterion.  For every pair of children of a polytomy the average gene-tree
   distance is computed — the mean patristic distance (sum of branch lengths)
   over all pairs of taxa taken from the two children, averaged over the gene
   trees.  The children are then agglomerated with average linkage, which
   yields a binary arrangement of the polytomy.

4. **Output.**  The resulting hierarchy is serialised as a Newick tree rooted
   on the outgroup.  If the inference fails for any reason the program falls
   back to writing a caterpillar tree over the leaf labels of the input, so
   that the output file always contains a syntactically valid tree with the
   correct taxon set.

The whole computation takes well under a second for a 250-taxon, 50-gene-tree
instance, so the runtime limit is never a constraint.

## Development

The development data in `/workspace/samples` contains two instances with their
true species trees and a self-check script:

```sh
dev=/workspace/samples
./run.sh "$dev/cases/001/genes.tre" "$PWD/pred.tre"
python3 "$dev/check_rf.py" "$dev/cases/001/truth.tre" "$PWD/pred.tre"
```

The check prints the number of leaves, the number of true internal edges that
were not recovered and the normalised Robinson–Foulds error.

On the two development instances the method recovers 211 and 192 of the 247
true internal edges, i.e. a normalised Robinson–Foulds error of 0.146 and
0.223.  Most of the remaining error consists of deep splits of the species
tree that occur in none of the gene trees and are therefore not recoverable
from them.
