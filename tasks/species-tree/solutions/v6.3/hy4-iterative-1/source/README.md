# Species-tree inference from 50 short-gene trees

## Task and data

Input: 50 Newick gene trees (unrooted, binary, 250 leaves, leaf `0` is the
outgroup).  Branch lengths are substitutions/site; the optional numeric label
after an internal `)` is an approximate Bayesian support in `[0,1]` (missing
labels are allowed).  Output: one unrooted binary species tree on exactly the
same labels.  Score: normalised Robinson–Foulds distance to the (hidden) true
species tree, i.e. the number of true internal splits that are recovered.

## Method

All constants below are fixed; nothing is fitted per instance and no
information is shared between instances.  The two supplied development cases
were used to choose the constants.

1. **Branch confidence.**  Each internal gene-tree branch gets
   `c = min(1, bl/0.008) * clip((support-0.3)/0.7, 0, 1)`
   (missing support is neutral, `c = min(1, bl/0.008)`).  Zero-length
   branches, which are arbitrary resolutions of polytomies, therefore get
   almost no weight and are effectively contracted.

2. **Locus-averaged internode distances.**  For a branch-confidence profile,
   the distance between two taxa in a gene tree is the sum of `c` over the
   internal branches separating them; averaging over loci gives a distance
   matrix (an ASTRID/NJst-like statistic, but confidence weighted).  Ten
   profiles are used, plus 24 bootstrap resamples of loci.  Neighbour joining
   (NJ) on each produces a tree.

3. **Consensus information.**  Greedy compatible-clade sets are built from
   summed branch confidence / branch length / support at three thresholds.
   Nine "constrained NJ" runs keep those clades and resolve the remaining
   polytomies with NJ on supertaxon distances (the outgroup is the extra
   supertaxon).

4. **Candidate splits** are the union of all above trees and all gene-tree
   splits.  For each one we compute
   * ensemble votes (total and per family of trees),
   * gene-tree frequency, summed confidence, and high-confidence count,
   * `qs`: confidence-weighted fraction of gene-tree quartet votes supporting
     the split, and its normalised raw count,
   * `delta`: the OLS-style internal branch length implied by the averaged
     distance matrix,
   * size and gene-tree-occurrence indicators.

   A fixed linear score combines these features.  An exact dynamic program
   over the allowed clusters then finds the maximum total score compatible
   split system (247 splits); a greedy fallback is used if the DP cannot
   finish inside the time budget.  The rooted ingroup tree is written as
   `(0,<ingroup tree>);`, which suppresses to a valid unrooted binary tree.

`main.py` also contains an exact quartet-support computation (validated
against brute force) and an optional NNI refinement, disabled in the default
configuration because they did not improve development accuracy.

## Requirements and running

* POSIX shell, Python 3 and `numpy` (present in the environment).  No network
  access and no compilation are needed.
* `run.sh` deliberately forces single-thread BLAS/OpenMP, chooses an
  interpreter that can import numpy, and respects `SEED`-style determinism by
  using only a fixed internal RNG seed (999).

```sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`build.sh` (no arguments) performs a write-free syntax check; it is optional.
If numpy is unexpectedly missing, `main.py` falls back to a pure-Python
weighted majority-rule consensus that still writes a valid tree.

Diagnostics go to stderr, the single Newick tree to the requested output path.
Typical runtime is about 8–10 s per instance and peak memory well below 1 GiB.
