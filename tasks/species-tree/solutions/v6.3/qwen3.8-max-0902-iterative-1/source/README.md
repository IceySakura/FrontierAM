# Species-tree inference from short-gene estimated trees

`stree` reads a file of gene trees (one Newick tree per line, 250 leaves each,
50 loci) and writes one unrooted binary species tree on the same leaf labels.

## Build / run

```sh
./build.sh                                  # optional; compiles stree.cpp -> ./stree
./run.sh /abs/input/genes.tre /abs/out/pred.tre
```

* `build.sh` takes no arguments, needs only a C++11 compiler (`g++`, `c++` or
  `clang++`) and libm, and finishes in a few seconds. A prebuilt `stree` binary
  is included as a fallback; `run.sh` uses it if compilation is unavailable.
* `run.sh INPUT OUTPUT` starts one process per instance, single threaded, and
  writes exactly one semicolon-terminated Newick tree to OUTPUT. Diagnostics go
  to stderr. Typical wall clock: ~1 s for 50 trees x 250 taxa (limit 135 s),
  peak RSS well under 100 MiB.
* No network access, no third-party packages, no runtime data files. Everything
  needed is `stree.cpp` (self-contained, ~700 lines).
* Deterministic: no randomness is used, so `SEED` is irrelevant.

## Method

The input gene trees are estimated from 200 bp loci, so they are very noisy:
only ~15-40 % of their internal edges match the species tree, and whole loci
sometimes degenerate (all branch lengths 0, no support reported). The method
therefore relies on *how often a bipartition is recovered across loci*, with
each occurrence weighted by how trustworthy that gene-tree edge looks.

1. **Parse** every gene tree, keep leaf labels as strings, orient each split
   canonically (the side that does not contain the outgroup label `0`).
2. **Reliability weight** `omega(e)` for every internal gene-tree edge:
   * `(near-)zero length` -> `omega = 0` (such an edge carries no signal; this
     also disables loci whose ML estimate collapsed),
   * `omega *= ((support - 1/3)/(1 - 1/3))^0.5` when support values are
     reported (the observed floor of the approximate-Bayes support is 1/3,
     i.e. "unresolved"); edges without a support value get `omega = 0`.
   * If a file reports no support values at all, support weighting is switched
     off automatically, and the same for branch lengths.
3. **Candidate splits** `X` = every bipartition that occurs in at least one
   gene tree (plus the trivial ones). For each `A in X` accumulate
   `c(A) = sum of omega(e)` over all gene-tree edges `e` whose split is `A`.
4. **Objective**: `score(A) = c(A)^0.5`. The species tree is the binary tree
   whose internal splits all lie in `X` and which maximises the sum of
   `score` over its internal edges - the maximum-weight compatible split set.
   The concave power keeps a single very frequent split from dominating the
   ~70 poorly supported splits that every 250-taxon tree still needs.
5. **Optimization**: the classical ASTRAL dynamic program over subsets in `X`,
   `F[A] = score(A) + max_{B u C = A, B,C in X} (F[B] + F[C])`, processed in
   increasing `|A|`, then the best root split of the full taxon set. This is
   exact for the restricted search space; the tree is written from the argmax
   pointers with a trifurcating outermost node.

A weighted-quartet (ASTRAL) score is also implemented in the same program
(`scoreW`, node-based so that every quartet is counted exactly once) and can be
mixed into the objective via environment variables, but on the development
cases the split-frequency objective was consistently more accurate, so the
quartet term is off by default.

## Tuning knobs (environment variables, all optional)

| variable | default | meaning |
|---|---|---|
| `ST_SUP_POW` | 0.5 | exponent on rescaled support in `omega` (0 = ignore support) |
| `ST_SUP_LO` | 1/3 | support value treated as "no signal" |
| `ST_KILL_ZEROLEN` | 1 | give zero-length edges weight 0 |
| `ST_KILL_NOSUP` | 0 | give edges without reported support weight 0 |
| `ST_LEN_REF`, `ST_LEN_POW`, `ST_LEN_EXP` | 0 | extra branch-length factors in `omega` |
| `ST_CNT_WEIGHT` | 1 | count occurrences with `omega` (0 = plain occurrence count) |
| `ST_CNTSCALE`, `ST_CNTPOW` | 1, 0.5 | scale/exponent of the split-frequency term |
| `ST_QMODE` | 0 | 0 = per-split full-complement quartets, 1 = ASTRAL-style restricted quartets per DP step |
| `ST_QSCALE`, `ST_NORM`, `ST_QBASE` | 0, 1, 1/3 | scale, normalisation and baseline of the weighted-quartet term |
| `ST_BUDGET` | 100 | wall-clock seconds after which the search stops and returns the best tree so far |
| `ST_VERBOSE` | 1 | progress on stderr |

A third command-line argument (a reference Newick tree) turns on development
diagnostics: RF distance and objective value of that tree versus the prediction.

## Self check on the provided samples

```sh
dev=/workspace/samples
./run.sh "$dev/cases/001/genes.tre" /tmp/pred.tre
python3 "$dev/check_rf.py" "$dev/cases/001/truth.tre" /tmp/pred.tre
# {"n_leaves": 250, "missed_splits": 76, "normalized_rf": 0.3077}
./run.sh "$dev/cases/002/genes.tre" /tmp/pred.tre
python3 "$dev/check_rf.py" "$dev/cases/002/truth.tre" /tmp/pred.tre
# {"n_leaves": 250, "missed_splits": 83, "normalized_rf": 0.3360}
```
