# Species-tree inference from 50 estimated gene trees (250 taxa)

## Method

`stree.cpp` is a single-file C++17 program with no third-party dependencies.
It implements a support-weighted quartet species-tree method in the spirit of
ASTRAL / weighted ASTRAL, plus a gene-bootstrap consensus step:

1. **Input.** The program parses each Newick gene tree, including labels, the
   approximate-Bayes support after `)` and branch lengths. It re-roots every tree
   on leaf `0`. This root is only a computational anchor; the trees are treated
   as unrooted. Supports are mapped from the aBayes range `[1/3, 1]` to
   `s' = (s-1/3)/(2/3)`. An edge with no reported support gets `s' = 0`; in the
   data these are the near-zero-length edges.
2. **Objective (weighted quartet score).** A gene-tree quartet `ab|cd` whose
   internal path is made of gene-tree edges `e1..ek` gets the weight
   `1 - prod_i (1 - s'_i)`, the probability that at least one path edge is
   correct. The species tree maximises the total weight of gene-tree quartets
   that agree with it. For each species-tree node (a tripartition), the score
   against one gene tree is computed exactly in O(n) with a post-order dynamic
   programme over pair sums with path products of `1 - s'`. Monochromatic
   subtrees are skipped. The scorer was checked against brute-force quartet
   enumeration on small random trees. Scores are stored per gene for every
   tripartition.
3. **Search (exact DP over a constrained cluster set X).** This is the
   ASTRAL-style dynamic programme. X is first built from:
   * all gene-tree clusters;
   * the clusters of a neighbour-joining tree on average internode distances
     (ASTRID-like).

   X then grows by:
   * *NNI closure*: the clusters of every NNI neighbour of the current optimum
     are added and the DP is solved again, until the score stops improving;
   * *bootstrap expansion*: for 24 gene-bootstrap replicates (fixed seed), the DP
     is solved under the replicate's gene multiplicities and NNI closure is run
     on it. All new clusters are kept. The full-data DP is then solved again on
     the enlarged X.

   Candidate splits are maintained incrementally, and tripartition scores are
   cached, so solving the DP again is cheap.
4. **Consensus.** 800 Bayesian gene-bootstrap replicates (exponential gene
   weights) are solved by DP over the final X, re-using the cached per-gene
   scores. The output is the binary tree in X
   that maximises the sum of its clusters' bootstrap frequencies plus a small
   multiple of the weighted quartet score. This targets expected RF accuracy.
   On the dev set it lowered the mean normalised RF from about 0.147 (plain
   optimum) to about 0.143.
5. **Output.** The result is a fully binary unrooted tree written as `(0,A,B);`.

Runtime safety: internal wall-clock checks stop the search phases early (DP
solving is aborted after 95 s). An alarm-based emergency writer outputs the
last complete tree at 108 s. The dev cases take about 3–20 s on one core.
Memory use is well under 1 GB. All randomness uses fixed seeds offset by the
`SEED` environment variable (default 1), so runs are deterministic.

## Build and run

```sh
./build.sh                       # offline, needs only g++ (C++17)
./run.sh /abs/input/genes.tre /abs/output/pred.tre
```

`build.sh` always builds a portable binary `stree_base`. If the CPU supports
AVX2/FMA, it also builds `stree_avx2`. `run.sh` picks the best available binary
and builds on the fly if the binaries are missing. Logs go to stderr.

## Sources

The algorithmic ideas follow the published descriptions of ASTRAL (Mirarab et
al. 2014; Zhang et al. 2018), weighted ASTRAL (Zhang & Mirarab 2022) and ASTRID
(Vachaspati & Warnow 2015). All code here was written from scratch for this
task. No external code or data is included.
