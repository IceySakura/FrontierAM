# Species tree from 50 short-locus gene trees (250 taxa)

## Method
`stree.cpp` is a self-contained C++17 program. It uses only the standard library, with no third-party code. The method is a re-implementation of ideas from the published methods
ASTRAL-III (Zhang et al. 2018), weighted ASTRAL / wASTRAL (Zhang & Mirarab 2022, MBE) and ASTRID/NJst
(Vachaspati & Warnow 2015; Liu & Yu 2011).

1. **Parse** all Newick gene trees. Degree-2 roots are suppressed and every tree is re-rooted at the outgroup `0`.
   Polytomies, if any, are resolved with zero-support edges.
   Missing approximate-Bayes supports are imputed from branch length (mean support of the 50 edges with the nearest lengths in the same input).
   In the samples they are almost all zero-length edges, so they become about 1/3, i.e. uninformative.
2. **Quartet weights (support weighting, as in wASTRAL).** Each support is normalised to `s' = ((s-1/3)/(2/3))^3`.
   A gene-tree quartet `ab|cd` whose internal path has edges `e1..ek` gets weight `1 - prod(1 - s'_ei)`.
   The alternative topologies get 0.
   The weighted score of a species-tree tripartition is computed exactly in O(n) per gene tree, using an up/down tree dynamic program with 4-wide SIMD vectors over the three parts.
   I checked this against brute-force quartet enumeration on small random inputs, and the scores match exactly.
3. **Candidate bipartitions X.** These come from:
   - all gene-tree clusters;
   - neighbour-joining trees on average internode distances (unweighted and support-weighted);
   - 100 NJ trees built on bootstrap resamples of the genes (deterministic RNG, seeded from `SEED`, default 1).
4. **Exact DP (ASTRAL algorithm).** The DP runs over X to find the tree that maximises the total weighted quartet support.
5. **Local refinement.** All NNI and radius-3 SPR neighbour clusters of the current best tree are added to X and the DP is rerun.
   This repeats until the score stops improving or 75 s have passed.

Typical runtime is about 8 s per instance (AVX2 build) and under 1 GB of memory.
The worst case I tested, 50 label-shuffled gene trees, took 27 s.

## Files
- `stree.cpp`: complete source.
- `build.sh`: offline build. It produces `stree_avx2` (`-mavx2 -mfma`) and `stree_generic` (plain x86-64), using g++ (tested with GCC 12).
- `run.sh INPUT OUTPUT`: picks the AVX2 binary if the CPU supports it.
  If the primary run fails without output, it retries with the generic binary.
  As a last resort it writes a caterpillar tree, so the output is always one valid Newick tree.
  Logs go to stderr.

## Running
```sh
./build.sh
./run.sh /abs/path/genes.tre /abs/path/pred.tre
```
The program is deterministic. `SEED` (default 1) seeds the bootstrap resampling.
