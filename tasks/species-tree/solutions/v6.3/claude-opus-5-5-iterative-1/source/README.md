# Species tree from estimated gene trees (weighted-quartet DP)

Single C++17 source file `stree.cpp`; no third-party code or libraries (only g++ and the C++
standard library). `fallback.py` (Python 3 standard library only) is a last-resort safety net.

* Build: `./build.sh` compiles `stree` (x86-64-v2) and `stree_generic` (portable). It never
  fails the build; `run.sh` compiles on the fly if no binary exists.
* Run: `./run.sh /abs/genes.tre /abs/pred.tre` (logs go to stderr; typically 2-4 s,
  < 10 s even for 50 completely conflicting gene trees; memory < 100 MB).
  If the main binary fails, `run.sh` retries with the portable build and finally writes the
  topology of the first input gene tree, so a valid tree is always produced.
* Randomness (gene resampling for extra candidate clusters) is seeded from `SEED` (default 1).
* Developer options (`./stree in out -lam 4 -alpha 0.5 ...`) exist for experiments only; the
  defaults are the submitted configuration.

## Method (own implementation of published ideas)

1. **Objective** – maximise the total weighted quartet support of the species tree, as in
   ASTRAL (Mirarab et al. 2014; Zhang et al. 2018), with gene-tree-uncertainty weighting in the
   spirit of weighted ASTRAL (Zhang & Mirarab 2022). A gene-tree quartet ab|cd with internal
   path P gets weight `(1 - prod_{e in P} (1 - w_e)) * prod_{e on its four terminal paths} h_e`:
   * `w_e` = geometric mean of the normalised aBayes support `(s - 1/3)/(2/3)` (clipped to
     [0,1]) and a length-based confidence `1 - exp(-L'/0.01)`; an edge without support uses
     the length-based value only (or 1 if the gene tree has no branch lengths at all);
   * `h_e = exp(-4 L')` discounts quartets whose leaves are far from the internal path;
   * `L'` = branch length / (mean branch length of that gene tree) * 0.03, which removes
     gene-specific rate differences.
   The weighted score of a species tripartition is computed exactly in O(#relevant nodes)
   per gene tree with a two-pass dynamic programme over the gene tree (discounted pair sums,
   balanced/caterpillar decomposition of quartets; subtrees containing only the largest colour
   class use precomputed values). It was checked against brute-force quartet enumeration.
2. **Search** – exact DP over a constrained set of clusters (species tree rooted at the
   outgroup `0`, or the first taxon if absent): all gene-tree bipartitions, splits of
   neighbour-joining trees built from support-weighted internode distances (ASTRID-like,
   Vachaspati & Warnow 2015) on all genes and on 30 gene-resampled replicates, then repeated
   NNI-neighbourhood expansion of the DP optimum until the score stops improving.
3. Output: the optimal binary tree, written with a degree-2 root on the outgroup edge.

Parameters (support/length blend, discount rate 4, scale 0.01) were chosen on the two
development cases with gene-resampled replicates; no instance-specific information is used.
