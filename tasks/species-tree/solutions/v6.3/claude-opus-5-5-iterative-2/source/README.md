# Species tree from 50 estimated gene trees (250 taxa)

## Files and running
* `st.cpp` is the complete solver: a single C++17 file using only the standard library, with no third-party code or data.
* `build.sh` (no arguments) compiles it offline: `g++ -O2 -march=x86-64-v3`, falling back to plain `-O2`.
* `./run.sh /abs/genes.tre /abs/pred.tre` runs the solver. The configuration variables exported in `run.sh`
  are also the program's built-in defaults. If the solver fails for any reason, `fallback.py`
  (Python 3, standard library only) writes the first gene tree's topology, so a valid tree is always written.
* Single-threaded and deterministic. No randomness is used in the default configuration; `SEED` defaults to 1.
  Logs go to stderr.
* Typical cost is 10-20 s and under 50 MB per instance on one core. Internal time guards keep even
  pathological inputs well under the limit.

## Method
Own implementation. It combines ideas from ASTRAL-III (exact dynamic programming over a cluster search space),
weighted ASTRAL / wASTRAL-hybrid (quartets weighted by branch support and terminal branch length),
ASTRID (internode distances) and neighbour joining.

1. **Parsing.** Gene trees are treated as unrooted, with degree-2 roots removed. For computation they are
   re-rooted at taxon `0`.
   * Support is normalised to p = (s - 1/3)/(2/3), clamped to [0,1].
   * A missing support is imputed from the branch length with a monotone (isotonic, binned) length-to-support
     map fitted on the same input file. Without any supports, a length-based pseudo-support is used.
2. **Objective.** Each gene-tree quartet ab|cd contributes to every species tree that displays it, with weight
   `w = S(q) * exp(-0.2 * T(q) / Lbar_g) + 0.2 * c * S(q)`, where:
   * S(q) is the maximum normalised support on the quartet's internal path, discretised with 3 contraction
     levels (thresholds 1/6, 1/2, 5/6).
   * T(q) is the total length of the four terminal paths of the quartet in gene g.
   * Lbar_g is the mean branch length of gene g, so the decay does not depend on that gene's rate.
   * c is a fixed calibration constant that sets the plain support-weighted term to 20% of the decayed term's
     magnitude.

   Decaying with terminal length emphasises quartets whose leaves are close to the branch in question,
   which are less affected by gene-tree estimation error. The plain term keeps distant quartets informative.

   For a species-tree tripartition the score is computed exactly by per-gene bottom-up/top-down dynamic
   programmes on the contracted gene trees, at O(n) per gene. The formula was verified against brute-force
   quartet enumeration.
3. **Search.** An ASTRAL-style exact DP maximises the total score over trees whose clusters (rooted at `0`)
   lie in a search space X.
   * X = gene-tree clusters with support >= 0.2, plus the clusters of 4 neighbour-joining trees. These are
     built on average gene-tree distances: internode distances with low-support edges contracted at p >= 0.05
     and p >= 0.5, support-weighted internode distances, and rate-normalised patristic distances.
   * X is then repeatedly enlarged with the NNI neighbourhood of the incumbent tree and the DP is re-run, until
     the score stops improving.
   * The NJ seeds are essential. With gene-tree clusters alone, the DP starts far from good trees.
4. **Time guard.** If the projected cost of hybrid scoring is too high, the run switches consistently to the plain
   support-weighted quartet score. NNI expansion also stops after 60 s.
5. **Output.** A fully resolved unrooted Newick tree, trifurcating at the node next to taxon `0`, with the input
   labels unchanged.
