# Species-tree inference from short-gene estimated gene trees

Input: `genes.tre` — one Newick gene tree per line (50 loci), each an unrooted binary
tree over the same 250 leaf labels (`0` = outgroup, `1..249` = ingroup). Branch
lengths and internal support labels are optional and are used only as described below.

Output: a single unrooted binary Newick tree over exactly the same 250 labels,
written to the second command-line argument, terminated by `;`.

```sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

## Method

The shipped default is **ASTRID**: average the internode distances over the gene
trees and run neighbour joining on the result. A quartet-score NNI refinement
stage is implemented but disabled by default because of a counting bug (see
*Known limitation* at the end).

1. **Parse** the gene trees. Each is converted to an internally rooted
   representation (root at a trifurcation) with a post-order node list, so all
   subtree statistics can be accumulated in one linear pass. The third child of
   the written root is handled explicitly; a degree-two serialisation root is
   suppressed so every gene tree contributes exactly `2n-3` edges.

2. **Starting tree (ASTRID).** For every gene tree, the internode (edge-count)
   distance between all pairs of taxa is computed by BFS from each leaf, and the
   distances are averaged over loci. Neighbour joining on that average matrix
   gives the initial species tree. This is the distance-based coalescent
   estimator of Vachaspati & Warnow; it is fast and robust to noisy gene trees.

3. **Exact weighted quartet objective.** For a bipartition `L|R` of the taxa the
   score is

   ```
   w(L|R) = sum over gene trees g  sum over edges e' of g
              sup(e') * ( C(a,2)*C(d,2) + C(c,2)*C(b,2) )
   ```

   where, for edge `e'`, `a = |below(e') ∩ L|`, `b = |below(e') ∩ R|`,
   `c = |above(e') ∩ L|`, `d = |above(e') ∩ R|`. This counts exactly the quartets
   `{x,y} ⊂ L, {z,w} ⊂ R` that `g` resolves as `xy|zw` — but **not** each once,
   which is the bug described in *Known limitation*. The
   cost is `O(m·n)` per bipartition (≈50·250 operations), so **the total quartet
   score of a whole 250-taxon tree — the sum over its 247 internal edges — costs
   only a few milliseconds**. No `O(n⁴)` quartet enumeration is performed.

   `sup(e')` is the reported approximate Bayesian support of that gene-tree edge,
   used as a weight (default; see *Options*). Edges with no reported value get
   weight 1. Very weakly supported gene-tree edges — which are numerous here,
   since 200 bp loci produce many 0.333-support edges — are thereby down-weighted
   instead of voting with full strength. This weighting is in the spirit of
   ASTRAL-Pro's use of branch support/lengths.

4. **NNI hill climbing.** On an unrooted binary tree an NNI move across internal
   edge `(u,v)` changes exactly one bipartition: `A∪B|C∪D` becomes `C∪B|A∪D`
   (or `A∪D|C∪B`). All other splits — including those of the four incident
   subtrees — are preserved as leaf bipartitions. So the score delta of a move is
   a single new split evaluation. The search applies every candidate NNI
   (2 per internal edge, 247 edges), evaluates the delta in `O(m·n)`, and takes
   the steepest improving move, repeating until no move improves the score.
   Split scores are memoised in a `NN×NN` table with an invalidation stamp that
   is bumped only for the edges actually rewired, so unaffected splits keep their
   cached values across moves.

5. **Perturbation restarts (optional).** After convergence the best tree can be
   randomly perturbed by a handful of NNI moves and re-climbed; the best scoring
   tree over all restarts is kept. Off by default (`ST_RESTARTS=0`) so the run is
   fully deterministic and finishes well inside the budget.

Because step 3 makes a full tree score essentially free, the search is not
restricted to bipartitions present in the gene trees (ASTRAL's set `X`); it can
move anywhere in tree space.

## Complexity / resources

* Time: distance matrix `O(m·n²)`; NJ `O(n³)`; each NNI round `O(n · m · n)`.
  A full run on 250 taxa × 50 genes converges in tens of seconds on one core.
* Memory: a few tens of MB (the `NN×NN` score/stamp tables are ≈2 MB each).
* Single threaded, no network, no dependencies beyond a C++ compiler.

## Build and run

```sh
./build.sh                       # optional; produces ./species (g++ -O3 -std=c++11)
./run.sh INPUT OUTPUT            # runs ./species INPUT OUTPUT
```

`run.sh` builds on demand if the binary is missing, so either order works.
Logs go to stderr; only the tree is written to `OUTPUT`.

A wall-clock budget guards the search: the program stops hill climbing and writes
the best tree found once the budget elapses, so it always produces output in time.

### Environment variables (all optional)

| Variable | Default | Meaning |
|---|---|---|
| `SEED` | `1` | RNG seed for perturbation restarts (unused when restarts are off) |
| `ST_BUDGET` | `0` | seconds of NNI hill climbing after the distance tree (0 = off, see *Known limitation*) |
| `ST_WEIGHT` | `1` | gene-tree edge weight: `0` none, `1` support, `2` `max(0,(s-1/3)*1.5)`, `3` `s²`, `4` indicator `s≥0.5` |
| `ST_RESTARTS` | `0` | number of perturbation restarts after the first climb |
| `ST_VERBOSE` | `0` | progress logging to stderr |

The defaults are the submission settings; the evaluator only ever calls
`./run.sh INPUT OUTPUT`.

## Dependencies and provenance

* Written from scratch for this task; no third-party code is bundled or fetched.
* Build requires only `g++` (any C++11 compiler) and the standard library.
* Run requires only POSIX `sh` and the built binary.
* Algorithms implemented: neighbour joining (Saitou & Nei 1987), the ASTRID
  average-internode-distance estimator (Vachaspati & Warnow 2015), and the
  weighted-quartet species-tree objective of ASTRAL (Mirarab et al. 2014)
  optimised by NNI hill climbing (disabled by default).

## Known limitation

The NNI hill-climbing stage (steps 3–4) uses a bipartition weight that sums over
*every* gene-tree edge separating a quartet's two pairs. A quartet's induced
middle edge can correspond to a **path** of several gene-tree edges, so such
quartets are counted more than once; the over-count varies between topologies, so
the objective is not the true quartet score and optimising it degrades accuracy
(measured on the dev cases: 0.91 vs 0.14 for the distance tree). It is therefore
**disabled by default** (`ST_BUDGET=0`) and the program returns the ASTRID/NJ
tree. A correct `O(m·n)` formulation exists — count each quartet at
`h = LCA_g(x,y,z,w)` via the per-node child distribution — but was not completed
within the time budget.

Default behaviour, verified on the two dev cases (250 taxa, 50 genes, ~150 ms):

| case | missed splits | normalised RF |
|---|---|---|
| 001 | 34 / 247 | 0.1377 |
| 002 | 82 / 247 | 0.3320 |
