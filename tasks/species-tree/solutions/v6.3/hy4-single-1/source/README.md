# Species-tree inference from 50 estimated gene trees

## Method

1. Parse the 50 Newick gene trees (250 leaves, taxon `0` is the outgroup). Branch
   lengths and Bayesian-support labels are read.
2. Suppress the degree-2 Newick serialization root and contract internal gene-tree
   edges whose support is below a threshold (and, optionally, whose branch length
   is tiny). This turns poorly resolved parts of the estimated gene trees into
   polytomies instead of arbitrary resolutions.
3. For every gene tree, root it at the outgroup. Every rooted triple of ingroup
   taxa `(a,b)|c` whose gene-tree LCA of `a,b` excludes `c` casts a vote; votes
   are weighted by the support of the corresponding gene-tree edge. The votes are
   aggregated as an O(#gene-tree nodes) query `h(S1,S2)`: the number of votes
   consistent with a species-tree node whose two children are `S1` and `S2`.
   Two such vote forests (two contraction settings), each normalised, are summed
   to give the objective. This is the quartet/triplet support objective used by
   ASTRAL-like methods, restricted to quartets containing the outgroup.
4. Build starting trees by neighbor joining on average topological and patristic
   distances, and by a support-weighted greedy split consensus.
5. Hill-climb with NNI swaps (each swap changes only two nodes' scores), take a
   weighted consensus of the local optima, and run iterated local search with
   random NNI perturbations. All randomness is seeded with 1.
6. Write the best tree as an unrooted binary Newick tree.

## Files

* `species_tree.cpp` — complete single-threaded C++17 source.
* `build.sh` — compiles it offline (`g++ -O3 -std=c++17 -mpopcnt`).
* `run.sh` — `./run.sh INPUT OUTPUT`; builds on demand if needed.
* `pipeline.py`, `lib2.py`, `trip.py` — the earlier Python/numpy prototype,
  kept as reference; not used by `run.sh`.

Only the C++ standard library and a C++17 compiler are needed at runtime/build.
No network, no third-party packages, no cross-instance state.

## Development

Validated with `samples/check_rf.py` on the two provided development cases:
normalized RF 0.121 and 0.146 (30 and 36 missed splits). Runtime is about 55 s
per case, comfortably inside the limit. The program prints progress to standard
error and writes exactly one tree to the output path. Optional third/fourth
command-line arguments (mode / time budget) are used only for development
tuning; `run.sh` uses the defaults. The `DUMP_STARTS`, `DUMP_CONS` environment
flags are debug-only.
