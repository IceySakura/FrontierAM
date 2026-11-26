# Species tree inference from short gene trees

## Method
The program infers an unrooted species tree from the 50 estimated gene trees by
maximizing the quartet similarity score (the ASTRAL objective): the number of
quartet topologies induced by the gene trees that are also induced by the
species tree. By default each gene-tree quartet counts equally; reported
Bayesian supports and branch lengths are used only to build alternative
starting trees.

Search starts from several fast estimates (neighbor joining on average
internode distances, neighbor joining on average branch-length distances, and
a greedy compatibility consensus) and then performs exact quartet-score NNI
hill climbing. An NNI move only changes quartets that sample one leaf from
each of the four subtrees around the moved edge, so its score delta is
computed exactly by a single pass over all gene-tree edges using bitset
intersections. Iterated local search with random NNI, leaf-regraft, and
subtree-regraft perturbations escapes local optima. The final tree is selected
by unweighted quartet score.

## Dependencies
* POSIX shell
* g++ (C++17) with hardware popcount (`-mpopcnt`)
* No external libraries, no network, no package downloads.

## Build
```sh
./build.sh
```
Builds the single binary `species` (well under the 300 s offline build limit).

## Run
```sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```
`run.sh` builds the binary if it is missing, then runs it. The program uses a
fixed internal time budget (about 130 s) and honors `SEED` for deterministic
randomness. It writes exactly one Newick tree covering all input leaves.

## Source
* `src/species.cpp` — complete implementation (Newick parsing, gene-tree
  split extraction, distance/NJ starts, quartet scoring, NNI search).

## Notes
* Leaf labels are treated as strings and preserved exactly.
* Only topology is output; branch lengths and internal labels are omitted.
* All computation is single-threaded and uses well under 4 GiB.
