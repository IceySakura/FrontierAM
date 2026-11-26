# Species tree inference from short gene trees

## Method

The program infers an unrooted 250-taxon species tree from 50 estimated gene
trees. It uses a weighted quartet-tripartition dynamic program motivated by the
multispecies coalescent and by ASTRAL-style search.

1. Parse the Newick gene trees, preserving topology, branch lengths, and reported
   internal supports. Very low-support internal edges (support < 0.1) are
   contracted by default.
2. Build several guide trees from the gene trees: NJst on average internode
   distances, a rank-based guide, a quartet-separation distance guide, a
   patristic-distance guide, and NJ/UPGMA/BIONJ guides on a blended distance.
3. Form a candidate cluster set from all nontrivial gene-tree clusters, the guide
   tree clusters, and clusters appearing in NNI neighbors of selected guide trees
   (NJst, quartet-separation, blended NJ, and BIONJ).
4. Score a candidate tripartition by counting gene-tree quartets with one taxon in
   each child cluster and two taxa outside the parent cluster. Contributions are
   weighted by `length^p * support^0.5`, where the internal gene-tree branch
   length and support describe the reliability of the corresponding clade. The
   exponent `p` is chosen adaptively from the observed gene-tree/guide-tree
   conflict, bounded between 0.2 and 0.5.
5. Assemble a fully resolved tree by dynamic programming over compatible
   candidate clusters, maximizing the total weighted tripartition score.

The outgroup label `0` is used to orient clusters and serialize the output. The
output is an unrooted binary tree written with a trifurcating root at the
outgroup; branch lengths and internal labels are omitted.

## Dependencies

* POSIX shell
* A C++17 compiler (g++ or clang++)

No external libraries, network access, package downloads, or randomness are
required. The program is deterministic.

## Build and run

```sh
./build.sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`build.sh` compiles `main.cpp` to `species_tree`. `run.sh` builds the binary if it
is missing, then runs it. The output is exactly one Newick tree ending in a
semicolon, with each input leaf label appearing once.

## Source

* `main.cpp`: Newick parser, guide-tree construction, candidate expansion,
  weighted quartet dynamic programming, and Newick writer.
* `build.sh`: offline one-step compilation.
* `run.sh`: entry point taking INPUT and OUTPUT paths.
