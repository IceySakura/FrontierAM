# Species-tree inference from short gene trees

## Method
The program implements a weighted ASTRAL-style quartet dynamic program.

1. Parse the 50 estimated gene trees, suppressing degree-two serialization
   roots.  Internal labels are read as branch supports when present.
2. Build a search space of taxon clusters containing:
   - all singletons;
   - clusters induced by the input gene trees;
   - clusters from a neighbor-joining tree on blended internode/patristic
     distances;
   - clusters from a neighbor-joining tree on average internode distances;
   - clusters from 10 fixed-seed bootstrap neighbor-joining trees on internode
     distances.
3. Score a candidate species-tree node by the number of induced quartet trees
   shared with the gene trees.  A gene-tree node contribution is weighted by
   `support^2 * length / (length + beta)`, where `length` is the estimated
   internal branch length and `beta` is a small fixed constant (0.007
   substitutions/site in the submitted configuration).  This downweights
   unsupported and very short branches, which are dominated by estimation
   error and coalescent noise.
4. Dynamic programming over the cluster search space finds the highest-scoring
   binary tree under this quartet objective.  The outgroup `0` is attached at
   the root and the result is written as an unrooted Newick tree.

The implementation is self-contained C++17, deterministic (bootstrap seed 1),
and uses only the standard library.  No network, external data, or per-instance
tuning is used.

## Build and run
```sh
./build.sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`build.sh` compiles `species.cpp` with `g++ -O3 -std=c++17`.  Runtime on the
development samples is roughly 10-15 seconds per instance, well under the
135-second limit.

## Files
- `species.cpp`: Newick parser, distance summaries, bootstrap cluster
  generation, weighted quartet scoring, and dynamic program.
- `build.sh`: offline build script.
- `run.sh`: entry point accepting INPUT OUTPUT.
