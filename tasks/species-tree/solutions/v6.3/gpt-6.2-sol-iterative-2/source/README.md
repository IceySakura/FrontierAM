# Species-tree inference

## Build and run

Requires a POSIX shell and a C++17 compiler on Linux amd64. The quartet kernel
uses AVX2 and FMA, available on the evaluation host. `CXX` defaults to `c++`.
No external libraries, data files, or network access are required.

```sh
./build.sh
SEED=1 ./run.sh /absolute/path/genes.tre /absolute/path/pred.tre
```

Each line of the input is one gene tree. The program writes a single unrooted,
fully resolved Newick tree with the same leaf labels and no branch annotations.
It runs in one thread and defaults to seed `1` if `SEED` is not set. Diagnostics
go to standard error.

## Method

1. Build a BIONJ guide from locus-normalized patristic distances and
   confidence-weighted internode distances.
2. Score the three resolutions around a candidate edge using weighted quartets.
   The weight discounts unreliable central paths and long pendant paths;
   missing support is replaced with a length-based estimate. A tree dynamic
   program sums these weights without enumerating all quartets. The scoring
   variants differ in how they weight branches, loci, and representatives of
   large clades. Some weights are estimated from agreement among the input loci.
3. Refine the guide with nearest-neighbor interchanges, bounded subtree
   relocations, and exact resolution of small connected regions of weak edges.
   Four passes use different scoring settings. One also resamples loci inside
   weak regions and incorporates rooted triplet and projected split evidence.
4. Combine the resulting trees by maximizing the weighted sum of split
   frequencies over compatible clades and child partitions.

The implementation is original. Method references: neighbor joining
(Saitou and Nei, 1987); BIONJ (Gascuel, 1997); internode-distance species-tree
inference in ASTRID (Vachaspati and Warnow, 2015); and weighted quartet evidence
in wASTRAL (Zhang and Mirarab, 2022).

## Check a sample

```sh
sample=/workspace/samples
./run.sh "$sample/cases/001/genes.tre" "$PWD/pred.tre"
python3 "$sample/check_rf.py" "$sample/cases/001/truth.tre" "$PWD/pred.tre"
```
