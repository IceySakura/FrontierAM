# Species tree inference

## Build and run

```sh
cd /workspace/submission
./build.sh
SEED=1 ./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

The build needs a POSIX shell, a C++17 compiler (default `g++`), and its
standard library on Linux amd64. It builds the executable for the local
machine. There are no external packages or downloads. Python is only
needed for the optional development check below.

The solver reads the gene-tree file and writes one fully resolved,
unrooted Newick tree on the same labels. It uses no files other than
the supplied input and output, runs on one thread with the inherited
CPU affinity, and defaults to `SEED=1`. A wall-clock guard ends optional
search stages before the 135-second instance limit.

## Method

The solver builds candidate trees using neighbor joining, BioNJ, and
local distance and quartet searches. Distance summaries use several
signals: supported internodes, rate-normalized branch lengths,
lower-tail distances across loci, and approximate clock ranks.
The balance between these signals depends on observed quartet agreement,
locus rate variation, and repeatability across disjoint sets of loci.

For each gene quartet, its topology is weighted by the confidence along
its internal path. This discounts poorly supported, near-zero resolutions
while still accepting edges without reported support. Dynamic programming
computes these scores for groups of taxa without enumerating all taxon
quartets. The solver explores several starts and whole-locus resamples,
then assembles a compatible set of splits. Small regions of weak support
are re-resolved by a second dynamic program, with votes pooled across
plausible resolutions rather than relying on a single local winner.

All code was implemented for this submission. The standard starting
methods are from Saitou and Nei (neighbor joining, 1987) and Gascuel
(BioNJ, 1997). Quartet concordance follows the species-tree summary
criterion used in ASTRAL (Mirarab et al., 2014); this implementation
does not use the ASTRAL software.

## Sample check

```sh
./run.sh /workspace/samples/cases/001/genes.tre /tmp/pred.tre
python3 /workspace/samples/check_rf.py /workspace/samples/cases/001/truth.tre /tmp/pred.tre
```
