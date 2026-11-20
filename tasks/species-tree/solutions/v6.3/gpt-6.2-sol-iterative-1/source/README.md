# Species-tree inference

## Build and run

From this directory:

```sh
./build.sh
SEED=1 ./run.sh /absolute/path/genes.tre /absolute/path/pred.tre
```

The build needs a POSIX shell and a C++17 compiler supporting GNU vector
arithmetic (GCC by default). It builds offline for the running machine.
There are no third-party libraries or data files. The program writes one
unrooted binary Newick tree, preserving all input leaf labels and omitting
edge lengths and internal labels.

## Method

1. Build neighbor-joining and BioNJ starting trees from normalized,
   trimmed summaries of gene-tree path lengths and internode distances.
   Missing edge support is handled using branch-length evidence.
2. Refine the starting trees by nearest-neighbor interchange. On each
   gene tree a traversal sums the quartet evidence for the three local
   arrangements without enumerating individual quartets. Branch lengths,
   edge support, and agreement between loci inform the weights.
3. Pool the refined trees, distance-only trees, and locus resamples. A
   dynamic program selects compatible clades that maximize the total
   split score. Distance-based estimates receive more weight when locus
   branch-length scales are comparable. Recurrent, well-supported gene
   splits provide a small additional vote.

The code is self-contained and single-threaded. Resampling starts from
`SEED` (default `1`). Only the supplied input is used, and no state is shared
between runs.

Method sources: neighbor joining (Saitou and Nei, 1987), BioNJ (Gascuel,
1997), and quartet agreement inspired by ASTRAL and weighted ASTRAL. The
implementation does not require those programs.
