# Species-Tree Inference

A self-contained, single-threaded C++17 estimator for the supplied gene-tree
format. It uses no external phylogenetics software or runtime data.

## Build and Run

Dependencies: POSIX shell and `g++` with C++17 support. Everything builds
offline; no package installation or network access is required.

```sh
./build.sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`build.sh` takes no arguments. `run.sh` reads the first path and writes exactly
one fully binary unrooted Newick tree to the second path. Original leaf labels
are preserved, each exactly once. Branch lengths and internal labels are
omitted from the output. Diagnostics go only to stderr.

## Method

The entry configuration constructs a support-weighted internode-distance
neighbor-joining tree and refines it with weighted-quartet NNI moves.
The source also supports locus resampling followed by an RF-optimal clade
consensus; `run.sh` specifies whether this aggregation is selected.

For a reported support `s`, the edge-retention probability is
`max(0, min(1, (3*s-1)/2))`. Missing support is not parsed as a reported value:
near-zero unreported edges are contracted, and positive-length unreported
edges use unweighted topology. A quartet's support weight is the probability
that at least one edge of its internal path is retained.

Quartets are additionally downweighted by their total pendant length. The
attenuation combines a per-locus relative rate and an absolute-length term.
Exact per-locus normalization preserves the original total support-weighted
quartet mass. Lightly balanced within-clade leaf weights reduce redundancy
from dense subclades.

The quartet counts are computed by a tree dynamic program, not by enumerating
all four-taxon sets. Local optimization is deterministic. Locus resampling and
other stochastic methods use `SEED` with default value `1`. Inference checks a
125-second wall-clock guard. No CPU affinity index or additional threads are
used.

## Source and State

All C++ source in this directory was newly implemented for this submission.
It uses standard neighbor-joining and quartet-summary inference ideas; no
third-party source, pretrained model, cached answer, or sample tree is bundled.
`main.cpp` also contains experimental inference variants, and `mpest.hpp`
contains a triplet-likelihood refinement that is not used by the default entry.

Every process is independent. All calculation caches are in memory and are
derived only from that invocation's input. No cross-instance state is read or
written.
