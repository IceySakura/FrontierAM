# Species-Tree Inference

This is a self-contained, single-threaded C++17 implementation. It needs a
POSIX shell, `g++`, and the ordinary C++ standard library. It does not use the
network, Python packages, external phylogenetics executables, or agent tools.

## Build And Run

```sh
./build.sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`build.sh` takes no arguments and builds the `infer` executable offline.
`run.sh` reads only its first argument and writes one fully resolved unrooted
Newick tree to its second argument. Diagnostics go to standard error.
All randomized search and resampling use `SEED`, defaulting to `1`.

## Method

The program parses the gene trees, including optional internal supports and
branch lengths. It builds support-aware distance trees with neighbor joining
and balanced minimum evolution, and weighted-quartet trees with
nearest-neighbor interchange, subtree pruning and regrafting, and a
candidate-clade dynamic program. Locus resampling provides additional
candidate trees.

An internal edge's reliability is calibrated relative to the one-third
uninformative support level. A quartet's central path combines this evidence
without counting unsupported, zero-length resolutions as confident evidence.
Optional missing supports are handled using whether the reported branch
length carries positive resolution. Both ordinary and more conservative
support calibrations are considered.

Candidate splits are compared using quartet support, locus-to-locus
stability, branch balance, mutation-length summaries, and agreement among
the candidate trees. A fixed global branch calibration orders compatible
splits. Remaining multifurcations are resolved using the reference quartet
tree, producing a binary unrooted result.

The coefficients in `calibration.hpp` are global parameters calibrated on
the public development data using case-level cross-validation. They do not
encode taxon labels, input fingerprints, file names, case IDs, or true-tree
splits. At runtime every instance uses only its supplied input and these
same parameters; there is no cross-instance storage or communication.

The implementation is original. Its algorithmic sources are the standard
neighbor-joining, Pauplin balanced-minimum-evolution, quartet-agreement,
and ASTRAL-style restricted-clade dynamic-programming ideas. No third-party
source, phylogenetics binaries, or precomputed species trees are included.

## Files And Limits

`infer.cpp` contains parsing, distances, quartet scoring, and topology search.
`ensemble.hpp` contains candidate construction, branch statistics, and binary
tree assembly. `calibration.hpp` contains the shared numerical parameters.

Inference is single-threaded and keeps the inherited CPU affinity. A
wall-clock guard retains a valid earlier tree if refinement approaches its
time allowance. All working tables are in process memory; no temporary
files or external services are needed.
