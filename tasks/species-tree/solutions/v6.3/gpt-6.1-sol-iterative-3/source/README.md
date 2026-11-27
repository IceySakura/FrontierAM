# Species-Tree Inference

This submission reads the estimated gene trees and writes one fully resolved
unrooted Newick species tree. Leaf labels are preserved as strings; output
branch lengths and internal labels are omitted.

## Build and Run

Dependencies are a POSIX shell and a C++17 compiler. GCC 12 is supported.
Everything builds offline using the standard C++ library. No Python packages,
external phylogenetics programs, network access, or reference data are needed.

```sh
./build.sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`build.sh` accepts no arguments and honors `CXX`, defaulting to `g++`.
`run.sh` starts the compiled executable without compiling or importing packages.
Inference is single-threaded and inherits the machine's CPU affinity. Random
choices use `SEED`, defaulting to 1. Diagnostics go to standard error.

## Method

Several neighbor-joining trees initialize the search, using mean internode,
support, branch-length, and square-root branch-length distances.

The objective is support-weighted gene-quartet agreement. Approximate Bayesian
support is mapped from its one-third baseline to a zero-to-one strength.
Unreported supports use a branch-length calibration fitted to reported
supports; near-zero unreported internal edges contribute no confidence.
The selected scheme uses the maximum support along each quartet's internal
path, represented by eight support intervals.

Nearest-neighbor interchanges and bounded-radius subtree pruning/reattachment
improve the objective. Exact colored-tree dynamic programs calculate move
scores without enumerating all four-taxon subsets. Per-locus contributions
are cached only in memory during the current instance. A monotonic-clock
budget leaves a margin below the stricter 135-second limit.

The source also contains alternative inference and weighting controls used
during development. The evaluator-facing configuration is fixed in `run.sh`.
The program reads no instance data other than its input file.

## Source

All code is original and contained in `infer.cpp`; no third-party source was
downloaded or vendored. The implementation uses established neighbor-joining
(Saitou and Nei, 1987), quartet-agreement/ASTRAL-style search principles
(Mirarab et al., 2014), and standard NNI and SPR topology moves.
