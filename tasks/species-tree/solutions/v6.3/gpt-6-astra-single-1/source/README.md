# Species-Tree Inference

This is an original, self-contained, single-threaded C++17 implementation.
Building requires a POSIX shell and `g++`. Running requires only the compiled
executable and the system C++ runtime. There are no Python packages, network
requests, downloaded dependencies, or external phylogenetics programs.

Build once with `./build.sh`, then run:

```sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`build.sh` takes no arguments and builds offline for Linux amd64 without
machine-specific instruction-set requirements. `run.sh` takes exactly the input
and output paths. The result is one fully resolved, unrooted Newick tree.
Diagnostics go to standard error.

## Method

1. Parse unrooted gene trees, suppressing a degree-two serialization root.
   Leaf names are preserved as strings.
2. Construct multiple neighbor-joining starts from topological, support-weighted,
   and rate-normalized branch-length distances.
3. Refine these trees using exact weighted quartet comparisons and nearest
   neighbor interchanges. For each quartet, internal-path confidence is
   `1 - product(1 - s)`, with support calibrated from the reported [0,1] values.
   Each internal edge on an external quartet path contributes a factor of
   `exp(-0.4)` (approximately 0.67),
   emphasizing more local gene-tree evidence.
4. Combine a fixed family of support calibrations and gene reliability weights
   to assess the stability of inferred splits.
5. Use dynamic programming to select a compatible binary tree maximizing
   aggregate split frequency.

Unreported supports are not treated as reported zeros. When reports are
available, a monotone within-gene branch-length calibration imputes only missing
values. With no reported supports, a substitution-length confidence model is
used; all-zero-length trees fall back to topology.

The search has an internal wall-clock budget and reserves time for consensus
and writing. Trees are structurally checked before output. The implementation
uses one thread and inherits CPU affinity; it does not select a CPU number.

## Reproducibility And Sources

Random choices use `SEED`, defaulting to 1. All estimates and resampling use only
the current input. No cross-instance files, caches, reference trees, or sample
identifiers are used.

The default uses the full-data calibration ensemble. Optional Bayesian gene
resampling can be enabled with `BOOTSTRAPS` set to a positive replicate count;
it is not needed by `run.sh INPUT OUTPUT`.

The methodological ingredients are standard neighbor joining, internode
distances, quartet amalgamation, nearest-neighbor interchange, Bayesian
bootstrap resampling, and compatible-split dynamic programming. The parser,
scorers, searches, consensus implementation, and all other source code were
written for this submission; no third-party source is bundled.
