# Species-Tree Inference

## Build and Run

Dependencies: the supplied Linux x86-64 environment, a C++17 compiler
(GCC 12 is used), a POSIX shell, and coreutils (`env`, `mktemp`, `rm`, `cp`).
There are no downloads,
Python dependencies, external phylogenetics programs, or agent-tool calls.

```sh
./build.sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

The build script takes no arguments. The executable and all source are local
to this directory. Output is one fully resolved, unrooted Newick tree with
the original taxon labels, without branch lengths. Diagnostics go to stderr.

## Method

The solver parses the input as unrooted graphs and constructs a starting
tree from averaged, rate-normalized gene distances using neighbor joining.
It then performs confidence-weighted quartet NNI and bounded subtree
prune-and-regraft searches.

Three views are combined: two support-based estimates with different weak
pendant-path discounts and balanced quartet contributions, and a normalized
branch-length-weighted estimate. The final dynamic program maximizes
agreement with their compatible splits. Ties prefer the first view and then
a canonical taxon-set ordering.

Unreported support is not treated as full confidence: missing supports on
near-zero internal branches are treated as maximally uncertain; other missing
supports are estimated from reported branches of the same gene. Randomized
search schedules start from `SEED`, defaulting to 1.

## Resources and Independence

Components run serially on the inherited CPU affinity, without threads.
Their wall-clock search caps total 120 seconds, leaving time for consensus,
file output, and process startup within the 135-second budget. Temporary
predictions are created in a unique directory and removed on exit. A
completed component remains available as a fallback.

All caches are process-local and use only the current input. There are no
cross-instance files, stored answers, or input-identity lookups.

## Source

`main.cpp` is an original implementation of standard neighbor-joining,
quartet-agreement, resampling, and compatible-cluster dynamic-programming
ideas. No third-party source code is bundled.
