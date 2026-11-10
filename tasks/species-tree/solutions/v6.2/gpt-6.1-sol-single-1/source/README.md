# Species-Tree Inference

## Build and Run

Requirements: Linux amd64, POSIX shell, and a C++17 compiler (`g++` by default).
Only the C++ standard library is used. No downloads, Python packages, agent
tools, or external phylogenetics programs are required.

```sh
./build.sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`build.sh` accepts no arguments. Set `CXX` to select another C++17 compiler.
`run.sh` also builds automatically if the executable is absent.
The program reads only its input file and writes exactly one Newick tree to
its output file. Diagnostics go to standard error.

## Inference

The input is parsed as unrooted gene trees. A degree-two serialization root
is suppressed. Quoted leaf names, scientific-notation branch lengths,
comments, and optional internal support labels are supported.

Each gene supplies a matrix of leaf-to-leaf path distances. Pendant edges
have weight one. Internal edges with reported support `s` have weight

```text
0.05 + 0.95 * max(0, min(1, 1.5*s - 0.5)).
```

An absent support label gives an internal edge weight of one. Thus a
reported support near one third contributes little without being confused
with a missing report. Branch lengths are parsed and validated but are not
used, avoiding sensitivity to locus-specific substitution rates.

Each locus's distances are divided by their mean before averaging over
loci. This prevents an unusually resolved or unresolved gene tree from
dominating the aggregate.

A row-sum-corrected neighbor-joining criterion then constructs the species
tree. For `m` active groups, it joins the pair with the largest value of

```text
(m - 2) * distance(i,j) - row_sum(i) - row_sum(j).
```

Larger corrected separation provides stronger evidence for the split
shared by the joined groups. Distances to the new group use the
neighbor-joining update, and the final three groups form the unrooted
serialization root. The result is fully binary and preserves every input
leaf exactly once.

There is no randomness, threading, persistent state, input-specific
parameter table, or use of development truths at runtime. The program
therefore gives identical results for every `SEED` value.

## Source

`solver.cpp` is an original implementation using support-weighted path
distances and the classical neighbor-joining distance update. No third-party
code is bundled.

At 250 taxa and 50 genes, storage is quadratic in the number of taxa and
the inference requires cubic time for joining plus gene-distance traversal.
The implementation uses one thread and does not alter CPU affinity.
