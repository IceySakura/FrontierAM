# Species-Tree Inference

## Build and Run

```sh
./build.sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

Requirements: Linux amd64, a POSIX shell, and GCC with C++17 support.
The implementation uses only the C++ standard library. It needs no network,
package installation, Python runtime, external phylogenetics software, or
agent tools. Build once; each invocation starts independently.

The output is exactly one fully binary, unrooted Newick tree. Leaf labels
are retained as strings; branch lengths and internal annotations are omitted.
Diagnostics go to standard error. No CPU affinity is changed and no threads
or child processes are created by the inference executable.

## Method

1. Parse the gene trees and suppress any degree-two serialization root.
2. Initialize by neighbor joining on mean support-weighted internode distances.
3. Improve quartet agreement by nearest-neighbor interchanges. A gene
   quartet's weight is `1 - product(1 - r_e)` along its internal path, with
   `r_e = max(0, min(1, (3*s_e - 1)/2))`. Missing supports are handled using
   branch lengths: edges of length at most `1e-5` are uninformative.
   A bounded local-information correction reduces differences in the
   fraction of resolved quartets between loci.
4. Infer up to 60 locus-bootstrap trees, mixing full-data and independently
   initialized starts and three support calibrations.
5. Use dynamic programming to maximize the sum of split frequencies over
   compatible binary trees in the ensemble's cluster set.

Randomness starts at `SEED=1` by default. Search uses a monotonic-clock
deadline of 122 seconds. Consensus has a separate 130-second guard and can
return the already valid full-data tree, leaving margin under the stricter
135-second instance limit.

## Source and Tests

`main.cpp` contains the complete inference implementation. `self_test.cpp`
checks quartet scores against exhaustive enumeration, support calibrations,
locus weights, the information correction, quoted-label Newick round trips,
and the consensus dynamic program against exhaustive six-taxon searches.

```sh
g++ -std=c++17 -O2 self_test.cpp -o /tmp/species-tree-self-test
/tmp/species-tree-self-test
```

All source was written for this submission. The algorithm uses standard
neighbor-joining, quartet-concordance, bootstrap, and consensus ideas; no
third-party implementation is bundled. There are no precomputed trees,
sample answers, trained parameters, input fingerprints, or cross-instance
caches in the program.
