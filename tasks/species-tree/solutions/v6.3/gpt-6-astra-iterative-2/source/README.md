# Species-Tree Inference

## Dependencies

A POSIX shell, a C++17 compiler (`c++` by default), and the standard C++
library. The program is self-contained and single-threaded. No downloads,
Python packages, external inference programs, or agent tools are required.

## Build and Run

Build once, without arguments:

```sh
./build.sh
```

`CXX` may select another installed C++17 compiler.

Run one independent instance:

```sh
./run.sh /absolute/path/genes.tre /absolute/path/pred.tre
```

The output file contains exactly one fully bifurcating, unrooted Newick tree.
Leaf labels are preserved as strings. Branch lengths and internal labels are
omitted from the output. Diagnostics go to standard error.

## Method

1. Parse the gene trees, including optional supports, quoted labels, comments,
   zero-length edges, and either common unrooted Newick root convention.
2. Average normalized path distances obtained from square-root branch lengths
   and construct a BIONJ starting tree.
3. Refine the topology with nearest-neighbor interchanges using a weighted
   quartet objective. Dynamic programming counts quartet agreement without
   enumerating all taxon quadruples. Weights use gene-edge supports, internal
   path lengths, and normalized external path lengths.

The selected configuration equally blends independent-path and maximum-edge
support models. The latter is integrated over 32 support thresholds.

Reported support is measured relative to the three-way uninformative
baseline. Unreported, effectively zero-length edges provide no support
evidence. Missing supports on longer edges use a length-confidence calibration
from reported edges in the same locus, with a length-based fallback.

The default random seed is 1; the `SEED` environment variable is honored.
Each process uses only its input trees and fixed algorithm parameters.
There is no cross-instance storage or communication. CPU affinity is inherited.
A 115-second internal search deadline leaves margin under the 135-second
process limit, and a complete valid tree is retained throughout refinement.

## Self-Tests

Optional tests require only the same compiler:

```sh
c++ -O2 -std=c++17 selftest.cpp -o selftest
./selftest
```

The tests compare quartet calculations against explicit enumeration, compare
optimized support scoring with its direct implementation, check topology-move
score changes, and exercise parsing, serialization, and split consensus.
They do not require sample files or any external data.

## Sources

All source code is original to this submission; no third-party implementation
is bundled. Algorithmic foundations include neighbor joining (Saitou and Nei,
1987), BIONJ (Gascuel, 1997), and weighted-quartet summary-tree inference in the
ASTRAL/wASTRAL family. The submitted implementation is independent of those
software packages.
