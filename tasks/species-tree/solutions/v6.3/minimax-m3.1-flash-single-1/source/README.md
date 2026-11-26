# Species tree inference from short gene trees (250 taxa)

## What this program does

It reads a file of estimated gene trees (`genes.tre`, one Newick tree per
line) and writes a single unrooted binary species tree in Newick format.

    ./run.sh /absolute/path/genes.tre /absolute/path/pred.tre

The output contains exactly one tree, terminated by `;`, with every input
leaf label appearing exactly once.

## Method

The data are 50 gene trees over 250 taxa, estimated from 200 bp loci.  Each
gene tree is therefore very noisy: on the development instances a single
gene tree agrees with the true species tree on only ~40% of its internal
edges, and only splits that are supported by a majority of the gene trees
are certainly correct.  The program exploits exactly that structure.

1. **Splits and support.**  Every internal edge of every gene tree gives an
   unrooted split.  Splits are expressed on the ingroup; the outgroup (`0`)
   is handled separately and re-attached at the end.  Counting how often
   each split occurs over the 50 gene trees gives its *support*.

2. **Greedy consensus over compatible splits.**  Candidate splits are
   processed in decreasing support (smaller split first on ties) and a
   split is kept when it is compatible with all splits kept so far.  Two
   splits are compatible when one side of the first is nested in or
   disjoint from one side of the second.  A set of pairwise compatible
   splits is realisable by exactly one tree, so the collected set always
   corresponds to a legal tree.  Incompatible candidates are skipped, which
   means a poorly supported but *correct* split can be dropped when it
   conflicts with a better supported one - this is the main source of
   residual error.

3. **Tree construction.**  For a compatible set of splits the smaller sides
   form a laminar family (nested or disjoint).  The tree is built by
   putting every one of them below the smallest cluster that contains it;
   the edge above such a cluster realises exactly the split it came from.
   Leaves that no cluster covers become children of the enclosing node, and
   any polytomy that is left is resolved so that the output is fully
   binary.

4. **Outgroup.**  The outgroup leaf is attached to the top level of the
   inferred ingroup tree.

### Why this objective

The score of the task is the normalised Robinson-Foulds distance computed
from the *missed* true splits only.  A wrong split that is compatible with
the correct ones therefore costs nothing directly, whereas dropping a
correct split costs one.  The greedy consequently keeps every split that
carries signal, instead of throwing away everything below a majority
support threshold - on the development instances that difference is large
(majority-rule consensus: 0.61 / 0.70; this program: 0.17 / 0.25).

Branch lengths and the support values written inside the input gene trees
are ignored; only the topologies are used.

## Requirements / dependencies

* Python 3 (standard library only - no third-party packages, no network
  access, no compilation step).
* POSIX shell for `run.sh`.

There is no build step, so no `build.sh` is provided.

## Layout

    run.sh              entry point: ./run.sh INPUT OUTPUT
    src/solve.py        the whole program (single file, standard library only)

## Development check

With the development instances in `/workspace/samples`:

    dev=/workspace/samples
    ./run.sh "$dev/cases/001/genes.tre" "$PWD/pred.tre"
    python3 "$dev/check_rf.py" "$dev/cases/001/truth.tre" "$PWD/pred.tre"

Results on the two development instances (lower is better):

| instance | normalised RF |
|---|---|
| 001 | 0.174 |
| 002 | 0.247 |

Runtime is about 0.3 s per instance and memory usage is far below 1 GiB,
i.e. well inside the per-instance limits (1 CPU, 4 GiB, 135 s).

## Notes on determinism

The program is fully deterministic: no randomness is used, so the
`SEED=1` convention of the task does not need to be consulted.
