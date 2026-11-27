# Species-tree inference from estimated gene trees (250 taxa, 50 loci)

`./run.sh INPUT OUTPUT` reads a file of Newick gene trees (one per line) and writes
exactly one unrooted binary Newick species tree on the same leaf labels.

## Method

**Objective.** The program maximises the ASTRAL quartet objective

    score(T) = sum over gene trees g of |Q(T) ∩ Q(g)|

the number of (gene tree, four-taxon subset) pairs for which the candidate species
tree `T` and the gene tree induce the same quartet topology. Only topology is used
from the input; branch lengths and support values are used optionally as edge
weights (see `WT` below), never as output.

This quantity is *not* additive over the branches of `T` (a quartet whose central
path spans several branches would be counted once per branch, which biases the
search towards degenerate pectinate trees), so it is optimised by local search
with **exact move deltas** rather than by a per-branch sum.

**Exact NNI delta.** For an internal edge whose four incident components are
`A,B,C,D`, an NNI changes the induced topology only for quartets that have exactly
one leaf in each of the four components. With

    N(P,Q|R,S) = sum_g #{ (p,q,r,s) in PxQxRxS : g displays pq|rs }

the two possible interchanges give

    delta(swap B,C) = N(A,C|B,D) - N(A,B|C,D)
    delta(swap B,D) = N(A,D|B,C) - N(A,B|C,D)

`N(P,Q|R,S)` is evaluated in a single pass over the internal nodes of all gene
trees. For an internal node `v` of a gene tree, whose three incident branches have
leaf sets `L1,L2,L3`, the quartets `pq|rs` whose p/q junction is exactly `v` are
those with `p,q` in two different branches and `r,s` together in the third:

    (p1 q2 + p2 q1) r3 s3 + (p1 q3 + p3 q1) r2 s2 + (p2 q3 + p3 q2) r1 s1

with `p_i = |P ∩ Li|` etc. Because `L1,L2,L3` partition the taxon set (and so do
`P,Q,R,S`), only six bitset intersections are needed per node. All three `N`
values for one NNI come out of one such pass. Term `i` carries the weight of the
gene-tree edge leaving `v` towards branch `i`, which is what makes support /
branch-length weighting possible; with all weights equal to 1 this is exactly
plain ASTRAL.

The delta machinery is verified against a brute-force quartet enumerator
(`SELFTEST=1`, small inputs only): after every random NNI the accumulated delta
sum is compared with a from-scratch recount of `sum_g |Q(T) ∩ Q(g)|`, and the
gene-node table itself is checked for consistency of branch sizes and bitsets.

The naive per-branch sum `sum_e w(A_e|B_e)` was tried first and rejected: it
counts a quartet once per branch on its central path in `T`, which is unbounded
and drives the search into degenerate pectinate trees (measured RF 0.94 versus
0.12 for the exact objective).

**Seed tree.** ASTRID: the average internode (topological path) distance between
every pair of taxa over all gene trees, followed by neighbour joining. Branch
lengths can be used instead of internode counts (`DIST=1`).

**Search.** Batch NNI hill climbing on the exact objective (all improving moves on
mutually non-interfering edges are applied per sweep — two moves are independent
whenever none of the six components of one is an endpoint of the other, and then
their affected quartet sets are provably disjoint, so the deltas add exactly),
followed by a ratchet loop: random NNI kicks of increasing size, each re-climbed,
keeping the best tree. Scores are tracked relative to the seed tree, which is all
a local search needs. Search stops at `DEADLINE*SFRAC` seconds; the tree is always
written before exit.

The climb from the ASTRID seed converges in about 3.5 s for 250 taxa x 50 gene
trees; the rest of the budget goes to ratchet rounds. Measured on the two
development cases (missed true splits out of 247):

| stage | case 001 | case 002 | mean normalised RF |
|---|---|---|---|
| ASTRID + NJ seed | 34 | 82 | 0.235 |
| + NNI climb | 29 | 41 | 0.142 |
| + ratchet (default settings) | 29 | 39 | 0.138 |

Larger random kicks were measured to be counter-productive here: with only 50
short loci the quartet objective is itself a noisy estimate of the true species
tree's score, so aggressive perturbation drives the search into optima of the
proxy that are worse in RF. `KICK0=2 KICKMAX=12` was the best of the schedules
tried (`1/6`, `2/12`, `3/20`, `4/30`, `8/80`, `16/48`), and running the search
for 90 s instead of 34 s did not improve the result, so the default budget is
kept low (`DEADLINE=40`, i.e. ~34 s of searching) which also leaves a large
margin under the per-instance wall-clock limit.

**Robustness.** Branch lengths and internal support labels are both optional: with
`WT=0` (the default) neither is used, so inputs stripped of one or both produce an
identical tree. Serialisation roots with two or three children, quoted labels,
CRLF line endings and a missing final newline are all handled. Every candidate
tree is validated (degrees, symmetric adjacency, connectivity, acyclicity) before
it can become the incumbent, and the seed tree is emitted if validation ever
fails, so the program always writes a legal 250-leaf tree.

## Build

```sh
./build.sh          # cc -O3 -std=gnu11 -o solver solver.c -lm   (~1 s, offline)
```

No dependencies beyond a C compiler and libc/libm. No network access, no data
files, no caches. Single threaded, respects the inherited CPU affinity, peak RSS
well under 100 MiB for 250 taxa × 50 gene trees.

## Run

```sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

Logs go to stderr; the output file holds one Newick tree ending in `;`, written
from a three-degree internal node, with every input leaf label exactly once.

## Environment variables (all optional, instance-independent)

| var | default | meaning |
|---|---|---|
| `SEED` | 1 | PRNG seed for the ratchet (the harness fixes `SEED=1`) |
| `DEADLINE` | 40 | wall-clock seconds after which searching stops |
| `SFRAC` | 0.85 | fraction of `DEADLINE` used for search |
| `EPS` | 1e-3 | score improvements below this are ignored (float noise floor) |
| `WT` | 0 | gene-tree edge weighting: 0 = uniform (plain ASTRAL), 1 = support, 2 = support², 3 = max(0, support−`WT_SHIFT`)/(1−`WT_SHIFT`), 4 = bl/(bl+`WT_B0`), 5 = support·bl/(bl+`WT_B0`), 6/7 = keep support ≥ 0.5 / ≥ 0.7, 8 = √support |
| `WT_B0` | 0.01 | branch-length half-saturation |
| `WT_SHIFT` | 1/3 | support floor for `WT=3,9` |
| `DIST` | 0 | ASTRID seed from internode distances (0) or branch lengths (1) |
| `RATCHET` | 1 | enable the ratchet restart loop |
| `BATCH` | 1 | apply several non-interfering NNI moves per sweep |
| `KICK0` / `KICKMAX` | 2 / 12 | initial / maximum ratchet kick size (small kicks measured best on the dev cases) |
| `SELFTEST` | 0 | verify deltas against brute force (small inputs only) |

## Provenance

All code in `solver.c` was written for this task. The method follows published
ideas — the ASTRAL quartet objective (Mirarab et al. 2014) and ASTRID average
internode distances with neighbour joining (Vachaspati & Warnow 2015) — but no
external source was copied and no third-party software is bundled or required.
