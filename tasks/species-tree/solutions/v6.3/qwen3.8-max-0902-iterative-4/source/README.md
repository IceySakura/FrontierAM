# Species tree inference from gene trees (ASTRAL-style quartet synthesis)

## Method

The program implements the ASTRAL objective from scratch: given k unrooted gene
trees on the same taxon set, output the binary species tree T maximizing

    |Q_G ∩ Q_T| = number of quartet topologies shared with the input gene trees

which is statistically consistent under the multi-species coalescent.

Implementation outline (single C++17 file, `species_tree.cpp`):

1. **Parsing** – iterative Newick parser (labels, optional internal support
   labels, optional branch lengths; degree-2 serialization roots are suppressed;
   multifurcations are supported). Each gene tree is re-rooted at the neighbour
   of the outgroup leaf `0`.
2. **Split set X** – all non-trivial splits of all gene trees (deduplicated,
   with frequencies). Canonical side = the side not containing the outgroup.
3. **Quartet counting** – for disjoint taxon sets X,Y and gene tree g,
   `N(g;X,Y)` = # quartets {x1,x2}|{y1,y2} (x's in X, y's in Y) displayed by g
   is computed exactly in O(#nodes) by a postorder recursion over LCA
   configurations (valid for any node degree). `M(A) = N(A, V\A)` is
   precomputed for every split in X.
4. **Dynamic programming** – the objective decomposes over the levels of the
   tree's clade recursion: for a level splitting clade S into A|B (R = V\S),

       levelScore(A,B;S) = M(A)+M(B) − N(A,B) − N(A,R) − N(B,R)

   and the total over levels equals |Q_G ∩ Q_T| exactly (each quartet is owned
   by exactly one level). The DP maximizes the sum of levelScores over subsets
   S that are sides of splits in X, enumerating bipartitions by size bucket.
5. **Beyond-X NNI local search** – the quartet objective is then improved by
   best-improvement NNI over the *unconstrained* split space: each move
   changes exactly the levels of the two affected nodes, and its delta is
   evaluated exactly with cached M/N quartet counts. This recovers true splits
   that appear in no gene tree (impossible for the X-restricted DP).
6. **Re-expansion** – splits of the NNI tree are added to X and the DP is run
   again (globally optimal over the enriched X), then NNI is repeated; this
   iterates 1-2 times under the time guard and is monotonically non-decreasing
   in the objective.
7. **Output** – the final tree is serialized as Newick with a trifurcating
   root (outgroup leaf first). Branch lengths/labels are omitted.

Robustness: any parse/pipeline failure falls back to writing a valid tree (the
first input gene tree) instead of crashing; time guards bound the M phase
(100 s), DP (85 s) and NNI (100 s), keeping the worst case well inside the
per-instance wall-clock limit.

Correctness self-tests (`./species_tree in out --selftest`):
- the N(g;X,Y) recursion is checked against quartet brute-force enumeration on
  hundreds of random trees (binary and multifurcating);
- the full DP is checked against brute-force enumeration of all binary trees
  (restricted to X) on random small instances.

## Dependencies / build

Only a C++17 compiler (g++). Offline build:

    ./build.sh          # produces ./species_tree  (~10 s)

## Run

    ./run.sh /absolute/input/genes.tre /absolute/output/pred.tre

Single-threaded, deterministic (no RNG use; SEED is irrelevant). Runtime on the
250-taxon / 50-gene instances: a few seconds; memory < 200 MiB. Logs go to
stderr; only the Newick tree is written to the output path.

## Options (development; defaults used by run.sh)

    --min-freq F       only keep splits seen in >= F gene trees
    --collapse-sup S   contract gene-tree edges with support < S
    --collapse-len L   contract gene-tree edges with length < L
    --nni R            R rounds of beyond-X NNI local search (run.sh: 5000)
    --rex R            R rounds of DP re-expansion with the NNI tree's splits
    --time-guard SEC   abort optional phases after SEC seconds
    --score-tree FILE  debug: report |Q_G ∩ Q_T| of FILE under the input
    --selftest         run correctness self-tests

## Provenance

Method: ASTRAL coalescent species-tree estimation (Mirarab & Warnow 2015;
Zhang, Mirarab & Warnow 2018, ASTRAL-III) — original from-scratch
implementation for this task; no external code or data used at runtime.
