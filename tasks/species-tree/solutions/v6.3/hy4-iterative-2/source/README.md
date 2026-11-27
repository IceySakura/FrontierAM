# Species-tree inference from 50 estimated gene trees (250 taxa)

## Method

`infer.cpp` is a self-contained C++17 program (single-threaded, no external
libraries).  It combines several coalescent-aware distance summaries of the
gene trees with a local tree search and a weighted split consensus.

1. **Parsing.**  Each Newick gene tree is parsed together with its edge
   supports (the optional number after `)`) and branch lengths.  Degree-2
   serialisation nodes are suppressed so every internal node has degree three.
   The splits of the gene trees are represented as 256-bit masks.

2. **Gene-tree distances.**  For each locus an all-pairs *internode* distance
   is computed: the number of reliable internal edges on the path between two
   leaves (pendant edges contribute nothing).  Four different summaries are
   used:
   * **soft**: each internal edge contributes its calibrated posterior support
     `clamp((p-1/3)/(2/3))`, and zero if its branch length is essentially 0;
   * **hard65 / hard80 / hard90**: edges with support below 0.65 / 0.80 / 0.90
     or with essentially zero length are contracted into polytomies; the
     remaining internal edges count 1.
   The four matrices are averaged over the 50 loci.  Under the multi-species
   coalescent, the expected number of internal gene-tree nodes on a path is
   monotone in species divergence, so these matrices are informative about
   species topology even for 200 bp loci.

3. **Initial trees.**  Neighbour joining (Saitou & Nei 1987) on each matrix,
   followed by steepest-ascent NNI hill climbing under the balanced minimum
   evolution criterion (Pauplin / FastME, Desper & Gascuel 2002).

4. **Adaptive weighted consensus.**  The gene trees also give an estimate of
   the amount of conflict: the number `m` of gene-tree splits occurring in at
   least half of the loci.  When `m` is small (high ILS / high estimation
   noise), the aggressively contracted summaries receive more weight; when
   `m` is large the support-weighted soft summary dominates.  All splits of
   the four trees are pooled with these weights and a greedy compatible-split
   (majority) consensus is built; ties are broken by the observed gene-tree
   split frequency.  The result is always a fully resolved unrooted binary
   tree on exactly the input labels.

Only the input file and instance-independent constants are used.  No
randomness, no network access, no cross-instance state.  All stderr output is
diagnostic only.

## Files

* `infer.cpp` — complete C++17 source.
* `build.sh` — builds `infer` with `g++ -O3 -std=c++17` (no arguments).
* `run.sh` — `./run.sh INPUT OUTPUT`; writes one Newick tree to `OUTPUT`.
  It builds `infer` on first use if the binary is absent.
* `infer` — prebuilt binary (rebuilt by `build.sh`).

## Dependencies / sources

POSIX shell, a C++17 compiler (`g++`) and the standard library only; no
third-party code is redistributed.  The algorithms are classical published
methods: neighbour joining (Saitou & Nei 1987), balanced minimum evolution /
Pauplin's formula (Desper & Gascuel 2002, FastME), internode distances for
coalescent-based species trees (NJst, Liu & Yu 2011; ASTRID, Vachaspati &
Warnow 2015), and majority-rule compatible-split consensus.  All code here
was written from scratch for this task.

## Development self-check

```sh
dev=/workspace/samples
./build.sh
./run.sh "$dev/cases/001/genes.tre" pred.tre
python3 "$dev/check_rf.py" "$dev/cases/001/truth.tre" pred.tre
```
