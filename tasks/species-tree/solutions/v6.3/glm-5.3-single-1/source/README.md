# Species-tree inference from 50 estimated gene trees (250 taxa)

## Running

    ./run.sh /absolute/path/genes.tre /absolute/path/pred.tre

`run.sh` invokes `python3 species_tree.py INPUT OUTPUT` (no build step is
required; `build.sh` is intentionally absent).  The only dependency is
Python 3 with NumPy, both present in the evaluation image.

The program reads the 50 Newick gene trees, infers one unrooted binary tree
covering all 250 taxa and writes it in Newick form terminated by `;`.
Leaf labels are reproduced exactly as they appear in the input.
Runtime is ~7 s and about 200 MB of RAM per instance, well inside the limits.
Randomness is not used; if it ever were, `SEED=1` is honoured.

## Method

1. **Parsing.**  Each gene tree is read with a small tokenising Newick parser
   (branch lengths and per-edge support values are captured, comments and
   quoted labels are tolerated).  Degree-two serialisation roots are
   suppressed so every tree is an unrooted binary tree.

2. **Distance estimate.**  For every pair of taxa the patristic (path)
   distance is averaged over the 50 gene trees.  Under the multispecies
   coalescent the expectation of this average is monotone in the species-tree
   distance, so it is a consistent summary even though individual loci are
   affected by incomplete lineage sorting and by estimation error from the
   short (200 bp) alignments.

3. **Species tree.**  Neighbour joining is run on the averaged distance
   matrix (implemented directly in NumPy).  NJ is statistically consistent
   for additive matrices and handles the 250-taxon scale in milliseconds.
   A second start from averaged topological (internode-count) distances is
   kept as a fallback in case the patristic matrix cannot be built.

4. **Model-based score.**  The exact ASTRAL-style quartet-agreement score
    F(T) = sum over gene trees of |{quartets displayed the same way by T and
    the gene tree}|
   is implemented exactly through the identity
    F(T) = sum over internal edges e of QS(e) - sum over internal nodes v of
           QS2(children of v),
   where QS(A|B) counts, over all gene trees, the quartets with two taxa in
   A and two in B that the gene tree resolves as the corresponding split.
   Both quantities are evaluated with vectorised 250-bit pop-count
   operations (`np.bitwise_count`), which costs about a millisecond per
   bipartition.  (Set `HILLCLIMB=1` and `SELECT=1` to additionally score the
   candidate starts with it and run NNI hill-climbing with exact incremental
   deltas; on the two development instances the plain distance-based start
   was at least as accurate, so these are off by default.)

## Files

* `species_tree.py` - the whole program (parser, NJ, quartet scorer, output)
* `run.sh`          - entry point

## Sources

Neighbour joining (Saitou & Nei 1987) and quartet-based species-tree
scoring in the spirit of ASTRAL (Mirarab et al. 2014); the implementation is
original and self-contained.
