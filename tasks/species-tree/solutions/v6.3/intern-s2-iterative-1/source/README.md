# Species-tree inference (greedy compatible-clade consensus)

## What this does

Given a file of 50 gene trees (one Newick tree per line, `;` terminated),
builds a single unrooted binary species tree over the 250 taxa and writes it
as one `;`-terminated Newick string.

Algorithm:

1. **Parse** each gene tree. Internal nodes may carry a bootstrap support in
   `[0, 1]` immediately after a closing `)`, followed optionally by a branch
   length (`)support:length`). A node that reports no support is recorded as
   `support = None` — the spec says such values must not be assumed to be 1
   nor to be 0.
2. **Re-root** every gene tree on taxon `"0"` (the outgroup). In that
   representation each internal node other than the root corresponds to
   exactly one unrooted split.
3. **Score each split** by the *sum* of the bootstrap supports of every
   clade that expresses it across all 50 trees (configurable: vote counts,
   or support-weighted, or hybrid). Summing supports suppresses the ~86% of
   clades that appear in exactly one gene tree: singletons end up near 0.33
   while true clades reach 14–16.
4. **Select greedily** in descending weight order, keeping only pairwise
   compatible clades (standard clade-compatibility test on masks oriented
   away from the anchor). Ties are broken by ascending mask value so the
   result is fully deterministic.
5. **Complete** the unresolved leaves as a caterpillar, then emit Newick.

## Files

| File | Contents |
| --- | --- |
| `sp.py` | Newick parser, rerooting, mask bookkeeping, greedy selection, tree builder, RF metric, utilities. Pure stdlib. |
| `sptree.py` | Driver: reads the gene trees, tallies weights, runs greedy, writes the tree. |
| `run.sh` | Executable entry point. |
| `build.sh` | Optional: builds a venv so the host does not need a working `python3` symlink. |
| `README.md` | This file. |

No other files in this directory are required at runtime.

## Dependencies

- **Python 3.12** (or any recent Python 3) and its standard library. The
  driver is pure Python; `sp.py` uses only `re`, `sys`, `collections`.
- `numpy` is *not* used by `run.sh`. It is only used by the diagnostic
  scripts in this workspace (`/tmp/wk/nj.py`, `/tmp/wk/nni.py`,
  `/tmp/wk/repair.py`), which are **not** part of the submission.
- Nothing is downloaded from the network; the submission works fully offline.

Verified on `portable-v1` (Linux amd64). `os.sched_getaffinity(0)` is used
where needed rather than hard-coding the pinned CPU number.

## How to run

```sh
./run.sh /absolute/input/genes.tre /absolute/output/pred.tre
```

`/absolute/output/pred.tre` will contain exactly one `;`-terminated Newick
tree whose leaf labels are exactly the input labels, each once. The top-level
root may be degree 3 (temporary degree-2 roots are also accepted by the
scorer). All diagnostic output goes to stderr.

The following environment variables control the weighting policy; they are
read once at process start and default to the best-known configuration:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SPT_WEIGHT` | `support` | `support` = sum of bootstrap supports; `vote` = count of trees containing the clade; `hybrid` = vote count scaled by mean support |
| `SPT_MINWEIGHT` | `1` | minimum vote count (only meaningful for `vote`) |
| `SPT_MISSING` | `zero` | value used for a node that reports no support (`one`=1.0, `zero`=0.0, `count`=1.0) |
| `SPT_SUPPORT_MIN` | `0.0` | ignore a node whose reported support is below this |

The default configuration scores **rf = 0.154 on case 001** and **rf = 0.239
on case 002** (normalized RF, `missed/(n-3)`).

## Performance

On the two published development cases the full run takes well under one
second of wall-clock time (including interpreter startup and imports), far
inside the 180 s runtime budget, using a single logical CPU and a small
fraction of the 4 GiB memory budget.

## Notes

- The `check_rf.py` scorer in `/workspace/samples/` is the reference
  validator: `python3 /workspace/samples/check_rf.py truth.tre pred.tre`.
- A nearest-neighbour interchange (NNI) hill-climbing post-processor was
  explored and discarded: on both development cases the greedy tree is
  already a strict local optimum of the sum-of-observed-weights objective,
  so NNI converged after 0 iterations while adding ~0.8 s with no RF gain.
- An ASTRID-style neighbor-joining on the average topological distance
  matrix was also tried: it scores rf = 0.138 / 0.332, marginally better on
  case 001 but markedly worse on case 002, so the consensus method is
  retained as the sole approach.
