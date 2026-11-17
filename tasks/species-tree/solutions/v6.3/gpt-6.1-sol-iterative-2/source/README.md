# Adaptive Weighted-Quartet Species-Tree Inference

## Build and Run

Dependencies are a POSIX shell, `g++` with C++17 support, and the standard C++
runtime. No downloads, Python packages, agent tools, or external phylogenetics
programs are needed.

```sh
./build.sh
./run.sh /absolute/path/genes.tre /absolute/path/pred.tre
```

`build.sh` takes no arguments. It builds `infer` offline. `run.sh` takes exactly
the input and output paths. The output is one fully resolved unrooted Newick
tree with every input leaf label exactly once. Branch lengths and internal
labels are omitted. Diagnostics go to standard error.

## Method

The program parses complete unrooted gene trees, including zero branch lengths,
optional numeric support labels, comments, and quoted taxon labels. A temporary
degree-two serialization root is suppressed. Labels are treated as strings.

Reported edge support `s` is converted to an evidence contrast
`e = max(0, min(1, (3*s - 1)/2))`. Missing support is estimated conservatively
from branch length relative to the locus's positive internal branch lengths;
missing support is never treated as a reported value. The residual uncertainty
is `r = sqrt(1-e)`. A gene's weight is proportional to the square root of
`0.05 + 0.95*mean(e)`.

An initial neighbor-joining tree uses locus-weighted, confidence-weighted
internode distances. Exact quartet scoring then refines this tree with
nearest-neighbor interchanges. A gene quartet's internal confidence is
`1 - product(r)` along its central path.

The preliminary tree's supported gene-split concordance selects a refinement:
collections with concordance at least 0.6 use local quartet weighting, adding
an `exp(-0.3)` factor for each internal edge on a quartet's pendant paths.
Less-concordant collections retain global quartet weighting and use a
confidence-calibrated distance initialization.

The quartet sums are computed by bottom-up and top-down dynamic programming
on the gene trees. All quartets affected by each interchange are counted
exactly; individual four-taxon combinations are not enumerated.

## Resources and Reproducibility

Inference is single-threaded and inherits CPU affinity. Random traversal order
uses `SEED`, defaulting to 1. The work cutoff is 120 seconds, with a valid tree
available throughout refinement. Memory use is well below the 4 GiB limit.
The program uses only the current input and fixed algorithm parameters. It
does not cache across instances or communicate with external services.

## Sources

All implementation code was written for this submission. The algorithmic
bases are standard neighbor joining (Saitou and Nei, 1987), quartet-summary
species-tree inference, and weighted quartet aggregation. The parser, exact
weighted counting, confidence calibration, and concordance-based adaptation
are implemented here. No third-party source, precomputed answers, or additional
data files are included.
