#!/bin/sh
#
# run.sh  --  species-tree inference: GENES_TRE -> PRED_TRE
#
# Usage: ./run.sh /path/to/genes.tre /path/to/pred.tre
#
# The species tree is written to the second argument (only), one Newick
# tree terminated by ";".  All diagnostics go to stderr, never to the
# output tree.
#
# Dependencies (already installed on the portable-v1 host):
#   Python 3.12 (stdlib only)  -- the driver is pure Python
#   NumPy                     -- used by the optional build.sh, not by run.sh
#   No network access is required; nothing is downloaded.
#
# Tuning knobs (environment variables, optional; defaults below are the
# configuration that scored best on the published development cases):
#   SPT_WEIGHT    support | vote | hybrid   how clades are weighted
#   SPT_MINWEIGHT int (1)                  minimum vote count
#   SPT_MISSING   one | zero | count       value when a node reports no support
#   SPT_SUPPORT_MIN float (0.0)            drop nodes below this support

set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 GENES_TRE PRED_TRE" >&2
    exit 2
fi

GENES_TRE="$1"
PRED_TRE="$2"

ME=$(cd "$(dirname "$0")" && pwd)

# SPT_MISSING=zero is the best-performing default on the two known cases.
SPT_MISSING="${SPT_MISSING:-zero}"
SPT_WEIGHT="${SPT_WEIGHT:-support}"
SPT_MINWEIGHT="${SPT_MINWEIGHT:-1}"
SPT_SUPPORT_MIN="${SPT_SUPPORT_MIN:-0.0}"
export SPT_MISSING SPT_WEIGHT SPT_MINWEIGHT SPT_SUPPORT_MIN

# Re-exec under the Python configured by build.sh, if it produced one.
if [ -x "$ME/.venv/bin/python" ]; then
    PYTHON="$ME/.venv/bin/python"
elif [ -x "$ME/python" ]; then
    PYTHON="$ME/python"
else
    PYTHON=python3
fi

exec "$PYTHON" "$ME/sptree.py" "$GENES_TRE" "$PRED_TRE"
