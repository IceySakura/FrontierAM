#!/usr/bin/env bash
#
# No-argument build step for the deconvolution submission.
#
# The submission is pure Python with only NumPy/SciPy pre-installed, so there
# is nothing to download or compile.  This script verifies that the required
# files are present and that the source compiles, then exits 0.  It takes no
# arguments and runs in well under 300 s.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for f in solution.py README.md; do
    if [[ ! -f "${HERE}/${f}" ]]; then
        echo "build: missing required file: ${HERE}/${f}" >&2
        exit 1
    fi
done

# Syntax-check the solver so that a corrupted edit is caught at build time
# rather than at evaluation time.
python3 -m py_compile "${HERE}/solution.py"

echo "build: ${HERE}/solution.py verified" >&2
exit 0
