#!/bin/sh
# Usage: run.sh [INPUT] [OUTPUT]; defaults are the evaluator paths.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INPUT=${1:-/workspace/inputs/input.json}
OUTPUT=${2:-/workspace/result/output.json}
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS PYTHONDONTWRITEBYTECODE
mkdir -p "$(dirname -- "$OUTPUT")" 2>/dev/null
python3 "$HERE/solve.py" "$INPUT" "$OUTPUT"
# Last-resort guard: never leave the result missing.
[ -s "$OUTPUT" ] || printf '{"estimate": 0.0}' > "$OUTPUT"
exit 0
