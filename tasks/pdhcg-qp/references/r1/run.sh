#!/bin/sh
set -eu
# Keep numerical libraries single-threaded even when the reference is invoked
# directly outside the evaluator's environment wrapper.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
input=
output=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --input) input=${2:?missing input path}; shift 2 ;;
    --output) output=${2:?missing output path}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$input" ] && [ -n "$output" ]
exec "${PYTHON:-python3}" "$(dirname "$0")/pdhcg.py" --input "$input" --output "$output"
