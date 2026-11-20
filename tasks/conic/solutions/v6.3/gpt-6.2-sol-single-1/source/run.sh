#!/bin/sh
set -eu
case "$0" in
    */*) cd -- "${0%/*}" ;;
esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ./conic "$@"
