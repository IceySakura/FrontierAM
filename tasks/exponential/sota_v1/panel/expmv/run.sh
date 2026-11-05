#!/bin/sh
# Optional first argument: an Octave struct literal overriding the precision policy (calibration only).
cd "$(dirname "$0")" || exit 1
exec octave-cli --no-init-file --no-site-file --no-history --quiet --eval "stream_expmv(${1:-})"
