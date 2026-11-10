#!/bin/sh
cd "$(dirname "$0")" || exit 1
# Optional first argument: the tolerance multiplier (calibration only).
exec octave-cli --no-init-file --no-site-file --no-history --quiet --eval "stream_kiops(${1:-})"
