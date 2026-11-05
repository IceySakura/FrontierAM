#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec octave-cli --no-init-file --no-site-file --no-history --quiet --eval stream_kiops
