#!/bin/sh
# Offline, one-shot build: pre-computes the polyphase transfer map that
# solution.py loads at run time.  Pure NumPy, no network, a few seconds.
set -e
cd "$(dirname "$0")"
exec python3 make_filter.py
