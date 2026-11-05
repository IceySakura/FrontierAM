#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec python3 -I stream_scipy.py
