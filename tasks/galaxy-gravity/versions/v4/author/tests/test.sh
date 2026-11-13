#!/bin/sh
set -eu
mkdir -p /logs/verifier
exec python3 -I /tests/bridge.py
