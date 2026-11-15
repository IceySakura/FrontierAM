#!/bin/sh
# Entry point: runs the FAMEXP2 session solver.
cd "$(dirname "$0")" || exit 1
exec python3 solve.py
