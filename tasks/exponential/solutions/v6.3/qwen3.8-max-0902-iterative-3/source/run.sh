#!/bin/bash
# Evaluator entry point: prefer the compiled solver, fall back to Python.
cd "$(dirname "$0")" || exit 1
if [ -x ./solver ]; then
    # sanity check (empty stdin -> protocol exit 1); a foreign/incompatible
    # binary would die with a signal instead, then we fall back to Python
    ./solver </dev/null >/dev/null 2>&1
    rc=$?
    if [ "$rc" -eq 1 ]; then
        exec ./solver
    fi
fi
exec python3 solve.py
