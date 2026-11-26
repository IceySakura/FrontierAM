#!/bin/sh
# Entry point: /bin/sh /workspace/submission/run.sh [INPUT OUTPUT]
# With no arguments it reads /workspace/inputs/input.json and writes
# /workspace/result/output.json.
set -u

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INPUT=${1:-/workspace/inputs/input.json}
OUTPUT=${2:-/workspace/result/output.json}

# Pick an interpreter that can import numpy.  The verifier puts the bundled
# runtime first on PATH; the absolute path is tried first as a belt-and-braces
# fallback, and nothing here installs or downloads anything.
PY=""
for cand in /workspace/runtime/bin/python3 python3 /usr/local/bin/python3 python; do
    found=0
    case "$cand" in
        /*) [ -x "$cand" ] && found=1 ;;
        *)  command -v "$cand" >/dev/null 2>&1 && found=1 ;;
    esac
    [ "$found" = 1 ] || continue
    if [ -z "$PY" ]; then PY="$cand"; fi
    if "$cand" -c 'import numpy' >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
[ -n "$PY" ] || PY=python3

exec "$PY" "$DIR/solve.py" "$INPUT" "$OUTPUT"
