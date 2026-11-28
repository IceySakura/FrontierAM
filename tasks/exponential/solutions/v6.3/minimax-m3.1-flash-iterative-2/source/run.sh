#!/bin/sh
# FAMEXP2 submission entry point.  The evaluator runs this script with the
# submission directory as the working directory.
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"

# Use the native solver when it is present and starts up correctly; otherwise
# fall back to the pure Python implementation (same algorithm, slower).
if [ -x bin/fastsolve ]; then
    probe=$(./bin/fastsolve < /dev/null 2>/dev/null | head -c 6)
    if [ "$probe" = "READY" ]; then
        exec ./bin/fastsolve
    fi
fi

exec python3 solve.py
