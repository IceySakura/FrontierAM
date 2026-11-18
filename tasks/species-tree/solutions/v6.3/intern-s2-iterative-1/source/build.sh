#!/bin/sh
#
# build.sh  --  optional offline bootstrap.
#
# Takes no arguments.  Creates a self-contained venv in this directory
# (./.venv) so that run.sh does not depend on a working `python3` symlink
# on the grading host.  Uses --system-site-packages and --without-pip so that
# pre-installed packages (numpy) remain available without any network access.
#
# Always exits 0: if the host already provides a usable python3, nothing
# needs to be built and run.sh falls back to `python3` automatically.

set -u

ME=$(cd "$(dirname "$0")" && pwd)
PYTHON="${PYTHON:-python3}"

if [ ! -x "$ME/.venv/bin/python" ]; then
    echo "[build.sh] creating venv (offline, system site packages)..." >&2
    "$PYTHON" -m venv --system-site-packages --without-pip "$ME/.venv" 2>/dev/null || {
        echo "[build.sh] venv creation failed; relying on system $PYTHON" >&2
        if [ -x "$PYTHON" ]; then
            exit 0
        fi
        echo "[build.sh] no usable python3 found" >&2
        exit 1
    }
    echo "[build.sh] venv ready at $ME/.venv" >&2
fi

echo "[build.sh] done" >&2
exit 0
