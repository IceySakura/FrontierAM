#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$DIR/bin/conic_solver" ] && [ -x "$DIR/bin/conic_solver.prebuilt" ]; then
    exec "$DIR/bin/conic_solver.prebuilt" "$@"
fi
exec "$DIR/bin/conic_solver" "$@"
