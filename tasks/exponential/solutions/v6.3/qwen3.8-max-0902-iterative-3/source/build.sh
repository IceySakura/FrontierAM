#!/bin/bash
# Offline build: compile solver.c into ./solver.
#
# Tries progressively more conservative flag sets (the first needs an x86-64-v3
# capable compiler target; later ones work on any host gcc).  A candidate binary
# is only installed if it passes a smoke run (empty stdin must make the solver
# exit 1 on a short read), and a pre-existing ./solver is never destroyed unless
# a working replacement exists, so run.sh can still fall back to it or to
# solve.py.
set -u
cd "$(dirname "$0")" || exit 1
CC=${CC:-gcc}
: > build.log
ok=0
for flags in "-O3 -march=x86-64-v3 -mtune=native" "-O3 -march=x86-64-v2" "-O3" "-O2"; do
    rm -f solver.tmp
    if $CC $flags -std=c99 -fno-math-errno -o solver.tmp solver.c -lm 2>>build.log; then
        ./solver.tmp </dev/null >/dev/null 2>&1
        if [ $? -eq 1 ]; then
            mv solver.tmp solver
            echo "built with: $CC $flags" >> build.log
            ok=1
            break
        fi
        echo "smoke run failed for: $CC $flags" >> build.log
    else
        echo "compile failed for: $CC $flags" >> build.log
    fi
done
rm -f solver.tmp
if [ "$ok" != "1" ]; then
    # Not fatal: run.sh still works from a previously built ./solver, or from
    # the pure Python implementation, so report success to the caller.
    echo "build failed; run.sh will use an existing ./solver or solve.py" >> build.log
fi
exit 0
