#!/bin/bash
# Offline build of the native solver.  Run once by the evaluator in a fresh
# container; no network access and no test inputs are available here.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"

CXX=${CXX:-g++}
mkdir -p bin

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "build.sh: no C++ compiler found, keeping the pure Python fallback" >&2
    exit 0
fi

FLAGS="-O2 -fno-math-errno -funroll-loops"
"$CXX" $FLAGS -o bin/fastsolve.new src/fastsolve.cpp -lm

# Self test: a tiny analytic case (L = diag(0, 4), t = 0.5) checked against a
# closed form.  If the binary misbehaves we keep the Python fallback.
if python3 tools/selftest.py ./bin/fastsolve.new; then
    mv bin/fastsolve.new bin/fastsolve
    echo "build.sh: native solver built and verified"
else
    rm -f bin/fastsolve.new
    echo "build.sh: native solver failed the self test, using the Python fallback" >&2
fi
