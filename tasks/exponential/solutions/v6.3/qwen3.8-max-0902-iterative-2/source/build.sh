#!/bin/sh
# Build the native solver.  Runs once, offline, in the frozen submission dir.
# Never fails: if no compiler works, run.sh falls back to solver_py.py.
set -u
cd "$(dirname "$0")" || exit 0
mkdir -p bin

CC="${CC:-}"
if [ -z "$CC" ]; then
    for c in gcc cc clang; do
        if command -v "$c" >/dev/null 2>&1; then CC="$c"; break; fi
    done
fi
if [ -z "$CC" ]; then
    echo "no C compiler found; python fallback will be used" >&2
    exit 0
fi

rm -f bin/famexp2
i=0
for FLAGS in "-O3 -march=native -mtune=native -fno-math-errno -fomit-frame-pointer" \
             "-O3 -march=x86-64-v3 -mfma -mavx2 -fno-math-errno" \
             "-O3 -march=x86-64-v2 -fno-math-errno" \
             "-O3 -fno-math-errno" \
             "-O2"; do
    i=$((i + 1))
    echo "== trying: $CC $FLAGS" >&2
    if $CC $FLAGS -o "bin/famexp2.try" solver.c -lm >"bin/build$i.log" 2>&1; then
        if sh smoke.sh "bin/famexp2.try" >&2; then
            mv "bin/famexp2.try" bin/famexp2
            chmod +x bin/famexp2
            echo "== built bin/famexp2 with: $FLAGS" >&2
            rm -f bin/famexp2.try bin/build*.log
            exit 0
        fi
        echo "== smoke test failed for: $FLAGS" >&2
    else
        echo "== compile failed for: $FLAGS" >&2
    fi
    rm -f bin/famexp2.try
done

echo "no working native build; python fallback will be used" >&2
rm -f bin/famexp2.try bin/build*.log
exit 0
