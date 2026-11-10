#!/bin/bash
# FAMEXP2 solver entry point.  Picks a prebuilt binary that runs on this CPU
# (validated with --check before READY, so outside the timed session), and
# falls back to compiling expo.c if no prebuilt binary is usable.  The chosen
# binary prints READY itself and speaks the wire protocol on stdin/stdout.
shopt -s execfail 2>/dev/null
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here" || exit 1

try_bin() {
    local b="$1"
    [ -f "$b" ] || return 1
    if [ ! -x "$b" ]; then
        chmod +x "$b" 2>/dev/null
    fi
    if [ ! -x "$b" ]; then
        local tdir
        tdir="$(mktemp -d 2>/dev/null)" || return 1
        cp "$b" "$tdir/expo" 2>/dev/null && chmod +x "$tdir/expo" 2>/dev/null || return 1
        b="$tdir/expo"
    fi
    "$b" --check </dev/null >/dev/null 2>&1 || return 1
    exec "$b"
}

if grep -qw avx2 /proc/cpuinfo 2>/dev/null && grep -qw fma /proc/cpuinfo 2>/dev/null; then
    try_bin "$here/expo_v3"
fi
try_bin "$here/expo_v1"

# Last resort: build from source into a scratch directory.
tdir="$(mktemp -d 2>/dev/null || echo /tmp)"
if command -v gcc >/dev/null 2>&1; then
    gcc -O2 -o "$tdir/expo_src" "$here/expo.c" -lm >/dev/null 2>&1 && try_bin "$tdir/expo_src"
fi
echo "run.sh: no usable solver binary" >&2
exit 1
