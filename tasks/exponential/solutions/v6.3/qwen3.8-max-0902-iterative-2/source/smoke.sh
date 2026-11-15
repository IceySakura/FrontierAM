#!/bin/sh
# Verify a freshly built binary actually runs and computes.  Exit 0 on success.
set -u
BIN="$1"
[ -x "$BIN" ] || exit 1

# 1) startup: must emit READY and must not die from an illegal instruction
"$BIN" </dev/null >bin/smoke.out 2>/dev/null
rc=$?
if [ "$rc" -ge 128 ]; then
    echo "smoke: binary died with signal (rc=$rc)" >&2
    exit 1
fi
printf 'READY\n' >bin/smoke.want
if ! cmp -s bin/smoke.out bin/smoke.want; then
    echo "smoke: unexpected startup output" >&2
    exit 1
fi

# 2) numerical self test on two tiny sessions (if python3 is available)
if command -v python3 >/dev/null 2>&1; then
    if ! python3 smoke.py "$BIN"; then
        echo "smoke: numerical self test failed" >&2
        exit 1
    fi
fi
rm -f bin/smoke.out bin/smoke.want
exit 0
