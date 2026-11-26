#!/bin/sh
# Offline build of the C kernel.  The pre-built libfamexp.so is shipped as
# well; this script only refreshes it and never fails the submission.
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 0
for CC in cc gcc clang; do
    if command -v "$CC" >/dev/null 2>&1; then
        "$CC" -O3 -fPIC -shared -o "$DIR/libfamexp.so.new" "$DIR/kernel.c" -lm 2>/dev/null \
            && mv -f "$DIR/libfamexp.so.new" "$DIR/libfamexp.so" && break
        rm -f "$DIR/libfamexp.so.new"
    fi
done
exit 0
