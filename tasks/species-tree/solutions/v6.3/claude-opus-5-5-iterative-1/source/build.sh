#!/bin/sh
# Builds stree (x86-64-v2) and stree_generic (portable). Never fails the build: if compilation
# is impossible, previously built binaries (if any) are kept and run.sh has further fallbacks.
cd "$(dirname "$0")" || exit 0
if g++ -O2 -std=c++17 -o stree_generic.new stree.cpp; then mv -f stree_generic.new stree_generic; fi
if g++ -O2 -march=x86-64-v2 -std=c++17 -o stree.new stree.cpp && ./stree.new --selftest >/dev/null 2>&1; then
  mv -f stree.new stree
elif [ -x stree_generic ]; then
  rm -f stree.new; cp -f stree_generic stree
fi
exit 0
