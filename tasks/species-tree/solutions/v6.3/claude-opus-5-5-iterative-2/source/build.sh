#!/bin/sh
# Offline build of the species-tree program (no network, no external deps).
cd "$(dirname "$0")"
g++ -O2 -march=x86-64-v3 -o st st.cpp 2>build.log || g++ -O2 -o st st.cpp 2>>build.log
test -x st
