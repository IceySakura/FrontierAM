#!/bin/sh
set -e
cd "$(dirname "$0")"
g++ -O2 -mavx2 -mfma -mtune=znver2 -DFINAL -DKERN='"kern_p11.h"' -o solve fmm.cpp
