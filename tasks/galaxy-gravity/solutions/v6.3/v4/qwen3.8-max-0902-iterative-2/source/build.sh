#!/bin/sh
# Offline build: produces ./solve in this directory.
set -e
cd "$(dirname "$0")"
CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
CFLAGS="-O3 -fno-math-errno -fno-trapping-math -fomit-frame-pointer -w"

# Probe for AVX2+FMA support on this machine (build host == run host).
cat > .avxprobe.c <<'PROBE'
#include <immintrin.h>
int main(void){
  __m256d a=_mm256_set1_pd(1.0),b=_mm256_set1_pd(2.0);
  __m256d c=_mm256_fmadd_pd(a,b,a);
  double o[4]; _mm256_storeu_pd(o,c);
  return o[0]==3.0?0:1;
}
PROBE
EXTRA=""
if $CC -O2 -mavx2 -mfma .avxprobe.c -o .avxprobe >/dev/null 2>&1 && ./.avxprobe >/dev/null 2>&1; then
  EXTRA="-mavx2 -mfma -DUSE_AVX2"
fi
rm -f .avxprobe .avxprobe.c
$CC $CFLAGS $EXTRA -o solve solver.c -lm
chmod +x solve
