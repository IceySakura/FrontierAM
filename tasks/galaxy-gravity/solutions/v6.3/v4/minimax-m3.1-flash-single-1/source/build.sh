#!/bin/sh
# Offline build of the galaxy-snapshot N-body solver.
# Produces ./solve in the submission root.  No network access is used.
set -e
cd "$(dirname "$0")"

CXX=${CXX:-g++}
FLAGS="-O3 -funroll-loops -fno-math-errno -std=c++17"

# Use AVX2+FMA when the host supports it (the grading host is AMD EPYC 7K62),
# otherwise fall back to a generic x86-64 build.
cat > /tmp/galaxy_cpu_check.c <<'CC'
#include <stdio.h>
int main(void){
  __builtin_cpu_init();
  printf("%d\n", (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 1 : 0);
  return 0;
}
CC
CC=${CC:-gcc}
$CC -O0 /tmp/galaxy_cpu_check.c -o /tmp/galaxy_cpu_check 2>/dev/null || true
HAVE_AVX2=0
if [ -x /tmp/galaxy_cpu_check ]; then
  HAVE_AVX2=$(/tmp/galaxy_cpu_check)
fi
if [ "$HAVE_AVX2" = "1" ]; then
  $CXX $FLAGS -march=x86-64-v3 -mtune=generic -o solve solve.cpp
else
  $CXX $FLAGS -march=x86-64 -mtune=generic -o solve solve.cpp
fi
chmod +x solve
rm -f /tmp/galaxy_cpu_check /tmp/galaxy_cpu_check.c
echo "built ./solve (avx2=$HAVE_AVX2)"
