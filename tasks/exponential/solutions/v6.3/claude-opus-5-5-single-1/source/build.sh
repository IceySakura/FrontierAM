#!/bin/bash
# Rebuild the compiled kernels (prebuilt copies are shipped; failures keep them).
cd "$(dirname "$0")"
gcc -O3 -march=x86-64 -mtune=generic -fPIC -shared -o libcheb_generic.so.tmp cheb.c -lm && mv -f libcheb_generic.so.tmp libcheb_generic.so
gcc -O3 -mavx2 -mfma -fPIC -shared -o libcheb_avx2.so.tmp cheb.c -lm && mv -f libcheb_avx2.so.tmp libcheb_avx2.so
rm -f libcheb_generic.so.tmp libcheb_avx2.so.tmp
exit 0
