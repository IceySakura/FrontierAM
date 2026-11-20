#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
blas=$(python -c 'import pathlib,sysconfig; print(next((pathlib.Path(sysconfig.get_path("platlib")) / "scipy.libs").glob("libscipy_openblas-*.so")))')
py_lib=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
py_name=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')
python dependencies.py > deps_autogen.hpp
cxx=${CXX:-g++}
flags="-std=c++17 -O3 -DNDEBUG -Wall -Wextra"
"$cxx" $flags -D_GLIBCXX_USE_CXX11_ABI=0 -c numeric.cpp -o numeric.o
"$cxx" $flags main.cpp model.cpp optimizer.cpp kkt.cpp reduce.cpp cuts.cpp split.cpp smooth.cpp diagsdp.cpp box.cpp cert.cpp logsum.cpp native.cpp numeric.o \
    -L"$(dirname "$blas")" -Wl,-rpath,"$(dirname "$blas")" -l:"$(basename "$blas")" \
    -L"$py_lib" -Wl,-rpath,"$py_lib" -Wl,--no-as-needed -l:"$py_name" -Wl,--as-needed \
    -l:libz.so.1 -ldl -o conic
rm -f numeric.o
