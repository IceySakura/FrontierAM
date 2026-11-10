#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 gen_unrolled.py >/dev/null
# 1. Default frozen paper-based baseline (p = 8, theta = 0.48, MAC = 0, unrolled = 0, split = 0)
g++ -O3 -std=c++17 -march=x86-64-v3 -funroll-loops -fno-math-errno -fno-trapping-math solve.cpp -o solve
# 2. Switchable ablation variant wrappers (using the same compiled binary with variant flags)
cat << 'EOF' > solve-A
#!/bin/sh
exec env FMM_UNROLLED=1 FMM_ORDER=8 FMM_THETA=0.45 FMM_MAC=0 FMM_NCRIT=96 FMM_NCC_POST=96 FMM_NCS=256 "$(dirname "$0")/solve" "$@"
EOF
cat << 'EOF' > solve-AB
#!/bin/sh
exec env FMM_UNROLLED=1 FMM_AUTOTUNE=1 FMM_TARGET_CPU=3.45 FMM_NCRIT=96 FMM_NCC_POST=96 FMM_NCS=256 "$(dirname "$0")/solve" "$@"
EOF
cat << 'EOF' > solve-C
#!/bin/sh
exec env FMM_SPLIT=1 FMM_UNROLLED=0 FMM_ORDER=8 FMM_RES_ORDER=6 FMM_THETA=0.44 FMM_MAC=0 FMM_NCRIT=96 FMM_NCC_POST=96 FMM_NCS=256 "$(dirname "$0")/solve" "$@"
EOF
cat << 'EOF' > solve-ABC
#!/bin/sh
exec env FMM_SPLIT=1 FMM_UNROLLED=1 FMM_AUTOTUNE=1 FMM_RES_ORDER=6 FMM_TARGET_CPU=3.25 FMM_NCRIT=96 FMM_NCC_POST=96 FMM_NCS=256 "$(dirname "$0")/solve" "$@"
EOF
chmod +x solve-A solve-AB solve-C solve-ABC
