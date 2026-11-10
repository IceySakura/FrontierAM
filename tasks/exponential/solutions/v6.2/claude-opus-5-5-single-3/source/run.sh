#!/bin/sh
# Entry point: start the solver with an interpreter that has NumPy and SciPy.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -x /usr/local/bin/python3 ]; then
    exec /usr/local/bin/python3 solver.py
fi
for py in python3 python3.12 /usr/bin/python3 python; do
    if "$py" -c "import numpy, scipy" >/dev/null 2>&1; then
        exec "$py" solver.py
    fi
done
exec python3 solver.py
