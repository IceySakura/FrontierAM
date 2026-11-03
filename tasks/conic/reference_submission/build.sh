#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1
rm -f build_manifest.json
python3 -m pip install --no-index --only-binary=:all: --no-deps --upgrade \
    --find-links "$PWD/dependencies/wheels" --target "$PWD/.deps" -r requirements.txt
export PYTHONPATH="$PWD/.deps:$PWD"
python3 - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

import clarabel
import numpy as np
import scipy
from scipy import sparse

if sys.version_info[:2] != (3, 12):
    raise RuntimeError('The reference environment requires Python 3.12')
expected = {'clarabel': '0.11.1', 'cffi': '2.0.0', 'pycparser': '2.23',
            'numpy': '2.3.5', 'scipy': '1.16.3'}
versions = {name: metadata.version(name) for name in expected}
if versions != expected:
    raise RuntimeError(f'Incorrect reference dependencies: {versions!r}')
if not Path(clarabel.__file__).resolve().is_relative_to(Path('.deps').resolve()):
    raise RuntimeError('Clarabel must load from the offline installation in .deps')

# min t subject to [[t, 1], [1, t]] PSD; optimum t = 1.
# An actual solve checks native SDP support, not just the cone class name.
settings = clarabel.DefaultSettings()
settings.verbose = False
settings.max_threads = 1
solver = clarabel.DefaultSolver(
    sparse.csc_matrix((1, 1)), np.array([1.0]),
    sparse.csc_matrix([[-1.0], [0.0], [-1.0]]),
    np.array([0.0, np.sqrt(2.0), 0.0]), [clarabel.PSDTriangleConeT(2)], settings)
solution = solver.solve()
if str(solution.status) != 'Solved' or abs(solution.x[0] - 1.0) > 1e-6:
    raise RuntimeError(f'Native SDP smoke check failed: {solution}')
report = {'python': platform.python_version(), 'platform': platform.platform(),
          'machine': platform.machine(), 'versions': versions,
          'clarabel_module': str(Path(clarabel.__file__).resolve()),
          'sdp_smoke': {'status': str(solution.status), 'x': solution.x,
                        'linear_solver': solver.get_info().linsolver.name},
          'wheel_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(Path('dependencies/wheels').glob('*.whl'))}}
Path('build_manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print('Offline reference build and native SDP smoke check passed.')
PY
