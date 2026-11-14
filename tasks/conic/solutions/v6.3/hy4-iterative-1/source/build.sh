#!/bin/sh
# Offline preparation: place a copy of SciPy's bundled HiGHS extension next
# to the sources so run.sh never depends on a SciPy import or its exact path.
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if ls "$DIR"/_core*.so >/dev/null 2>&1; then
    exit 0
fi
SOLVER_DIR="$DIR" python - <<'PY'
import glob, os, shutil, sys
here = os.environ.get("SOLVER_DIR", os.getcwd())
tgt = glob.glob(os.path.join(here, "_core*.so"))
if tgt:
    sys.exit(0)
for root in sys.path:
    for p in glob.glob(os.path.join(root, "scipy", "optimize", "_highspy", "_core*.so")):
        try:
            shutil.copy2(p, os.path.join(here, os.path.basename(p)))
            sys.exit(0)
        except OSError:
            pass
# Failure is non-fatal: lpqp.py can still find the extension through sys.path.
PY
exit 0
