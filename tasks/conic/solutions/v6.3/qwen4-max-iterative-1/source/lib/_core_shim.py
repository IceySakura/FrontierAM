import os
import sys
import glob
for p in glob.glob("/usr/local/lib/python3*/site-packages/scipy/optimize") + glob.glob("/usr/lib/python3*/site-packages/scipy/optimize"):
    sys.path.insert(0, p)
from _highspy import _core
core = _core
