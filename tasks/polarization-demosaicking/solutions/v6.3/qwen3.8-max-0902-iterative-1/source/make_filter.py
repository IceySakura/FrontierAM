"""Offline step: pre-compute and store the polyphase transfer map.

The map depends only on the padded grid size and the prior weights, never on
the scene, so it is computed once (see build.sh) and memory-mapped at run time.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import solution  # noqa: E402


def main():
    H = W = 1024
    Mh, Mw = (H + 2 * solution.PAD) // 2, (W + 2 * solution.PAD) // 2
    m = solution.build_map(Mh, Mw)
    if not np.isfinite(m.view(np.float32)).all():
        raise SystemExit("non-finite filter map")
    path = solution._map_path(Mh, Mw)
    np.save(path, m)
    print("wrote %s  shape=%s  %.1f MB" % (path, m.shape, m.nbytes / 1e6))


if __name__ == "__main__":
    main()
