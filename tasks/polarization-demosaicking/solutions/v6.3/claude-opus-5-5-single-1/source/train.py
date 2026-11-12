"""Offline training of the gated filters (not needed at run time).

Usage: python3 train.py [samples_dir]   (default: ../samples)
Fits per-phase least-squares weights on the public development scenes
wen-001 and wen-002 and writes gated_r4.npz next to this file.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solution as S  # noqa: E402

LAM = 3e-5  # ridge strength, relative to mean diagonal of the normal matrix


def collect(scenes):
    acc = {}
    for raw, ref in scenes:
        for pr, pc, X, g, _ in S.phase_inputs(raw):
            Y = ref[:, pr::2, pc::2].reshape(4, -1).T.astype(np.float64)
            a = acc.setdefault((pr, pc), [0, 0])
            for s in range(0, X.shape[0], 32768):
                Xc = X[s:s + 32768].astype(np.float64)
                Xc = np.hstack([Xc, np.ones((Xc.shape[0], 1))])
                F = (Xc[:, :, None] * g[s:s + 32768, None, :]).reshape(Xc.shape[0], -1)
                a[0] = a[0] + F.T @ F
                a[1] = a[1] + F.T @ Y[s:s + 32768]
    return acc


def solve(acc):
    out = {}
    for (pr, pc), (XtX, XtY) in acc.items():
        n = XtX.shape[0]
        w = np.linalg.solve(XtX + LAM * np.trace(XtX) / n * np.eye(n), XtY)
        c_obs = S._PHASE_OBS[(pr, pc)]
        cs = [c for c in range(4) if c != c_obs]
        G = 6
        nx1 = n // G
        w = w.reshape(nx1, G, 4)[:, :, cs]
        out["w%d%d" % (pr, pc)] = w[:-1].reshape(nx1 - 1, G * 3).astype(np.float32)
        out["b%d%d" % (pr, pc)] = w[-1].reshape(G * 3).astype(np.float32)
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "samples")
    scenes = []
    for name in ["wen-001", "wen-002"]:
        scenes.append((np.load(os.path.join(root, name, "raw.npy")),
                       np.load(os.path.join(root, name, "reference.npy"))))
    np.savez(os.path.join(HERE, "gated_r4.npz"), **solve(collect(scenes)))
    print("wrote gated_r4.npz")
