#!/usr/bin/env python3
"""Reproduce the model identification and score solver.py on the public dev pairs.

    /workspace/runtime/bin/python3 /workspace/submission/dev/selftest.py
"""
import json
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import solver  # noqa: E402

DEV = '/workspace/public/dev'


def load(cid):
    inp = json.load(open(f'{DEV}/{cid}.input.json'))
    tr = json.load(open(f'{DEV}/{cid}.truth.json'))
    return np.array(inp['x'], float), np.array(inp['y'], float), tr


def identify():
    print('=== identification: residual of y = t + a x^p + b x^q (+ c x^(q+3/2)) ===')
    for f in sorted(glob.glob(f'{DEV}/*.input.json')):
        cid = os.path.basename(f)[:-len('.input.json')]
        x, y, tr = load(cid)
        t, p, q = tr['truth'], tr['p'], tr['q']
        A2 = np.vstack([x ** p, x ** q]).T
        c2 = np.linalg.lstsq(A2, y - t, rcond=None)[0]
        r2 = np.abs(A2 @ c2 - (y - t)).max()
        A3 = np.vstack([np.ones_like(x), x ** p, x ** q, x ** (q + 1.5)]).T
        c3 = np.linalg.lstsq(A3, y, rcond=None)[0]
        r3 = np.abs(A3 @ c3 - y).max()
        print(f'{cid} {tr["family"]:18s} n={len(x)}  2-term {r2:.2e}   '
              f'3-term(r=q+3/2) {r3:.2e}   a={c3[1]:+.5f} b={c3[2]:+.5f} c={c3[3]:+.5f}')


def score():
    print('\n=== scoring solver.estimate on the dev pairs ===')
    tot, k = [], []
    for f in sorted(glob.glob(f'{DEV}/*.input.json')):
        cid = os.path.basename(f)[:-len('.input.json')]
        x, y, tr = load(cid)
        z = solver.estimate(x, y)
        t = tr['truth']
        L = abs(z - t) / max(1.0, abs(t))
        tag = '' if tr['family'] == 'smooth-primary' else '   (audit family, not scored)'
        print(f'{cid} n={len(x)} true={t:+.8f} est={z:+.8f} loss={L:.3e}{tag}')
        if tr['family'] == 'smooth-primary':
            tot.append(L)
        k.append(L)
    print(f'mean loss over the {len(tot)} smooth-primary cases : {np.mean(tot):.4e}')
    print(f'mean loss over all {len(k)} cases                 : {np.mean(k):.4e}')


if __name__ == '__main__':
    identify()
    score()
