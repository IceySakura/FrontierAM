"""Independent direct-sum check of the galaxy-gravity v4 (B.4) public development references.

For every snapshot in tasks/galaxy-gravity/versions/v4/data/samples (smoke and the five 65,536-particle cases):
checks the manifest SHA-256 and lengths, recomputes all softened accelerations by pairwise summation in float64
(G = 1, eps = 0.01, chunked), and reports the RMS relative difference to the shipped reference with the statement's
formula. Run on a dev lease: PYTHONPATH=. FRONTIERAM_RUNTIME=... scripts/ops/with_lease.py -- python3 THIS
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

SAMPLES = Path(__file__).resolve().parents[2] / 'tasks/galaxy-gravity/versions/v4/data/samples'
EPS2 = 0.01 ** 2


def direct(body: np.ndarray, chunk: int = 128) -> np.ndarray:
    m, x = body[:, 0], body[:, 1:]
    out = np.empty_like(x)
    for start in range(0, len(x), chunk):
        d = x[None, :, :] - x[start:start + chunk, None, :]
        r2 = np.einsum('ijk,ijk->ij', d, d) + EPS2
        w = m[None, :] / (r2 * np.sqrt(r2))
        for i in range(min(chunk, len(x) - start)):
            w[i, start + i] = 0.0
        out[start:start + chunk] = np.einsum('ij,ijk->ik', w, d)
    return out


def rms_relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)) ** 2)))


def main() -> None:
    manifest = json.loads((SAMPLES / 'data/manifest.json').read_text())
    cases = [('smoke', SAMPLES / 'smoke.in', SAMPLES / 'smoke.ref', None, None)]
    cases += [(c['id'], SAMPLES / 'data' / f'{c["id"]}.in', SAMPLES / 'data' / f'{c["id"]}.ref',
               c['input_sha256'], c['reference_sha256']) for c in manifest['cases']]
    for name, inp, ref, sha_in, sha_ref in cases:
        raw_in, raw_ref = inp.read_bytes(), ref.read_bytes()
        hashes_ok = sha_in is None or (hashlib.sha256(raw_in).hexdigest() == sha_in
                                       and hashlib.sha256(raw_ref).hexdigest() == sha_ref)
        body = np.frombuffer(raw_in, dtype='<f8').reshape(-1, 4)
        want = np.frombuffer(raw_ref, dtype='<f8').reshape(-1, 3)
        start = time.monotonic()
        got = direct(body)
        print(json.dumps({'case': name, 'n': len(body), 'hashes_ok': hashes_ok, 'total_mass': float(body[:, 0].sum()),
                          'equal_mass': bool(np.all(body[:, 0] == body[0, 0])),
                          'rms_relative_vs_reference': rms_relative(got, want),
                          'seconds': round(time.monotonic() - start, 1)}), flush=True)


if __name__ == '__main__':
    main()
