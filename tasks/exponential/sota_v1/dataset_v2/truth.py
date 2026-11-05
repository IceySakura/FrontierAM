"""Reference solutions for dataset-v2 cases, each verified by an independent second method.

Primary: exp(-tL) b0 and t*phi1(-tL) b1 are expanded separately in Chebyshev series on the PSD spectral interval
[0, lam], lam = ||L||_inf (a rigorous bound); both functions are bounded there (by 1 and t), so no cancellation
arises at large t*lam. Coefficients come from samples at Chebyshev points and the whole recurrence runs in x87
extended precision (eps ~1.1e-19), so rounding stays far below the targets; the series is truncated where the
coefficient envelope falls below TAIL x the case's acceptance threshold.

Audit: the Al-Mohy--Higham truncated-Taylor action (scipy.sparse.linalg.expm_multiply, binary64) on the augmented
operator t*[[-L, b1],[0, 0]] applied to [b0; 1], an unrelated method family. A case qualifies when, for every
request, ||primary - audit||_inf <= AGREE x (atol + rtol*||primary||_inf). On disagreement, operators with
n <= ARBITRATE_MAX_N are arbitrated by dense eigendecomposition and qualify if the primary is within AGREE of it (the
binary64 Taylor audit itself loses accuracy on some log-spectrum requests, where primary and eigh agree to 1e-4).
Qualified cases get y_ref = primary; any other case stays unqualified and must not be scored.

Validation on the public smoke cases against dense eigendecomposition: both methods within 0.001 of the threshold
except the deliberately ill-conditioned log-spectrum operator, where they agree with each other to 0.003 and the
dense eigensolver itself is the inaccurate one. (A binary64 Chebyshev primary was tried first and lost accuracy at
large t*lam through rounding amplified by max f = t; a Lanczos audit stagnated before converging on stiff operators.)

Usage: truth.py MANIFEST [--jobs N] [--cpus LIST]
"""

import argparse
import hashlib
import json
import os
import time

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from scipy.fft import dct
from scipy.sparse.linalg import expm_multiply

TAIL = 1e-3
AGREE = 0.02
ARBITRATE_MAX_N = 6000
EPS = np.finfo(np.longdouble).eps


class LongCSR:
    """CSR product in x87 extended precision (eps ~1.1e-19); SciPy's sparse kernels are binary64 only."""

    def __init__(self, L: sp.csr_matrix):
        self.data = L.data.astype(np.longdouble)
        self.indices = L.indices
        self.starts = L.indptr[:-1].astype(np.int64)
        self.empty = np.diff(L.indptr) == 0
        self.n = L.shape[0]

    def __matmul__(self, x):
        products = self.data * x[self.indices]
        if products.size == 0:
            return np.zeros(self.n, dtype=np.longdouble)
        # reduceat returns the start element for empty segments; those rows are exactly zero.
        out = np.add.reduceat(products, np.minimum(self.starts, products.size - 1))
        out[self.empty] = 0
        return out


def chebyshev(f, lam, target: float) -> np.ndarray:
    """Coefficients of f on [0, lam]: sample until the series has converged to rounding level, then truncate where
    the coefficient envelope is below target (or at the noise floor if rounding prevents reaching target)."""
    size = 64
    while True:
        nodes = np.cos(np.pi * (np.arange(size, dtype=np.longdouble) + 0.5) / size)
        samples = f(np.longdouble(0.5) * lam * (nodes + 1))
        c = dct(samples, type=2) / size
        c[0] *= 0.5
        # Computed coefficients carry rounding noise of order eps * max|f|, not eps * max|c|.
        floor = 64 * EPS * float(np.abs(samples).max())
        if float(np.abs(c[size // 2:]).max()) <= floor:
            break
        if size > 2**21:
            raise RuntimeError('Chebyshev series did not converge within 2M terms')
        size *= 2
    # Past the knee the coefficients of these entire functions decay super-geometrically, so the largest remaining
    # coefficient bounds the tail up to a small factor; summing would add up rounding noise instead.
    envelope = np.maximum.accumulate(np.abs(c)[::-1])[::-1]
    below = np.nonzero(envelope <= max(target / 4, floor))[0]
    return c[: max(int(below[0]), 1)]


def clenshaw(L, lam, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """sum_k c_k T_k(s) v with s = (2/lam) L - I."""
    def apply(w):
        return (2 / lam) * (L @ w) - w

    b_next = np.zeros_like(v)
    b_cur = np.zeros_like(v)
    for ck in c[:0:-1]:
        b_cur, b_next = 2 * apply(b_cur) - b_next + ck * v, b_cur
    return apply(b_cur) - b_next + c[0] * v


def primary(L: sp.csr_matrix, b0: np.ndarray, b1: np.ndarray, t: float, threshold: float) -> tuple[np.ndarray, int]:
    if t == 0.0 or L.nnz == 0 or not (b0.any() or b1.any()):
        return b0 + t * b1, 0
    lam = np.longdouble(max(float(abs(L).sum(axis=1).max()), 1e-300))
    tl = np.longdouble(t)
    op = LongCSR(L)
    y = np.zeros(b0.shape, dtype=np.longdouble)
    degree = 0
    if b0.any():
        c = chebyshev(lambda x: np.exp(-tl * x), lam, TAIL * threshold / (2 * float(np.abs(b0).max())))
        y += clenshaw(op, lam, c, b0.astype(np.longdouble))
        degree = max(degree, len(c))
    if b1.any():
        def f(x):
            safe = np.where(x > 0, x, np.longdouble(1))
            return np.where(x > 0, -np.expm1(-tl * safe) / safe, tl)
        c = chebyshev(f, lam, TAIL * threshold / (2 * float(np.abs(b1).max())))
        y += clenshaw(op, lam, c, b1.astype(np.longdouble))
        degree = max(degree, len(c))
    return y.astype(np.float64), degree


def audit(L: sp.csr_matrix, b0: np.ndarray, b1: np.ndarray, t: float) -> np.ndarray:
    n = L.shape[0]
    if t == 0.0 or L.nnz == 0:
        return b0 + t * b1
    operator = sp.bmat([[-t * L, sp.csr_matrix((t * b1).reshape(n, 1))], [None, sp.csr_matrix((1, 1))]], format='csr')
    start = np.concatenate([b0, [1.0]])
    return expm_multiply(operator, start, traceA=float(-t * L.diagonal().sum()))[:n]


def qualify(job):
    path, case = job
    z = dict(np.load(path, allow_pickle=False))
    n = int(z['shape'][0])
    L = sp.csr_matrix((z['data'], z['indices'].astype(np.int64), z['indptr'].astype(np.int64)), shape=(n, n))
    t, rtol, atol = float(z['t']), float(z['rtol']), float(z['atol'])
    started = time.process_time()
    ys, worst, degree = [], 0.0, 0
    for b0, b1 in zip(z['b0'], z['b1']):
        guess = audit(L, b0, b1, t)
        y, d = primary(L, b0, b1, t, atol + rtol * float(np.abs(guess).max()))
        degree = max(degree, d)
        allowed = atol + rtol * float(np.abs(y).max())
        worst = max(worst, float(np.abs(y - guess).max()) / allowed)
        ys.append(y)
    arbitration = None
    if worst > AGREE and n <= ARBITRATE_MAX_N:
        w, V = np.linalg.eigh(L.toarray())
        w = np.clip(w, 0, None)
        safe = np.where(w > 0, w, 1)
        phi = np.where(w > 0, -np.expm1(-t * safe) / safe, t)
        arbitration = 0.0
        for y, b0, b1 in zip(ys, z['b0'], z['b1']):
            dense = V @ (np.exp(-t * w) * (V.T @ b0) + phi * (V.T @ b1))
            arbitration = max(arbitration, float(np.abs(y - dense).max()) / (atol + rtol * float(np.abs(y).max())))
    agreed = worst <= AGREE or (arbitration is not None and arbitration <= AGREE)
    qualified = bool(agreed and all(np.isfinite(y).all() for y in ys))
    result = {'id': case['id'], 'qualified': qualified, 'audit_disagreement_fraction': worst, 'chebyshev_degree': degree,
              'cpu_seconds': time.process_time() - started}
    if arbitration is not None:
        result['dense_arbitration_fraction'] = arbitration
    if qualified:
        z['y_ref'] = np.asarray(ys, dtype='<f8')
        temp = path.with_name(path.stem + '.tmp.npz')
        np.savez_compressed(temp, **z)
        os.replace(temp, path)
        result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--cpus', help='comma-separated CPUs to pin the workers to')
    args = parser.parse_args()
    if args.cpus:
        os.sched_setaffinity(0, {int(c) for c in args.cpus.split(',')})
    manifest = json.loads(args.manifest.read_text())
    todo = [(args.manifest.parent / c['file'], c) for c in manifest['cases']
            if c.get('role') == 'scored' and c.get('truth_status') != 'chebyshev_audited_qualified']
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = {r['id']: r for r in pool.map(qualify, todo)}
    for case in manifest['cases']:
        r = results.get(case['id'])
        if r is None:
            continue
        case['truth'] = {k: v for k, v in r.items() if k not in ('id', 'sha256')}
        if r['qualified']:
            case.update(truth_status='chebyshev_audited_qualified', sha256=r['sha256'])
        else:
            case['truth_status'] = 'audit_disagreement_unqualified'
    manifest['truth_qualified'] = all(c.get('role') != 'scored' or c.get('truth_status') == 'chebyshev_audited_qualified'
                                      for c in manifest['cases'])
    manifest['truth_method'] = {'primary': 'extended-precision Chebyshev, exp and phi1 expanded separately, envelope <= 1e-3 of threshold',
                                'audit': 'Al-Mohy--Higham expm_multiply on the augmented operator (binary64)',
                                'arbitration': f'dense eigendecomposition when the audit disagrees and n <= {ARBITRATE_MAX_N}',
                                'agreement': f'<= {AGREE} of the acceptance threshold per request'}
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    bad = [r for r in results.values() if not r['qualified']]
    print(json.dumps({'cases': len(results), 'qualified': len(results) - len(bad),
                      'worst_disagreement': max((r['audit_disagreement_fraction'] for r in results.values()), default=None),
                      'max_degree': max((r['chebyshev_degree'] for r in results.values()), default=None),
                      'cpu_seconds': round(sum(r['cpu_seconds'] for r in results.values()), 1),
                      'unqualified': [r['id'] for r in bad]}))


if __name__ == '__main__':
    main()
