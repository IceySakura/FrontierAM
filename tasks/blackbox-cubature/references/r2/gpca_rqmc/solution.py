"""GPCA-RQMC (Xiao and Wang 2019; the active-subspace basis of Liu and Owen 2023): the frozen 50-point reference of
blackbox-cubature v3, a protocol port of versions/v3/baselines/methods.py:as_rqmc.

1. Pilot: forward-difference gradients (step 0.05) at min(64, floor(N/8 / (d+1))) scrambled Sobol' points.
2. Main phase: one scrambled Sobol' net of the largest power-of-two size that fits in the remaining budget, mapped to
   N(0, I_d) and expressed in the eigenbasis of C = mean(grad grad^T), leading eigenvector first; the queries left
   over before it buy more gradients (they only refine the basis).
3. Estimate: the mean over the net. No error bound (one net gives no variance estimate).
"""
import warnings

import numpy as np
from scipy import special, stats

from cubature_protocol import Session

FD_STEP = 0.05
PILOT_FRACTION = 1 / 8
PILOT_GRADIENTS = 64


def sobol_normal(d: int, n: int, seed: int) -> np.ndarray:
    sampler = stats.qmc.Sobol(d, scramble=True, rng=np.random.default_rng(seed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = sampler.random(n)
    return special.ndtri(np.clip(u, 1e-16, 1 - 1e-16))


def pow2(n: int) -> int:
    return 1 << (int(n).bit_length() - 1)


def gradients(session: Session, n: int, seed: int) -> np.ndarray:
    d = session.d
    z = sobol_normal(d, n, seed)
    vals = session.query(np.concatenate([z] + [z + FD_STEP * e for e in np.eye(d)])).reshape(d + 1, n)
    return (vals[1:] - vals[0]).T / FD_STEP


def basis(grads: np.ndarray) -> np.ndarray:
    eig, vec = np.linalg.eigh(grads.T @ grads / len(grads))
    return vec[:, np.argsort(eig)[::-1]]


def main():
    session = Session()
    d, seed = session.d, session.seed
    grads = gradients(session, max(1, min(PILOT_GRADIENTS, int(session.budget * PILOT_FRACTION) // (d + 1))), seed + 1)
    main_size = pow2(session.remaining)
    extra = (session.remaining - main_size) // (d + 1)
    if extra:
        grads = np.vstack([grads, gradients(session, extra, seed + 2)])
    session.result(float(session.query(sobol_normal(d, main_size, seed) @ basis(grads).T).mean()))


if __name__ == '__main__':
    main()
