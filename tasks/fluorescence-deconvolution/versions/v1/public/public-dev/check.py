#!/usr/bin/env python3
"""Development-only checker for one fixed example."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('input', type=Path)
    ap.add_argument('truth', type=Path)
    ap.add_argument('estimate', type=Path)
    args=ap.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        if set(data.files) != {'observed','psf'}:
            raise SystemExit('input fields must be exactly observed and psf')
        observed=np.asarray(data['observed']); psf=np.asarray(data['psf'])
    truth=np.asarray(np.load(args.truth, allow_pickle=False)); estimate=np.asarray(np.load(args.estimate, allow_pickle=False))
    if observed.dtype != np.float32 or psf.dtype != np.float32:
        raise SystemExit('input arrays must be float32')
    if estimate.dtype not in (np.float32, np.float64):
        raise SystemExit('estimate must be float32 or float64')
    if observed.ndim != 3 or estimate.shape != observed.shape or truth.shape != observed.shape:
        raise SystemExit('shape mismatch')
    if not np.isfinite(observed).all() or not np.isfinite(psf).all() or not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise SystemExit('non-finite array')
    if (observed < 0).any() or (psf < 0).any() or (truth < 0).any() or (estimate < 0).any():
        raise SystemExit('negative array')
    if abs(float(psf.sum()) - 1.0) > 1e-5:
        raise SystemExit('psf does not sum to one')
    estimate=estimate.astype(np.float64); truth=truth.astype(np.float64)
    mse=float(np.mean((estimate-truth)**2))
    nmse=mse/max(float(np.mean(truth**2)),1e-12)
    print(f'mse={mse:.9g} nmse={nmse:.9g}')

if __name__ == '__main__':
    main()
