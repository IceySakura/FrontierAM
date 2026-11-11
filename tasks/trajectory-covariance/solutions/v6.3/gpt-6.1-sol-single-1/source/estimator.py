#!/usr/bin/env python3
"""Adaptive overlapping-batch covariance estimation for one trajectory."""

import argparse

import numpy as np


def _batch_width(centered, covariance):
    n, d = centered.shape
    values, vectors = np.linalg.eigh(covariance)
    largest = float(values[-1])
    if largest <= 0.0:
        return 1
    active = values > largest * (64.0 * np.finfo(np.float64).eps)
    if not np.any(active):
        return 1
    transform = vectors[:, active] / np.sqrt(values[active])
    lag = centered[:-1].T @ centered[1:] / (n - 1)
    lag = transform.T @ lag @ transform
    lag = (lag + lag.T) * 0.5
    persistence = float(np.max(np.abs(np.linalg.eigvalsh(lag))))
    persistence = min(0.995, max(0.0, persistence))
    noise_floor = np.sqrt(np.count_nonzero(active) / n)
    if persistence < noise_floor:
        return 1
    factor = 6.0 * n * persistence**2 / (1.0 - persistence**2) ** 2
    return max(1, min(n // 10, int(np.ceil(np.cbrt(factor)))))


def _overlapping_batches(centered, width, cumulative):
    n = len(centered)
    if width == 1:
        return centered.T @ centered / (n - 1)
    sums = cumulative[width:] - cumulative[:-width]
    count = n - width + 1
    factor = n / (width * count * (n - width))
    return (sums.T @ sums) * factor


def estimate_long_run_covariance(trajectory):
    x = np.asarray(trajectory, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or len(x) < 2:
        raise ValueError("input must have shape (M, d), with M >= 2 and d >= 1")
    if not np.isfinite(x).all():
        raise ValueError("input contains nonfinite observations")
    n, d = x.shape
    # Subtract an observed anchor first to keep centering stable after translation.
    centered = x - x[0]
    centered -= centered.mean(axis=0)
    covariance = centered.T @ centered / n
    width = _batch_width(centered, covariance)
    cumulative = np.empty((n + 1, d), dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(centered, axis=0, out=cumulative[1:])
    widths = sorted(
        {max(1, min(n // 4, int(round(width * scale))))
         for scale in (0.75, 1.0, 1.5)}
    )
    estimate = np.zeros((d, d), dtype=np.float64)
    for batch in widths:
        estimate += _overlapping_batches(centered, batch, cumulative)
    estimate /= len(widths) * n
    estimate = (estimate + estimate.T) * 0.5
    if not np.isfinite(estimate).all():
        raise ValueError("the estimated covariance is not finite")
    return np.ascontiguousarray(estimate, dtype="<f8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="input trajectory, a C-order float64 NPY array")
    parser.add_argument("output", help="output covariance NPY file")
    args = parser.parse_args()
    trajectory = np.load(args.input, allow_pickle=False)
    result = estimate_long_run_covariance(trajectory)
    with open(args.output, "wb") as stream:
        np.save(stream, result, allow_pickle=False)


if __name__ == "__main__":
    main()
