#!/usr/bin/env python3
"""Single-threaded, noise-aware fluorescence deconvolution."""
import os
import sys
import time

START = time.monotonic()
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def solve(observed, psf, deadline):
    from scipy import fft, ndimage
    shape = observed.shape
    full = tuple(fft.next_fast_len(n + m - 1) for n, m in zip(shape, psf.shape))
    crop = tuple(slice(0, n) for n in shape)
    buffer = np.zeros(full, dtype=np.float32)
    buffer[tuple(slice(0, n) for n in psf.shape)] = psf
    kernel = np.roll(buffer, tuple(-(n // 2) for n in psf.shape), axis=(0, 1, 2))
    transfer = fft.rfftn(kernel, workers=1)
    transpose = transfer.conj()

    def convolve(x, adjoint=False):
        buffer.fill(0)
        buffer[crop] = x
        spectrum = fft.rfftn(buffer, workers=1)
        spectrum *= transpose if adjoint else transfer
        return fft.irfftn(spectrum, full, workers=1)[crop].copy()

    smoothed = ndimage.gaussian_filter(observed, 1.4)
    background = observed[smoothed < np.quantile(smoothed, 0.35)]
    positive = background[background > 0]
    sigma = float(np.median(positive) / 0.67448975) if positive.size else 0.008
    sigma = max(sigma, 1e-6)
    alpha = 0.0125 if sigma < 0.015 else 1.0 / 22
    shift = sigma * sigma / alpha
    bias = sigma / np.sqrt(2 * np.pi)
    sensitivity = convolve(np.ones(shape, dtype=np.float32), True)
    sensitivity = np.maximum(sensitivity, 1e-6)
    x = np.maximum(convolve(np.maximum(observed - bias, 0)), 1e-6)
    variance = 0.0
    for axis in range(3):
        marginal = psf.sum(axis=tuple(a for a in range(3) if a != axis))
        positions = np.arange(psf.shape[axis]) - psf.shape[axis] // 2
        variance += float(np.dot(marginal, positions * positions)) / 3
    iterations = int(np.clip((4.0 if sigma < 0.015 else 2.4) * variance * variance, 5, 50))
    for _ in range(iterations):
        if time.monotonic() > deadline - 0.045:
            break
        predicted = np.maximum(convolve(x), 1e-8)
        clipping_bias = bias * np.exp(-predicted / alpha)
        ratio = np.maximum(observed + shift - clipping_bias, 0) / (predicted + shift)
        x *= np.maximum(convolve(ratio, True), 0) / sensitivity
    return np.maximum(np.nan_to_num(x, copy=False), 0).astype(np.float32, copy=False)


def main():
    deadline = min(START + 1.05, float(os.environ.get("DECONV_DEADLINE_MONOTONIC", "inf")) - .055)
    with np.load(sys.argv[1], allow_pickle=False) as data:
        observed = np.asarray(data["observed"], dtype=np.float32)
        psf = np.asarray(data["psf"], dtype=np.float32)
    try:
        from restoration import restore
        result = restore(observed, psf, deadline)
    except Exception:
        if time.monotonic() < deadline - .06:
            result = solve(observed, psf, deadline)
        else:
            result = observed.copy()
    result = np.asarray(result, dtype=np.float32)
    np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
    np.maximum(result, 0, out=result)
    with open(sys.argv[2], "wb") as stream:
        np.save(stream, result, allow_pickle=False)


if __name__ == "__main__":
    main()
