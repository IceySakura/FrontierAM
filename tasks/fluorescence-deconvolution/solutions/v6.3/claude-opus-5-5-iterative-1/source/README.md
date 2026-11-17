# 3-D fluorescence deconvolution (32³ volume, 9³ PSF) on one CPU core

```sh
python3 -I -B solution.py /abs/raw.npz /abs/pred.npy
```

Pure numpy at run time (no scipy/torch import; interpreter + numpy import ≈ 0.1 s).
Total wall time ≈ 0.83 s per case on the development core (limit 1.25 s), adaptive to the deadline.

## Method

1. **Exact forward model.** `same` convolution with zero extension. When the PSF is separable
   (checked numerically; all observed PSFs are axis-aligned Gaussians) the operator is applied as
   three small Toeplitz-matrix products along z, y, x (exact boundary handling, ~0.3 ms per A/Aᵀ
   pair). Otherwise an FFT of size N+(K−1)/2 per axis is used (no wrap-around inside the crop).
2. **Noise estimate.** The background is zero-mean Gaussian noise clipped at 0, so about half of
   the background voxels are exact zeros and the background positives are half-normal:
   σ ≈ (median of the smallest `n_zero` positive values) / 0.6745. The Poisson scale is tied to the
   regime (σ > 0.015 → 24.5 photons per unit intensity, else 86), giving the shift `s = σ²k` of a
   shifted-Poisson likelihood.
3. **Richardson–Lucy features.** Shifted-Poisson RL, `x ← x · Aᵀ((y+s)/(Ax+s)) / Aᵀ1`, iterates
   at 4, 12 and 36 iterations, plus the observation and four constant conditioning channels
   (σ and the PSF standard deviation along each axis). RL-12 is written to the output first, as a
   fallback.
4. **Learned refinement.** A small 3-level 3-D U-Net (16/32/64 channels, 3×3×3 convolutions, ReLU,
   stride-2 convolution down-sampling, stride-2 transposed-convolution up-sampling, skip
   concatenations, output = RL-12 + correction) maps the 8 feature channels to the clean volume.
   Inference is float32 numpy (flat-shift im2col + BLAS GEMM, ≈ 0.16 s per pass). The result is
   clipped at 0.
5. **Flip test-time augmentation.** The prediction is averaged over the identity and y-, x- and
   yx-flipped inputs (flips along y/x are symmetries of the training distribution; z is not, because
   of the boundary sheet). Each extra pass is started only if the measured pass time leaves a wide
   margin before `DECONV_DEADLINE_MONOTONIC` (absent → assume start + 1.0 s); the single-pass result
   is written first, and any exception leaves the last written result in place.

## Training (offline, before submission)

`model.npz` was produced by our own training script with PyTorch 2.2 (CPU, one thread, ~3 h in
total, Adam, cosine learning-rate decay, per-case normalized-MSE loss, ~36k on-the-fly samples);
the shipped weights are the average of the last three checkpoints of the low-learning-rate phase.
Training data come from a synthetic generator written after inspecting the public examples:
straight capsule tubes with Gaussian cross-section, isotropic Gaussian blobs, mixtures of both,
and a boundary scene (thin Gaussian sheet next to the z = 0 face, a wall with a linear intensity
ramp, small blobs), each normalized to max 1; Gaussian PSFs truncated to 9³ (mostly the three
shapes seen publicly, 20 % random isotropic/anisotropic widths); noise
`max(0, Poisson(k·Ax)/k + N(0, σ²))` in the two observed regimes with jitter; random y/x flips.
The public truth volumes were used only for validation, never for training.

Development results (12 public examples, mean nMSE): shifted-Poisson RL with oracle stopping 0.152,
RL-12 alone 0.175, this method 0.026.

## Files

- `solution.py` – self-contained solver (numpy only).
- `model.npz` – U-Net weights (float32, ~0.43 M parameters), from our own training run.
- `README.md` – this file.

## Dependencies

Python 3.11 and numpy 1.26 as installed in the provided environment. No build step, no network.
