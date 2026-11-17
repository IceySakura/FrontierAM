# 3D fluorescence deconvolution submission

`solution.py` is a self-contained NumPy-only solver for one `(Z,Y,X)` case.
It reads `observed` and `psf` from the input `.npz`, writes one float32 `.npy`
of the same shape, and respects the supplied absolute monotonic deadline.

## Algorithm

1. The clipped-Gaussian noise regime is classified from the low-intensity tail
   of the observation (two fixed regimes: sigma 0.008 / Poisson gain 85 and
   sigma 0.025 / gain 25). The censoring bias `sigma/sqrt(2*pi)` is removed;
   the continuous tail estimate is also kept as a feature.
2. A separable nonnegative Richardson-Lucy iteration is run with exact
   zero-extension boundaries (the supplied Gaussian PSFs are separable into
   their three 1-D marginal profiles). Reconstructions at iterations 4, 8, 15,
   20, 25 and 30 are retained.
3. A 106-channel feature vector is computed per voxel: 42 multi-scale features
   (debiased data, RL snapshots, Gaussian smooths, local maxima/means/standard
   deviations at radii 1-3, band-pass details, peakiness, sensitivity, PSF
   widths, boundary distance), 54 raw 3x3x3 neighborhood values, and 10 coarse
   pooled features.
4. Five small offline-trained MLPs are averaged with fixed weights. They differ
   in feature subset, capacity and training seed; separate weights are used for
   the two noise regimes where applicable. All weights are shipped in
   `ensemble_weights.npz` (NumPy `.npz`, no pickle). If the weights cannot be
   read, or if the PSF is not separable, a physics-only RL estimate with mild
   smoothing and a noise-scaled threshold is emitted instead.

The models were trained offline on synthetic cases produced from the twelve
public development truths (random flips, axis permutations, translations,
intensity changes and mixtures), procedurally generated point/tube/sheet
volumes (including boundary-adjacent sheets), Gaussian PSFs with isotropic and
anisotropic widths, and simulated Poisson + clipped-Gaussian noise. Bright
voxels are oversampled and the loss is weighted to approximate the evaluation
nMSE. No hidden data, network access or private files are used. Training code
is not part of the submission; only the fitted weights are shipped.

## Dependencies and sources

- Python 3.11 and NumPy 1.26, both preinstalled in the offline image.
- `scipy` and `torch` were used only offline while developing and training the
  weights; they are not needed at evaluation time.
- All solver and feature code was written for this task. The public development
  examples in `/workspace/public-dev` are the only external data source.
- No compiled assets, symlinks or network resources are used.

Typical full-process runtime on the development core is about 0.43 s, inside
the 0.500 s per-case limit; the isolated grading core is faster.
