# Offline deconvolution submission

Algorithm:
1. Detect a rank-one separable PSF and run 30 nonnegative Richardson-Lucy
   iterations with exact zero-extension convolution in a small C helper.
   A NumPy FFT fallback handles nonseparable PSFs.
2. Apply an ensemble of compact learned patch denoisers (57 inputs, one ReLU
   hidden layer) to multi-iteration RL features and the observed volume.
   Inference and two-flip test-time augmentation run in the C helper.
3. For very sparse point-like outputs, normalize the peak; clip to the unit
   intensity range, remove a small relative floor, and save float32.

The weights in `mlp_weights.npz`, `mlp_torch.npz`, and `mlp_torch128.npz`
were trained offline only on the supplied public development examples and
their orientation augmentations. No hidden data, network access, or external
datasets are used.

Dependencies: Python 3, NumPy, and a C compiler for the optional build.
`build.sh` recompiles `deconv.c`; a prebuilt shared object is also included.
The program honors `DECONV_DEADLINE_MONOTONIC` and `TIME_BUDGET_SECONDS`.
