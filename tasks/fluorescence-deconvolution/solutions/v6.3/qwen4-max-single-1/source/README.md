# Three-dimensional fluorescence deconvolution

This submission reconstructs a nonnegative volume from the observed blurred
volume and centered odd PSF. It uses only the current input and files in this
directory.

## Method

1. Estimate a robust noise level from second differences of the observation.
2. If the PSF is separable (the public PSFs are), run a short nonnegative
   Richardson-Lucy iteration with Huber-total-variation regularization in a
   small C library. The forward model uses zero extension and the adjoint
   accounts for boundary loss. A brief probe run estimates energy concentration
   and adapts the regularization strength.
3. Apply an edge-preserving bilateral post-filter whose range scale follows the
   estimated noise level, followed by a small nonnegative total-variation
   proximal cleanup.
4. If the PSF is not separable or the C library is unavailable, fall back to a
   NumPy/SciPy FFT implementation of the same regularized iteration.

The iteration count is deliberately short; early stopping acts as an additional
regularizer and keeps the whole process well under the strict 0.5-second limit.

## Dependencies

- Python 3 with NumPy (required).
- SciPy is imported only for the nonseparable-PSF fallback.
- A C compiler is optional at build time. `build.sh` recompiles `solver.c` when
  `gcc` is available; a prebuilt Linux x86-64 `solver.so` is also included.

No network access, downloaded packages, or trained weights are used.

## Build and run

```sh
./build.sh
python3 -I -B solution.py INPUT.npz OUTPUT.npy
```

The output is a finite nonnegative float32 `.npy` array with the input shape.
