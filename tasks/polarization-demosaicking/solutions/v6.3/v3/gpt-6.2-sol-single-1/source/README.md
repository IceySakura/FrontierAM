# Polarization demosaicking

`solution.py` provides `reconstruct(raw)` for a `1024 x 1024` `uint8` mosaic.
It returns an array of shape `(4, 1024, 1024)` and dtype `float32` in the order
0, 45, 90, 135 degrees. Each measured sample is copied back exactly at its
original coordinate and angle.

## Running

No build step is required. The only dependencies are the supplied NumPy and
SciPy. FFTs use one worker, the linear algebra libraries are limited to one
thread, and the program does not change CPU affinity. It runs offline and
does not use any files or state from other scenes.

## Method

1. Estimate relative offsets between angle planes from their cross-phase in
   low spatial frequencies. Refine the translations into a regularized
   affine field with a bounded local correction. Estimate relative contrast
   from cross-spectral agreement after registration.
2. Analyze overlapping `64 x 64` Fourier tiles on a 48-pixel grid. Work in a
   four-component basis with one shared brightness field and three angular
   differences. Fit nonnegative power spectra from the mosaic, using a
   scene-wide estimate to stabilize each local fit. The sum of opposite
   angles is not constrained to be identical for both pairs.
3. Solve a regularized four-alias Wiener system that models the offsets of
   the shared brightness field. Undo the local contrast and offset
   corrections, and combine the tiles with normalized overlap weights.
   A small spatial residual correction controls noise leakage and restores
   detail where the remaining mosaic error is large.

To reproduce the public interface and accuracy check:

```sh
python /workspace/public-dev/check.py /workspace/submission/solution.py
```
