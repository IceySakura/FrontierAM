# Polarization demosaicking

`solution.reconstruct(raw)` takes a 1024-by-1024 NumPy `uint8` mosaic
and returns a `(4, 1024, 1024)` `float32` array in 0, 45, 90, 135 degree
order. It preserves the exact observed intensity at each measured location.

The decoder uses the period-two modulation identity
`raw = L + (-1)^x X + (-1)^y Y + (-1)^(x+y) D`. Here `L` is the common
intensity, and `X`, `Y`, and `D` are signed contrasts between directions.
After estimating brightness-dependent noise from the mosaic, the decoder
applies a square-root intensity transform. It fits nonnegative, locally
directional spectra for the four components and a noise floor in overlapping
96-by-96 tiles. A Wiener allocation with a cross-carrier covariance
correction separates the components. Synthesis uses the same tapers as
analysis, with squared tapers summing to one at overlaps.

Where angle samples have weaker local agreement, the decoder gives more
weight to independent denoised cubic interpolation of each sampling lattice.
It then restores the measured samples.

Only NumPy and SciPy are needed. There is no build, model file, network
access, training, or persistent scene state. The reconstruction is
deterministic and the FFT uses one worker.

To run the development checker:

```bash
python /workspace/public-dev/check.py /workspace/submission/solution.py
```
