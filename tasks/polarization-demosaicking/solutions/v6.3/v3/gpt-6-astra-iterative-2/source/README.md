# Polarization Demosaicking

`solution.py` defines `reconstruct(raw)` for the specified `(1024, 1024)`
`uint8` mosaic. It returns a contiguous `float32` array of shape
`(4, 1024, 1024)`, in the order 0, 45, 90, 135 degrees.
Measured samples are preserved exactly, the input is not modified, and
predictions are clipped to the valid intensity range `[0, 255]`.

The method packs the four mosaic phases onto a half-resolution grid,
computes bilinear interpolation, and adds predicted residuals from a
seven-layer convolutional model with 24 hidden features. Reflection padding
on the raw mosaic preserves the filter phases and supplies boundary
context.

The shared filters in `weights.npz` were trained only on the provided
public development images, with seed 1, geometric and photometric
augmentation, and multi-scale crops. The reconstruction does not enforce
an exact polarization sum identity. No scene identification, reference
images, or cross-scene state is used at inference.

## Offline Execution

Keep `solution.py` and `weights.npz` together. The module locates its
weights relative to its own filename, independently of the working
directory.

Only Python 3.11 and NumPy are required at runtime. No compilation,
downloads, build script, SciPy, or PyTorch import is needed. BLAS is
explicitly restricted to one thread. Convolution uses small strips and
reusable buffers; inference is deterministic.
