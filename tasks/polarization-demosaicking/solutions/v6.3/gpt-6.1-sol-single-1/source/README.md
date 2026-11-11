# Polarization Demosaicking

`solution.py` exports `reconstruct(raw)`. It returns a contiguous `float32`
array with shape `(4, H, W)`, ordered as 0, 45, 90, and 135 degrees. Input
samples are not modified; measured values are preserved in their channels.

## Method

A separable, phase-correct low-pass filter estimates local angular
differences. These differences transfer the high-resolution spatial detail
from the mosaic to all four channels. A four-layer, 16-feature residual
convolutional filter then corrects interpolation artifacts and noise. It
operates on the half-resolution sensor phases and has 7,692 parameters.
Its zero-DC output preserves constant angular offsets. Local affine regression
against a shared intensity guide refines the missing samples. All filtering
uses reflected boundaries, and final intensities are clipped to `[0, 255]`.

The bundled `weights.npz` contains scene-independent filter parameters fitted
offline using only the two supplied public Wen reference scenes, with spatial
and Stokes-consistent polarization augmentation. No reference images, scene identifiers, lookup
tables, or cross-scene state are used at inference.

## Dependencies And Execution

Python 3.11, NumPy, and SciPy are required. The installed `threadpoolctl` is used
when available to enforce single-threaded BLAS. No compilation, downloads,
PyTorch runtime, or build script is needed.

```python
import numpy as np
from solution import reconstruct

raw = np.load("raw.npy")
prediction = reconstruct(raw)
np.save("prediction.npy", prediction)
```

Keep `solution.py` and `weights.npz` in the same directory. Computation uses
the inherited CPU affinity and does not write any runtime state.

## Sources

The reconstruction and NumPy inference code are original to this submission.
NumPy and SciPy provide array operations and native filtering routines.
PyTorch was used only for offline fitting of the small filters; it is not
imported by the submitted program. The only image data used during fitting
were the public samples identified above.

## Verification

The public interface checker and clean-directory subprocess tests pass.
Full-resolution checks include real scenes, random data, constant images,
checkerboards, stripes, and ramps. The tests also verify read-only input
handling, finite output, channel order, and exact preservation of measured
samples. No hidden-scene results were available during development.
