# Four-Angle Polarization Reconstruction

`solution.py` provides `reconstruct(raw)`. The returned C-contiguous `float32`
array has shape `(4, H, W)` and channel order 0, 45, 90, 135 degrees.
The input is never modified.

## Method

A low-pass pseudo-intensity image supplies a common guide. For each of the
four sensor sublattices, a local least-squares model separates shared image
detail from slowly varying angle-dependent intensity. Orthogonal spatial
polynomials (constant, slopes, and axis curvatures) are removed before
estimating the local gain, reducing bias from polarization gradients and
curvature during texture transfer. A second polynomial smoothing of the gain and
residual coefficients reduces false polarization while preserving smooth
angular intensity changes. These fields are interpolated to the original
coordinates. Coefficient smoothing uses one-sided polynomial fits at the
image boundaries. A regularized measurement back-projection supplies
full-resolution intensity detail to all four channels.

The four channel models are independent: exact ideal Stokes equalities are
not assumed, allowing nonlinear encodings and saturated measurements.
Observed samples are retained exactly. Predictions are clipped to [0, 255].

## Dependencies and Execution

Python 3.11, NumPy 1.26.4, and the installed SciPy 1.12.0 filtering extension.
All processing is single-threaded. No network access, package installation,
compiler, model weights, runtime data files, or build step is required.
The SciPy native extension is loaded directly to avoid importing unrelated
SciPy modules during the one-second process budget.

```python
from solution import reconstruct
pred = reconstruct(raw)
```

Public interface check:

```sh
python3 /workspace/public-dev/check.py /workspace/submission/solution.py
```

## Sources

The reconstruction and interpolation code is original. The method uses
standard local affine guided filtering and Savitzky-Golay polynomial
regression ideas. Numerical filtering is provided by the installed
BSD-licensed SciPy library; array operations are provided by NumPy.
No reference-scene data, learned parameters, or scene-identification
tables are included.
