# Offline Fluorescence Deconvolution

Build once, then run:

```sh
./build.sh
python3 -I -B solution.py INPUT.npz OUTPUT.npy
```

## Method

The solver obtains a nonnegative, boundary-corrected, smoothed Richardson-Lucy
estimate, then detects coherent Gaussian points, straight Gaussian-profile
tubes, and axis-aligned sheets. It jointly fits their parameters using the
supplied zero-extended convolution operator. The final objective marginalizes
Poisson counts and accounts for Gaussian read noise and censoring at zero.
Shape priors regularize poorly identified parameters; residual significance
and likelihood improvement control point selection. Point reconstructions
include a small localization-uncertainty correction.
Coherent residuals of the forward model trigger a local blend with the
iterative estimate, protecting against structures not described well by the
geometric model.

Noise-regime calibration and shape priors were developed using the twelve
public examples. Additional generated geometric scenes and new noise draws
were used for development validation. No example volumes, stored answers,
external datasets, or trained neural-network weights are included or read
at runtime.

The native backend handles the fixed 32-cubed problem with separable symmetric
PSFs. A general FFT-based iterative fallback handles other PSFs and shapes.
All numerical work is single-threaded and inherits the process CPU affinity.
The supplied absolute monotonic deadline limits refinement; a valid
reconstruction is retained throughout. The default local safety limit also
leaves time for output and interpreter shutdown under the 1.250-second limit
in `problem.md`.

## Dependencies And Sources

- Python 3, NumPy 1.26.4, and SciPy 1.12.0 from the provided offline environment.
- A system C++17 compiler (`g++`) and its standard runtime.
- `solution.py` and `native.cpp` contain the reconstruction implementation.
  The small reverse-communication optimizer driver uses the interface of
  SciPy's installed L-BFGS-B extension, following SciPy 1.12's
  `scipy/optimize/_lbfgsb_py.py`. Its BSD notice is included in
  `SCIPY-LICENSE.txt`. There are no downloaded or vendored binary dependencies.
- `build.sh` compiles `native.cpp` locally into `native.so`; it does not access
  the network.

The output is a finite, nonnegative float32 NumPy array with the input shape.
