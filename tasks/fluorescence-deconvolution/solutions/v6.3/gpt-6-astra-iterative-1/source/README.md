# Fluorescence Deconvolution

Build once with `./build.sh`, then run:

```sh
python3 -I -B solution.py INPUT.npz OUTPUT.npy
```

The solver combines a voxel-wise reconstruction with a fitted geometric
regularizer for smooth fluorescent points, tubes, and sheets. It estimates
background noise using leave-one-out neighborhoods, initializes with
noise-corrected Richardson-Lucy iterations, detects structures in the current
observation, and fits their parameters using the supplied forward operator.
Residual matched filtering finds additional point components, and significant
residual anisotropy can trigger a penalized split/refit. Localization uncertainty
is averaged over a small Gaussian-component ensemble. Weak geometric and
intensity priors regularize poorly determined parameters. A data-consistency
check falls back toward the voxel estimate when the geometric model fits poorly.
All convolutions use zero extension and the specified same-grid crop.
Nonseparable PSFs use the FFT-based voxel solver.

The native C++ routines implement separable convolution, geometric rendering,
analytic derivatives, and line scoring. Numerical libraries are single-threaded.
Bound-constrained damped least squares uses analytic derivatives. Iterative
stages check the absolute monotonic deadline and retain a usable estimate.
Parameter-count-dependent reserves and a conservative internal time cap leave
time for output and normal process exit.

Dependencies are the installed Python 3, NumPy 1.26.4, SciPy 1.12.0, and g++.
The build requires no network access or package installation.

All source is original code developed for this task. The supplied public
examples were used for algorithm development and calibration of weak geometric
and intensity priors. Independently generated Gaussian structures were used for
additional validation. There are no pretrained weights, copied volumes, external
data, or cross-case caches or state. Runtime data access is limited to the
specified input and this submission; installed numerical libraries supply the
remaining dependencies.
