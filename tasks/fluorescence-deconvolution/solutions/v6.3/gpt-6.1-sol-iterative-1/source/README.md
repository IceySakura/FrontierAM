# Fluorescence Deconvolution

Run `./build.sh` once, then:

```
python3 -I -B solution.py INPUT.npz OUTPUT.npy
```

The program first computes a nonnegative, noise-bias-corrected, accelerated
curvature-regularized estimate. It then fits smooth Gaussian spots, capped
Gaussian tubes, or sheet-like structures identified from the current volume.
Point regions receive adaptive multi-spot initialization; compact elongated
regions can use a rotated anisotropic Gaussian. The fit uses the supplied PSF,
zero extension, analytic derivatives, and a clipped Poisson/Gaussian noise
response. Significant residual spots are fitted in a second pass. A residual
check retains the voxel estimate when the structural model does not adequately
explain the observation.

Separable PSFs use three one-dimensional convolutions. Other PSFs use the full
supplied kernel and the voxel method. Computation checks the supplied monotonic
deadline and reserves time to close the output and exit.

Dependencies: Python 3 standard library and the system C/C++ runtime. The
included Linux amd64 libraries are rebuilt offline when `g++` is available.
The launcher selects the AVX2/FMA version only after checking CPU support;
otherwise it uses the generic x86-64 version. All numerical work is
single-threaded. C-order, Fortran-order, and either-endian float32 inputs are
supported.

No packages, network access, weights, or downloaded code are used. All
implementation source was written for this submission.
