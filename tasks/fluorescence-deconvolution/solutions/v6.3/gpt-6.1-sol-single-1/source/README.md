# Offline 3-D Deconvolution

## Build And Run

```sh
./build.sh
python3 -I -B solution.py INPUT.npz OUTPUT.npy
```

The build uses the installed `g++` compiler and the C++17 standard library.
Runtime dependencies are the installed Python 3, NumPy, glibc, and libstdc++.
No network, downloads, agent tools, SciPy, or PyTorch are required.

## Method

The observation and PSF determine every reconstruction. The solver estimates
read noise from spatially independent background samples. It uses an exact
zero-extended, cropped convolution and its adjoint, exploiting separability
when supported by the supplied PSF.

A nonnegative, regularized accelerated least-squares reconstruction provides
an initialization and fallback. Gaussian points, finite straight Gaussian
tubes, and smoothly varying Gaussian sheets are detected and fitted directly
through the supplied forward operator. The observation model accounts for
Poisson noise, Gaussian read noise, and clipping at zero. Conservative priors
and source-selection tests limit noise amplification. Significant unexplained
regions retain the numerical reconstruction.

The native code is single-threaded and retains inherited CPU affinity.
Monotonic deadline checks bound iterative work, leaving time for writing,
closing the output, and normal interpreter exit. A NumPy FFT fallback handles
nonseparable PSFs or an unavailable native library. The output is a finite,
nonnegative float32 NumPy array, without pickle.

## Sources

All submitted Python and C++ code is original implementation for this task.
Algorithm settings were developed using the twelve supplied public examples
and standard convolution, likelihood, and constrained-optimization formulas.
There are no trained weights, copied clean volumes, cached answers, external
data files, or cross-case state.
