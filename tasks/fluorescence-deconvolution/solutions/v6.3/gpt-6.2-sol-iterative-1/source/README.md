# Fluorescence deconvolution

Run `./build.sh` once from the submission, then invoke
`python3 -I -B solution.py INPUT.npz OUTPUT.npy`. The build uses the installed
C++17 compiler and single-precision FFTW 3. The Python entry point uses NumPy
and the standard library. No network access or package downloads are
needed. All submitted code was written for this task. There are no trained
weights, external datasets, or stored reconstructions.

The solver estimates the noise floor from quiet regions and corrects the
mean bias at clipped voxels. With a separable PSF on the nominal grid it
fits a nonnegative mixture of Gaussian spots, finite tubular segments, and
thin sheets. A regularized inverse filter supplies candidate filament directions.
Candidates are fitted and compared in observation space using the supplied
PSF, a signal-dependent noise model, and a complexity penalty. Spot and
filament estimates are refined by alternating component updates. Broad
landmarks may be split into jointly fitted pairs when the observation
supports two emitters. If a fit is dominated by atypically broad spots, a
second search can try alternative filament directions; it replaces the
first fit only if the penalized data fit improves. The final refinement
uses the clipped Poisson-Gaussian score when the noise estimates support
it, and averages the spot estimates over local position and width uncertainty.

The forward operator is centered convolution with zero exterior values.
The native solver uses three one-dimensional convolutions for separable
kernels. The generic fallback uses variance-weighted projected gradient
iterations with zero-padded FFTs. All computations are single-threaded.
The solver checks the supplied monotonic deadline and reserves time to close
the NumPy output and exit.
