# 3-D fluorescence deconvolver

Build and run from this directory:

```sh
./build.sh
python3 -I -B solution.py /absolute/path/input.npz /absolute/path/output.npy
```

The build is offline and takes no arguments. The installed dependencies are
Python 3, NumPy, a C++17 compiler (`g++`), and the single-precision FFTW 3
headers and library. The native solver uses AVX2 and FMA on the supplied
Linux amd64 host. It uses the inherited CPU affinity and does not create
worker threads.

The solver estimates the noise regime from low-signal blocks. A
nonnegative, sparsity-regularized inverse provides a guide for locating
finite tubular ridges; matched filters find compact spots, and robust
slice statistics find extended sheets. The solver fits these shapes
jointly in observation space, accounting for the spatially varying
Poisson variance, background Gaussian noise, and the point mass from
clipping at zero. It compares nearby spot and tube hypotheses, and
averages over position and width uncertainty for compact spots.

Both the forward operator and its adjoint use zero extension and a centered
crop. Separable PSFs use direct passes. For other PSFs, spot fitting
contracts the full kernel with the spot's one-dimensional factors; other
convolutions use padded FFTW transforms. When the shape model leaves
substantial evidence unexplained, the solver can use a regularized
nonnegative inverse for the residual or the whole volume. The absolute
monotonic deadline governs optional refinement, with time reserved for
writing the array and exiting.

All solver code was written for this task, based on standard convolution,
Poisson likelihood, and nonlinear least-squares methods. The shape and
noise priors were informed by the public development examples; there are
no trained weights, embedded example volumes, or external data. Only the
requested input is read at inference.

In a clean local build, all 12 public examples passed the output checks.
Their mean nMSE was 0.01449. Across the public and 72 independently
generated stress inputs, the slowest fresh-process run under the formal
deadline was 1.025 seconds. These are development measurements, not
held-out results.
