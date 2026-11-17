# Task name

Three-Dimensional Fluorescence Deconvolution under a Fixed CPU Budget

# Problem description

A fluorescence microscope produces a blurred, noisy three-dimensional intensity volume. You receive that volume and its point-spread function (PSF). Recover a nonnegative intensity volume on the same grid. Cases contain tubular, point, mixed and boundary-adjacent structures, isotropic or anisotropic PSFs, and two noise regimes. The clean volume is hidden. The task measures reconstruction quality under a CPU limit.

The forward operator is three-dimensional linear convolution with zero extension outside the clean grid, cropped to the input grid (`same` convolution with a centered odd PSF), followed by Poisson and Gaussian noise and clipping at zero. Input intensities and output intensities use the same scale. Formal inputs are 32×32×32; the supplied PSF is 9×9×9. Noise parameters are not provided.

# Algorithm objective

Implement one reusable algorithm that recovers the hidden intensity as accurately as possible across the cases. Determine stopping and per-case parameters from the current observation, PSF and resource limit. You may use numerical libraries or ship offline weights and dependencies.

# Algorithm constraints

Linux x86-64, one logical CPU, single-threaded numerical libraries, 8 GiB RAM, no swap and no network. **Every case has a hard wall-clock limit of 1.250 seconds.** This includes submission process/interpreter startup, imports, reading input, computing, writing and closing output, and normal exit. Container orchestration and trusted grading are outside the timed interval. Each case starts a new process. `SEED=1`, `TIME_BUDGET_SECONDS=0.5` and an absolute `DECONV_DEADLINE_MONOTONIC` are supplied. The monotonic deadline refers to Python's `time.monotonic()` clock. The process must exit normally within the limit even if it has already written an output. A timeout on any case makes the entire submission score zero.

Use only the current input and your own submission. Do not access private files, another case, the evaluator, external paths or network services. Cross-case communication and cached hidden answers are prohibited.

# Delivery requirements

Put `solution.py`, `README.md` and any offline dependencies in your submission directory. At most 4096 regular files and 128 MiB are allowed; symbolic links and hard links are not supported. An optional executable `build.sh` runs once before evaluation, with no arguments, under the same CPU/memory limits for at most 300 seconds. State dependencies and code/weight sources in README.

Each case invokes:

```sh
python3 -I -B solution.py /absolute/path/raw.npz /absolute/path/pred.npy
```

The input `.npz` contains exactly `observed` and `psf`. Both are finite, nonnegative float32 arrays. `observed` has shape `(Z,Y,X)`; `psf` has three odd side lengths, is centered, and sums to one within absolute tolerance 1e-5. Write one `.npy` array of shape `(Z,Y,X)`, dtype float32 or float64, finite and nonnegative, with no pickle. Flush/close it and exit normally by the deadline. Standard output is for diagnostics only.

# Evaluation metric

The evaluator computes each case's nMSE using float64 arithmetic:

`mean((estimate-clean)**2) / max(mean(clean**2), 1e-12)`.

All voxels, including boundaries, count. The metric E is the equal-weight mean over 96 cases, comprising 16 equally populated structure/PSF/noise strata. Lower is better. One seed/run is evaluated per case.

A fixed basic reference defines U and a fixed independently implemented accelerated reference defines W, with U>W>0. Score is 0 for E≥U; `5*(U-E)/(U-W)` for W≤E<U; and `10-5*E/W` for 0≤E<W. Zero error is the ideal value. Any submission-caused build or case failure, including timeout, out-of-memory, abnormal exit, missing output, wrong dtype/shape, nonfinite or negative values, makes the entire submission score zero. Failed cases are not removed from the batch. Infrastructure errors are reported without a score.

# Development materials and data entry

`public-dev/` contains twelve fixed example inputs, clean volumes, manifest and a checker. It provides no generator, private seeds or hidden cases. Run your program on an example, then use:

```sh
python3 public-dev/check.py public-dev/case_01.npz public-dev/case_01_truth.npy pred.npy
```

Public checks report raw error and do not establish a formal score. Deliver the source after checking at least one public example.
