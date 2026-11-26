# 3-D fluorescence deconvolution under a fixed CPU budget

`solution.py` reconstructs a nonnegative 32x32x32 intensity volume from a
blurred, noisy observation and its point spread function.

```sh
python3 -I -B solution.py /abs/path/raw.npz /abs/path/pred.npy
```

## Method

**Forward model.** The observation is a `same` convolution with zero extension
outside the grid, i.e. `A x = full[4:36]` where `full = linear_conv(x, psf)`
for a 9x9x9 PSF. The adjoint `A^T` used by the solver is the exact transpose
of that operator (verified to ~1e-13 against `scipy.signal.fftconvolve`).

**Estimator.** The observation noise is signal dependent: the residuals of the
supplied cases satisfy `Var(y_i) ~ sigma^2 + c * (A x)_i` (a Poisson-like
counting term plus an additive Gaussian term; the two noise regimes of the
task differ by roughly 3x in `sigma`). The maximum-likelihood estimator for
that model is the Richardson-Lucy multiplicative update

```
x <- x * A^T( y / (A x) ) / A^T 1
```

with the boundary-corrected denominator `A^T 1` (the zero extension makes the
column sums smaller than one near the faces of the volume). Plain RL is an
unregularised maximum-likelihood method: its iterates improve for a while and
then start to absorb noise, so *the number of iterations is the regulariser*.
On the twelve public examples the per-case optimum of the iteration count
gives a mean nMSE of 0.161, versus 0.174 for the automatic rule below and
0.498 for the raw observation.

**Choosing the iteration count (no ground truth is used).** The stopping point
is picked by cross validation *on the observed volume itself*: a deterministic
random 10% of the voxels is held out, the multiplicative update is run with
the likelihood restricted to the remaining voxels (the masked column sums
`A^T mask` are used in the denominator), and the prediction error on the held
out voxels is recorded after every iteration. The iteration with the smallest
held-out error is where the implicit smoothing of RL still predicts unseen
data best; the final answer is one RL run over all voxels with that many
iterations. On the public set this reaches a mean nMSE of 0.174 (0.161 for the
oracle iteration count), and it needs no knowledge of the noise parameters.

**Regularisation alternatives that were tried and rejected** (all measured on
the twelve public examples, oracle parameter per case): Tikhonov/gradient
smoothing 0.204, second-difference smoothing 0.213, non-negative L1 0.576,
RL with an explicit quadratic prior 0.213, Poisson-weighted FISTA 0.710,
RL with momentum acceleration 0.279. Plain early-stopped RL wins because its
multiplicative update regularises bright and dark voxels differently, which
matches the heteroscedastic noise.

**Speed.** The PSF of the task is separable; when that is detected the
convolution and its adjoint are applied as three 32x32 matrix products
(~0.3 ms per convolution), otherwise an exact zero-padded FFT operator is
used. Only NumPy is imported (no SciPy), all BLAS threading is disabled, and
the run takes ~0.29 s per case on the development machine (including ~0.10 s
of interpreter start-up and NumPy import).

**Budget handling.** The solver reads `DECONV_DEADLINE_MONOTONIC` and
`TIME_BUDGET_SECONDS` and computes an absolute stop time
(`min(deadline - 0.045 s, start + 0.35 s, start + TIME_BUDGET_SECONDS -
0.045 s)`). The iteration loops check the clock, so the process always exits
normally and in time, writing a valid output even if the budget is nearly
exhausted. If the deadline has already passed the solver falls back to a
minimal run and still writes a valid file.

## Output

A single `.npy` array of shape `(Z, Y, X)`, dtype float32, finite and
nonnegative, written with `np.save` (no pickle).

## Dependencies

* Python 3.11 (standard library only for the solver logic).
* NumPy 1.26.4 - the only third-party import (`numpy`, `numpy.fft`).
* No SciPy, no compiled extensions, no model weights, no network access, no
  build step. `build.sh` is therefore not needed; an empty executable stub is
  included for completeness.

## Code and weight sources

All code in this directory was written for this task; there are no external
code fragments and no pretrained weights. The algorithm (Richardson-Lucy
multiplicative updates with a cross-validated stopping rule) is standard
image-restoration methodology implemented from scratch.

## Public-example check

```sh
python3 solution.py public-dev/case_01.npz /tmp/pred.npy
python3 public-dev/check.py public-dev/case_01.npz public-dev/case_01_truth.npy /tmp/pred.npy
```
