# 3-D fluorescence deconvolution under a fixed CPU budget

`solution.py` reconstructs a non-negative fluorescence volume (32x32x32 in the
formal cases) from a blurred/noisy observation and its PSF.

```sh
python3 -I -B solution.py INPUT.npz OUTPUT.npy
```

The input `.npz` must contain exactly `observed` (Z,Y,X, float32) and `psf`
(odd sided, non-negative, sums to one).  The program writes a float32 array of
the same shape as `observed` to `OUTPUT.npy`.

## Forward model

`y = H x`, where `H` is the zero-extension `same` convolution with the PSF
(cropped to the input grid), followed by Poisson and Gaussian noise and
clipping at zero.  Both `H` and its adjoint are implemented exactly with
rFFTs: the zero padded signal is convolved linearly with the (flipped) kernel
on a grid of size `n + k - 1` per axis, which is exactly the length of the
linear convolution and is therefore free of wrap-around.  The crop offsets of
the forward and of the adjoint operator implement the zero extension of the
two operators (`<Hx, y> == <x, H^T y>` was verified numerically).  When the
PSF factorises into three 1-D kernels the very same operators are evaluated
with three 1-D passes instead (see the implementation notes).

## Algorithm

The noise parameters are **not** assumed - they are identified from the
current observation, and every other quantity of the method is derived from
the observation and the PSF as well, so that one and the same algorithm works
for all structure types, PSF widths and noise regimes.

1. **Noise identification.**  A short non-negative Tikhonov reconstruction
   (FISTA, ~10 iterations) gives a first model `b = H x` of the blurred image.
   The variance of the residual `y - b` is regressed against `b`, which
   recovers the Poisson gain `g` of the model `var(y|b) = g*b + sigma^2`
   (within a few percent of the truth on the development cases).  The read
   noise level `sigma` is obtained from the background samples of the
   observation, which are clipped Gaussian noise: for a half-normal sample
   `E[y^2 | y > 0] = sigma^2`.  The reconstruction is insensitive to a 25%
   error in `sigma` (checked on the development cases), so the estimate only
   has to be roughly right.

2. **Reconstruction.**  The hidden volume is the minimiser of

   ```
   min_{x >= 0}   1/2 * || H x - y ||_2^2  +  lam * TV(x)  +  mu * ||x||_1
   ```

   * `TV` is the total variation of the volume; it is handled with the
     standard IRLS re-weighting (each sub-problem is then quadratic) over two
     re-weighting sweeps.
   * `||x||_1` is handled by its exact proximal operator (a soft threshold),
     which on the non-negative orthant also enforces the non-negativity.
   * The whole problem is solved with FISTA (accelerated projected gradient),
     two IRLS sweeps of up to 40 inner iterations each.

   The data term is deliberately *unweighted*.  A Poisson-Gaussian weighting
   (`w_i = 1/(g b_i + sigma^2)`) puts a very large weight on the signal free
   background, where the observation is pure read noise; the fit then follows
   that noise and leaves a haze over the whole volume.  That haze is the
   dominant error term for the sparse cases, and the plain quadratic term
   (plus the L1 term, which suppresses the background) was markedly more
   accurate on every development stratum (0.147 versus 0.268 mean nMSE).

3. **Adaptive regularisation.**  The weight of the TV term follows from the
   identified noise level, from the width of the PSF and from the strength of
   the observed signal:

   ```
   lam = clip( sigma^1.567 * width(PSF)^-1.598 * (p99(obs)/sigma + 0.5)^0.871 ,
                0.0006, 0.014 )
   lam = lam^0.75 * 0.0025^0.25          # shrink towards the default weight
   mu  = 0.7 * sigma
   ```

   `width(PSF)` is the geometric mean of the per-axis second moments of the
   PSF, i.e. how wide the instrument blur is.  A wider blur has to be
   inverted further and its structure content lives at lower spatial
   frequencies, so it needs a different amount of regularisation; a bright,
   dense volume carries a lot of total variation and can afford a stronger
   prior, while a sparse volume is penalised relatively harder by the TV term
   and needs a smaller weight.  The exponents were fitted on the public
   development examples (leave-one-out cross validated: 0.114 nMSE against
   0.120 for a two-parameter rule and 0.131 for a fixed weight) and the value
   is clipped to a safe band and shrunk towards the fixed default, so the
   rule degrades gracefully for PSFs or noise levels outside the fitted
   range.

4. **Shrinkage compensation.**  The TV and L1 terms shrink the recovered
   amplitudes (the L1 term subtracts a constant `mu` from every voxel and the
   TV term smooths the peaks).  A small gain `1 + 4.5*sigma` compensates that
   shrinkage; it was fitted on the development examples, where the optimum is
   flat (a gain of 1.03 for the low-noise and 1.11 for the high-noise regime).

## Results on the public development examples

Mean nMSE over the twelve public examples (lower is better):

| variant | mean nMSE |
| --- | --- |
| observed image (no deconvolution) | 0.508 |
| blurred truth (no deconvolution) | 0.355 |
| non-negative Tikhonov + background shrinkage | 0.199 |
| + Poisson-Gaussian weighted data term | 0.268 (worse: background haze) |
| + IRLS-TV | 0.167 |
| + L1 term instead of post-hoc shrinkage | 0.147 |
| + adaptive TV weight | 0.135 |
| + two IRLS sweeps instead of three | 0.121 |
| + shrinkage compensation (submitted) | 0.113 |

The public evaluation of the submitted version reports an nMSE ratio of
0.576 against the accelerated reference (score 71.2/100): the sparse "point"
stratum reaches 0.39, the boundary stratum 0.68, and the dense "mixed" and
"tubular" strata are at parity (0.99 and 1.00).

## Implementation notes

* The PSFs of this task factorise into three 1-D kernels.  When that is the
  case (it is verified numerically before the fast path is taken) the
  convolution is done with three 1-D passes instead of two 3-D FFTs, which
  avoids importing `scipy.fft` and saves ~0.13 s of process start-up.  For a
  general PSF the exact rFFT operator is used instead.
* One FISTA iteration costs two convolutions (~1.8 ms).  Convergence was
  checked on the development cases: ~80 iterations are enough, which is what
  the solver runs (40 per IRLS sweep).
* `DECONV_DEADLINE_MONOTONIC` is honoured: the iteration counts are derived
  from the wall clock time that is still available and every loop checks the
  clock, so the program always writes a valid output and exits normally
  before the deadline (the reconstruction quality degrades smoothly if the
  budget is smaller - verified for deadlines down to 0.25 s).
* A self-imposed cap of 0.42 s (wall clock, from process start) keeps a
  comfortable margin below the 1.25 s hard limit; the measured end-to-end
  runtime of the development cases is ~0.32 s (of which ~0.11 s is the
  `numpy` import).  Under heavy CPU contention (two competing processes on
  the same core) the program still finishes in 0.49 s, i.e. it degrades
  gracefully instead of timing out.
* The solver was checked on synthetic inputs with PSFs of 3x3x3 up to
  13x13x13, on volumes of 16^3 up to 32^3, on non-cubic volumes, on a
  non-separable PSF (which exercises the FFT fallback) and on degenerate (all
  zero) inputs: it always produces a finite, non-negative array of the right
  shape.  It was also checked on synthetic volumes at noise levels outside the
  development range (10 to 400 photon counts per unit intensity): the
  reconstruction degrades smoothly with the noise level, as it should.

## Dependencies

Only `numpy` (1.26.4) and `scipy` (1.12.0), both already installed in the
evaluation image (`scipy.fft`, used only for the non-separable fallback, is
the only `scipy` submodule needed).  No
weights, no data files and no network access are used, so there is nothing to
build and `build.sh` is not provided.

## Files

* `solution.py` - the complete solver (a single file, no auxiliary assets).
