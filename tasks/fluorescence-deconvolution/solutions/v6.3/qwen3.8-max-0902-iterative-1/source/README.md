# 3D Fluorescence Deconvolution under a Fixed CPU Budget

    python3 -I -B solution.py INPUT.npz OUTPUT.npy

Recovers a nonnegative intensity volume from a blurred, Poisson–Gaussian,
zero-clipped observation and its PSF.

**Mean nMSE over the twelve `public-dev` examples: 0.0941.**
**Wall clock per case: 120–133 ms** end-to-end (measured around the whole
`python3 -I -B solution.py …` process), against the 0.500 s limit.

## Dependencies and sources

* **Python 3 standard library** (`math`, `os`, `sys`, `time`) plus **NumPy**.
  NumPy 1.26.4 is already in the image (`requirements.txt`). Nothing is
  downloaded and nothing is compiled, so **no `build.sh` is shipped** and there
  are **no offline assets** — no weights, no data files, no vendored code.
* **scipy is deliberately not imported.** On this machine `import scipy.signal`
  costs ~0.39 s versus ~0.085 s for `import numpy`; that would consume most of
  the budget before any work is done. Everything scipy would have provided
  (FFT convolution, `erf`/`ppf`) is implemented here instead.
* The inverse-normal-CDF is **Acklam's rational approximation** (a standard
  published formula, |relative error| < 1.15e-9), used only to fill a small
  scalar look-up table once at import time.
* All tuned constants were fitted by the author on `public-dev/` with the
  supplied `check.py` as the scorer. No hidden data, private file or external
  service is accessed; each run reads only its own `INPUT.npz`.

## Forward model, measured from `public-dev`

`observed = clip(blur(clean) + noise, 0)`, where `blur` is `same`-mode 3D linear
convolution with **zero extension** outside the grid. Comparing against the
supplied clean volumes, binned by the true blurred signal `s`:

    Var(observed | s) ~= sigma^2 + a*s,      a ~= 1.6 * sigma

with two noise regimes across the examples, `sigma ~= 0.0080` and
`sigma ~= 0.0251` (the Poisson gain is ~82 and ~26, and the read noise is
~0.6 counts in both, i.e. the two regimes differ only in gain). Zero-clipping
makes the background mean `sigma/sqrt(2*pi) ~= 0.399*sigma` rather than 0 —
between 20% and 90% of `mean(observed)` on these examples, so it has to be
removed rather than ignored.

## Algorithm

**1. Exact operator, in Kronecker form.** The task PSFs are anisotropic
Gaussians and are exactly rank-1 separable (measured second/first singular
value ~2e-8 of the 9×9×9 kernel). Hence

    A = Mz (x) My (x) Mx

where each `Ma` is the small banded Toeplitz matrix of the 1D convolution by
the corresponding PSF marginal (each normalised to sum 1, which reproduces `h`
exactly because `sum(h) = 1`). A matvec is then three small BLAS `gemm` calls:
**~0.2 ms at 32^3 in float32 versus ~1.3 ms for the equivalent padded
`rfftn`/`irfftn`** measured here — the ~6x saving is what makes the iteration
counts below affordable. `A^T` uses the transposed matrices and is exact
(verified `<Ax,y> - <x,A^T y>` = 0.0 to the last bit, and `fwd` matches
`scipy.signal.convolve(..., 'same')` to 2.8e-9). Separability is **tested at
runtime** against the supplied PSF (`KronOp.__init__` reconstructs the outer
product and compares); a non-separable PSF raises and `make_op` falls back to
`FftOp`, an exact zero-padded real-FFT implementation of the same operator.

Being a Kronecker product, `A^T A = Gz (x) Gy (x) Gx` is diagonalised by the
Kronecker eigenbasis, so `(A^T A + lam I)^-1` is six matmuls plus an elementwise
divide. That made an *exact* Tikhonov/Wiener solve cost ~2 ms and is how the
ridge baseline below was measured — it is reported as rejected, not shipped.

**2. Noise level from the data alone** (`estimate_sigma`). Because of the
clipping, the smallest strictly positive voxels are background voxels whose
value is `|N(0,sigma)|`, so their `p`-th quantile is `sigma * ppf(0.5 + p*w)`,
where `w = n_positive/n_background >= 0.5` absorbs signal leaking into the tail.
`(sigma, w)` are fitted to five low quantiles (p = 0.02, 0.06, 0.12, 0.25, 0.35)
by relative weighted least squares over a grid of `w`, and `sigma` is taken from
the best-fitting grid point. The grid stops where `0.5 + p_max*w` would leave
the model's domain, so **every grid point is scored against the same set of
quantiles** — this matters: an earlier version let high-`w` points fit only two
quantiles with two free parameters, which drove the residual to zero and
collapsed `sigma` by ~10x on 8 of the 12 examples. Against the true `sigma`
recovered from the clean volumes the estimator lands within 0.89x–1.32x
(geometric mean 1.15x, sd of log-ratio 0.10). Statistics that look more obvious
— high-frequency spectral power, MAD of adjacent differences — fail here,
because clipping is nonlinear and generates broadband content (they were
off by up to 5x and their two regimes overlapped).

**3. Clipping-bias correction.** Subtract `b = sigma/sqrt(2*pi)` and re-clip at
0. On its own this moves mean nMSE from 0.192 to 0.147.

**4. Richardson–Lucy** (Poisson ML, multiplicative, so nonnegativity is free)
with an **iteration count predicted from two observables**:

    n = round(2.0 * B^2.25 * (0.015/sigma)^0.30),   clamped to [5, 45]

where `B` is the PSF's RMS width, `sqrt(sum of the three axial second moments)`
— more blur needs more iterations to undo, more noise needs fewer. This was the
single largest win: the per-case optimal `n` is not a noise effect but a *blur*
effect (the narrow isotropic PSF, B=1.99, wants n≈8 on all four of its examples;
the widest, B=2.99, wants n≈20–40). The rule reaches **0.110** where the best
single fixed count gives 0.131 and choosing `n` per case with the clean volume
in hand gives 0.108 — i.e. it recovers nearly all of the oracle gain from
observables alone. The exponent surface is a broad plateau (0.110–0.111 across
`B^1.75…B^2.75` and a noise exponent of 0.1…0.5).

**5. Boundary-aware firm-threshold shrinkage** (`shrink`): zero below
`0.8*sigma*w`, a linear ramp up to `4.8*sigma*w`, identity above, then a global
gain of 1.06 to undo the shrinkage bias. Implemented branchlessly as
`min((x-t1)+ * t2/(t2-t1), x)`: that ramp has slope `t2/(t2-t1)` and so meets
the identity exactly at `x = t2`, making the expression precisely
zero/ramp/identity with a single temporary. Firm
beats soft thresholding at a uniform threshold (0.1001 vs 0.1099) because soft
biases the bright structures that dominate the metric, and beats hard
thresholding (0.1081) because the ramp keeps it continuous.

The local weight is `w = (max(A^T 1) / (A^T 1))^2`, capped at `4^2`. `A^T 1`
measures how much of the PSF still lands inside the grid, so it falls off near
the faces, edges and corners (down to 0.23–0.31 of its interior value for these
PSFs, i.e. `w` up to ~11–18 at the eight corners). Those voxels are informed by
fewer observations *and* the RL update divides by `A^T 1`, so the reconstruction
is genuinely noisier there and should be thresholded harder. This is worth ~7%
mean nMSE (0.0928 vs 0.1001 with a uniform threshold). **Leave-one-out
refitting over the twelve examples selects the same parameters — or adjacent
grid points — in all twelve folds** (LOO mean 0.0946 vs in-sample 0.0928), and
every held-out case improves or is unchanged, so this is a real effect and not a
fit to particular cases.

## Rejected alternatives

Mean nMSE over the twelve `public-dev` examples, each at its own best
hyper-parameter (lower is better). Everything below was measured, not assumed:

| method | mean nMSE |
|---|---|
| identity (`x = observed`) | 0.498 |
| exact Tikhonov/Wiener, ridge, oracle-tuned lambda | 0.285 |
| plain RL, best fixed iteration count | 0.192 |
| Gaussian FISTA with nonnegativity, best iteration count | 0.206 |
| Gaussian LS + L1 penalty, oracle-tuned mu | 0.406 |
| Poisson RL + L1 penalty, oracle-tuned mu | 0.428 |
| Chambolle–Pock, Poisson KL + TV, oracle-tuned lambda | 0.157 |
| multiplicative RL-TV, oracle-tuned lambda | 0.169 |
| RL + clipping-bias correction, best fixed count | 0.147 |
| iterative (per-round) clipping-bias correction | 0.158 |
| averaging RL over a range of iteration counts | 0.147 |
| RL + soft threshold | 0.130 |
| adaptive n + soft threshold | 0.110 |
| adaptive n + firm shrinkage, uniform threshold | 0.1001 |
| **+ boundary-aware weight (shipped)** | **0.0941** |

Two findings shaped the design. First, *total variation never won at a global
strength*: it helped the noisiest point-source volumes (0.24 vs RL's 0.35) and
hurt the easy ones (0.09 vs 0.03), and no global lambda beat plain early-stopped
RL — capturing its benefit would need a per-case lambda the data does not
reliably supply. Second, *L1 penalties converge to over-sparse solutions* whose
amplitude bias costs more MSE than the noise they remove. Thresholding *inside*
the RL loop is catastrophic (best 0.96) because it destroys the flux
conservation that makes RL well behaved.

A discrepancy-principle stopping rule was also tried and rejected: the censored
chi-square per degree of freedom is already below 1 at n=6 and nearly flat
afterwards, so its crossing point carries almost no information about the
optimal `n` (best achievable that way 0.176 vs 0.131 for a fixed count).
Fitting `n` from PSF width instead works because blur, not noise, is what sets
the useful iteration count.

## Timing and safety

* ~110 ms of the 120–133 ms is interpreter start plus `import numpy`; the
  reconstruction itself is ~10–20 ms. Margin against the 0.5 s limit is ~4x.
* The solve must return by `min(DECONV_DEADLINE_MONOTONIC - 0.09 s,
  start + 0.33 s)`, falling back to `TIME_BUDGET_SECONDS` and then to 0.5 s if
  the deadline variable is absent or unparsable, and never to less than
  `start + 0.02 s`. RL re-checks `time.monotonic()` every 4 iterations, so an
  unexpectedly slow machine degrades to fewer iterations instead of overrunning.
  If the deadline is already in the past on entry, the clamp still yields a
  valid (less converged) answer.
* **No CPU number is hard-coded** and nothing touches affinity, processes,
  threads or the network; the inherited affinity mask is used as-is. The BLAS
  thread-pool variables are set to 1 *before* `import numpy` (via
  `os.environ.setdefault`, so an explicit environment value is respected) to
  honour the single-threaded requirement even if the grading core exposes a
  wider mask.
* `main` wraps load + solve in `try/except`; on any failure it writes a
  correctly shaped, nonnegative, finite array rather than nothing. The result is
  always passed through `nan_to_num`, `maximum(., 0)` and a shape check before
  `np.save`, and the process exits normally on every path.
* Extra guards for degenerate input: the bias shift is capped at half of
  `max(observed)`, and if shrinkage would zero the whole volume the unshrunk
  reconstruction is returned instead. Verified on all-zero, constant,
  delta-PSF, non-separable-PSF, non-cubic (20,16,20) and larger (48^3) inputs:
  all produce finite, nonnegative, correctly shaped `float32` output in
  120–170 ms. `boundary_weight` also returns all-ones if `A^T 1` is degenerate,
  and the `w` cap keeps a very wide PSF from producing an unbounded threshold.
* Output is `float32`, shape `(Z,Y,X)`, finite, nonnegative, written with
  `np.save` and no pickle.

## Reproducing

```sh
for i in 01 02 03 04 05 06 07 08 09 10 11 12; do
  TIME_BUDGET_SECONDS=0.5 python3 -I -B solution.py \
      ../public-dev/case_$i.npz /tmp/pred_$i.npy
  python3 -I -B ../public-dev/check.py ../public-dev/case_$i.npz \
      ../public-dev/case_${i}_truth.npy /tmp/pred_$i.npy
done
```

Per-case nMSE: 0.0231 0.0769 0.0262 0.1977 0.0922 0.0991 0.0432 0.0814 0.0203
0.2373 0.0895 0.1424 — mean **0.0941**.
