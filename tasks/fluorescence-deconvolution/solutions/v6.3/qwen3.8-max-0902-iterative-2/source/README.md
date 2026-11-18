# 3D fluorescence deconvolution under a fixed CPU budget

Single-file solver: `solution.py`. Invoked as

```sh
python3 -I -B solution.py /abs/raw.npz /abs/pred.npy
```

No `build.sh` is needed — the submission is pure Python + NumPy with no
compiled artefacts, no weights and no network access.

## Dependencies and sources

* Python 3.11 standard library (`os`, `sys`, `time`).
* `numpy` (1.26.4, already present in the image). No SciPy, no torch.
* All code in `solution.py` was written for this task. No third-party code,
  no pretrained weights, no data files.

## Forward model

The observation is modelled as

```
y = clip( Poisson(g * (A t)) / g + N(0, s^2), 0 )
```

where `A` is the `same` 3D convolution with the supplied PSF (zero extension
outside the grid). `g` (photon gain) and `s` (additive/read noise) are not
supplied; only `s` is needed and it is estimated from the data.

## Algorithm

1. **Operator.** `A` and its adjoint `A^T` are applied separably when the PSF
   is numerically rank-1 separable — true to float32 precision for the
   Gaussian PSFs of this task (relative rank-1 residual ≈ 3e-8), which makes
   each application ~1.7 ms instead of ~2.5 ms for a zero-padded real FFT.
   A non-separable PSF transparently falls back to the FFT path, so the code
   is not tied to the separable structure.
2. **Noise estimation.** Voxels whose true signal is ≈ 0 are observed as
   `clip(N(0,s),0)`, i.e. a half-normal with a point mass at zero. With
   `f0 = P(y = 0)`, the probability mass in `(0, u]` equals `0.5 * f0` exactly
   when `erf(u/(s*sqrt(2))) = 0.5`, i.e. `u = 0.6745 * s`. So
   `s_hat = quantile(y, 1.5*f0)/0.6745`. On the twelve public examples this
   recovers `s = 0.008` as `0.0073…0.0077` and `s = 0.025` as `0.023…0.025`.
   If a volume has essentially no clipped voxels, a robust high-pass MAD
   estimate is used instead.
3. **Reconstruction.** Flux-normalised Richardson–Lucy with an additive-noise
   offset in the denominator:

   ```
   x <- x * A^T( y / (A x + beta) ) / (A^T 1)^gamma ,   beta = 0.62 * s_hat
   ```

   `beta` stops the iteration from trying to explain the rectified read noise
   as signal. `A^T 1 < 1` within one PSF radius of the grid boundary because
   the forward model extends the volume with zeros, so those voxels converge
   much more slowly; dividing by `(A^T 1)^gamma` preconditions them. The full
   correction `gamma = 1` measurably overshoots the truth on the outermost
   shell (mean estimate 0.087 against a mean truth of 0.039 on the public
   boundary examples); `gamma = 0` leaves it far too dark. `gamma = 0.84` is
   the measured optimum and was the single largest improvement found, worth
   13 % of the total nMSE.
4. **Initialisation.** The iteration starts from the observation smoothed with
   a Gaussian of width 2.0 rather than from a constant; at a fixed iteration
   count this is worth ~1 %.
5. **Stopping.** The number of iterations follows the estimated noise level,
   `N = clip(round(18 * (0.008/s_hat)^0.22), 8, 40)` — about 18 iterations in
   the quiet regime, about 14 in the noisy one. Early stopping *is* the
   regulariser here: explicit L1 and TV penalties in the update, and TV via a
   one-step-late denominator, were all measurably worse at every strength and
   iteration count tried.
6. **Post-processing.** The early-stopped iteration systematically loses flux
   (measured `sum(x)/sum(t)` of 0.68–0.98), so the estimate is compressed with
   `x_hat = m^0.1 * min(x,1)^0.9` where `m = max(min(x,1))`. The clean volumes
   are normalised to a maximum of 1, so values are clipped at 1 first.

## Timing and safety

Measured full-process wall clock (launch → exit, `python3 -I -B`, including
interpreter start, the NumPy import, decompressing the input, the solve and
writing the output) is **≈150 ms median / 155 ms worst case** on the
development core for the public examples, i.e. a factor ≥3 margin under the
0.500 s budget named in `public-dev/README.md`.

Guards, because a single failed case scores zero:

* `DECONV_DEADLINE_MONOTONIC` and `TIME_BUDGET_SECONDS` are honoured, and a
  self-imposed cap (`_START + 0.26 s` for the solve, 40 ms reserved for the
  write) is applied on top, so the process finishes well inside the limit even
  if the environment variables are absent or inconsistent.
* The iteration loop re-checks the monotonic clock every iteration and stops
  with the current iterate.
* Thread-count environment variables are pinned to 1 before NumPy is imported.
* The whole solve is wrapped; any exception falls back to the (valid,
  non-negative) observation itself, and if even loading fails, to zeros of the
  input shape.
* Output is written to a temporary file and `os.replace`d into place, then the
  file is flushed and closed, so the prediction on disk is always complete.
  A direct-write fallback covers the case where the temporary file cannot be
  created.
* The result is cast to float32, `nan_to_num`ed and clipped at zero before
  writing, satisfying the finiteness/non-negativity/dtype/no-pickle contract.
* No CPU index is hard-coded; no affinity or scheduler call is made.

## Tuning data

Hyper-parameters (`beta 0.62`, `gamma 0.84`, `init 2.0`, `N_ref 18`,
`N_exp 0.22`, `p 0.9`, `clip 1.0`) were chosen on

* the 12 public examples, and
* 48 synthetic validation cases built in development from those same clean
  volumes: flips / 90° rotations / translations of the truth, PSF sigmas
  perturbed by up to ±25 % (and one isotropised variant per case), gain
  perturbed by ±35 % and additive noise by ±15 %, then re-noised through the
  forward model above with fresh seeds.

Only the second set has independent noise realisations, so the parameters were
selected on it and cross-checked on the first; the two agree to within 2 %.
On the twelve public examples the solver reaches a mean nMSE of **0.0832**
(0.0202 best case, 0.1623 worst); on the 48 synthetic validation cases,
0.0823. For scale, returning the observation gives 0.498 and a plain
Richardson–Lucy without the noise offset, boundary preconditioner or
compression gives 0.184 on the same twelve examples.

Alternatives that were measured and rejected: Wiener/Tikhonov (nMSE ≈ 0.25–1.0,
never better than returning the observation), L1 or TV regularisation inside
the RL update, a spatially capped boundary correction, two-stage and annealed
`beta` schedules, local LMMSE / sqrt-domain shrinkage, ensembling over
iteration counts, and flux-matched global rescaling.
