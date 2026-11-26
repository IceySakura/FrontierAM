# Long-run covariance estimation from a single MCMC trajectory

## Task

Given one trajectory `x` of shape `(M, d)` (rows = sampling time, columns = the
components of the sampled vector), estimate the *long-run covariance*

```
H = sum_{k=-inf}^{+inf} Gamma(k),      Gamma(k) = E[(x_t - mu)(x_{t+k} - mu)^T]
```

i.e. the matrix such that `H / M` estimates the covariance of the mean of `M`
samples and `sqrt(H_jj / M)` estimates the standard error of the `j`-th mean.
The submitted matrix is `H`.

## Method

The estimate combines a non-parametric and a model based ingredient, both
computed from the given trajectory only (no model of the sampling scheme is
assumed, no lookup table, no state carried between calls).

### 1. Non-parametric ingredient

1. **Centring.** The data are centred by their sample mean.  (The task allows
   an arbitrary translation of the coordinates, so centring is mandatory.)

2. **Autocovariances.** The matrix valued empirical autocovariances

   ```
   r[k] = (1/M) sum_{t=1}^{M-k} (x_t - x_bar)(x_{t+k} - x_bar)^T,   k = 0..K_max
   ```

   are computed with FFT based cross-correlations (all `d^2` pairs at once).
   `r[0]` estimates the marginal covariance, `r[k]` the lag-`k` covariance.

3. **Whitening.** The lag matrices `lam[k] = (r[k] + r[k]^T)/2` are divided by
   `r[0]^{1/2}` on both sides.  In that metric the sampling noise of the
   autocovariance estimates is (to first order) the same for every lag and
   every coordinate, so one scalar noise level suffices for the whole matrix
   valued series.

4. **Noise level.** For a stationary series the sampling covariance of the lag
   `k` estimate is

   ```
   Var(r[k]) ~ (1/M) [ G0 G0^T + 2 sum_j G(j) G(j)^T ]
   ```

   so the mean squared norm of the whitened noise is `(d^2 + 2 A)/M`, where
   `A = sum_j ||R(j)||_F^2` is estimated on the lags that are still resolved.

5. **Truncation.** Lags are kept while the whitened autocovariance stays above
   `2` times that noise level; a lag is dropped only when it stays below the
   level for two consecutive lags.  This guards against a single unlucky
   fluctuation and against oscillating autocovariances, for which the pair
   sums of Geyer's sequence turn negative although the autocovariance is still
   large.

6. **Flat-top window.** The weights are

   ```
   w(k) = 1                      for k <= K
   w(k) = (1 - (k-K)/(K2-K))^2   for K < k <= K2 = 2 K
   ```

   i.e. the hard cut at `K` is replaced by a quadratic taper.  The taper
   averages over the remaining lags, which removes the phase sensitivity of a
   hard cut.  This matters when the autocovariance oscillates (mixed sign
   eigenvalues of the dynamics): there a hard cut can leave a large residual
   that depends on where the cut is placed, while the tapered window is
   stable.  For a monotone autocovariance the taper costs almost nothing.

   ```
   H_np = r[0] + sum_{k=1}^{K2} w(k) (r[k] + r[k]^T)
   ```

### 2. Model based ingredient

The long-run covariance is a functional of the autocovariance function only, so
any model that reproduces the autocovariances well also gives a good long-run
covariance - with a much smaller sampling error than the raw autocovariance
sum.

7. A VAR(p) (`p = 1..3`, order chosen by BIC) is fitted by least squares.  Its
   companion system `F` and innovation covariance `Q` give the stationary
   covariance `Sig` (discrete Lyapunov equation) and hence the long-run
   covariance of the companion system

   ```
   Slr = Sig + F (I-F)^{-1} Sig + Sig (I-F^T)^{-1} F^T
   ```

   of which the upper-left `d x d` block is the estimate `H_model`.  Fits
   whose companion system is not stable (an eigenvalue too close to one) are
   discarded because their implied long-run covariance is numerically
   unreliable.

### 3. Combining the two

8. The weight `w` of the model based estimate is one when the fitted model
   reproduces the empirical autocovariance function (a chi-square like
   comparison on the lags that are above the noise) and when the residuals of
   the fit are white (a mean squared residual autocorrelation close to one);
   otherwise it is reduced.  The final answer is

   ```
   H = (1 - w) H_np + w H_model
   ```

   symmetrised and projected onto the cone of positive semi-definite matrices
   (zero eigenvalues are allowed by the task).

## Files

| file | role |
|---|---|
| `run.sh` | entry point: `./run.sh INPUT.npy OUTPUT.npy` |
| `main.py` | reads the `.npy` input, writes the `.npy` output |
| `lrv.py` | the estimator itself (NumPy only) |
| `build.sh` | optional; there is nothing to compile, it only self-checks the sources |

## Dependencies

Python 3 and NumPy only (both are present in the offline evaluation image).
No SciPy, no network access, no compilation step is required.

## How to run

```sh
./run.sh /absolute/input/input.npy /absolute/output/pred.npy
```

The output is a C-ordered little-endian `float64` array of shape `(d, d)`,
finite, symmetric and positive semi-definite.  The whole run (including the
interpreter start-up) takes well under a second for `M = 80000`, `d = 6`.

Self-check on the two development trajectories:

```sh
./run.sh ../../samples/01/input.npy /tmp/pred.npy
python3 ../../samples/check.py ../../samples/01/reference.npy /tmp/pred.npy
```
